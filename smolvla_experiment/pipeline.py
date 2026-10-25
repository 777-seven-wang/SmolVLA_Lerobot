# 统一工作流调度器，由根入口 run.py 调用。
# 负责命令解析、实验工作流锁、后台启动、短跑选 batch、预算、续训与批量评测。
# 准备、训练和评测通过独立 Python 子进程执行，精度补丁及重型依赖各自在子进程内生效。
# 报告在当前进程内生成，只处理已经保存的记录，不启动新的训练或仿真。
# 成功后的自动清理仅面向已知短跑目录，正式历史检查点不参与自动删除。

"""Prepare, train, resume, evaluate and report using an explicit experiment profile."""

import argparse
import csv
import fcntl
import math
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from smolvla_experiment.common import (
    EXPERIMENTS,
    ExperimentContext,
    configure_environment,
    read,
)
from smolvla_experiment.records import (
    experiment_record,
    read_step_timing,
    record_evaluation_protocol,
    run_record,
    set_training_values,
    step_timing_path,
    temporary_config,
    training_values,
)

from .train import validate_training_config


# 仅在计时和训练状态均为 completed 时整理短跑产物。
# 预先校验所有短跑名称、路径和链接，精简管理记录后删除 probe_b8/b4/b2 及其指标目录。
# 失败、中断或超时运行返回 False，保留现场；正式 train/checkpoints 不在删除范围。
def compact_completed_run(ctx, run):
    record = run_record(ctx, run)
    document = record.read()
    if (
        document["timing"].get("status") != "completed"
        or document["runtime"].get("status") != "completed"
    ):
        return False
    # 清理范围只含可丢弃的原生短跑输出，正式历史检查点始终保留。
    for name in document["probes"]:
        if name not in {"probe_b8", "probe_b4", "probe_b2"}:
            raise ValueError(f"Unexpected probe name: {name}")
        for suffix in ("", "_metrics"):
            path = run / f"{name}{suffix}"
            ctx.owned(path)
            if path.is_symlink():
                raise ValueError(f"Refusing to remove linked staging directory: {path}")

    # 保留短跑状态、错误和更新校验结论，去掉大型逐参数采样诊断。
    # 正式梯度检查也只保留关键结论，便于长期保存已完成实验的记录。
    def compact(value):
        for probe in value["probes"].values():
            runtime = probe.get("runtime", {})
            gradient = probe.get("gradient_check", {})
            verified = gradient.get("verified_updates", probe.get("verified_updates", False))
            probe.clear()
            probe.update(
                runtime={
                    key: runtime[key]
                    for key in ("step", "status", "elapsed_s", "error")
                    if key in runtime
                },
                verified_updates=verified,
            )
        gradient = value["diagnostics"].get("gradient_check")
        if gradient:
            value["diagnostics"]["gradient_check"] = {
                key: gradient[key]
                for key in (
                    "grad_norm",
                    "trainable_parameters",
                    "total_parameters",
                    "expected_groups",
                    "verified_updates",
                )
                if key in gradient
            }

    record.update(compact)
    for name in document["probes"]:
        for suffix in ("", "_metrics"):
            path = run / f"{name}{suffix}"
            if path.exists():
                shutil.rmtree(path)
    return True


# 构造传给原生训练器的配置字典，不直接启动训练。
# 固定数据划分、随机种子、优化器与验证约定；冻结设置和参数存储方式来自实验配置。
# 新训练的 warmup/衰减随目标 steps 设置，续训不调用此函数重建原有学习率计划。
def config(ctx, output, steps=20000, batch=8):
    split = read(ctx.root / "data/split.json")
    revision = experiment_record(ctx).read()["provenance"]["resources"]["dataset"]["revision"]
    return {
        "dataset": {
            "repo_id": "lerobot/libero",
            "root": str(ctx.root / "data/libero"),
            "revision": revision,
            "episodes": split["episodes"],
            "eval_split": 0.1,
            "video_backend": "pyav",
            "image_transforms": {"enable": False},
        },
        "policy": {
            "path": str(ctx.root / "models/base_spatial"),
            "push_to_hub": False,
            "load_vlm_weights": False,
            **ctx.flags,
            **({"dtype": "float32"} if ctx.profile["parameter_storage"] == "float32" else {}),
            "optimizer_lr": 1e-4,
            "optimizer_grad_clip_norm": 10,
            "scheduler_warmup_steps": max(1, int(steps * 0.05)),
            "scheduler_decay_steps": steps,
            "scheduler_decay_lr": 2.5e-6,
        },
        "output_dir": str(ctx.owned(output)),
        "steps": steps,
        "batch_size": batch,
        "seed": 0,
        "num_workers": 2,
        "prefetch_factor": 2,
        "log_freq": 50,
        "save_freq": min(1000, steps),
        "env_eval_freq": 0,
        "eval_steps": 1000 if steps >= 1000 else 0,
        "max_eval_samples": 1000,
        "accelerator": {"mixed_precision": "bf16", "gradient_accumulation": {"steps": 1}},
        "wandb": {"enable": False},
    }


# 启动一个子进程，把标准输出和错误追加到指定日志，并等待结束。
# 若传入 state，登记 child_pid 和日志路径；非零退出码转为包含日志位置的错误。
def execute(ctx, command, logfile, state=None):
    logfile = ctx.owned(logfile)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("a") as stream:
        process = subprocess.Popen(
            command, cwd=ctx.project_root, stdout=stream, stderr=subprocess.STDOUT
        )
        if state is not None:
            state.update(child_pid=process.pid, log=str(logfile))
            experiment_record(ctx).set("activity", state)
        code = process.wait()
    if code:
        raise RuntimeError(f"Command failed ({code}); see {logfile}")


# 保证同一种实验一次只有一个工作流运行。
# 默认遇到占用立即报错；wait=True 时循环等待，退出上下文无论成功失败都释放锁。
@contextmanager
def workflow_lock(ctx, wait=False):
    ctx.owned(ctx.root / "workflow.lock")
    ctx.root.mkdir(parents=True, exist_ok=True)
    with (ctx.root / "workflow.lock").open("a") as stream:
        while True:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not wait:
                    raise RuntimeError(f"Another workflow is running for {ctx.name}") from None
                time.sleep(2)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


# 列出运行中名称为步数的实际检查点目录，按整数步数排序。
# 排除 last 等符号链接，避免同一模型被重复计数。
def checkpoints(ctx, run):
    return sorted(
        (
            p
            for p in ctx.owned(run / "train/checkpoints").glob("[0-9]*")
            if p.name.isdigit() and p.is_dir() and not p.is_symlink()
        ),
        key=lambda p: int(p.name),
    )


# 返回指定步数的 pretrained_model 目录，并检查模型文件和冻结配置。
# 评测或恢复前先确认模型属于当前实验，避免只凭目录名称使用错误策略。
def checkpoint_path(ctx, run, step):
    policy = ctx.owned(run / "train/checkpoints" / f"{step:06d}" / "pretrained_model")
    if not (policy / "model.safetensors").is_file():
        raise FileNotFoundError(policy / "model.safetensors")
    model = read(policy / "config.json")
    for key, expected in ctx.flags.items():
        if model.get(key) is not expected:
            raise ValueError(f"Checkpoint has incompatible {key} for {ctx.name}")
    return policy


# 校验续训来源并返回恢复步数加 additional_steps 的新目标。
# 只接受同一运行的最新实际检查点，核对保存步数、运行状态及指标／计时历史。
# 最后检查原生训练配置和完整优化器、调度器、随机状态；全过程在修改运行记录前完成。
def continuation_target(ctx, run, checkpoint, additional_steps):
    run, checkpoint = ctx.owned(run), ctx.owned(checkpoint)
    if additional_steps <= 0 or checkpoint.parent != run / "train/checkpoints":
        raise ValueError("Resume requires a checkpoint from the same run and positive steps")
    available = checkpoints(ctx, run)
    if not available or checkpoint != available[-1]:
        raise ValueError("Resume only the latest checkpoint in this run")
    restored = read(checkpoint / "training_state/training_step.json")["step"]
    if (
        int(checkpoint.name) != restored
        or run_record(ctx, run).read()["runtime"].get("step") != restored
    ):
        raise ValueError("Runtime and checkpoint steps differ")
    checkpoint_path(ctx, run, restored)
    for name in ("train", "validation"):
        path = run / "train_metrics" / f"{name}.csv"
        if path.exists():
            with path.open() as stream:
                if any(int(row["steps"]) > restored for row in csv.DictReader(stream)):
                    raise ValueError("Metric history extends past the resume checkpoint")
    timing = read_step_timing(step_timing_path(ctx, run / "train"))
    if timing and timing[-1]["steps"] > restored:
        raise ValueError("Timing history extends past the resume checkpoint")
    source = checkpoint / "pretrained_model/train_config.json"
    validate_training_config(ctx, read(source), resume=True, config_path=source)
    return restored + additional_steps


# 评测指定模型，或在已有完整结果通过校验后返回 False 表示跳过。
# 不完整目录保留并报错；新评测按 batch 生成独立目录、登记协议，再启动原生评测子进程。
# 新结果必须完整覆盖 10 个任务各 10 回合，否则失败；成功启动并完成时返回 True。
def evaluate_checkpoint(ctx, run, step, state, batch_size=1):
    from smolvla_experiment.report import summarize_evaluation

    policy = checkpoint_path(ctx, run, step)
    label = f"{step:06d}" + (f"_b{batch_size}" if batch_size != 1 else "")
    output = ctx.owned(run / "eval" / label)
    if output.exists():
        info = output / "eval_info.json"
        if info.is_file() and summarize_evaluation(info)[0]["complete"]:
            print(f"Keeping completed evaluation {label}", flush=True)
            return False
        raise FileExistsError(f"Incomplete evaluation exists; preserve and inspect: {output}")
    record_evaluation_protocol(ctx, run, label, f"{step:06d}", batch_size)
    state.update(stage="evaluation", checkpoint=step, eval_batch_size=batch_size)
    experiment_record(ctx).set("activity", state)
    command = ctx.command(
        "smolvla_experiment.evaluate",
        f"--policy.path={policy}",
        "--policy.n_action_steps=1",
        "--policy.use_amp=true",
        "--env.type=libero",
        "--env.task=libero_spatial",
        "--env.episode_length=220",
        "--env.fps=20",
        "--env.observation_height=256",
        "--env.observation_width=256",
        "--env.control_mode=relative",
        "--env.init_states=true",
        "--env.hard_reset=true",
        "--env.max_parallel_tasks=1",
        f"--eval.batch_size={batch_size}",
        "--eval.use_async_envs=true",
        "--eval.n_episodes=10",
        "--seed=0",
        f"--output_dir={output}",
    )
    execute(ctx, command, run / "logs" / f"eval_{label}.log", state)
    if not summarize_evaluation(output / "eval_info.json")[0]["complete"]:
        raise RuntimeError("Evaluation did not complete all 10 tasks x 10 episodes")
    return True


# 把当前实验的运行目录交给统一报告函数，转发可选对比运行和 CSV 导出标志。
def report(ctx, run, compare_with=None, export_csv=False):
    from smolvla_experiment.report import generate_report

    return generate_report(
        ctx.owned(run), ctx=ctx, compare_with=compare_with, export_csv=export_csv
    )


# 通过同一解释器启动准备子进程，日志放在当前实验 logs/prepare.log。
def prepare(ctx):
    execute(ctx, ctx.command("smolvla_experiment.prepare"), ctx.root / "logs/prepare.log")


# 构造新训练或续训子进程命令，统一传入原生配置和累计训练预算。
# 续训加 --resume，新目标通过 --steps 覆盖，原检查点配置文件保持原样。
def train_command(ctx, source, budget_s, target=None, resume=False):
    arguments = ["--config", str(source), "--max-training-time-s", str(budget_s)]
    if resume:
        arguments += ["--resume"]
    if target is not None:
        arguments += ["--steps", str(target)]
    return ctx.command("smolvla_experiment.train", *arguments)


# 正式训练前执行 200 步短跑，验证训练可完成且预期模块确有更新。
# 动作专家只在显存不足时按 8→4→2 尝试 batch；联合微调固定为 8。
# 累计所有已尝试短跑的循环耗时，返回成功 batch 与耗时，其他错误保留日志并上抛。
def run_probe(ctx, run, budget_s, state):
    used = 0.0
    # 只有动作专家配置允许显存不足后降低 batch；其他异常不能被当成显存回退继续尝试。
    batches = (8, 4, 2) if ctx.name == "expert_only" else (8,)
    for batch in batches:
        name = f"probe_b{batch}"
        output, log = run / name, run / "logs" / f"{name}.log"
        probe_config = config(ctx, output, 200, batch)
        set_training_values(ctx, output, "config", probe_config)
        try:
            with temporary_config(ctx, probe_config) as source:
                execute(ctx, train_command(ctx, source, budget_s - used), log, state)
        except RuntimeError:
            runtime = training_values(ctx, output, "runtime")
            used += runtime.get("elapsed_s", 0)
            if "out of memory" not in log.read_text().lower() or batch == batches[-1]:
                raise
            continue
        runtime = training_values(ctx, output, "runtime")
        used += runtime["elapsed_s"]
        gradient = training_values(ctx, output, "gradient_check")
        if (
            runtime["status"] != "completed"
            or runtime["step"] != 200
            or not gradient["verified_updates"]
        ):
            raise RuntimeError(
                "Probe failed gradient/update checks or exhausted the training budget"
            )
        return batch, used
    raise RuntimeError("No probe configuration completed")


# 在同一次记录更新中合并阶段计时，并在给出 sessions 时更新会话列表。
def save_timing(ctx, run, timing, sessions=None):
    # 只修改计时和可选会话字段，保留最新的配置、训练状态和其他记录。
    def update(document):
        document["timing"].update(timing)
        if sessions is not None:
            document["sessions"] = sessions

    run_record(ctx, run).update(update)


# 根据互斥操作参数执行报告、评测、续训或新训练流程。
# 续训与指定评测的历史校验先于状态修改；新训练依次准备、短跑、正式训练、评测和报告。
# 各阶段维护 activity/timing/sessions，发生异常时保存失败原因后继续抛出。
def run_pipeline(ctx, args):
    # 先核对恢复／评测历史，再修改活动、计时或配置，错误输入不会覆盖有效运行记录。
    if args.resume:
        checkpoint = ctx.owned(args.resume)
        continuation_target(ctx, checkpoint.parents[2], checkpoint, args.additional_steps or 5000)
    if args.evaluate_only:
        candidate = ctx.owned(args.evaluate_only)
        available = checkpoints(ctx, candidate)
        if not available:
            raise FileNotFoundError("No trained checkpoints")
        for step in args.checkpoint or [int(available[-1].name)]:
            checkpoint_path(ctx, candidate, step)
    # 工作流状态描述调度阶段；实际训练进度和模型编号分别从运行记录读取。
    state = {"pid": os.getpid(), "status": "running", "experiment": ctx.name}
    run, timing, sessions, session = None, {}, None, None
    training_started = None
    try:
        # 报告分支只读取已保存实验并更新派生产物，finally 不覆盖最近的工作流活动记录。
        if args.report_only:
            report(ctx, args.report_only, args.compare_with, args.export_csv)
            return
        # 批量评测去重后依次处理指定检查点，完整结果校验通过便跳过，随后统一刷新报告。
        if args.evaluate_only:
            run = ctx.owned(args.evaluate_only)
            available = checkpoints(ctx, run)
            steps = list(dict.fromkeys(args.checkpoint or [int(available[-1].name)]))
            state.update(run_dir=str(run), stage="evaluation")
            for step in steps:
                evaluate_checkpoint(ctx, run, step, state, args.eval_batch_size)
            report(ctx, run)
        # 续训复用历史会话和剩余预算，追加目标不会重建学习率调度或修改旧检查点配置。
        elif args.resume:
            checkpoint = ctx.owned(args.resume)
            run = checkpoint.parents[2]
            target = continuation_target(ctx, run, checkpoint, args.additional_steps or 5000)
            record = run_record(ctx, run)
            document = record.read()
            timing, sessions = document["timing"], document["sessions"]
            budget = args.budget_s - timing.get("probe_s", 0)
            # 原生训练子进程按累计循环时间判断预算；短跑开销先由父进程从总预算中扣除。
            previous_elapsed = document["runtime"].get("elapsed_s", 0)
            if budget - previous_elapsed <= 60:
                raise ValueError("Training time budget exhausted")
            state.update(run_dir=str(run), stage="training", target_step=target)
            session = {
                "from_step": int(checkpoint.name),
                "target_step": target,
                "source_checkpoint": str(checkpoint.relative_to(run)),
                "status": "running",
            }
            # 先登记正在运行的续训会话，成功或异常退出后再补充实际结束步数、耗时和状态。
            sessions.append(session)
            timing["status"] = "running"
            save_timing(ctx, run, timing, sessions)
            training_started = time.perf_counter()
            execute(
                ctx,
                train_command(
                    ctx, checkpoint / "pretrained_model/train_config.json", budget, target, True
                ),
                run / "train.log",
                state,
            )
            runtime = record.read()["runtime"]
            if runtime["status"] == "completed" and runtime["step"] != target:
                raise RuntimeError("Continuation did not reach its target")
            checkpoint_path(ctx, run, runtime["step"])
            session.update(
                training_wall_s=time.perf_counter() - training_started,
                runtime_s=runtime.get("elapsed_s", previous_elapsed) - previous_elapsed,
                end_step=runtime["step"],
                status=runtime["status"],
            )
            timing["status"] = runtime["status"]
            if timing.get("training_wall_s") is not None:
                timing["training_wall_s"] += session["training_wall_s"]
            record.update(lambda doc: doc["config"].__setitem__("steps", target))
            save_timing(ctx, run, timing, sessions)
            report(ctx, run)
            state["status"] = runtime["status"]
        else:
            # 新工作流先准备；只有 --prepare-only 分支在准备完成后直接返回。
            prepare(ctx)
            if args.prepare_only:
                state.update(status="prepared", stage="prepared")
                return
            run = (
                ctx.root
                / "runs"
                / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
            )
            run.mkdir(parents=True, exist_ok=False)
            state.update(run_dir=str(run), stage="probe")
            timing = {"status": "running", "training_wall_s": 0, "probe_s": 0}
            save_timing(ctx, run, timing)
            historical = (
                experiment_record(ctx).read()["provenance"].get("historical_validation_s", 0)
            )
            batch, probe_s = run_probe(ctx, run, args.budget_s - historical, state)
            timing["probe_s"] = probe_s + historical
            record = run_record(ctx, run)
            train_config = config(ctx, run / "train", args.max_steps, batch)
            record.set("config", train_config)
            record.set("budget", {"training_budget_s": args.budget_s})
            save_timing(ctx, run, timing)
            state["stage"] = "training"
            training_started = time.perf_counter()
            # 临时 JSON 只用于把有效配置交给原生训练器；管理记录和检查点仍保存可恢复配置。
            with temporary_config(ctx, train_config) as source:
                execute(
                    ctx,
                    train_command(ctx, source, args.budget_s - timing["probe_s"]),
                    run / "train.log",
                    state,
                )
            runtime = record.read()["runtime"]
            timing.update(
                training_wall_s=time.perf_counter() - training_started, status=runtime["status"]
            )
            sessions = [
                {
                    "from_step": 0,
                    "target_step": runtime["step"],
                    "training_wall_s": timing["training_wall_s"],
                    "runtime_s": runtime.get("elapsed_s"),
                }
            ]
            save_timing(ctx, run, timing, sessions)
            steps = sorted(
                {step for step in ctx.profile["evaluation_steps"] if step <= runtime["step"]}
                | {runtime["step"]}
            )
            for step in steps:
                evaluate_checkpoint(ctx, run, step, state)
            # 正式训练及评测完成后才整理短跑目录，正式检查点均保留。
            if runtime["status"] == "completed":
                compact_completed_run(ctx, run)
            report(ctx, run)
            state["status"] = runtime["status"]
        if state["status"] == "running":
            state["status"] = "completed"
        state["stage"] = state["status"]
    # 包含进程中断在内的异常都登记到活动及会话状态，再继续抛出，保留失败诊断。
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        if run is not None and timing:
            timing["status"] = "failed"
            if session is not None:
                session["status"] = "failed"
                if training_started is not None:
                    session["training_wall_s"] = time.perf_counter() - training_started
            save_timing(ctx, run, timing, sessions)
        raise
    finally:
        state.pop("child_pid", None)
        if not args.report_only:
            experiment_record(ctx).set("activity", state)


# 定义根入口的所有命令行参数和互斥操作模式。
# 这里只声明参数，跨参数约束及实验默认值在 main 中处理，解析帮助不加载模型。
def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--resume", type=Path, help="Latest complete checkpoint to continue")
    mode.add_argument("--evaluate-only", type=Path, help="Existing run directory")
    mode.add_argument(
        "--report-only", type=Path, help="Generate a report without training/evaluation"
    )
    parser.add_argument("--compare-with", type=Path, help="Explicit second run for --report-only")
    parser.add_argument(
        "--export-csv", action="store_true", help="Export tables with --report-only"
    )
    parser.add_argument("--additional-steps", type=int)
    parser.add_argument("--checkpoint", nargs="+", type=int)
    parser.add_argument("--eval-batch-size", type=int, choices=(1, 2, 5, 10), default=1)
    parser.add_argument("--background", action="store_true")
    parser.add_argument(
        "--queue", action="store_true", help="Wait for this experiment's workflow lock"
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--budget-s", type=float)
    return parser


# 根入口调用的调度主函数：解析、校验、设环境，再选择后台启动或当前进程执行。
# 后台父进程只记录启动信息，子进程取得实验工作流锁后运行真正的工作流。
def main():
    parser = build_parser()
    args = parser.parse_args()
    ctx = ExperimentContext.load(args.experiment)
    args.max_steps = args.max_steps if args.max_steps is not None else ctx.profile["max_steps"]
    args.budget_s = args.budget_s if args.budget_s is not None else ctx.profile["budget_s"]
    if args.max_steps <= 0 or not math.isfinite(args.budget_s) or args.budget_s <= 60:
        parser.error("Steps must be positive and training budget must exceed 60 seconds")
    if args.additional_steps is not None and (not args.resume or args.additional_steps <= 0):
        parser.error("--additional-steps requires --resume and positive steps")
    if args.checkpoint is not None and (
        not args.evaluate_only or not args.checkpoint or min(args.checkpoint) <= 0
    ):
        parser.error("--checkpoint requires --evaluate-only and positive steps")
    if args.eval_batch_size != 1 and not args.evaluate_only:
        parser.error("--eval-batch-size requires --evaluate-only")
    if args.compare_with and not args.report_only:
        parser.error("--compare-with requires --report-only")
    if args.export_csv and not args.report_only:
        parser.error("--export-csv requires --report-only")
    # 在启动后台进程前先拒绝越界路径和不合法参数组合，避免后台才发现基本输入错误。
    for path in (args.resume, args.evaluate_only, args.report_only):
        if path is not None:
            ctx.owned(path)
    configure_environment(ctx)
    # 后台父进程移除 --background 后重新启动同一根入口，防止子进程再次递归转入后台。
    # start_new_session 使子进程拥有独立会话，标准输出与错误均追加到本实验的 pipeline.log。
    if args.background:
        logfile = ctx.owned(ctx.root / "logs/pipeline.log")
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with logfile.open("a") as stream:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    str(ctx.project_root / "run.py"),
                    *[arg for arg in sys.argv[1:] if arg != "--background"],
                ],
                cwd=ctx.project_root,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        experiment_record(ctx).set("launcher", {"pid": process.pid, "log": str(logfile)})
        print(f"Pipeline PID: {process.pid}; log: {logfile}")
        return
    with workflow_lock(ctx, args.queue):
        run_pipeline(ctx, args)


if __name__ == "__main__":
    main()
