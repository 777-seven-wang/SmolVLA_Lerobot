# 训练执行层：在当前子进程内为原生 LeRobot 训练器增加观测、校验和中断管理。
# 模型构造、优化器更新、调度器恢复与完整检查点保存仍由原生训练器执行。
# 本模块包装日志、更新、梯度裁剪及保存钩子，记录原始指标与真实参数变化证据。
# 正式运行写入 run.json 的配置、状态和诊断分区，短跑则写入 probes 对应条目。
# 续训先核验完整状态，SIGINT/SIGTERM 或预算到期时等待保存完成后再退出。

"""Instrument the installed trainer in this process; all outputs stay in this workspace."""

import argparse
import csv
import json
import logging
import math
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import MethodType

from smolvla_experiment.common import (
    EXPERIMENTS,
    ExperimentContext,
    configure_environment,
    ensure_owned,
    read,
)
from smolvla_experiment.records import (
    append_step_timing,
    read_step_timing,
    run_record,
    set_training_values,
    step_timing_path,
    training_values,
)


# 在训练子进程内追加训练、离线验证和 GPU 原始指标。
# 每次创建实例才启动 GPU 采样线程，导入模块和查看命令帮助不会触发采样。
class Metrics:
    # 校验指标目录、建立线程锁，并启动可随训练一起停止的后台采样线程。
    def __init__(self, directory, root=None):
        self.root = root
        self.directory = ensure_owned(directory, root)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.gpu, daemon=True)
        self.thread.start()

    # 追加一条带时间戳的原始指标，限定 train/validation/gpu 三种类型。
    # 写入前拒绝非有限数值，已有表头必须与当前字段一致，避免续训时改变 CSV 模式。
    # 同一个实例的采样线程和训练日志线程通过互斥锁串行写入。
    def log(self, kind, values):
        if kind not in {"train", "validation", "gpu"}:
            raise ValueError(f"Unknown metric kind: {kind}")
        row = {"timestamp": time.time(), **values}
        json.dumps(row, allow_nan=False)
        with self.lock:
            path = ensure_owned(self.directory / f"{kind}.csv", self.root)
            exists = path.exists() and path.stat().st_size > 0
            fields = list(row)
            if exists:
                with path.open(newline="") as stream:
                    fields = next(csv.reader(stream))
                if set(fields) != set(row):
                    raise ValueError(f"Metric schema changed for {kind}: {set(row) ^ set(fields)}")
            with path.open("a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                if not exists:
                    writer.writeheader()
                writer.writerow(row)

    # 每约 10 秒通过 nvidia-smi 采集整卡利用率、显存和功率。
    # 命令缺失、暂时失败或字段异常时跳过本次采样；采样峰值与 PyTorch 分配峰值含义不同。
    def gpu(self):
        fields = ["gpu_index", "utilization_pct", "memory_used_mib", "memory_total_mib", "power_w"]
        while not self.stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=5,
                )
                for row in csv.reader(output.splitlines()):
                    self.log("gpu", dict(zip(fields, [float(x) for x in row], strict=True)))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self.stop.wait(10)

    # 通知后台采样线程退出，并等待至多 6 秒；由训练 finally 分支统一调用。
    def close(self):
        self.stop.set()
        self.thread.join(timeout=6)


# 在模型构造前检查输出范围、冻结策略、batch、梯度累积和模型配置。
# 续训额外要求来自同一运行的原生 train_config.json，并存在完整训练步数、优化器、
# 调度器与随机状态文件。返回已验证的训练输出目录，校验失败时不开始训练。
def validate_training_config(ctx, cfg, output=None, resume=False, config_path=None):
    output = ctx.owned(output or cfg["output_dir"])
    if output != ctx.owned(cfg["output_dir"]):
        raise ValueError("Cannot change output directory when invoking the trainer")
    for key, expected in ctx.flags.items():
        if cfg["policy"].get(key) is not expected:
            raise ValueError(f"policy.{key} must be {expected}")
    if (
        cfg.get("batch_size") not in ((8, 4, 2) if ctx.name == "expert_only" else (8,))
        or cfg.get("accelerator", {}).get("gradient_accumulation", {}).get("steps", 1) != 1
    ):
        raise ValueError("Invalid batch or gradient accumulation for this experiment")
    ctx.owned(cfg["dataset"]["root"])
    if resume:
        if config_path is None:
            raise ValueError("Resume requires checkpoint train_config.json")
        path = ctx.owned(config_path)
        checkpoint = path.parent.parent
        if path.name != "train_config.json" or path.parent.name != "pretrained_model":
            raise ValueError("Resume from checkpoint/pretrained_model/train_config.json")
        if checkpoint.parent != output / "checkpoints":
            raise ValueError("Resume checkpoint must belong to the same run")
        for filename in (
            "training_step.json",
            "optimizer_state.safetensors",
            "optimizer_param_groups.json",
            "scheduler_state.json",
            "rng_state.safetensors",
        ):
            required = ctx.owned(checkpoint / "training_state" / filename)
            if not required.is_file():
                raise FileNotFoundError(f"Incomplete resumable checkpoint: {required}")
        model_path = path.parent
    else:
        model_path = ctx.owned(cfg["policy"].get("path") or cfg["policy"].get("pretrained_path"))
    model_config = read(model_path / "config.json")
    for key, expected in ctx.flags.items():
        if model_config.get(key) is not expected:
            raise ValueError(f"Model config {key} must be {expected} before construction")
    return output


# 按参数名归入需要观测的代表性模块组，未纳入采样的参数返回 None。
# 语言层使用第 0 和第 14 层作为早／后段代表，动作相关投影分别归入 state/action。
def gradient_group(name):
    if ".vlm.model.connector." in name:
        return "connector"
    if ".vlm.model.text_model.embed_tokens." in name:
        return "embedding"
    if ".vlm.model.text_model.layers.0." in name:
        return "language_early"
    if ".vlm.model.text_model.layers.14." in name:
        return "language_late"
    if ".lm_expert." in name:
        return "expert"
    if ".state_proj." in name:
        return "state"
    if ".action_in_proj." in name or ".action_out_proj." in name:
        return "action"
    return None


# 在梯度裁剪钩子中统计可训练／总参数量、模块组梯度平方范数及异常梯度。
# 冻结参数意外带梯度或存在 NaN/Infinity 会进入诊断列表，供训练入口立即拒绝异常更新。
def summarize_gradients(policy, norm):
    import torch

    report = {
        "grad_norm": float(norm),
        "trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in policy.parameters()),
        "groups": {},
        "frozen_with_gradient": [],
        "nonfinite_gradients": [],
    }
    for name, parameter in policy.named_parameters():
        if parameter.grad is None:
            continue
        if not parameter.requires_grad:
            report["frozen_with_gradient"].append(name)
        grad = parameter.grad.detach()
        if not bool(torch.isfinite(grad).all()):
            report["nonfinite_gradients"].append(name)
        group = gradient_group(name)
        if group:
            values = report["groups"].setdefault(
                group, {"parameters_with_gradient": 0, "squared_norm": 0.0}
            )
            values["parameters_with_gradient"] += 1
            values["squared_norm"] += float(grad.float().square().sum())
    return report


# 内部控制异常：表示预算或信号要求结束，且原生检查点及 last 更新已经完成。
# 由本模块捕获，不作为训练实现崩溃记录。
class GracefulStop(Exception):
    pass


# 训练子进程入口：先验证配置与续训历史，再设置原生训练器观测钩子并执行。
# 新增训练使用暂存配置，续训读取检查点配置并只通过命令覆盖目标步数。
# finally 中持久化状态、追加本会话计时并关闭 GPU 采样，失败原因保留在管理记录。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-training-time-s", type=float, default=28800)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--steps", type=int, help="Override total target steps without editing checkpoint config"
    )
    args = parser.parse_args()
    ctx = ExperimentContext.load(args.experiment)
    configure_environment(ctx)
    if not math.isfinite(args.max_training_time_s) or args.max_training_time_s <= 0:
        parser.error("Training budget must be positive")
    cfg = read(ctx.owned(args.config))
    output = validate_training_config(ctx, cfg, args.output_dir, args.resume, args.config)
    # 恢复时先取原运行累计值，再检查已有逐步历史不能超过检查点，防止过期恢复污染指标。
    previous = training_values(ctx, output, "runtime") if args.resume else {}
    restored = (
        read(args.config.parent.parent / "training_state/training_step.json")["step"]
        if args.resume
        else 0
    )
    if args.steps is not None and args.steps <= restored:
        parser.error("--steps must exceed the restored checkpoint step")
    prior_steps = read_step_timing(step_timing_path(ctx, output)) if args.resume else []
    if any(row["steps"] > restored for row in prior_steps):
        raise ValueError("Timing records extend past the resume checkpoint")
    wall_started = time.perf_counter()
    if not args.resume:
        set_training_values(ctx, output, "config", cfg)
    metrics = Metrics(output.parent / f"{output.name}_metrics", root=ctx.root)
    # 这些累计字段跨会话继承；session_wall_s 和初始化计时则从本次进程开始重新测量。
    state = {
        "step": restored,
        "status": "initializing",
        "session_started_at": time.time(),
        "elapsed_s": previous.get("elapsed_s", 0.0),
        "initialization_s": previous.get("initialization_s", 0.0),
        "validation_s": previous.get("validation_s", 0.0),
        "checkpoint_s": previous.get("checkpoint_s", 0.0),
        "optimizer_update_s": previous.get("optimizer_update_s", 0.0),
        "cuda_allocated_peak_gib": previous.get("cuda_allocated_peak_gib", 0.0),
        "session_count": previous.get("session_count", 0) + 1,
    }
    training_start = None
    previous_elapsed = state["elapsed_s"]
    stop_requested = False
    # evidence 保存可训练模块的代表性更新前后值；frozen_samples 保存应保持不变的采样值。
    evidence = {}
    frozen_samples = {}
    gradient_summary = None
    step_rows = []
    policy_ref = None

    # 更新累计循环时间、本会话墙钟和更新时间，保存当前训练状态。
    # 循环累计以首次更新开始为界，续训累加历史值；会话墙钟从本次初始化开始计算。
    def persist():
        if training_start is not None:
            state["elapsed_s"] = previous_elapsed + time.perf_counter() - training_start
        state["updated_at"] = time.time()
        state["session_wall_s"] = time.perf_counter() - wall_started
        set_training_values(ctx, output, "runtime", state)

    # 信号处理器只设置退出标志，不在异步信号回调中写文件或直接终止进程。
    # 后续保存判断会触发完整检查点，再由 update_last 安全退出。
    def request_stop(*_):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    persist()
    try:
        import torch
        from lerobot.utils.logging_utils import MetricsTracker

        from lerobot.scripts import lerobot_train as trainer
        from smolvla_experiment.common import install_fp32_policy_storage

        # 在原生训练器创建模型／优化器之前安装 FP32 存储补丁，计算混合精度仍由配置控制。
        if ctx.profile["parameter_storage"] == "float32":
            install_fp32_policy_storage()

        # 将原生训练日志转为项目原始 CSV 的适配器，保留原生训练器本身的日志行为。
        class Handler(logging.Handler):
            # 识别 MetricsTracker 训练指标与固定格式的验证 loss 日志。
            # 分别追加 train.csv／validation.csv，训练窗口还会触发运行状态持久化。
            def emit(self, record):
                if isinstance(record.msg, MetricsTracker):
                    values = record.msg.to_dict()
                    metrics.log(
                        "train",
                        {
                            **values,
                            "elapsed_s": previous_elapsed + time.perf_counter() - training_start,
                        },
                    )
                    persist()
                elif isinstance(record.msg, str):
                    match = re.fullmatch(r"step (\d+): eval_loss=([\deE+.-]+)", record.msg)
                    if match:
                        metrics.log(
                            "validation",
                            {
                                "steps": int(match[1]),
                                "eval_loss": float(match[2]),
                                **(
                                    {"n_samples": min(cfg["max_eval_samples"], 5927)}
                                    if ctx.name == "joint"
                                    else {}
                                ),
                                "elapsed_s": previous_elapsed
                                + time.perf_counter()
                                - training_start,
                            },
                        )

        original_logging = trainer.init_logging

        # 先安装原生日志配置，再追加项目指标处理器，避免被原生初始化清掉自定义 handler。
        def init_logging(*a, **kw):
            original_logging(*a, **kw)
            logging.getLogger().addHandler(Handler())

        trainer.init_logging = init_logging
        original_update = trainer.update_policy

        # 包装原生 update_policy：首次更新验证模型设置并安装一次性梯度观测。
        # 后续每步只执行原生更新、同步 CUDA 并记录耗时和显存，不重建优化器或调度器。
        # 记录的 update_s 覆盖原生更新调用及同步等待，并非仅 AdamW.step 的纯计算时间。
        def update(*a, **kw):
            nonlocal training_start, policy_ref, gradient_summary
            accelerator = kw["accelerator"]
            policy = a[1]
            if training_start is None:
                policy_ref = policy
                if any(getattr(policy.config, k) is not v for k, v in ctx.flags.items()):
                    raise RuntimeError("Constructed policy has incorrect trainability flags")
                count = sum(p.numel() for p in policy.parameters() if p.requires_grad)
                if count != ctx.profile["expected_trainable_parameters"]:
                    raise RuntimeError(
                        f"Expected {ctx.profile['expected_trainable_parameters']} trainable parameters, got {count}"
                    )
                state["trainable_parameters"] = count
                state["total_parameters"] = sum(p.numel() for p in policy.parameters())
                state["parameter_storage_dtype"] = ctx.profile["parameter_storage"]
                state["compute_mixed_precision"] = "bf16"
                if ctx.profile["parameter_storage"] == "float32" and any(
                    p.dtype != torch.float32 for p in policy.parameters() if p.is_floating_point()
                ):
                    raise RuntimeError("All floating parameters must use FP32 storage before AdamW")
                # 首次见到实际模型后再绑定模式切换与梯度观测，避免在模块导入阶段修改模型类行为。
                original_mode = policy.train
                validation_start = None

                # 包装 train/eval 模式切换，以切换边界累计验证阶段耗时。
                # 原始 train(mode) 的调用和返回值保持一致，包装通过 MethodType 绑定到当前模型实例。
                def train_mode(this, mode=True):
                    nonlocal validation_start
                    result = original_mode(mode)
                    if not mode and validation_start is None:
                        validation_start = time.perf_counter()
                    elif mode and validation_start is not None:
                        state["validation_s"] += time.perf_counter() - validation_start
                        validation_start = None
                    return result

                policy.train = MethodType(train_mode, policy)
                original_clip = accelerator.clip_grad_norm_

                # 先执行原生梯度裁剪，再检查预期组是否有非零有限梯度。
                # 为每组选择一个最大绝对梯度位置记录更新前值，并采样应冻结模块用于之后对照。
                # 这些是代表性变化证据；完成首次检查后恢复原裁剪函数，避免每步重复扫描所有参数。
                def clip(parameters, max_norm):
                    nonlocal gradient_summary
                    norm = original_clip(parameters, max_norm)
                    gradient_summary = summarize_gradients(policy, norm)
                    missing = [
                        g
                        for g in ctx.required_groups
                        if gradient_summary["groups"].get(g, {}).get("squared_norm", 0) <= 0
                    ]
                    if (
                        not math.isfinite(float(norm))
                        or missing
                        or gradient_summary["frozen_with_gradient"]
                        or gradient_summary["nonfinite_gradients"]
                    ):
                        raise RuntimeError(
                            f"Gradient verification failed: missing={missing}, report={gradient_summary}"
                        )
                    for name, p in policy.named_parameters():
                        group = gradient_group(name)
                        if (
                            p.requires_grad
                            and p.grad is not None
                            and group
                            and group not in evidence
                        ):
                            # 挑选每个代表组中观测到的非零梯度位置，保存当前参数值供保存检查点前进行对照。
                            index = int(p.grad.detach().flatten().abs().argmax())
                            if float(p.grad.detach().flatten()[index]) != 0:
                                evidence[group] = {
                                    "parameter": name,
                                    "index": index,
                                    "before": float(p.detach().flatten()[index]),
                                }
                        if not p.requires_grad and (
                            ".vision_model." in name
                            or ".text_model.layers.15." in name
                            or ".lm_head." in name
                            or ".text_model.norm." in name
                        ):
                            category = (
                                "vision"
                                if ".vision_model." in name
                                else "last_language_layer"
                                if ".layers.15." in name
                                else "lm_head"
                                if ".lm_head." in name
                                else "final_norm"
                            )
                            frozen_samples.setdefault(
                                category,
                                {"parameter": name, "before": float(p.detach().flatten()[0])},
                            )
                    set_training_values(
                        ctx,
                        output,
                        "gradient_check",
                        {
                            **gradient_summary,
                            "expected_groups": list(ctx.required_groups),
                            "samples": evidence,
                            "verified_updates": False,
                        },
                    )
                    # 梯度结构检查和采样完成后恢复原函数，后续训练步骤继续使用原生裁剪实现。
                    accelerator.clip_grad_norm_ = original_clip
                    return norm

                accelerator.clip_grad_norm_ = clip
                state["initialization_s"] += time.perf_counter() - wall_started
                state["status"] = "running"
                training_start = time.perf_counter()
                persist()
            start = time.perf_counter()
            result = original_update(*a, **kw)
            # CUDA 运算是异步提交的，先同步再停止计时，保证耗时包含该更新实际完成的等待。
            torch.cuda.synchronize()
            duration = time.perf_counter() - start
            state["optimizer_update_s"] += duration
            state["cuda_allocated_peak_gib"] = max(
                state["cuda_allocated_peak_gib"], torch.cuda.max_memory_allocated() / 1024**3
            )
            row = {"steps": state["step"] + 1, "update_s": duration}
            # 每步耗时先缓存在本会话内存，退出 finally 时统一追加 CSV，减少训练热路径的写盘。
            step_rows.append(row)
            return result

        trainer.update_policy = update
        original_should_save = trainer.should_save_checkpoint

        # 同步当前步数并在累计训练预算接近上限或收到停止信号时要求保存。
        # 预算预留最多 60 秒用于保存，正常情况下仍采用原生周期／最终步保存规则。
        def should_save(step, frequency, total):
            state["step"] = step
            used = previous_elapsed + (
                time.perf_counter() - training_start if training_start else 0
            )
            if used >= args.max_training_time_s - min(60, args.max_training_time_s * 0.05):
                state["status"] = "time_limit"
                return True
            if stop_requested:
                state["status"] = "interrupted"
                return True
            return original_should_save(step, frequency, total)

        trainer.should_save_checkpoint = should_save
        original_save = trainer.save_checkpoint

        # 在保存前比较采样参数的前后值，确认所有预期组发生更新、冻结样本保持不变。
        # 保存详细诊断；本会话已训练至少 200 步仍未满足更新检查时，明确中止而不是标记成功。
        def check_updates():
            named = dict(policy_ref.named_parameters())
            for values in evidence.values():
                values["after"] = float(
                    named[values["parameter"]].detach().flatten()[values["index"]]
                )
                values["changed"] = values["after"] != values["before"]
            for values in frozen_samples.values():
                values["after"] = float(named[values["parameter"]].detach().flatten()[0])
                values["unchanged"] = values["after"] == values["before"]
            changed = all(v["changed"] for v in evidence.values()) and set(evidence) == set(
                ctx.required_groups
            )
            frozen = all(v["unchanged"] for v in frozen_samples.values())
            set_training_values(
                ctx,
                output,
                "gradient_check",
                {
                    **gradient_summary,
                    "samples": evidence,
                    "frozen_samples": frozen_samples,
                    "expected_groups": list(ctx.required_groups),
                    "verified_updates": changed and frozen,
                },
            )
            if state["step"] >= restored + 200 and not (changed and frozen):
                raise RuntimeError("Expected modules did not update or frozen sample changed")

        # 先核对参数变化，再调用原生完整检查点保存并累计保存耗时。
        # 原生保存负责模型、优化器、调度器、随机状态和进度，包装不删减恢复状态。
        def save_checkpoint(*a, **kw):
            check_updates()
            started = time.perf_counter()
            result = original_save(*a, **kw)
            state["checkpoint_s"] += time.perf_counter() - started
            persist()
            return result

        trainer.save_checkpoint = save_checkpoint
        original_last = trainer.update_last_checkpoint

        # 先完成原生 last 引用更新并落盘状态，再根据退出标志抛出 GracefulStop。
        # 将退出点放在此处，保证用户恢复时能找到刚完成的完整检查点。
        def update_last(*a, **kw):
            result = original_last(*a, **kw)
            persist()
            if stop_requested or state["status"] == "time_limit":
                raise GracefulStop()
            return result

        trainer.update_last_checkpoint = update_last
        # 把经过项目校验的配置路径交给原生解析器。续训覆盖目标只作用于本次调用，不重写源文件。
        sys.argv = ["lerobot-train", f"--config_path={args.config.resolve()}"]
        if args.resume:
            sys.argv.append("--resume=true")
        if args.steps is not None:
            sys.argv.append(f"--steps={args.steps}")
        trainer.main()
        state["status"] = "completed"
    # 完整保存后的受控停止有独立状态，不与未知异常导致的 failed 混为一类。
    except GracefulStop:
        if state["status"] != "time_limit":
            state["status"] = "interrupted"
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        # 退出时先持久化已知状态和计时；最外层 finally 即使追加失败也会关闭采样线程。
        try:
            persist()
            if step_rows:
                append_step_timing(ctx, output, step_rows, restored)
                if output.name == "train":
                    # 在正式运行的 timing 分区登记本会话恢复起点和逐步计时文件来源。
                    # 具体逐步记录已通过专用 CSV 接口追加，不把数组重复写入 run.json。
                    def timing_metadata(document):
                        document["timing"].update(
                            step_timing_resume_from=restored,
                            step_timing_source="train_metrics/step_timing.csv",
                        )

                    run_record(ctx, output.parent).update(timing_metadata)
        finally:
            metrics.close()
    if state["status"] not in {"completed", "time_limit"}:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
