# 统一结果计算与报告模块：消费原始记录，生成 Markdown、曲线和可选 CSV。
# 单次报告复用已读取的运行记录、划分、指标与逐回合评测，绘图函数只接收这些数据。
# 成功率从 successes 重算，只有默认协议下完整的 10 任务×10 回合参与默认选模。
# 完整统计可由函数返回；run.json 只持久化关键汇总、来源引用及模型编号。
# 派生产物清理限定为本模块登记的文件名，原始指标、评测与用户自建文件不在清理范围。

"""Render actual run metrics, checkpoint comparisons and a Chinese experiment report."""

import copy
import csv
import math
import statistics
from pathlib import Path

from smolvla_experiment.common import ExperimentContext, ensure_owned, read
from smolvla_experiment.records import EVALUATION_PROTOCOL, evaluation_protocol, run_record

from . import plots as charts

# 报告展示默认计划步数与已有评测步数的并集，缺少实际结果的检查点显示为待评测。
EXPECTED_STEPS = (5000, 10000, 20000)
# 受管派生产物白名单，清理只匹配这些名称；退休的性能图名称保留用于清除旧输出。
MANAGED_FIGURES = (
    "loss_training_validation.png",
    "checkpoint_comparison.png",
    "loss_strategy_comparison.png",
    "performance_summary.png",  # 已退休的旧输出，重建报告时清理。
)
MANAGED_CSV = ("evaluation_summary.csv", "per_task.csv", "loss_strategy_comparison.csv")


# 读取可选 JSON：仅文件缺失时返回空字典，存在但损坏的文件仍明确报错。
# 先检查路径边界，主要用于模型配置和数据划分的读取。
def optional_json(path):
    path = ensure_owned(path)
    return read(path) if path.is_file() else {}


# 读取一份原始指标 CSV，保留字段字符串供统计函数显式转换。
# 文件未生成时返回空列表，以便为尚未开始或未完成的运行生成如实报告。
def csv_rows(path):
    path = ensure_owned(path)
    if not path.is_file():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


# 从完整返回结果中选择允许长期写入 run.json 的关键统计与来源引用。
# 不复制逐任务数组、完整运行状态或计时明细，导出引用只列出本次实际生成的 CSV。
def compact_summary(summary):
    return {
        **{
            key: summary[key]
            for key in ("completed_steps", "training_loss", "strategy_comparison", "figures")
        },
        "performance": {
            key: summary["performance"][key]
            for key in ("torch_peak_allocated_gib", "sampled_gpu_memory_max_mib", "gpu_samples")
        },
        "sources": {
            "report": "REPORT.md",
            "exports": summary["exports"],
            "metrics": "train_metrics/",
            "raw_evaluations": "eval/*/eval_info.json",
        },
    }


# 从已读取的训练与验证记录计算最终值及最低验证 loss 所在步数。
# 任一类记录缺失则返回空字典；直接使用原始记录，不额外平滑或推断未记录的值。
def loss_statistics(metrics):
    train, validation = metrics["train"], metrics["validation"]
    if not train or not validation:
        return {}
    best = min(validation, key=lambda row: float(row["eval_loss"]))
    return {
        "training_points": len(train),
        "validation_points": len(validation),
        "final_step": int(train[-1]["steps"]),
        "minimum_validation_step": int(best["steps"]),
        "minimum_validation_loss": float(best["eval_loss"]),
        "final_training_loss": float(train[-1]["loss"]),
        "final_validation_loss": float(validation[-1]["eval_loss"]),
        "sources": ["train_metrics/train.csv", "train_metrics/validation.csv"],
        "extra_smoothing": False,
    }


# 结合运行记录和原始指标计算阶段耗时、窗口吞吐及显存统计。
# 深拷贝会话后补充统计，保持传入 document 不变；历史缺失会话时按现有记录构造描述。
# 训练循环与真实墙钟分别保留，整卡采样显存与 PyTorch 分配峰值也分别报告。
def performance_statistics(document, metrics):
    rows = metrics["train"]
    runtime, timing = document["runtime"], document["timing"]
    final_step = int(rows[-1]["steps"]) if rows else runtime.get("step", 0)
    sessions = copy.deepcopy(document["sessions"])
    if not sessions and final_step:
        sessions = [
            {
                "from_step": 0,
                "target_step": final_step,
                "training_wall_s": timing.get("training_wall_s"),
            }
        ]
    # 按会话步数区间选取已有的记录窗口；窗口均值／中位数不等同于全流程墙钟吞吐。
    for session in sessions:
        selected = [
            r for r in rows if session["from_step"] < int(r["steps"]) <= session["target_step"]
        ]
        session["recorded_windows"] = len(selected)
        for field, output, function in (
            ("step_s", "mean_window_step_s", statistics.mean),
            ("samples_per_s", "median_window_samples_per_s", statistics.median),
            ("gpu_mem_gb", "torch_recorded_allocated_peak_gib", max),
        ):
            values = [float(row[field]) for row in selected if field in row]
            session[output] = function(values) if values else None
    gpu = metrics["gpu"]
    memory = [float(row["memory_used_mib"]) for row in gpu if "memory_used_mib" in row]
    # 标记 GPU 采样的大空档供解释使用，不把空档内的利用率、功率或能耗插值补造。
    gaps = [
        float(b["timestamp"]) - float(a["timestamp"])
        for a, b in zip(gpu, gpu[1:])
        if float(b["timestamp"]) - float(a["timestamp"]) > 60
    ]
    return {
        "training_wall_s": timing.get("training_wall_s"),
        "training_loop_s": runtime.get("elapsed_s"),
        "sessions": sessions,
        "torch_peak_allocated_gib": runtime.get(
            "cuda_allocated_peak_gib",
            max((float(row["gpu_mem_gb"]) for row in rows if "gpu_mem_gb" in row), default=None),
        ),
        "sampled_gpu_memory_max_mib": max(memory) if memory else None,
        "gpu_samples": len(gpu),
        "gpu_sampling_gaps_s": gaps,
        "measurement_note": "Window metrics are recorded, not smoothed. Session wall times exclude pauses; GPU memory is a sampled whole-card maximum, distinct from PyTorch allocation. No energy estimate.",
    }


# 一次加载需要的指标类型；默认包括训练、验证和 GPU 三份 CSV。
# 对比运行只需训练与验证两类，由调用者通过 kinds 指定。
def load_metrics(run, kinds=("train", "validation", "gpu")):
    return {kind: csv_rows(run / f"train_metrics/{kind}.csv") for kind in kinds}


# 校验两个不同实验的数据划分与共同范围内的 loss 记录，并构造对比数据。
# 可复用当前运行的 ctx/split/metrics；对比方的划分和指标各自读取一次。
# 共同范围取两边 train/validation 最后步数的最小值，拒绝非有限值、重复或错位步数。
# 返回绘图／CSV 共用的数据以及可写入管理汇总的精简比较统计。
def collect_comparison(run, other, *, ctx=None, split=None, metrics=None):
    contexts = [ctx or ExperimentContext.for_run(run), ExperimentContext.for_run(other)]
    splits = [
        split if split is not None else optional_json(contexts[0].root / "data/split.json"),
        optional_json(contexts[1].root / "data/split.json"),
    ]
    if contexts[0].name == contexts[1].name:
        raise ValueError("Strategy comparison requires two different experiment profiles")
    for key in ("train_episodes", "validation_episodes", "train_frames", "validation_frames"):
        if splits[0][key] != splits[1][key]:
            raise ValueError(f"Dataset split mismatch: {key}")
    experiments = []
    for ctx, path in zip(contexts, (run, other), strict=True):
        data = (
            {kind: metrics[kind] for kind in ("train", "validation")}
            if metrics is not None and path == run
            else load_metrics(path, ("train", "validation"))
        )
        if not all(data.values()):
            raise ValueError("Both runs require recorded training and validation loss")
        experiments.append({"name": ctx.name, "run": path, "data": data})
    # 对比截止于两次运行的训练和验证都已有记录的步数范围，避免一边缺失却继续比较。
    limit = min(
        int(e["data"][kind][-1]["steps"]) for e in experiments for kind in ("train", "validation")
    )
    summary = {}
    for exp in experiments:
        for kind, column in (("train", "loss"), ("validation", "eval_loss")):
            values = [
                (int(r["steps"]), float(r[column]))
                for r in exp["data"][kind]
                if int(r["steps"]) <= limit
            ]
            if not values or any(not math.isfinite(y) for _, y in values):
                raise ValueError("Invalid comparison loss values")
            if len({x for x, _ in values}) != len(values) or values != sorted(values):
                raise ValueError("Loss steps must be unique and ordered")
            exp[kind] = values
        best = min(exp["validation"], key=lambda pair: pair[1])
        summary[exp["name"]] = {
            "run": str(exp["run"]),
            "final_training_loss": exp["train"][-1][1],
            "final_validation_loss": exp["validation"][-1][1],
            "minimum_validation_step": best[0],
            "minimum_validation_loss": best[1],
        }
    for kind in ("train", "validation"):
        if [x for x, _ in experiments[0][kind]] != [x for x, _ in experiments[1][kind]]:
            raise ValueError("Recorded comparison steps differ")
    return experiments, {
        "comparison_steps": limit,
        "shared_split_verified": True,
        "extra_smoothing": False,
        "experiments": summary,
        "caveat": "Trainable parameter scope and parameter storage precision both differ.",
    }


# 计算成功率的 95% Wilson 置信区间，返回百分比上下界。
# 零回合返回 [None, None]，成功数越界则报错；全失败／全成功的端点固定为 0／100。
def wilson(successes, total):
    """Return a 95% Wilson score interval in percentage points."""
    if not total:
        return [None, None]
    if not 0 <= successes <= total:
        raise ValueError("Invalid success count")
    z = 1.959963984540054
    proportion = successes / total
    divisor = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / divisor
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total**2)) / divisor
    return [
        0.0 if successes == 0 else 100 * max(0, center - margin),
        100.0 if successes == total else 100 * min(1, center + margin),
    ]


# 对已记录的数值求算术平均，空记录返回 None，保留未记录与零的区别。
def mean(values):
    return sum(values) / len(values) if values else None


# 从一个检查点的逐回合原始结果重算整体及逐任务统计。
# result/protocol 可传已读数据，单独传 path 时仍按原有方式读取并推导协议。
# 核对 successes、成功数、回合数、任务唯一性及整体比例，发现不一致立即报错。
# 返回整体摘要和逐任务列表，完整性与协议资格分别判断，供默认选模和分组展示使用。
def summarize_evaluation(path, *, result=None, protocol=None):
    path = ensure_owned(path)
    result = read(path) if result is None else result
    step = int(path.parent.name.split("_b")[0])
    tasks, all_successes, all_lengths = [], [], []
    for task in result.get("per_task", []):
        metrics = task["metrics"]
        successes = metrics["successes"]
        n_success = sum(bool(value) for value in successes)
        if any(not isinstance(value, (bool, int)) or value not in (0, 1) for value in successes):
            raise ValueError(f"Invalid success observations in {path}")
        if n_success != task.get("n_success", n_success):
            raise ValueError(f"Inconsistent success count in {path}: task {task['task_id']}")
        lengths = metrics.get("episode_lengths", [])
        if lengths and len(lengths) != len(successes):
            raise ValueError(f"Inconsistent episode lengths in {path}")
        all_successes.extend(successes)
        all_lengths.extend(lengths)
        tasks.append(
            {
                "checkpoint": path.parent.name,
                "step": step,
                "task_group": task.get("task_group"),
                "task_id": task["task_id"],
                "n_success": n_success,
                "n_episodes": len(successes),
                "pc_success": 100 * n_success / len(successes) if successes else None,
                "pc_success_ci95": wilson(n_success, len(successes)),
                "avg_episode_length": mean(lengths),
                "eval_s": metrics.get("eval_s"),
            }
        )
    overall = result.get("overall", {})
    # 任务身份必须唯一；完整性要求 Spatial 的 0–9 任务都各有 10 个回合。
    identities = {(task["task_group"], task["task_id"]) for task in tasks}
    if len(identities) != len(tasks):
        raise ValueError(f"Duplicate tasks in evaluation: {path}")
    n_success, n_episodes = sum(bool(value) for value in all_successes), len(all_successes)
    if (
        overall.get("n_success", n_success) != n_success
        or overall.get("n_episodes", n_episodes) != n_episodes
    ):
        raise ValueError(f"Task and overall episode counts differ: {path}")
    if overall.get("pc_success") is not None and not math.isclose(
        overall["pc_success"], 100 * n_success / n_episodes if n_episodes else 0, abs_tol=1e-6
    ):
        raise ValueError(f"Task and overall success rates differ: {path}")
    if protocol is None:
        protocol = evaluation_protocol(path.parent.parent.parent, path.parent.name)
    return {
        "checkpoint": path.parent.name,
        "protocol": protocol,
        "selection_eligible": all(
            protocol.get(key) == value for key, value in EVALUATION_PROTOCOL.items()
        ),
        "step": step,
        "n_success": n_success,
        "n_episodes": n_episodes,
        "pc_success": 100 * n_success / n_episodes if n_episodes else None,
        "pc_success_ci95": wilson(n_success, n_episodes),
        "avg_episode_length": mean(all_lengths),
        "eval_s": overall.get("eval_s"),
        "eval_ep_s": overall.get("eval_ep_s"),
        "complete": identities == {("libero_spatial", task) for task in range(10)}
        and all(task["n_episodes"] == 10 for task in tasks),
    }, tasks


# 按明确字段顺序导出表格，把区间数组拆成 ci95_low/ci95_high 两列。
# 忽略不属于表格的内部字段；该函数仅用于可重建的派生 CSV。
def write_csv(path, rows, fields):
    with ensure_owned(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            ci = row.get("pc_success_ci95", [None, None])
            writer.writerow({**row, "ci95_low": ci[0], "ci95_high": ci[1]})


# 导出检查点汇总和逐任务表；提供对比数据时再导出策略 loss 对照表。
# 对比表按训练记录的步数排列，只有验证记录存在的步数才填写验证 loss。
# 返回实际导出文件的相对路径映射，供报告文本和管理来源引用复用。
def export_tables(run, evaluations, tasks, comparison):
    columns = [
        "checkpoint",
        "step",
        "n_success",
        "n_episodes",
        "pc_success",
        "ci95_low",
        "ci95_high",
        "avg_episode_length",
        "eval_s",
        "eval_ep_s",
        "complete",
    ]
    write_csv(run / MANAGED_CSV[0], evaluations, columns)
    write_csv(
        run / MANAGED_CSV[1], tasks, ["checkpoint", "step", "task_group", "task_id", *columns[2:-1]]
    )
    exported = {"checkpoints": MANAGED_CSV[0], "per_task": MANAGED_CSV[1]}
    if comparison is not None:
        experiments, _ = comparison
        fields = [
            "step",
            *[f"{e['name']}_{kind}_loss" for e in experiments for kind in ("train", "validation")],
        ]
        maps = {
            (e["name"], kind): dict(e[kind])
            for e in experiments
            for kind in ("train", "validation")
        }
        rows = [
            {
                "step": step,
                **{
                    f"{name}_{kind}_loss": values.get(step, "")
                    for (name, kind), values in maps.items()
                },
            }
            for step, _ in experiments[0]["train"]
        ]
        write_csv(run / MANAGED_CSV[2], rows, fields)
        exported["strategy_comparison"] = MANAGED_CSV[2]
    return exported


# 分别确定最新保存模型与默认协议下的最佳完整评测模型。
# 最新按保存步数，最佳按成功率且同分取更早步数；不把最佳观测当作独立测试成绩。
# 返回最佳评测行和精简模型编号记录，此处不写入磁盘。
def select_checkpoints(run, evaluations):
    """Keep concrete references inside this run; ties prefer the earlier step."""
    checkpoints = []
    for path in (run / "train/checkpoints").glob("[0-9]*"):
        if path.name.isdigit() and (path / "pretrained_model/model.safetensors").is_file():
            checkpoints.append(ensure_owned(path))
    latest = max(checkpoints, key=lambda path: int(path.name)) if checkpoints else None
    # 默认最佳模型只从完整且协议一致的评测里挑选，其他 batch／seed 的结果仍保留展示。
    completed = [
        row
        for row in evaluations
        if row["complete"] and row["n_episodes"] and row.get("selection_eligible", True)
    ]
    best = max(completed, key=lambda row: (row["pc_success"], -row["step"])) if completed else None
    selection = {
        "latest": {
            "checkpoint": latest.name if latest else None,
            "selection": "latest_saved_step",
            "independent_test": False,
            "pc_success": None,
        },
        "best_observed": {
            "checkpoint": f"{best['step']:06d}" if best else None,
            "pc_success": best["pc_success"] if best else None,
            "selection": "same_evaluation_batch",
            "independent_test": False,
        },
    }
    return best, selection


# 协调绘图并收集当前报告实际生成的相对图片路径。
# 成功率图仅传入完整且满足默认协议的结果，跨协议数据仍留在原始记录和表格中。
# 最后只移除已登记但本次未生成的旧派生图片，其他文件保持原样。
def render_curves(run, ctx, config, metrics, summary, evaluations, comparison=None):
    curves = ensure_owned(run / "curves")
    curves.mkdir(exist_ok=True)
    generated = []
    if summary:
        generated.append(charts.render_loss(run, ctx, config, metrics, summary)[1])
    if comparison is not None:
        generated.append(charts.render_comparison(run, comparison)[1])
    rows = [
        row
        for row in evaluations
        if row["complete"] and row["pc_success"] is not None and row.get("selection_eligible", True)
    ]
    if rows:
        generated.append(charts.render_success(run, rows))
    # 仅删除这次报告没有生成的已知图片，禁止按后缀遍历删除用户添加的其他图。
    for name in MANAGED_FIGURES:
        path = curves / name
        if str(path.relative_to(run)) not in generated:
            ensure_owned(path).unlink(missing_ok=True)
    return generated


# 把报告单元格转换为文字：缺失显示“未记录”，小量使用适当有效位数。
# 仅影响 Markdown 的呈现，不修改原始数值或 CSV 中的精度。
def display(value, digits=2):
    if value is None:
        return "未记录"
    if isinstance(value, float) and 0 < abs(value) < 10 ** (-digits):
        return f"{value:.4g}"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


# 将已记录秒数同时显示为分钟与秒；缺失时显示“未记录”。
# 计时口径由对应字段决定，这个格式化函数不把循环时间重命名为墙钟时间。
def duration(seconds):
    return (
        f"{seconds / 60:.2f} 分钟（{seconds:.1f} 秒）"
        if isinstance(seconds, (float, int))
        else "未记录"
    )


# 生成一次运行的完整报告，返回包含详细统计的临时 summary 字典。
# run 为运行目录；plots 控制绘图，ctx 可显式指定实验，compare_with 指定对比运行。
# export_csv 默认为 False；根据本次参数更新已知派生文件集合，原始输入始终保留。
# 先读取并校验输入，再生成派生内容；选模与关键汇总通过一次带锁分区更新提交。
def generate_report(run, plots=True, ctx=None, compare_with=None, export_csv=False):
    ctx = ctx or ExperimentContext.for_run(run)
    run = ctx.owned(Path(run).resolve())
    if compare_with is not None:
        compare_with = Path(compare_with).resolve()
    if not run.is_dir():
        raise FileNotFoundError(run)
    record = run_record(ctx, run)
    # 本次统计使用一个运行记录快照；提交结果时 Record.update 会在锁内再次读取最新版本。
    document = record.read()
    config = document["config"]
    policy = config.get("policy", {}).copy()
    if policy.get("path"):
        policy = {**optional_json(Path(policy["path"]) / "config.json"), **policy}
    split = optional_json(ctx.root / "data/split.json")
    timing, runtime = document["timing"], document["runtime"]
    protocol = document["evaluation"].get("protocol", {})
    metrics = load_metrics(run)
    # 缓存每轮原始 JSON，同时用于统计与后面的恢复说明，避免同一文件被重复读取。
    raw_evaluations = {}
    paths = sorted((run / "eval").glob("*/eval_info.json"))
    evaluations, tasks = [], []
    for path in paths:
        if path.parent.name.split("_b")[0].isdigit():
            raw_evaluations[path.parent.name] = read(ensure_owned(path))
            overall, per_task = summarize_evaluation(
                path,
                result=raw_evaluations[path.parent.name],
                protocol=evaluation_protocol(run, path.parent.name, document=document),
            )
            evaluations.append(overall)
            tasks.extend(per_task)
    # 所有来源校验完成后再替换输出；原始结果损坏时保留上一次有效报告供排查。
    comparison = (
        collect_comparison(run, compare_with, ctx=ctx, split=split, metrics=metrics)
        if compare_with
        else None
    )
    steps = sorted(set(EXPECTED_STEPS) | {row["step"] for row in evaluations})
    best, selection = select_checkpoints(run, evaluations)
    # 完整统计只作为返回值和报告渲染输入；落盘时通过 compact_summary 选择必要字段。
    summary = {
        "expected_steps": steps,
        "completed_steps": [row["step"] for row in evaluations if row["complete"]],
        "checkpoints": evaluations,
        "per_task": tasks,
        "selection_uses_same_evaluation_batch": True,
        "independent_test": False,
        "training_loss": loss_statistics(metrics),
        "strategy_comparison": comparison[1] if comparison else None,
        "performance": performance_statistics(document, metrics),
        "evaluation_protocol": protocol,
        "training_runtime": runtime,
        "timing": timing,
    }
    # CSV 是可选派生输出。省略导出参数时按明确清单移除上次导出的表格，
    # 原始 train_metrics 与 eval 目录的 CSV 不受影响。
    exported = export_tables(run, evaluations, tasks, comparison) if export_csv else {}
    for name in MANAGED_CSV:
        if name not in exported.values():
            ensure_owned(run / name).unlink(missing_ok=True)
    curves = (
        render_curves(run, ctx, config, metrics, summary["training_loss"], evaluations, comparison)
        if plots
        else []
    )
    summary["exports"] = exported
    summary["figures"] = curves

    # 提交评测分区时复用锁内最新记录，避免覆盖并发写入的 runtime 等分区。
    # 只有选模内容有语义变化才替换 selection，保留历史整数／浮点数等既有数值表示。
    def save_result(value):
        evaluation = value["evaluation"]
        if evaluation.get("selection") != selection:
            evaluation["selection"] = selection
        evaluation["summary"] = compact_summary(summary)

    # 选模与精简汇总在同一次记录事务中提交；任何其他分区仍以锁内最新内容为准。
    record.update(save_result)
    # 之后只组织 Markdown 文本，使用已计算结果，不再重新执行仿真或读取训练指标。
    text = [
        "# LIBERO-Spatial 实验结果",
        "",
        f"实验目录：`{run}`。状态：{timing.get('status', runtime.get('status', '未记录'))}；"
        f"正式训练已完成 {runtime.get('step', 0)} 步。",
        "",
        "## 实际配置",
        "",
        "| 配置 | 值 |",
        "|---|---|",
    ]
    settings = [
        (key, policy.get(key))
        for key in (
            "train_expert_only",
            "freeze_vision_encoder",
            "train_state_proj",
            "optimizer_lr",
            "scheduler_warmup_steps",
            "scheduler_decay_steps",
            "scheduler_decay_lr",
            "chunk_size",
        )
    ]
    settings += [
        ("精度", config.get("accelerator", {}).get("mixed_precision")),
        ("Batch", config.get("batch_size")),
        ("目标步数", config.get("steps")),
        ("随机种子", config.get("seed")),
        ("仿真 n_action_steps", protocol.get("shared", protocol).get("n_action_steps")),
        ("训练轨迹", len(split["train_episodes"]) if "train_episodes" in split else None),
        ("训练帧", split.get("train_frames")),
        ("验证轨迹", len(split["validation_episodes"]) if "validation_episodes" in split else None),
        ("验证帧", split.get("validation_frames")),
        ("可训练参数", runtime.get("trainable_parameters")),
        ("总参数", runtime.get("total_parameters")),
    ]
    text += [f"| {key} | {display(value)} |" for key, value in settings]
    if not runtime:
        text += ["", "尚未记录训练指标；未完成实验不推断效果。"]
    loss = summary["training_loss"]
    if loss:
        text += [
            "",
            f"最终训练 loss：{loss['final_training_loss']:.4f}；最终验证 loss："
            f"{loss['final_validation_loss']:.4f}；最低验证 loss：{loss['minimum_validation_loss']:.4f}"
            f"（{loss['minimum_validation_step']} 步）。验证 flow noise 仍随机。",
        ]
    text += [
        "",
        "## 实测耗时与资源",
        "",
        "| 指标 | 实测值 |",
        "|---|---:|",
        f"| 正式训练总墙钟 | {duration(timing.get('training_wall_s'))} |",
        f"| 训练循环累计（含循环内验证和保存） | {duration(runtime.get('elapsed_s'))} |",
        f"| 优化器更新累计 | {duration(runtime.get('optimizer_update_s'))} |",
        f"| 离线验证 | {duration(runtime.get('validation_s'))} |",
        f"| 检查点保存 | {duration(runtime.get('checkpoint_s'))} |",
    ]
    for session in summary["performance"]["sessions"]:
        text += [
            f"| {session['from_step']}–{session['target_step']} 步阶段墙钟 | {duration(session.get('training_wall_s'))} |"
        ]
    perf = summary["performance"]
    text += [
        f"| PyTorch 分配峰值 | {display(perf['torch_peak_allocated_gib'])} GiB |",
        f"| 整卡采样显存最大值 | {display(perf['sampled_gpu_memory_max_mib'])} MiB |",
        "",
        "阶段墙钟不含隔夜暂停，包含初始化、验证和保存。各子项存在包含关系，不能直接相加。"
        "整卡显存为采样观测，与 PyTorch 分配峰值含义不同；不跨采样空档推算利用率或能耗。",
        "",
        "## 检查点仿真结果",
        "",
        "| 步数 | 成功数/回合数 | 成功率 | 95% Wilson 区间 | 平均回合步数 | 评测时间 | 状态 |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    by_step = {row["step"]: row for row in evaluations if row.get("selection_eligible", True)}
    for step in steps:
        row = by_step.get(step)
        if not row:
            text.append(f"| {step} | — | — | — | — | — | 待评测 |")
            continue
        low, high = row["pc_success_ci95"]
        text.append(
            f"| {step} | {row['n_success']}/{row['n_episodes']} | {display(row['pc_success'])}% | "
            f"[{display(low)}, {display(high)}]% | {display(row['avg_episode_length'])} | "
            f"{duration(row['eval_s'])} | {'完成' if row['complete'] else '不完整'} |"
        )
    if best:
        text += [
            "",
            f"本轮最佳观测检查点：{best['checkpoint']}，成功率 {best['pc_success']:.1f}%；同分取更早检查点。",
        ]
    text += [
        "",
        "只有符合默认评测协议且完成的10任务×每任务10回合参与选模；不同 batch 或 seed 的结果单独保留。同批评测用于比较和选模，最佳观测成绩并非独立测试成绩；"
        "单种子实验不衡量训练随机性的稳定性。平均回合步数包含失败回合。",
    ]
    if tasks:
        text += [
            "",
            "## 逐任务结果",
            "",
            "| 检查点 | 任务 | 成功数/回合数 | 成功率 | 95% Wilson 区间 | 平均回合步数 | 耗时 |",
            "|---|---|---:|---:|---|---:|---:|",
        ]
        for row in tasks:
            low, high = row["pc_success_ci95"]
            text.append(
                f"| {row['checkpoint']} | {row['task_group']}/{row['task_id']} | "
                f"{row['n_success']}/{row['n_episodes']} | {display(row['pc_success'])}% | "
                f"[{display(low)}, {display(high)}]% | {display(row['avg_episode_length'])} | {duration(row['eval_s'])} |"
            )
    text += [
        "",
        "## 原始数据与结果图",
        "",
        "train_metrics/ 的 CSV 保存训练、验证、GPU 原始指标；"
        + (
            "逐步耗时保存于 train_metrics/step_timing.csv。"
            if (run / "train_metrics/step_timing.csv").is_file()
            else "本次运行未记录逐步计时 CSV。"
        )
        + "run.json 统一保存运行记录、关键汇总及最佳／最新模型编号。",
        "原始 eval/*/eval_info.json 保留每回合成功、奖励与步数；"
        "详细逐回合数据只保存在原始评测文件中，不复制到管理记录。",
    ]
    if exported:
        text += [
            "",
            "已显式导出表格："
            + "、".join(f"[{path}]({path})" for path in exported.values())
            + "。",
        ]
    else:
        text += [
            "",
            "需要 CSV 表格时，使用根入口 --report-only RUN --export-csv 重新生成；策略对比需同时指定 --compare-with。",
        ]
    text += ["", *[f"![{Path(path).stem}]({path})" for path in curves]]
    # 恢复说明直接使用已缓存的原始 recovery 字段，保留奖励缺失、随机状态及耗时口径局限。
    for row in evaluations:
        info = raw_evaluations[row["checkpoint"]]
        recovery = info.get("recovery")
        if recovery:
            original = recovery["original_completed_episodes"]
            text += [
                "",
                f"## {row['step']} 步评测恢复说明",
                "",
                f"前 {original} 回合结果由原日志累计成功率还原，长度取自原视频；"
                f"后 {recovery['completed_episodes'] - original} 回合为续测。原视频未修改。"
                "原回合奖励未保存，整体 reward 统计留空。策略噪声按已完成步数重建，"
                "不承诺与未中断运行位级一致；评测时间部分来自原日志估计，不含隔夜停机。",
            ]
            source = run / "eval" / row["checkpoint"] / "recovery_source.log.gz"
            if source.is_file():
                text += [
                    f"恢复所用[原日志压缩归档]({source.relative_to(run).as_posix()})；"
                    "解压 SHA256 保留于原始评测结果 recovery 字段。"
                ]
    ensure_owned(run / "REPORT.md").write_text("\n".join(text) + "\n")
    return summary
