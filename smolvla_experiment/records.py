# 统一管理 JSON、评测协议与逐步计时的存储层。
# 实验级信息写入 metadata/experiment.json，单次运行信息写入 runs/<运行名>/run.json。
# 管理文件使用独立短时锁：锁内重新读取、修改指定分区、校验后原子替换。
# 这些短时锁保护父子进程的记录更新；整个工作流的互斥由 pipeline.py 另行管理。
# 模型、分词器及检查点中的原生 JSON 保持原生格式，不经过此管理记录封装。

"""Locked, atomic management records; native model JSON stays framework-owned."""

import copy
import csv
import fcntl
import hashlib
import json
import math
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .common import ExperimentContext, read

# 管理格式版本与必需分区：版本变化应显式迁移，不能静默按空记录处理。
SCHEMA_VERSION = 1
EXPERIMENT_SECTIONS = ("provenance", "assets", "preparation", "activity", "launcher")
RUN_SECTIONS = ("config", "runtime", "timing", "budget", "evaluation", "probes", "diagnostics")


# 管理一个实验级或运行级 JSON 的结构校验和并发更新。
# ctx 限定可写范围，path 指定文件，kind 决定应有的分区结构。
class Record:
    # 保存上下文和记录类型，构造时即检查目标路径属于当前实验。
    def __init__(self, ctx, path, kind):
        self.ctx, self.path, self.kind = ctx, ctx.owned(path), kind

    # 构造尚未落盘时的完整空记录，所有管理分区默认为字典。
    # 运行记录额外使用 sessions 列表保存训练与续训会话，不与分区字典混用。
    def empty(self):
        sections = EXPERIMENT_SECTIONS if self.kind == "experiment" else RUN_SECTIONS
        value = {"schema_version": SCHEMA_VERSION, **{key: {} for key in sections}}
        if self.kind == "run":
            value["sessions"] = []
        return value

    # 校验版本、必需分区类型和所有数值的 JSON 可序列化性。
    # 保留额外历史字段；布尔值不能冒充整数版本号，非有限浮点数也会被拒绝。
    def validate(self, value):
        expected = self.empty()
        if (
            not isinstance(value, dict)
            or type(value.get("schema_version")) is not int
            or value["schema_version"] != SCHEMA_VERSION
        ):
            raise ValueError(f"Invalid management record schema: {self.path}")
        for key, default in expected.items():
            if key != "schema_version" and not isinstance(value.get(key), type(default)):
                raise ValueError(f"Invalid management record section {key}: {self.path}")
        json.dumps(value, allow_nan=False)
        return value

    # 缺失时返回规范空结构，已有文件则严格解析并校验。
    # 损坏文件明确报错，避免用默认空记录覆盖仍有诊断价值的原文件。
    def read(self):
        if not self.path.exists():
            return self.empty()
        try:
            return self.validate(read(self.ctx.owned(self.path)))
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"Corrupt management record {self.path}: {error}") from error

    # 在独立短时锁内读取最新内容，再对深拷贝执行 change 回调。
    # 回调只修改需要的分区；校验通过且内容有变化时才原子保存。
    # 返回本次修改后的文档，避免父进程、训练子进程和后台启动信息互相覆盖。
    def update(self, change):
        """Reload while holding a separate short-lived lock, then replace atomically."""
        with file_lock(self.ctx, self.path.with_suffix(".lock")):
            before = self.read()
            value = copy.deepcopy(before)
            # 回调可修改副本，原始 before 用于判断是否真的需要写盘；回调失败不会触发保存。
            change(value)
            self.validate(value)
            if value != before or not self.path.exists():
                self.ctx.save(self.path, value)
            return value

    # 替换一个合法的顶层分区，内部仍走完整的加锁读改写流程。
    # 禁止通过这个接口重写 schema_version 或添加拼写错误的分区名。
    def set(self, section, value):
        if section not in self.empty() or section == "schema_version":
            raise ValueError(f"Unknown management section: {section}")
        return self.update(lambda document: document.__setitem__(section, value))


# 在当前实验内取得排他文件锁，退出上下文时一定释放。
# 锁文件与被原子替换的 JSON 分开，避免替换文件后出现两组进程锁住不同 inode。
@contextmanager
def file_lock(ctx, path):
    path = ctx.owned(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


# 创建实验级记录接口，路径固定为 metadata/experiment.json。
def experiment_record(ctx):
    return Record(ctx, ctx.metadata / "experiment.json", "experiment")


# 创建某次运行的记录接口，路径固定为该运行目录下的 run.json。
def run_record(ctx, run):
    return Record(ctx, ctx.owned(run) / "run.json", "run")


# 仅对 provenance 与 assets 两个来源分区计算规范化 SHA256。
# 排序键和固定分隔符消除 JSON 排版影响；排除活动状态及准备记录本身，避免循环校验。
def sources_digest(document):
    """Exclude mutable activity/preparation fields and never hash our own file."""
    source = {key: document[key] for key in ("provenance", "assets")}
    content = json.dumps(
        source, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(content.encode()).hexdigest()


# 把 train 或 probe_b* 输出目录映射到其所属运行的 run.json。
def training_record(ctx, output):
    output = ctx.owned(output)
    return run_record(ctx, output.parent)


# 读取正式训练或短跑的指定信息，屏蔽两者在管理记录中的存储位置差异。
# 正式梯度诊断位于 diagnostics；短跑信息位于 probes/<短跑名称>。
def training_values(ctx, output, section):
    document = training_record(ctx, output).read()
    if output.name == "train":
        return (
            document["diagnostics"].get("gradient_check", {})
            if section == "gradient_check"
            else document.get(section, {})
        )
    return document["probes"].get(output.name, {}).get(section, {})


# 按正式训练／短跑类别更新对应位置，其余分区由锁内重读保留。
# 短跑不会额外创建独立的配置或诊断 JSON。
def set_training_values(ctx, output, section, value):
    record = training_record(ctx, output)
    if output.name == "train":
        if section == "gradient_check":
            return record.update(
                lambda document: document["diagnostics"].__setitem__(section, value)
            )
        return record.set(section, value)

    # 仅更新指定短跑的一个字段，保留其他短跑及正式训练记录。
    def update(document):
        document["probes"].setdefault(output.name, {})[section] = value

    return record.update(update)


# 为一次原生训练调用提供临时 train_config.json 路径。
# 退出上下文时由 TemporaryDirectory 清理，即使子进程失败也不留下零散交接配置。
# 训练使用过的有效配置另行保存在运行记录，完整检查点保留自己的原生配置。
@contextmanager
def temporary_config(ctx, config):
    """Only the native trainer handoff uses a transient standalone config JSON."""
    directory = ctx.owned(ctx.root / ".cache/configs")
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="native_", dir=directory) as temporary:
        path = Path(temporary) / "train_config.json"
        ctx.save(path, config)
        yield path


# 由训练输出名推导对应的逐步计时路径，如 train_metrics/step_timing.csv。
def step_timing_path(ctx, output):
    return ctx.owned(output.parent / f"{output.name}_metrics/step_timing.csv")


# 读取并校验两列计时 CSV，返回整数步数和浮点耗时组成的列表。
# 文件尚不存在时返回空列表；列名、字段值或历史顺序错误都会报错。
def read_step_timing(path):
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["steps", "update_s"]:
            raise ValueError(f"Invalid step timing columns: {path}")
        try:
            rows = [
                {"steps": int(row["steps"]), "update_s": float(row["update_s"])} for row in reader
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid step timing record: {path}") from error
    validate_step_rows(rows)
    return rows


# 要求步数为正数、严格递增且不重复，耗时为有限的非负数。
# 该校验同时用于已有历史和本次追加数据，防止不合法历史进入续训。
def validate_step_rows(rows):
    steps = [row["steps"] for row in rows]
    if steps != sorted(set(steps)) or any(step <= 0 for step in steps):
        raise ValueError("Step timing steps must be positive, ordered and unique")
    if any(not math.isfinite(row["update_s"]) or row["update_s"] < 0 for row in rows):
        raise ValueError("Step timing must contain finite nonnegative durations")


# 在独立 CSV 锁内追加本会话的逐步更新耗时，只在空文件写一次表头。
# restored 是恢复检查点步数：历史不得超过它，新记录必须从它之后开始。
# 已有末尾大于恢复步数时拒绝追加，避免把过期检查点接到较新的训练历史上。
def append_step_timing(ctx, output, rows, restored):
    path = step_timing_path(ctx, output)
    validate_step_rows(rows)
    # 追加前必须在同一把锁内检查历史，否则两个进程可能同时通过步数检查并重复追加。
    with file_lock(ctx, path.with_suffix(".lock")):
        previous = read_step_timing(path)
        if previous and previous[-1]["steps"] > restored:
            raise ValueError("Timing history extends past the resume checkpoint")
        if rows and rows[0]["steps"] <= restored:
            raise ValueError("New timing rows overlap restored steps")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["steps", "update_s"])
            if path.stat().st_size == 0:
                writer.writeheader()
            writer.writerows(rows)


# 默认可参与选模的基准协议。并行 batch 等不同设置通过每轮 overrides 单独记录。
# 图像、动作执行频率、随机种子和回合设置共同决定结果是否可直接比较。
EVALUATION_PROTOCOL = {
    "suite": "libero_spatial",
    "tasks": 10,
    "episodes_per_task": 10,
    "initial_state_ids": list(range(10)),
    "control_hz": 20,
    "recording_fps": 20,
    "observation_height": 256,
    "observation_width": 256,
    "control_mode": "relative",
    "hard_reset": True,
    "init_states": True,
    "num_steps_wait": 10,
    "n_action_steps": 1,
    "num_inference_steps": 10,
    "episode_length": 220,
    "seed": 0,
    "eval_batch_size": 1,
    "max_parallel_tasks": 1,
    "include_base": False,
}


# 合并默认协议、运行共享协议及单轮 overrides，后者优先。
# 报告可传入已读 document 避免反复读取；独立调用时再按 run 读取管理记录。
def evaluation_protocol(run, label, *, document=None, ctx=None):
    if document is None:
        ctx = ctx or ExperimentContext.for_run(run)
        document = run_record(ctx, run).read()
    document = document["evaluation"].get("protocol", {})
    shared = document.get("shared", document)
    overrides = document.get("evaluations", {}).get(label, {}).get("overrides", {})
    return {**EVALUATION_PROTOCOL, **shared, **overrides}


# 在启动评测前登记该轮检查点和相对共享协议的差异。
# 兼容已有的扁平共享协议内容，并通过带锁更新保留其他评测轮次。
def record_evaluation_protocol(ctx, run, label, checkpoint, batch):
    # 将旧协议转换为 shared/evaluations 组织，只存本轮与共享设置不同的字段。
    # 批量大小写入本轮 overrides，避免把并行评测误计入默认串行选模。
    def update(record):
        previous = record["evaluation"].get("protocol", {})
        document = (
            previous
            if "shared" in previous
            else {
                "shared": {**EVALUATION_PROTOCOL, **previous},
                "evaluations": {},
            }
        )
        desired = {**EVALUATION_PROTOCOL, "eval_batch_size": batch}
        document["evaluations"][label] = {
            "checkpoint": checkpoint,
            "overrides": {
                key: value for key, value in desired.items() if document["shared"].get(key) != value
            },
        }
        record["evaluation"]["protocol"] = document

    run_record(ctx, run).update(update)
