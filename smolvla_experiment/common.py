# 公共基础层：实验上下文、路径边界、离线运行环境和联合微调精度设置。
# ExperimentContext 将选中的配置与 experiments/<实验名>/ 绑定，避免两种实验互相写入。
# 原始数据和基础权重位于共享下载目录，运行产物与缓存按实验隔离。
# 精度修复仅在显式调用时导入 PyTorch，并仅修改当前子进程内的模型构造器。
# 本模块不依赖记录、训练或报告模块，作为其他模块的公共依赖。

"""Explicit experiment paths, guarded writes and append-only metric recording."""

import json
import os
import sys
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

# 项目根目录基于源码位置推导，因此从其他工作目录启动根入口也能定位配置。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ("expert_only", "joint")


# 校验并返回解析后的可写路径。
# path 可以含 ~ 或符号链接；先解析实际目标，再检查是否位于 root 内且不属于 lerobot。
# root 缺省为项目根目录；需要更严格的实验边界时由 ExperimentContext.owned 再次约束。
def ensure_owned(path, root=None):
    """Resolve symlinks before permitting an output under the selected root."""
    root = Path(root or PROJECT_ROOT).resolve()
    path = Path(path).expanduser().resolve()
    if not path.is_relative_to(root) or path.is_relative_to((root / "lerobot").resolve()):
        raise ValueError(f"Output must be inside {root} and outside lerobot: {path}")
    return path


# 读取一个 JSON 文件并返回解析后的 Python 对象。
# 缺失、损坏或解析失败直接抛出异常；是否允许文件缺失由调用者按业务决定。
def read(path):
    return json.loads(Path(path).read_text())


# 将 value 写为缩进 JSON，保留中文字符，并拒绝 NaN/Infinity。
# 先在同目录写 .tmp 文件，再 replace 目标，使读者看到完整的旧版本或新版本。
# 此函数本身不提供并发锁；管理记录的读改写需由 records.py 在持锁范围内调用。
def save(path, value, root=None):
    path = ensure_owned(path, root)
    content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


# 保存单个实验的名称、项目根目录和配置字典，统一推导输入与输出路径。
# frozen=True 防止上下文字段被重新赋值；profile 内容仍由调用者按只读约定使用。
@dataclass(frozen=True)
class ExperimentContext:
    """One profile and its writable workspace; shared inputs are read-only."""

    name: str
    project_root: Path
    profile: dict

    # 从 configs/<name>.json 建立上下文，核对文件名与配置中的实验名一致。
    # project_root 可显式传入，用于隔离的验证工作区；缺省使用当前项目。
    @classmethod
    def load(cls, name, project_root=None):
        if name not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment: {name}")
        root = Path(project_root or PROJECT_ROOT).resolve()
        profile = read(root / "configs" / f"{name}.json")
        if profile["name"] != name:
            raise ValueError("Experiment name and configuration disagree")
        return cls(name, root, profile)

    # 根据运行目录位于哪一个 experiments/<name>/runs/ 下推导实验。
    # 会解析链接并执行边界检查，项目外目录或无法归属的运行路径会被拒绝。
    @classmethod
    def for_run(cls, run, project_root=None):
        root = Path(project_root or PROJECT_ROOT).resolve()
        run = Path(run).resolve()
        for name in EXPERIMENTS:
            ctx = cls.load(name, root)
            if run.is_relative_to(ctx.root / "runs"):
                ctx.owned(run)
                return ctx
        raise ValueError(f"Run is outside the experiment workspaces: {run}")

    # 当前实验的工作目录：准备产物、运行记录及缓存均以此为写入边界。
    @property
    def root(self):
        return self.project_root / "experiments" / self.name

    # 实验级管理目录，用于来源记录、准备校验和环境快照。
    @property
    def metadata(self):
        return self.root / "metadata"

    # 底层 LeRobot 仓库路径；项目代码不在此处写入实验产物。
    @property
    def repo(self):
        return self.project_root / "lerobot"

    # 共享的原始下载目录，包含数据集、基础模型及处理器资源。
    @property
    def resources(self):
        return self.repo / "downloads/smolvla_libero"

    # 两个实验共享的 LIBERO 仿真素材目录。
    @property
    def assets(self):
        return self.project_root / "resources/libero_assets"

    # 返回冻结策略字典的副本，避免调用者修改返回值时影响原始实验配置。
    @property
    def flags(self):
        return self.profile["policy"].copy()

    # 给出梯度及参数变化检查必须覆盖的代表性模块组。
    # 动作专家检查 expert/state/action；联合微调还需覆盖连接器、嵌入和代表性语言层。
    @property
    def required_groups(self):
        groups = ("expert", "state", "action")
        return (
            groups
            if self.flags["train_expert_only"]
            else ("connector", "embedding", "language_early", "language_late", *groups)
        )

    # 把通用路径约束收紧到当前实验目录，并禁止实验根目录重定向到其他位置。
    # 返回真实路径，供后续创建、写入或删除操作使用。
    def owned(self, path):
        ensure_owned(path, self.project_root)
        if self.root.resolve() != self.root:
            raise ValueError("Experiment root must not redirect through a symlink")
        return ensure_owned(path, self.root)

    # 在当前实验边界内保存 JSON，复用公共原子替换逻辑。
    def save(self, path, value):
        save(self.owned(path), value, self.project_root)

    # 构造当前解释器的子进程参数列表，并显式传递实验名称。
    # -B 禁止生成字节码缓存；列表参数直接交给 subprocess，不经过 shell 拼接。
    def command(self, module, *arguments):
        return [
            sys.executable,
            "-B",
            "-m",
            module,
            "--experiment",
            self.name,
            *[str(argument) for argument in arguments],
        ]


# 在导入模型、数据集或仿真库之前设置运行环境。
# 缓存和 LIBERO 配置按实验隔离，下载库启用离线模式，仿真使用 EGL 无界面后端。
# 本函数只设置当前进程环境变量，不主动创建目录或加载模型。
def configure_environment(ctx):
    sys.dont_write_bytecode = True
    cache = ctx.root / ".cache"
    # 将运行库常见的缓存位置收束到当前实验，离线标志阻止缺失输入时自动联网补齐。
    values = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HOME": str(cache / "huggingface"),
        "HF_DATASETS_CACHE": str(cache / "datasets"),
        "HF_LEROBOT_HOME": str(cache / "lerobot"),
        "HF_MODULES_CACHE": str(cache / "modules"),
        "XDG_CACHE_HOME": str(cache),
        "TORCH_HOME": str(cache / "torch"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "NUMBA_CACHE_DIR": str(cache / "numba"),
        "MPLCONFIGDIR": str(cache / "matplotlib"),
        "LIBERO_CONFIG_PATH": str(ctx.root / "libero_config"),
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "WANDB_MODE": "disabled",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "2",
    }
    os.environ.update(values)


# 安装仅限当前进程的模型构造包装，用于联合微调的 FP32 参数存储。
# 在原生外部权重加载及优化器创建之前转换参数，防止先按 BF16 接收权重时丢失精度。
# 重复调用会直接返回；BF16 autocast 计算策略由训练器配置，仍保持不变。
def install_fp32_policy_storage():
    """Cast before loading weights/creating AdamW so small updates remain representable.

    Nested Transformers configs otherwise construct the language/expert layers in
    BF16. Casting after checkpoint loading would already have rounded FP32 resume
    weights, so the constructor is wrapped in this process only.
    """
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    if getattr(SmolVLAPolicy, "_local_fp32_storage", False):
        return
    # 保留原构造器，包装只在本进程安装一次；动作专家子进程不会调用这个安装函数。
    original_init = SmolVLAPolicy.__init__

    # 原构造器完成后立即转换参数，并同步配置中的 dtype。
    # 此处处理参数存储；实际检查点读取仍由原生 from_pretrained 流程完成。
    @wraps(original_init)
    def initialize(policy, *args, **kwargs):
        original_init(policy, *args, **kwargs)
        policy.float()
        policy.config.dtype = torch.float32

    SmolVLAPolicy.__init__ = initialize
    SmolVLAPolicy._local_fp32_storage = True
