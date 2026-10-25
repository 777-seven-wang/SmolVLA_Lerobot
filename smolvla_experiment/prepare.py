# 离线准备模块：把共享下载资源适配为当前实验可使用的数据视图和模型输入。
# 首次准备会复制小型元数据、链接原始数据与基础权重，生成划分、归一化统计和处理器。
# 已有准备记录时优先核验摘要与产物，不重新覆盖已完成的输入。
# 实验元数据及本地配置只写入当前实验目录，共享下载目录保持只读。
# 轨迹数量和帧数按当前固定 LIBERO-Spatial 数据版本校验，数据变化会明确报错。

"""Prepare an offline Spatial data view without modifying downloaded inputs."""

import argparse
import fcntl
import hashlib
import importlib.util
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from .common import EXPERIMENTS, ExperimentContext, configure_environment, read
from .records import experiment_record, sources_digest

# 准备版本约束产物的生成约定，与管理 JSON 的 schema_version 分别维护。
PREPARATION_VERSION = 4


# 组合当前冻结设置与离线模型覆盖项。
# 禁用额外 VLM 权重加载和上传；闭环评测默认每次仅执行预测动作块的第一步。
def policy_overrides(ctx):
    return {**ctx.flags, "load_vlm_weights": False, "push_to_hub": False, "n_action_steps": 1}


# 按二进制流计算文件摘要，用于准备产物及来源校验，不改变文件内容。
def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# 仅在内容不同或文件缺失时写入配置。
# 已有完成准备记录时拒绝覆盖变化的产物，防止静默改变历史实验的有效输入。
def _save_if_changed(ctx, path, value):
    ctx.owned(path)
    if not Path(path).exists() or read(path) != value:
        if bool(experiment_record(ctx).read()["preparation"]):
            raise RuntimeError(f"Refusing to change a completed preparation: {path}")
        ctx.save(path, value)


# 给现有共享输入建立符号链接；已有链接必须解析到同一来源。
# 不覆盖普通文件或错误链接，来源缺失也直接失败，避免生成悬空输入链接。
def _link_input(ctx, link, source):
    """Link an existing read-only input, refusing unexpected existing targets."""
    source = Path(source).resolve(strict=True)
    if link.is_symlink() or link.exists():
        if not link.is_symlink() or link.resolve() != source:
            raise RuntimeError(f"Refusing to replace existing input path: {link}")
        return
    ctx.owned(link)
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(source, target_is_directory=source.is_dir())


# 校验共享素材、LIBERO 安装目录和素材完成状态，并配置本实验的仿真路径。
# 不解压或重新下载素材；仅在需要时建立 assets 链接和本地 config.yaml。
def prepare_assets(ctx):
    """Validate shared offline assets and configure LIBERO before importing it."""
    configure_environment(ctx)
    folder = ctx.assets
    for relative in ("scenes", "textures", "articulated_objects", "wall.xml"):
        if not (folder / relative).exists():
            raise FileNotFoundError(folder / relative)
    _link_input(ctx, ctx.root / "assets", folder)
    manifest = experiment_record(ctx).read()["assets"]
    if manifest.get("status") != "complete":
        raise RuntimeError("LIBERO assets are incomplete")
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("LIBERO is not installed in the current Python environment")
    package = Path(next(iter(spec.submodule_search_locations))) / "libero"
    for relative in ("bddl_files", "init_files"):
        if not (package / relative).is_dir():
            raise FileNotFoundError(package / relative)
    _save_if_changed(
        ctx,
        ctx.root / "libero_config/config.yaml",
        {
            "benchmark_root": str(package),
            "bddl_files": str(package / "bddl_files"),
            "init_states": str(package / "init_files"),
            "datasets": str(ctx.root / "data"),
            "assets": str(folder),
        },
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(ctx.root / "libero_config")


# 按每个 Spatial 任务的完整轨迹划分训练与验证，末尾 ceil(10%) 轨迹留作验证。
# numeric 是逐帧数值记录，names 将数据任务编号映射到语言，spatial 将语言映射到仿真编号。
# 返回 split 字典和轨迹到数据任务编号的映射；同时拒绝任务缺失、轨迹跨任务和集合重叠。
def build_split(numeric, names, spatial):
    """Hold out the final ceil(10%) complete episodes separately for each task."""
    task_ids = {task for task, name in names.items() if name in spatial}
    if len(task_ids) != len(spatial):
        raise ValueError("Dataset task names do not match the requested benchmark")
    episode_task, groups = {}, defaultdict(set)
    for row in numeric:
        episode, task = row["episode_index"], row["task_index"]
        if episode in episode_task and episode_task[episode] != task:
            raise ValueError(f"Episode {episode} contains multiple tasks")
        episode_task[episode] = task
        if task in task_ids:
            groups[task].add(episode)
    if set(groups) != task_ids:
        raise ValueError("Some Spatial tasks have no episodes")
    train, validation = [], []
    for _, episodes in sorted(groups.items()):
        episodes = sorted(episodes)
        held_out = math.ceil(len(episodes) * 0.1)
        if len(episodes) <= held_out:
            raise ValueError("Each task must have both training and validation episodes")
        train.extend(episodes[:-held_out])
        validation.extend(episodes[-held_out:])
    train_ids, validation_ids = set(train), set(validation)
    if train_ids & validation_ids:
        raise ValueError("Training and validation episodes overlap")
    return {
        "spatial_task_indices": sorted(task_ids),
        "train_episodes": sorted(train),
        "validation_episodes": sorted(validation),
        "episodes": sorted(train + validation),
        "train_frames": sum(r["episode_index"] in train_ids for r in numeric),
        "validation_frames": sum(r["episode_index"] in validation_ids for r in numeric),
        "task_mapping": {
            str(task): {"language": names[task], "env_task_id": spatial[names[task]]}
            for task in sorted(task_ids)
        },
    }, episode_task


# 只用训练轨迹中的 state/action 帧重算 min/max/mean/std/count。
# 保留原统计中的其他特征，使用 float64 计算并检查空数据与非有限值，防止验证集泄漏。
def training_stats(numeric, train_episodes, original_stats):
    """Compute state/action statistics exclusively from training trajectories."""
    import numpy as np

    stats = dict(original_stats)
    train_ids = set(train_episodes)
    for feature in ("observation.state", "action"):
        array = np.asarray(
            [row[feature] for row in numeric if row["episode_index"] in train_ids],
            dtype=np.float64,
        )
        if not array.size or not np.isfinite(array).all():
            raise ValueError(f"Invalid training values for {feature}")
        stats[feature] = {
            "min": array.min(0).tolist(),
            "max": array.max(0).tolist(),
            "mean": array.mean(0).tolist(),
            "std": array.std(0).tolist(),
            "count": [len(array)],
        }
    return stats


# 核验已经完成的准备：来源摘要、准备版本、各产物哈希及冻结设置必须一致。
# 同时确认大数据和权重链接仍指向固定输入；全部通过后跳过首次准备逻辑。
# 返回 False 表示尚无完成记录，返回 True 表示现有准备可复用。
def _completed_preparation(ctx):
    """Validate completed products without rewriting files used by a running job."""
    document = experiment_record(ctx).read()
    prepared = document["preparation"]
    if not prepared:
        return False
    if prepared.get("sources_sha256") != sources_digest(document):
        raise RuntimeError("Prepared source metadata changed")
    if prepared.get("preparation_version") != PREPARATION_VERSION:
        raise RuntimeError("Preparation version changed; use a new experiment directory")
    for relative, digest in prepared["output_sha256"].items():
        path = ctx.root / relative
        ctx.owned(path)
        if not path.is_file() or path.is_symlink() or sha256(path) != digest:
            raise RuntimeError(f"Prepared file changed or disappeared: {path}")
    config = read(ctx.root / "models/base_spatial/config.json")
    if any(config.get(key) != value for key, value in policy_overrides(ctx).items()):
        raise RuntimeError("Prepared model has incompatible fine-tuning settings")
    _link_input(ctx, ctx.root / "data/libero/data", ctx.resources / "datasets/libero/data")
    _link_input(ctx, ctx.root / "data/libero/videos", ctx.resources / "datasets/libero/videos")
    _link_input(
        ctx,
        ctx.root / "models/base_spatial/model.safetensors",
        ctx.resources / "models/smolvla_base/model.safetensors",
    )
    print("Preparation already complete; verified existing products without rewriting", flush=True)
    return True


# 执行持有准备锁后的准备主体；先尝试复用已有产物。
# 首次运行依次建立独立元数据、生成划分与统计、适配模型处理器，再写入来源和校验记录。
# 只有真实产物均完成后才登记 preparation，防止中途失败被误认为准备成功。
def _prepare_data_locked(ctx):
    if _completed_preparation(ctx):
        return
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from libero.libero import benchmark

    source = ctx.resources / "datasets/libero"
    target = ctx.root / "data/libero"
    ctx.owned(target / "meta")
    for path in (target / "meta").rglob("*"):
        ctx.owned(path)
        if path.is_symlink():
            raise RuntimeError(f"Existing metadata must consist of owned copies: {path}")
    # 解引用来源链接，建立真正独立的元数据副本，避免通过共享链接改写下载资源。
    shutil.copytree(source / "meta", target / "meta", dirs_exist_ok=True, symlinks=False)
    for path in (target / "meta").rglob("*"):
        ctx.owned(path)
        if path.is_symlink():
            raise RuntimeError(f"Metadata must consist of owned copies: {path}")
    for name in ("data", "videos"):
        _link_input(ctx, target / name, source / name)
    tasks = pq.read_table(source / "meta/tasks.parquet").to_pylist()
    names = {row["task_index"]: row["__index_level_0__"] for row in tasks}
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    spatial = {suite.get_task(i).language: i for i in range(suite.n_tasks)}
    columns = ["episode_index", "task_index", "observation.state", "action"]
    # 只读取划分和 state/action 统计需要的数值列，不在准备阶段解码全部视频。
    numeric = pa.concat_tables(
        [
            pq.read_table(path, columns=columns)
            for path in sorted((source / "data").rglob("*.parquet"))
        ]
    ).to_pylist()
    split, episode_task = build_split(numeric, names, spatial)
    # 固定数据版本的训练／验证轨迹数与帧数；不同来源不能沿用本实验的历史划分解释。
    expected = (384, 48, 47043, 5927)
    actual = (
        len(split["train_episodes"]),
        len(split["validation_episodes"]),
        split["train_frames"],
        split["validation_frames"],
    )
    if actual != expected:
        raise RuntimeError(f"Pinned dataset split changed: expected {expected}, got {actual}")
    # 修改的是实验内独立复制的 episodes 元数据，补充语言任务列以适配原生数据读取器。
    for path in sorted((target / "meta/episodes").rglob("*.parquet")):
        ctx.owned(path)
        table = pq.read_table(path)
        values = [[names[episode_task[episode]]] for episode in table["episode_index"].to_pylist()]
        if "tasks" in table.column_names:
            table = table.drop(["tasks"])
        pq.write_table(
            table.append_column("tasks", pa.array(values, type=pa.list_(pa.string()))), path
        )
    # 划分完成后再计算训练统计，验证轨迹仅用于离线验证，不参与归一化估计。
    stats = training_stats(numeric, split["train_episodes"], read(source / "meta/stats.json"))
    ctx.save(target / "meta/stats.json", stats)
    ctx.save(ctx.root / "data/split.json", split)

    # 生成当前实验独立的小型模型配置／处理器；体积大的基础权重通过只读输入链接复用。
    base = ctx.resources / "models/smolvla_base"
    adapted = ctx.root / "models/base_spatial"
    ctx.owned(adapted)
    adapted.mkdir(parents=True, exist_ok=True)
    config = read(base / "config.json")
    config.update(policy_overrides(ctx))
    config["vlm_model_name"] = str(ctx.resources / "models/smolvlm_processor")
    config["input_features"] = {
        "observation.state": {"type": "STATE", "shape": [8]},
        **{
            f"observation.images.{cam}": {"type": "VISUAL", "shape": [3, 256, 256]}
            for cam in ("image", "image2")
        },
    }
    config["output_features"] = {"action": {"type": "ACTION", "shape": [7]}}
    ctx.save(adapted / "config.json", config)
    _link_input(ctx, adapted / "model.safetensors", base / "model.safetensors")
    # load_vlm_weights=False 已先写入配置，后续读取配置和构造处理器无需额外下载模型。
    cfg = SmolVLAConfig.from_pretrained(str(adapted))
    pre, post = make_smolvla_pre_post_processors(cfg, dataset_stats=stats)
    pre.save_pretrained(adapted)
    post.save_pretrained(adapted)

    manifest = read(ctx.resources / "download_manifest.json")
    experiment_record(ctx).set(
        "provenance",
        {
            "code_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ctx.repo, text=True
            ).strip(),
            "resources": {
                key: {field: value for field, value in entry.items() if field != "files"}
                for key, entry in manifest["resources"].items()
            },
            "base_weight_sha256": sha256(base / "model.safetensors"),
            "base_config_sha256": sha256(base / "config.json"),
            "stats_sha256": sha256(target / "meta/stats.json"),
            "split_sha256": sha256(ctx.root / "data/split.json"),
            "source_info_sha256": sha256(source / "meta/info.json"),
            "source_tasks_sha256": sha256(source / "meta/tasks.parquet"),
            "dataset_fps": read(source / "meta/info.json")["fps"],
            "simulation_control_hz": 20,
            "policy_overrides": policy_overrides(ctx),
        },
    )
    # 冻结本次准备环境的包列表，作为可校验产物保存；之后的准备复用不会重写历史快照。
    environment = ctx.metadata / "environment.txt"
    ctx.owned(environment)
    environment.write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    # 收集真正写入当前实验的准备文件。共享符号链接不作为输出文件重新哈希，
    # 来源摘要单独记录，管理 JSON 自身不纳入产物列表，避免记录自己的哈希形成循环。
    products = [
        ctx.root / "data/split.json",
        environment,
        ctx.root / "libero_config/config.yaml",
        *sorted((target / "meta").rglob("*")),
        *sorted(adapted.rglob("*")),
    ]
    experiment_record(ctx).set(
        "preparation",
        {
            "preparation_version": PREPARATION_VERSION,
            "sources_sha256": sources_digest(experiment_record(ctx).read()),
            "output_sha256": {
                str(path.relative_to(ctx.root)): sha256(path)
                for path in products
                if path.is_file() and not path.is_symlink()
            },
        },
    )
    print(
        f"Prepared {actual[0]} train / {actual[1]} validation episodes ({actual[2]}/{actual[3]} frames)",
        flush=True,
    )


# 先配置仿真素材，再用 prepare.lock 串行执行数据和处理器准备。
# 这个锁只保护准备阶段，与管理 JSON 的短时更新锁及整个工作流锁分别使用。
def prepare_data(ctx):
    prepare_assets(ctx)
    lock = ctx.metadata / "prepare.lock"
    ctx.owned(lock)
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        _prepare_data_locked(ctx)


# 准备模块的内部命令入口，接收实验名和准备范围。
# 由调度器调用时执行完整离线准备，也可单独校验素材；不会启动训练或仿真回合。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--assets-only", action="store_true")
    group.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    ctx = ExperimentContext.load(args.experiment)
    if args.assets_only:
        prepare_assets(ctx)
    else:
        prepare_data(ctx)


if __name__ == "__main__":
    main()
