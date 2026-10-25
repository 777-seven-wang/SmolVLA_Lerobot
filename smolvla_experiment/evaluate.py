# 仿真评测执行层：校验路径后调用当前环境安装的原生 LeRobot 评测器。
# 共享 LIBERO 素材由实验上下文指定，联合微调按配置启用 FP32 参数存储。
# 包装原生 rollout 仅补充回合控制步数和耗时，不改变动作选择、reset 或成功判定。
# 视频渲染帧率按环境实际控制频率设置，避免播放时长与控制步数不一致。
# 本文件由调度器作为子进程调用，其他原生评测参数继续传递给 LeRobot。

"""Use external assets and the native evaluator without modifying the library."""

import argparse
import sys
import threading
import time
from functools import wraps
from pathlib import Path

from smolvla_experiment.common import EXPERIMENTS, ExperimentContext, configure_environment, read


# 从原生命令列表提取一个必需参数，兼容 --name=value 和 --name value。
# 缺失、空值或重复出现会报错，避免原生解析器接受含糊的模型／输出路径。
def argument_value(arguments, name):
    """Read required native CLI flags without consuming other native arguments."""
    values = []
    for index, value in enumerate(arguments):
        if value.startswith(name + "="):
            values.append(value.split("=", 1)[1])
        elif value == name:
            if index + 1 == len(arguments) or arguments[index + 1].startswith("--"):
                raise ValueError(f"Missing value for {name}")
            values.append(arguments[index + 1])
    if len(values) != 1 or not values[0]:
        raise ValueError(f"Exactly one explicit {name} is required")
    return values[0]


# 校验原生评测参数中的模型路径、输出路径和冻结策略。
# 两条路径必须归属当前实验；模型配置及显式 train_expert_only 覆盖不得与实验配置冲突。
def validate_paths(ctx, arguments):
    """Reject old experiments and symlink escapes before loading simulation code."""
    output = ctx.owned(Path(argument_value(arguments, "--output_dir")).resolve())
    policy = ctx.owned(Path(argument_value(arguments, "--policy.path")).resolve())
    config = read(ctx.owned(policy / "config.json"))
    if config.get("train_expert_only") is not ctx.flags["train_expert_only"]:
        raise ValueError("Policy does not belong to the selected experiment")
    for index, arg in enumerate(arguments):
        if arg.startswith("--policy.train_expert_only="):
            setting = arg.split("=", 1)[1]
        elif arg == "--policy.train_expert_only":
            setting = arguments[index + 1] if index + 1 < len(arguments) else ""
        else:
            continue
        if setting.lower() != str(ctx.flags["train_expert_only"]).lower():
            raise ValueError("Override conflicts with selected experiment")
    for key, expected in ctx.flags.items():
        if config.get(key) is not expected:
            raise ValueError(f"Policy has incompatible {key}")
    return output, policy


# 把 rollout 的 done 批次转换为每回合实际控制步数。
# 遇到第一个终止标志时取其一基位置；没有终止标志则使用整段长度，张量先转到 CPU。
def episode_lengths(done):
    """Count executed control steps up to the first terminal transition."""
    if hasattr(done, "detach"):
        done = done.detach().cpu().tolist()
    return [next((i + 1 for i, terminal in enumerate(row) if terminal), len(row)) for row in done]


# 包装评测器 rollout/run_one，累计每个任务的回合长度和总耗时。
# 使用线程局部状态隔离并行任务；保留原生成功记录，并校验回合数量能一一对应。
def record_episode_lengths(evaluator):
    """Observe native rollout results; leave action selection and reset behavior intact."""
    original_rollout = evaluator.rollout
    original_run_one = evaluator.run_one
    # 并发任务各自累积回合长度；一个任务的 rollout 不能混入另一个任务的统计。
    context = threading.local()

    # 原样调用并返回原生 rollout，只在当前任务上下文存在时采集 done 对应的长度。
    @wraps(original_rollout)
    def rollout(*args, **kwargs):
        result = original_rollout(*args, **kwargs)
        if hasattr(context, "lengths"):
            context.lengths.extend(episode_lengths(result["done"]))
        return result

    # 为一个任务建立长度收集区，将观测到的步数和耗时补充到原生 metrics。
    # 即使任务失败也清理线程局部状态，避免后续任务继承旧回合记录。
    @wraps(original_run_one)
    def run_one(*args, **kwargs):
        context.lengths = []
        started = time.perf_counter()
        try:
            group, task_id, metrics = original_run_one(*args, **kwargs)
            n_episodes = len(metrics["successes"])
            lengths = context.lengths[:n_episodes]
            if len(lengths) != n_episodes:
                raise RuntimeError("Native rollout and task episode counts differ")
            metrics["episode_lengths"] = lengths
            metrics["avg_episode_length"] = sum(lengths) / max(1, n_episodes)
            metrics["eval_s"] = time.perf_counter() - started
            return group, task_id, metrics
        finally:
            del context.lengths

    evaluator.rollout = rollout
    evaluator.run_one = run_one


# 包装环境构造器，将实例的 render_fps 对齐到实际 control_freq。
# 复制实例 metadata 再修改，避免影响其他环境的类级共享字典。
def align_recording_fps(env_class):
    """Record one frame per control step at its actual simulation rate."""
    original_init = env_class.__init__

    # 先完成原生环境初始化，再修改当前实例的视频帧率元数据。
    @wraps(original_init)
    def initialize(env, *args, **kwargs):
        original_init(env, *args, **kwargs)
        # 复制为实例字典，避免直接修改类级 metadata 而影响其他环境。
        env.metadata = {**env.metadata, "render_fps": env.control_freq}

    env_class.__init__ = initialize


# 取出本项目的 --experiment，剩余参数交给原生评测器解析。
# 先做路径校验和离线环境配置，再导入 LIBERO/模型库并安装对应精度及观测钩子。
def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    # 项目只消费实验选择参数；原生参数仍交给原生解析器，保持其评测配置语义。
    args, native = parser.parse_known_args()
    ctx = ExperimentContext.load(args.experiment)
    configure_environment(ctx)
    validate_paths(ctx, native)
    sys.argv = [sys.argv[0], *native]
    from libero import libero

    # 显式指向共享运行素材，避免原生包默认使用安装目录内的另一套资产路径。
    libero._assets_path_cache = str(ctx.assets)
    from lerobot.envs.libero import LiberoEnv

    from lerobot.scripts import lerobot_eval
    from smolvla_experiment.common import install_fp32_policy_storage

    if ctx.profile["parameter_storage"] == "float32":
        install_fp32_policy_storage()
    align_recording_fps(LiberoEnv)
    record_episode_lengths(lerobot_eval)
    lerobot_eval.main()


if __name__ == "__main__":
    main()
