# 统一命令行入口。
# 常用形式：python -B run.py --experiment expert_only|joint <操作参数>。
# 这里只在直接执行时导入调度器，由调度器校验参数并选择准备、训练、续训、评测或报告分支。
# 将导入放在主入口判断中，便于其他程序读取本文件时保持无启动副作用。

"""Unified local SmolVLA experiment entry point."""

if __name__ == "__main__":
    from smolvla_experiment.pipeline import main

    main()
