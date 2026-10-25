# SmolVLA / LIBERO-Spatial

在 LIBERO-Spatial 的 10 个任务上微调 SmolVLA，对比动作专家微调和语言骨干与动作专家联合微调。统一代码负责资源准备、训练、续训、仿真评测和报告，LeRobot 提供底层模型与训练器。

| 实验配置 | 可训练参数 | 已保留检查点 | 最佳观测 |
|---|---:|---|---|
| `expert_only`：动作专家微调 | 99,880,992 | 5k、10k、15k、20k、25k | **20k，76%** |
| `joint`：语言骨干与动作专家联合微调 | 306,471,072 | 3k、5k、10k、15k、20k、25k | **25k，67%** |

两轮均已完成 25k 训练，保留 11 份完整检查点、1100 个原始评测视频及完整指标。动作专家使用原生 BF16/FP32 混合参数存储，联合微调使用 FP32 参数存储；已有实验的差异不能只归因于解冻范围。

[完整结果与案例](docs/results.md) · [使用与维护指南](docs/guide.md)

## 使用

使用现有 Python 3.12 Conda 环境，在本项目根目录运行：

```bash
conda activate lerobot
python -B run.py --help
python -B run.py --experiment expert_only --prepare-only
python -B run.py --experiment joint --prepare-only
```

`--prepare-only` 校验现有离线资源，不启动训练或仿真。共享原始数据、基础模型仍依赖 `lerobot/downloads/smolvla_libero/`；已有环境可直接运行，无需重新安装或下载。

GitHub 仓库包含代码、`configs/` 和项目文档。`lerobot/`、`experiments/` 及仿真运行素材保存在本地，不随代码上传；下面命令中的历史运行路径需要本地已有对应产物。克隆代码后还需另行准备底层框架、依赖和离线资源。

新建训练：默认先短跑 200 步，再训练至 20k，并评测指定检查点及最终模型。累计训练预算默认 8 小时，准备与仿真另计。

```bash
python -B run.py --experiment expert_only --background
python -B run.py --experiment joint --max-steps 20000 --budget-s 28800 --background
```

从最新完整检查点追加 5k 步；续训不会自动启动仿真：

```bash
python -B run.py --experiment expert_only --resume experiments/expert_only/runs/20261023_123852/train/checkpoints/025000 --additional-steps 5000 --background
python -B run.py --experiment joint --resume experiments/joint/runs/20261023_193559/train/checkpoints/025000 --additional-steps 5000 --background
```

评测与生成报告：

```bash
python -B run.py --experiment expert_only --evaluate-only experiments/expert_only/runs/20261023_123852 --checkpoint 20000 25000
python -B run.py --experiment joint --report-only experiments/joint/runs/20261023_193559
python -B run.py --experiment joint --report-only experiments/joint/runs/20261023_193559 --compare-with experiments/expert_only/runs/20261023_123852
python -B run.py --experiment joint --report-only experiments/joint/runs/20261023_193559 --export-csv
```

默认报告包含 Markdown、训练／验证 loss 图和成功率图。`--compare-with` 显式增加策略对比图，`--export-csv` 仅与 `--report-only` 配合导出汇总表；两者可同时使用。报告清理本次未启用的旧派生产物，仅处理已知文件名，原始指标和评测数据继续保留。

已完整评测的结果校验后跳过；不完整结果保留并报错。并行评测可加 `--eval-batch-size 2|5|10`，输出单独保存。省略 `--checkpoint` 时选择最新保存模型。每种实验各有工作流锁，`--queue` 等待锁释放；使用同一 GPU 时请根据可用显存安排两个实验。

## 项目结构

| 目录 | 内容 |
|---|---|
| `smolvla_experiment/` | 扁平组织的 8 个功能模块及包初始化文件，覆盖准备、训练、评测和报告 |
| `configs/` | 两种实验的冻结策略、存储精度及默认运行参数 |
| `experiments/` | 按实验隔离的数据视图、模型配置、元数据和运行产物 |
| `resources/` | 共享 LIBERO 仿真素材 |
| `docs/` | 操作文档、结果图表、案例 GIF 和参考论文 |
| `lerobot/` | 底层框架及现有下载资源；本次未修改 |

## 开发验证

```bash
python -B -m ruff check --no-cache .
python -B -m ruff format --check smolvla_experiment run.py
```

现有回归检查在本轮整理验收后移除，项目不再提供测试目录；验证记录见[维护指南](docs/guide.md#整理历史与验证)。

依赖声明集中在 `pyproject.toml`，各实验 `metadata/environment.txt` 保留原实验环境快照。来源与工作流状态集中在 `metadata/experiment.json`，每次运行的配置、计时、协议及选模信息集中在 `run.json`；逐步耗时单独保存为 CSV。后续运行产生的缓存、锁、日志和检查点按 `.gitignore` 管理；程序不会自动删除未来生成的历史检查点。
