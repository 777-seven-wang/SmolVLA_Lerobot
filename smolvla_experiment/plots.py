# 内部绘图模块：将报告层提供的数据绘制成固定名称的 PNG。
# 本模块不读取运行 JSON 或原始指标 CSV，也不写入实验管理记录。
# 曲线采用原始记录，不添加额外平滑；跨实验数据的可比性由报告层事先校验。
# Matplotlib 使用无界面的 Agg 后端，适用于服务器和后台生成报告。
# 每次保存后关闭图对象，避免批量生成时累积占用内存。

"""Render supplied observations; computation and record updates belong to report."""

from smolvla_experiment.common import ensure_owned


# 按需导入 Matplotlib 并统一使用 Agg 后端，返回 pyplot 接口。
# 若系统装有指定中文字体则显式注册；设置公共坐标轴样式，不在模块导入时执行。
def plot_style():
    import matplotlib

    matplotlib.use("Agg")
    from pathlib import Path

    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )
    return plt


# 根据已读 train/validation 记录绘制完整训练曲线及后期细节。
# ctx/config 提供实验名称、存储精度和 batch；summary 提供最终步数与最低验证 loss。
# 保存固定名称图片，返回原 summary 和相对运行目录的图片路径，不修改管理记录。
def render_loss(run, ctx, config, metrics, summary):
    run = ensure_owned(run)
    if not summary:
        raise ValueError("Training and validation loss are required")
    plt = plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    train, validation = metrics["train"], metrics["validation"]
    for ax in axes:
        ax.plot(
            [int(r["steps"]) for r in train],
            [float(r["loss"]) for r in train],
            label="训练 loss",
            linewidth=1.3,
        )
        ax.plot(
            [int(r["steps"]) for r in validation],
            [float(r["eval_loss"]) for r in validation],
            label="验证 loss",
            marker=".",
        )
        ax.set(
            xlabel="训练步数", ylabel="Flow-matching loss", xlim=(0, summary["final_step"] * 1.025)
        )
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    axes[0].set_title("完整训练过程")
    axes[1].set_title("后期细节")
    axes[1].set_xlim(min(1000, summary["final_step"] / 2), summary["final_step"] * 1.025)
    # 后期视图按实际记录值确定纵轴，保留训练与验证原始差异，不补点或平滑。
    values = [float(r["loss"]) for r in train if int(r["steps"]) >= 1000]
    values += [float(r["eval_loss"]) for r in validation]
    axes[1].set_ylim(0, max(values) * 1.15)
    storage = (
        "FP32 参数存储"
        if ctx.profile["parameter_storage"] == "float32"
        else "原生 BF16/FP32 混合参数存储"
    )
    fig.suptitle(
        f"SmolVLA / LIBERO-Spatial：{ctx.name}，batch={config.get('batch_size', '未记录')}，{storage}"
    )
    fig.text(
        0.05,
        0.02,
        f"无额外平滑；最低验证 loss {summary['minimum_validation_loss']:.4f}（{summary['minimum_validation_step']:,} 步）。",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    # 使用固定文件名覆盖同一派生图；是否清理其他旧图由报告层的已知文件清单决定。
    output = ensure_owned(run / "curves/loss_training_validation.png")
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return summary, str(output.relative_to(run))


# 绘制两种实验在共同记录范围内的训练／验证 loss 对比。
# comparison 由报告层提前完成划分和步数校验；图上注明解冻范围与存储精度同时不同。
# 只输出图片，CSV 导出仍由报告层统一决定。
def render_comparison(run, comparison):
    run = ensure_owned(run)
    experiments, summary = comparison
    plt = plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    labels = {
        "expert_only": "动作专家（原生参数存储）",
        "joint": "语言骨干与动作专家（FP32 参数存储）",
    }
    for ax, kind in zip(axes, ("train", "validation"), strict=True):
        for exp in experiments:
            ax.plot(*zip(*exp[kind], strict=True), label=labels[exp["name"]], linewidth=1.5)
        ax.set(
            xlabel="训练步数",
            ylabel="Flow-matching loss",
            title="训练 loss" if kind == "train" else "验证 loss",
        )
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(f"SmolVLA 微调策略对比（共同记录范围：0–{summary['comparison_steps']:,} 步）")
    fig.text(
        0.05,
        0.02,
        "使用原始记录，无额外平滑；参数解冻范围和参数存储精度同时不同，不能单独归因于解冻范围。",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    output = ensure_owned(run / "curves/loss_strategy_comparison.png")
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return summary, str(output.relative_to(run))


# 绘制报告层筛选后的完整、同协议检查点成功率，误差线为 95% Wilson 区间。
# rows 中已包含成功率及上下界，直接使用原始检查点步数，不在绘图阶段重新选模。
def render_success(run, rows):
    plt = plot_style()
    curves = ensure_owned(run / "curves")
    curves.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    # Matplotlib 的非对称 yerr 接收“到下界的距离／到上界的距离”，而不是区间端点本身。
    errors = [
        [row["pc_success"] - row["pc_success_ci95"][0] for row in rows],
        [row["pc_success_ci95"][1] - row["pc_success"] for row in rows],
    ]
    ax.errorbar(
        [row["step"] for row in rows],
        [row["pc_success"] for row in rows],
        yerr=errors,
        fmt="o-",
        capsize=5,
        color="#2563eb",
    )
    for row in rows:
        ax.annotate(
            f"{row['pc_success']:g}%",
            (row["step"], row["pc_success"]),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
        )
    ax.set(
        xlabel="训练步数",
        ylabel="成功率（%）及 95% Wilson 区间",
        ylim=(0, 100),
        title="LIBERO-Spatial 检查点成功率（每个检查点100回合）",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    destination = ensure_owned(curves / "checkpoint_comparison.png")
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return str(destination.relative_to(run))
