"""Draw two explanatory diagrams: (1) project workflow, (2) MLP architecture."""
import json
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from config import EVAL_DIR, EDA_DIR, set_style, save_fig, INK, INK2


def box(ax, x, y, w, h, text, fc="#e8f0fb", ec="#2a78d6", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=INK)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.3))


def workflow():
    fig, ax = plt.subplots(figsize=(10, 3.9)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4)
    top = [("1. Acquire\nUCI dataset\n4,424 × 37", 0.1), ("2. Parse &\nclean\ntypes, duplicates,\nmissing values", 2.1),
           ("3. EDA &\nstatistics\nunivariate →\nmultivariate", 4.1), ("4. Visualise\n12 figures +\n2 interactive", 6.1),
           ("5. Split\n70 / 15 / 15\nstratified", 8.1)]
    for t, x in top:
        box(ax, x, 2.3, 1.75, 1.35, t, fs=8.5)
    for i in range(4):
        arrow(ax, top[i][1] + 1.77, 2.97, top[i + 1][1] - 0.02, 2.97)
    bot = [("6. Preprocess (train-fit)\nwinsorise → scale,\none-hot, binary", 0.1), ("7. Tune MLP\n12 configurations on\nvalidation set", 2.65),
           ("8. Train final MLP\nearly stopping,\nclass weights", 5.2), ("9. Evaluate on test\nmetrics, ROC, confusion\nmatrix, importance", 7.75)]
    for t, x in bot:
        box(ax, x, 0.35, 2.1, 1.25, t, fc="#fdeee6", ec="#eb6834", fs=8.5)
    for i in range(3):
        arrow(ax, bot[i][1] + 2.12, 0.97, bot[i + 1][1] - 0.02, 0.97)
    arrow(ax, 8.97, 2.28, 1.15, 1.62)
    ax.text(5.0, 3.95, "Diagram 1  End-to-end project workflow", ha="center", fontsize=12, fontweight="bold")
    save_fig(fig, EDA_DIR / "diagram1_workflow.png")


def architecture():
    s = json.load(open(EVAL_DIR / "training_summary.json"))
    n_in = s["sizes"]["n_input_features"]; hid = s["best_config"]["hidden"]; dr = s["best_config"]["dropout"]
    fig, ax = plt.subplots(figsize=(10, 3.4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3.4)
    blocks = [(f"Input\n{n_in} features", "#e8f0fb", "#2a78d6")]
    for h in hid:
        blocks.append((f"Dense {h}\nReLU + L2\nBatchNorm\nDropout {dr}", "#eaf6f0", "#1baf7a"))
    blocks.append(("Dense 3\nSoftmax", "#fdeee6", "#eb6834"))
    blocks.append(("Output\nDropout /\nEnrolled /\nGraduate", "#e8f0fb", "#2a78d6"))
    n = len(blocks); w = 1.4; gap = (10 - 0.3 - n * w) / (n - 1)
    for i, (t, fc, ec) in enumerate(blocks):
        x = 0.15 + i * (w + gap); box(ax, x, 0.7, w, 1.6, t, fc=fc, ec=ec, fs=8.5)
        if i < n - 1:
            arrow(ax, x + w + 0.02, 1.5, x + w + gap - 0.02, 1.5)
    ax.text(5, 3.0, f"Diagram 2  MLP architecture ({s['n_params']:,} parameters)", ha="center", fontsize=12, fontweight="bold")
    save_fig(fig, EVAL_DIR / "diagram2_mlp_architecture.png")


if __name__ == "__main__":
    set_style(); workflow(); architecture()
