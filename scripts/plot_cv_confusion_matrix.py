from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
SUMMARY = ROOT / "results" / "cv_summary.parquet"
SAVE = ROOT / "figures" / "cv_confusion_dual.pdf"


def confusion_heatmap(ax, summary, rule, title):
    cm = summary[
        (summary["kind"] == "confusion_matrix") & (summary["decision_rule"] == rule)
    ].copy()
    cm["true_k"] = cm["true_k"].astype(int)
    cm["inferred_k"] = cm["inferred_k"].astype(int)

    ks = sorted(set(cm["true_k"]) | set(cm["inferred_k"]))
    idx = {k: i for i, k in enumerate(ks)}
    mat = np.zeros((len(ks), len(ks)))
    for _, r in cm.iterrows():
        mat[idx[r["true_k"]], idx[r["inferred_k"]]] = float(r["fraction"])

    im = ax.imshow(mat, vmin=0, vmax=1, cmap="Blues", aspect="equal")

    labels = [f"K={k}" for k in ks]
    ax.set_xticks(range(len(ks)))
    ax.set_yticks(range(len(ks)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Inferred K", fontsize=10)
    ax.set_ylabel("Ground-truth K", fontsize=10)
    ax.set_title(title, fontsize=11, pad=8)

    for i in range(len(ks)):
        for j in range(len(ks)):
            v = mat[i, j]
            if v == 0.0:
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if v > 0.55 else "black")
        ax.add_patch(plt.Rectangle(
            (i - 0.5, i - 0.5), 1, 1,
            fill=False, edgecolor="gold", lw=1.8, zorder=3,
        ))
    return im


summary = pd.read_parquet(SUMMARY)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.subplots_adjust(right=0.87)

im = confusion_heatmap(ax1, summary, "1se", "1-SE rule\n(smallest K within 1 SE of best)")
confusion_heatmap(ax2, summary, "max", "Max rule\n(argmax CV lppd)")

cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.68])
fig.colorbar(im, cax=cbar_ax, label="Fraction of trials")
fig.suptitle("CV model-selection confusion matrix (5-fold, 10 trials/K)", fontsize=11, y=1.01)

SAVE.parent.mkdir(exist_ok=True)
fig.savefig(SAVE, bbox_inches="tight")
plt.show()
