import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(os.environ.get("GENERATED_FIGURES_DIR", "outputs/generated_figures")).resolve()
SRC = BASE / "fig6_whitebox_source_subset.csv"
OUT_PNG = BASE / "fig7_built2built_growth_auc_audit.png"
OUT_PDF = BASE / "fig7_built2built_growth_auc_audit.pdf"
OUT_SUMMARY = BASE / "fig7_built2built_growth_auc_audit_summary.csv"


METRICS = [
    ("original_built2built_auc", "Built-to-built\nredevelopment"),
    ("growth_auc", "Greenfield\ngrowth"),
    ("one_step_cell_auc", "One-step cell\nbaseline"),
]


def main():
    df = pd.read_csv(SRC)
    for col, _ in METRICS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    rows = []
    for col, label in METRICS:
        vals = df[col].dropna()
        rows.append(
            {
                "metric": col,
                "label": label.replace("\n", " "),
                "n": int(vals.shape[0]),
                "mean_auc": float(vals.mean()),
                "median_auc": float(vals.median()),
                "q25": float(vals.quantile(0.25)),
                "q75": float(vals.quantile(0.75)),
            }
        )
    pd.DataFrame(rows).to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9,
        }
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    data = [df[col].dropna().to_numpy() for col, _ in METRICS]
    labels = [label for _, label in METRICS]
    colors = ["#2B7896", "#C8793A", "#6F55B5"]

    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.52,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 2},
        boxprops={"linewidth": 1.1},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)

    rng = np.random.default_rng(42)
    label_gap = 0.028
    for i, vals in enumerate(data, start=1):
        x = rng.normal(i, 0.045, size=len(vals))
        ax.scatter(x, vals, s=14, color=colors[i - 1], alpha=0.35, edgecolor="none")
        upper_whisker_y = max(bp["whiskers"][2 * (i - 1) + 1].get_ydata())
        ax.text(
            i,
            upper_whisker_y + label_gap,
            f"n={len(vals)}\nmedian={np.nanmedian(vals):.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#333333",
        )

    ax.axhline(0.5, color="#888888", linewidth=1.0, linestyle="--", alpha=0.65)
    ax.set_title(
        "Figure 7. Predictability contrast between redevelopment and growth",
        loc="left",
        fontweight="bold",
        pad=22,
    )
    ax.set_ylabel("AUC")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.35, 1.08)
    ax.grid(axis="y", alpha=0.28)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(OUT_PNG)
    print(OUT_PDF)
    print(OUT_SUMMARY)


if __name__ == "__main__":
    main()
