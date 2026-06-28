import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(os.environ.get("GENERATED_FIGURES_DIR", "outputs/generated_figures")).resolve()
SRC = BASE / "table_kruskal_wallis_industrial_city_patterns.csv"
OUT_PNG = BASE / "fig9_kruskal_industrial_city_patterns.png"
OUT_PDF = BASE / "fig9_kruskal_industrial_city_patterns.pdf"


METRIC_LABELS = {
    "industrial_residential_reciprocal_share": "Industrial-residential\nreciprocal share",
    "to_commercial_public_share": "Transitions to commercial\nor public service",
    "industrial_to_commercial_public_share": "Industrial to commercial\nor public service",
}

REGION_COLS = [
    ("median_east", "East"),
    ("median_central", "Central"),
    ("median_west", "West"),
    ("median_northeast", "Northeast"),
]


def fmt_p(p):
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"


def main():
    df = pd.read_csv(SRC)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=False)
    colors = {"all_periods": "#2B7896", "2015->2024": "#C84A4A"}
    labels = {"all_periods": "All periods", "2015->2024": "2015-2024"}

    for ax, metric in zip(axes, METRIC_LABELS):
        sub = df[df["metric"] == metric].set_index("period_scope")
        x = np.arange(len(REGION_COLS))
        width = 0.35

        for i, scope in enumerate(["all_periods", "2015->2024"]):
            vals = [sub.loc[scope, col] for col, _ in REGION_COLS]
            ax.bar(
                x + (i - 0.5) * width,
                vals,
                width=width,
                color=colors[scope],
                alpha=0.9,
                label=labels[scope],
            )

        p_all = float(sub.loc["all_periods", "p_value"])
        p_recent = float(sub.loc["2015->2024", "p_value"])
        ax.set_title(f"{METRIC_LABELS[metric]}\n{fmt_p(p_all)}; recent {fmt_p(p_recent)}", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([name for _, name in REGION_COLS], rotation=28, ha="right")
        ax.set_ylabel("Median share of updates")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, max(ax.get_ylim()[1], 0.30))

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Figure 9. Regional contrasts in industrial and upgrading transition patterns", fontsize=14, fontweight="bold", y=1.12)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
