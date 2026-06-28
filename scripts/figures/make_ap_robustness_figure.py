import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE = Path(os.environ.get("GENERATED_FIGURES_DIR", "outputs/generated_figures")).resolve()
SRC = BASE / "table_period_ap_multimodel_robustness.csv"
OUT_PNG = BASE / "fig2_ap_robustness_multimodel.png"
OUT_PDF = BASE / "fig2_ap_robustness_multimodel.pdf"


def main():
    df = pd.read_csv(SRC)
    df = df.sort_values(["classifier", "period_order"])

    periods = (
        df[["period", "period_short", "period_order"]]
        .drop_duplicates()
        .sort_values("period_order")
    )
    x_labels = periods["period_short"].tolist()

    colors = {
        "GBDT": "#C83E4D",
        "Random forest": "#1F7A9A",
        "L2 logistic regression": "#6F55B5",
    }
    labels = {
        "GBDT": "Gradient boosting",
        "Random forest": "Random forest",
        "L2 logistic regression": "L2 logistic regression",
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(7.6, 4.4))

    for clf in ["GBDT", "Random forest", "L2 logistic regression"]:
        sub = df[df["classifier"] == clf].sort_values("period_order")
        y = sub["mean_ap"].to_numpy()
        ax.plot(
            range(len(y)),
            y,
            marker="o",
            linewidth=2.4,
            markersize=5.5,
            color=colors[clf],
            label=labels[clf],
        )
        ax.text(
            len(y) - 0.78,
            y[-1],
            f"{y[0]:.3f}→{y[-1]:.3f}",
            color=colors[clf],
            va="center",
            ha="left",
            fontsize=8.5,
        )

    ax.set_title("Average precision declines across model classes", loc="left", fontweight="bold")
    ax.set_ylabel("Mean average precision (AP)")
    ax.set_xlabel("Period")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=35, ha="right")
    ax.set_ylim(0.18, 0.82)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.6, alpha=0.55)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.98))

    ax.annotate(
        "robust decline",
        xy=(7, df[(df["classifier"] == "GBDT") & (df["period_short"] == "20-24")]["mean_ap"].iloc[0]),
        xytext=(6.15, 0.405),
        arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9, shrinkA=2, shrinkB=4),
        fontsize=8.5,
        color="#555555",
        ha="left",
        va="center",
    )

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
