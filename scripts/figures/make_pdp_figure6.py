import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE = Path(os.environ.get("GENERATED_FIGURES_DIR", "outputs/generated_figures")).resolve()
CURVES = BASE / "fig5_pdp_contemporary_top6_curves.csv"
FEATURES = BASE / "fig5_pdp_contemporary_top6_features.csv"
OUT_PNG = BASE / "fig6_pdp_contemporary_top6.png"
OUT_PDF = BASE / "fig6_pdp_contemporary_top6.pdf"
OUT_CURVES = BASE / "fig6_pdp_contemporary_top6_curves.csv"
OUT_FEATURES = BASE / "fig6_pdp_contemporary_top6_features.csv"


def main():
    curves = pd.read_csv(CURVES)
    features = pd.read_csv(FEATURES)
    selected = features["feature"].tolist()

    OUT_CURVES.write_text(CURVES.read_text(encoding="utf-8-sig"), encoding="utf-8-sig")
    OUT_FEATURES.write_text(FEATURES.read_text(encoding="utf-8-sig"), encoding="utf-8-sig")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
    for ax, feat in zip(axes.ravel(), selected):
        g = curves[curves["feature"] == feat].sort_values("grid_value")
        ax.plot(g["grid_value"], g["mean_pred_update_prob"], color="#bd3f45", marker="o", lw=2)
        ax.fill_between(
            g["grid_value"].to_numpy(),
            g["mean_pred_update_prob"].to_numpy(),
            0,
            color="#bd3f45",
            alpha=0.13,
        )
        ax.set_title(feat, fontweight="bold")
        ax.set_ylabel("avg P(update)")
        ax.grid(alpha=0.22)

    fig.suptitle(
        "Figure 6. Marginal effects of contemporary top renewal predictors (2020-2024)",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
