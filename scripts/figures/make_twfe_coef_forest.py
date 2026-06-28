import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(os.environ.get("GENERATED_FIGURES_DIR", "outputs/generated_figures")).resolve()
SRC = BASE / "table2_twfe_auc_economic_covariates.csv"
OUT_PNG = BASE / "fig_twfe_coef_forest.png"
OUT_PDF = BASE / "fig_twfe_coef_forest.pdf"
OUT_CSV = BASE / "fig_twfe_coef_forest_data.csv"


VAR_LABELS = {
    "tertiary_10pp": "Tertiary share\n(+10 pp)",
    "ln_gdp_pc": "GDP per capita\n(log)",
    "ln_builtup_stock": "Built-up stock\n(log)",
}

VAR_COLORS = {
    "tertiary_10pp": "#2B7896",
    "ln_gdp_pc": "#C8793A",
    "ln_builtup_stock": "#6F55B5",
}


def main():
    df = pd.read_csv(SRC)
    for col in ["coefficient", "std_error_hc1", "p_value", "n", "within_r2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ci_low"] = df["coefficient"] - 1.96 * df["std_error_hc1"]
    df["ci_high"] = df["coefficient"] + 1.96 * df["std_error_hc1"]
    df["variable_label"] = df["variable"].map(VAR_LABELS)
    df["plot_label"] = df["model"] + "  |  " + df["variable_label"].str.replace("\n", " ", regex=False)
    df["p_label"] = df["p_value"].map(lambda x: f"p={x:.3f}")
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    plot_df = df.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot_df)) * 1.28

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 8.6,
        }
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.axvline(0, color="#777777", lw=1.1, ls="--", zorder=0)

    for i, row in plot_df.iterrows():
        yi = y[i]
        color = VAR_COLORS.get(row["variable"], "#333333")
        x = row["coefficient"]
        left = x - row["ci_low"]
        right = row["ci_high"] - x
        ax.errorbar(
            x,
            yi,
            xerr=np.array([[left], [right]]),
            fmt="o",
            ms=6.0,
            mfc=color,
            mec="white",
            mew=0.8,
            ecolor=color,
            elinewidth=2.0,
            capsize=3.5,
            alpha=0.95,
            zorder=3,
        )
        ax.text(
            x,
            yi + 0.24,
            f"{row['coefficient']:+.3f} ({row['ci_low']:+.3f}, {row['ci_high']:+.3f}); {row['p_label']}",
            va="bottom",
            ha="center",
            fontsize=7.7,
            color="#333333",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["plot_label"])
    ax.set_xlabel("Coefficient on within-period AUC")
    ax.set_title("Figure X. Two-way fixed-effects estimates for period-level AUC", loc="left", fontweight="bold", pad=18)
    ax.set_xlim(-0.024, 0.024)
    ax.set_ylim(-0.7, y[-1] + 0.75)
    ax.grid(axis="x", alpha=0.22)
    fig.text(
        0.56,
        0.02,
        "Points show coefficients; horizontal bars show 95% confidence intervals. All models include city and period fixed effects.",
        ha="center",
        va="bottom",
        fontsize=8.4,
        color="#666666",
    )
    fig.tight_layout(rect=[0, 0.055, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(OUT_PNG)
    print(OUT_PDF)
    print(OUT_CSV)


if __name__ == "__main__":
    main()
