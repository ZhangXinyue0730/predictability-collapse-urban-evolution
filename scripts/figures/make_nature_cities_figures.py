from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT = Path(os.environ.get("PROJECT_ROOT", ".")).resolve()
ANALYSIS = PROJECT / "city_extract_national" / "national_analysis_dynamic_t1"
WHITEBOX = PROJECT / "whitebox_redev_suite_runs" / "whitebox_redev_suite_all79_summary_greenfield_rechecked.csv"

PERIODS = [
    "1984->1990",
    "1990->1995",
    "1995->2000",
    "2000->2005",
    "2005->2010",
    "2010->2015",
    "2015->2020",
    "2020->2024",
]
PERIODS_DASH = [p.replace("->", "-") for p in PERIODS]
SHORT = {
    "1984->1990": "84-90",
    "1990->1995": "90-95",
    "1995->2000": "95-00",
    "2000->2005": "00-05",
    "2005->2010": "05-10",
    "2010->2015": "10-15",
    "2015->2020": "15-20",
    "2020->2024": "20-24",
}
PERIOD_MID = {
    "1984->1990": 1987,
    "1990->1995": 1992.5,
    "1995->2000": 1997.5,
    "2000->2005": 2002.5,
    "2005->2010": 2007.5,
    "2010->2015": 2012.5,
    "2015->2020": 2017.5,
    "2020->2024": 2022,
}


def norm_period(s: object) -> object:
    if pd.isna(s):
        return s
    return str(s).replace("-", "->")

MODEL_ORDER = [
    "M0 cell only",
    "M1 cell+Neighborhood",
    "M2 cell+SyntaxDecay",
    "M3 cell+Functional",
    "M4 cell+Neighborhood+Syntax",
    "M5 cell+Neighborhood+Functional",
    "M6 cell+Syntax+Functional",
    "M7 AllFeatures",
]
MODEL_LABELS = {
    "M0 cell only": "M0\nCell",
    "M1 cell+Neighborhood": "M1\nCell+Nb",
    "M2 cell+SyntaxDecay": "M2\nCell+Syntax",
    "M3 cell+Functional": "M3\nCell+Func",
    "M4 cell+Neighborhood+Syntax": "M4\nCell+Nb+Syntax",
    "M5 cell+Neighborhood+Functional": "M5\nCell+Nb+Func",
    "M6 cell+Syntax+Functional": "M6\nCell+Syntax+Func",
    "M7 AllFeatures": "M7\nAll",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )


def save_df(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    try:
        df.to_excel(path.with_suffix(".xlsx"), index=False)
    except Exception:
        pass


def write_inventory(out: Path) -> None:
    rows = [
        {
            "manuscript_location": "Chapter 2 / Figure 1",
            "suggested_file": "fig1_predictability_multimodel_auc.png",
            "data_source": "appendix_A_multimodel_auc_city_period.csv; appendix_A_table_A1_predictability_by_classifier.csv",
            "status": "generated",
            "note": "RF, GBDT, L2 logistic regression AUC robustness; AP robustness is not present in the current multi-model table.",
        },
        {
            "manuscript_location": "Chapter 2 / Table 1",
            "suggested_file": "table1_multimodel_auc_robustness.csv/xlsx",
            "data_source": "appendix_A_table_A1_predictability_by_classifier.csv",
            "status": "exported",
            "note": "Direct table for multi-model AUC robustness.",
        },
        {
            "manuscript_location": "Chapter 3 / Figure 2",
            "suggested_file": "fig2_expansion_renewal_crossover.png",
            "data_source": "all79_city_period_update_rates.csv",
            "status": "generated",
            "note": "Uses dynamic t1 non-vacant renewal definition; expansion is outside t1 built-up extent.",
        },
        {
            "manuscript_location": "Chapter 4 / Figure 3",
            "suggested_file": "fig3_feature_ablation_auc_ap.png",
            "data_source": "all79_city_ablation_results.csv",
            "status": "generated",
            "note": "Overall city-level ablation across M0-M7, not period-specific.",
        },
        {
            "manuscript_location": "Chapter 4 / Figure 4",
            "suggested_file": "fig4_driver_shift_permutation_importance.png",
            "data_source": "all79_city_period_shap_group.csv; all79_city_period_shap_top_features.csv",
            "status": "generated",
            "note": "This is permutation importance, not classical SHAP value beeswarm.",
        },
        {
            "manuscript_location": "Chapter 4 / Figure 5",
            "suggested_file": "not_generated_pdp_requires_curve_data.txt",
            "data_source": "No national PDP curve table found in current summary folder.",
            "status": "not generated",
            "note": "Existing pipeline creates per-city PDP PNGs, but no unified 79-city PDP curve table was found.",
        },
        {
            "manuscript_location": "Chapter 5 / White-box mechanism figure",
            "suggested_file": "fig5_whitebox_mechanism_audit.png",
            "data_source": "whitebox_redev_suite_all79_summary_greenfield_rechecked.csv",
            "status": "generated",
            "note": "Uses latest greenfield-rechecked white-box results, replacing the earlier overly-high AUC version.",
        },
        {
            "manuscript_location": "Chapter 6 / Heterogeneity figure",
            "suggested_file": "fig6_city_heterogeneity_renewal_regimes.png",
            "data_source": "all79_city_period_update_rates.csv; all79_city_period_transition_counts.csv",
            "status": "generated",
            "note": "Shows renewal dominance heterogeneity and road/municipal sink share by period.",
        },
    ]
    inv = pd.DataFrame(rows)
    save_df(inv, out / "figure_table_inventory")


def fig1_predictability(out: Path) -> None:
    df = pd.read_csv(ANALYSIS / "appendix_A_multimodel_auc_city_period.csv")
    table = pd.read_csv(ANALYSIS / "appendix_A_table_A1_predictability_by_classifier.csv")
    save_df(table, out / "table1_multimodel_auc_robustness")

    df["period"] = df["period"].map(norm_period)
    df = df[df["period"].isin(PERIODS)].copy()
    df["period"] = pd.Categorical(df["period"], PERIODS, ordered=True)
    df["period_short"] = df["period"].astype(str).map(SHORT)
    df["midyear"] = df["period"].astype(str).map(PERIOD_MID)

    gbdt = df[df["classifier"] == "Gradient boosting"].copy()
    mean = df.groupby(["classifier", "period"], observed=True)["auc"].mean().reset_index()
    mean["period_short"] = mean["period"].astype(str).map(SHORT)

    slopes = []
    for (clf, city), sub in df.dropna(subset=["auc"]).groupby(["classifier", "city_name"]):
        if sub["period"].nunique() < 3:
            continue
        x = sub["midyear"].to_numpy(dtype=float)
        y = sub["auc"].to_numpy(dtype=float)
        m = np.polyfit(x, y, 1)[0] * 10.0
        slopes.append({"classifier": clf, "city_name": city, "slope_per_decade": m})
    slopes = pd.DataFrame(slopes)
    save_df(slopes, out / "fig1_city_auc_slopes")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.6), constrained_layout=True)

    ax = axes[0]
    for _, sub in gbdt.groupby("city_name"):
        sub = sub.sort_values("period")
        ax.plot(sub["period_short"], sub["auc"], color="#c9c9c9", lw=0.7, alpha=0.45)
    m_gbdt = mean[mean["classifier"] == "Gradient boosting"].sort_values("period")
    ax.plot(m_gbdt["period_short"], m_gbdt["auc"], color="#b63b3f", marker="o", lw=2.4, label="79-city mean")
    ax.set_title("a  Predictability declines within periods")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.55, 1.0)
    ax.legend(frameon=False)

    ax = axes[1]
    colors = {"Gradient boosting": "#b63b3f", "Random forest": "#2b7b9f", "Logistic regression (L2)": "#6a51a3"}
    for clf, sub in mean.groupby("classifier"):
        sub = sub.sort_values("period")
        ax.plot(sub["period_short"], sub["auc"], marker="o", lw=2, color=colors.get(clf), label=clf)
    ax.set_title("b  Robust across model classes")
    ax.set_ylabel("Mean AUC")
    ax.set_ylim(0.65, 0.98)
    ax.legend(frameon=False, loc="lower left")

    ax = axes[2]
    s = slopes[slopes["classifier"] == "Gradient boosting"]["slope_per_decade"].dropna()
    ax.hist(s, bins=18, color="#5b91a8", edgecolor="white")
    ax.axvline(0, color="#b63b3f", ls="--", lw=1.6)
    ax.axvline(s.mean(), color="#111111", lw=1.2)
    ax.set_title("c  City-level AUC slopes")
    ax.set_xlabel("AUC change per decade")
    ax.set_ylabel("Number of cities")
    ax.text(0.02, 0.92, f"mean = {s.mean():.3f}", transform=ax.transAxes)

    fig.savefig(out / "fig1_predictability_multimodel_auc.png", bbox_inches="tight")
    plt.close(fig)


def fig2_expansion_renewal(out: Path) -> None:
    df = pd.read_csv(ANALYSIS / "all79_city_period_update_rates.csv")
    df = df[df["period"].isin(PERIODS)].copy()
    df["period"] = pd.Categorical(df["period"], PERIODS, ordered=True)
    df["period_short"] = df["period"].astype(str).map(SHORT)
    df["renewal_dominance"] = df["update_area_km2"] / (df["update_area_km2"] + df["expansion_area_km2"])

    agg = df.groupby("period", observed=True).agg(
        update_area_km2=("update_area_km2", "sum"),
        expansion_area_km2=("expansion_area_km2", "sum"),
        mean_renewal_dominance=("renewal_dominance", "mean"),
        pct_renewal_dominant=("renewal_dominance", lambda x: (x > 0.5).mean() * 100),
    ).reset_index()
    agg["period_short"] = agg["period"].astype(str).map(SHORT)
    save_df(agg, out / "fig2_expansion_renewal_period_summary")

    cross = []
    for city, sub in df.groupby("city_name"):
        sub = sub.sort_values("period")
        hit = sub[sub["renewal_dominance"] > 0.5]
        cross.append(
            {
                "city_name": city,
                "crossover_period": hit.iloc[0]["period"] if len(hit) else "not_crossed",
                "max_renewal_dominance": sub["renewal_dominance"].max(),
            }
        )
    cross = pd.DataFrame(cross)
    save_df(cross, out / "fig2_city_renewal_crossover_period")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
    ax = axes[0]
    ax.plot(agg["period_short"], agg["expansion_area_km2"], color="#d1783a", marker="s", lw=2.2, label="Expansion")
    ax.plot(agg["period_short"], agg["update_area_km2"], color="#2a6f8f", marker="o", lw=2.2, label="Renewal")
    ax.set_title("a  Expansion versus renewal")
    ax.set_ylabel("Area (km², 79 cities)")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(agg["period_short"], agg["mean_renewal_dominance"], color="#2a6f8f", marker="o", lw=2.2)
    ax.axhline(0.5, color="#888888", ls=":", lw=1.2)
    ax.set_ylim(0, 1)
    ax.set_title("b  Mean renewal dominance")
    ax.set_ylabel("Renewal / (renewal + expansion)")

    ax = axes[2]
    ax.bar(agg["period_short"], agg["pct_renewal_dominant"], color="#7ecf6a")
    ax.axhline(50, color="#888888", ls=":", lw=1.2)
    ax.set_ylim(0, 100)
    ax.set_title("c  Cities dominated by renewal")
    ax.set_ylabel("Cities with dominance > 0.5 (%)")
    fig.savefig(out / "fig2_expansion_renewal_crossover.png", bbox_inches="tight")
    plt.close(fig)


def fig3_ablation(out: Path) -> None:
    df = pd.read_csv(ANALYSIS / "all79_city_ablation_results.csv")
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    summary = df.groupby("model", observed=True).agg(
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
        ap_mean=("ap", "mean"),
        ap_std=("ap", "std"),
        n=("city_name", "nunique"),
    ).reset_index()
    summary["auc_se"] = summary["auc_std"] / np.sqrt(summary["n"])
    summary["ap_se"] = summary["ap_std"] / np.sqrt(summary["n"])
    save_df(summary, out / "fig3_feature_ablation_summary")

    x = np.arange(len(summary))
    labels = [MODEL_LABELS.get(m, m) for m in summary["model"].astype(str)]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.8), constrained_layout=True)
    axes[0].bar(x, summary["auc_mean"], yerr=summary["auc_se"], color="#5b91a8", capsize=2)
    axes[0].set_xticks(x, labels, rotation=35, ha="right")
    axes[0].set_ylabel("AUC")
    axes[0].set_title("a  Feature ablation: AUC")
    axes[0].set_ylim(max(0.45, summary["auc_mean"].min() - 0.04), min(1.0, summary["auc_mean"].max() + 0.06))
    axes[1].bar(x, summary["ap_mean"], yerr=summary["ap_se"], color="#d1783a", capsize=2)
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    axes[1].set_ylabel("Average precision")
    axes[1].set_title("b  Feature ablation: AP")
    axes[1].set_ylim(0, summary["ap_mean"].max() + 0.08)
    fig.savefig(out / "fig3_feature_ablation_auc_ap.png", bbox_inches="tight")
    plt.close(fig)


def fig4_importance(out: Path) -> None:
    g = pd.read_csv(ANALYSIS / "all79_city_period_shap_group.csv")
    g = g[g["period"].isin(PERIODS)].copy()
    g["period"] = pd.Categorical(g["period"], PERIODS, ordered=True)
    gsum = g.groupby(["period", "group"], observed=True)["delta_auc"].mean().reset_index()
    gsum["period_short"] = gsum["period"].astype(str).map(SHORT)
    wide = gsum.pivot(index="period_short", columns="group", values="delta_auc").loc[[SHORT[p] for p in PERIODS]]
    save_df(wide.reset_index(), out / "fig4_group_permutation_importance_summary")

    top = pd.read_csv(ANALYSIS / "all79_city_period_shap_top_features.csv")
    top = top[top["period"].isin(PERIODS)].copy()
    top_period = (
        top.groupby(["period", "feature", "group"], observed=True)["delta_auc"]
        .mean()
        .reset_index()
    )
    top_features = (
        top_period.groupby("feature")["delta_auc"].mean().sort_values(ascending=False).head(12).index.tolist()
    )
    heat = top_period[top_period["feature"].isin(top_features)].pivot_table(
        index="feature", columns="period", values="delta_auc", aggfunc="mean"
    )
    heat = heat.reindex(top_features).reindex(columns=PERIODS)
    save_df(heat.reset_index(), out / "fig4_top_feature_period_importance")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.2), constrained_layout=True)
    ax = axes[0]
    palette = {"Cell class": "#c8753b", "Functional": "#7aa03f", "Neighborhood": "#2a6f8f", "Syntax": "#8d6bb2"}
    for group in ["Cell class", "Functional", "Neighborhood", "Syntax"]:
        if group in wide.columns:
            ax.plot(wide.index, wide[group], marker="o", lw=2.2, color=palette[group], label=group)
    ax.set_title("a  Renewal drivers reshuffle")
    ax.set_ylabel("Mean ΔAUC (permutation importance)")
    ax.legend(frameon=False)

    ax = axes[1]
    im = ax.imshow(heat.values, aspect="auto", cmap="YlGnBu")
    ax.set_title("b  Top features across periods")
    ax.set_xticks(np.arange(len(PERIODS)), [SHORT[p] for p in PERIODS], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(heat.index)), heat.index)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Mean ΔAUC")
    fig.savefig(out / "fig4_driver_shift_permutation_importance.png", bbox_inches="tight")
    plt.close(fig)

    (out / "not_generated_pdp_requires_curve_data.txt").write_text(
        "Current national summary tables do not contain PDP curve coordinates. "
        "The pipeline produced per-city PDP PNGs, but a unified 79-city PDP figure "
        "requires rerunning/exporting PDP curves from city-level models.\n",
        encoding="utf-8",
    )


def fig5_whitebox(out: Path) -> None:
    if not WHITEBOX.exists():
        (out / "whitebox_missing.txt").write_text(f"Missing: {WHITEBOX}\n", encoding="utf-8")
        return
    df = pd.read_csv(WHITEBOX)
    save_df(df, out / "fig5_whitebox_source_subset")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True)
    ax = axes[0, 0]
    vals = [
        df["original_built2built_auc"].dropna().values,
        df["growth_auc"].dropna().values,
        df["one_step_cell_auc"].dropna().values,
    ]
    ax.boxplot(vals, labels=["Built-to-built\nwhite-box", "Greenfield\ngrowth", "One-step\ncell"], patch_artist=True,
               boxprops={"facecolor": "#dce8ef", "color": "#44778f"}, medianprops={"color": "#b63b3f"})
    ax.set_title("a  White-box discrimination")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.45, 1.0)

    ax = axes[0, 1]
    ax.hist(df["remodel_cyclic_fraction"].dropna(), bins=18, color="#7aa03f", edgecolor="white")
    ax.set_title("b  Cyclic flow among renewal transitions")
    ax.set_xlabel("Remodel cyclic fraction")
    ax.set_ylabel("Cities")

    ax = axes[1, 0]
    s = df[["allen_cahn_slope_predicted", "allen_cahn_slope_measured"]].dropna()
    ax.scatter(s["allen_cahn_slope_predicted"], s["allen_cahn_slope_measured"], s=22, alpha=0.75, color="#6a51a3")
    if len(s) >= 2:
        lo = min(s.min())
        hi = max(s.max())
        ax.plot([lo, hi], [lo, hi], color="#888888", ls=":", lw=1.2)
        m, b = np.polyfit(s["allen_cahn_slope_predicted"], s["allen_cahn_slope_measured"], 1)
        x = np.linspace(lo, hi, 50)
        ax.plot(x, m * x + b, color="#b63b3f", lw=1.8)
    ax.set_title("c  Allen-Cahn slope consistency")
    ax.set_xlabel("Predicted slope")
    ax.set_ylabel("Measured slope")

    ax = axes[1, 1]
    ax.hist(df["robust_critical_beta"].dropna(), bins=18, color="#d1783a", edgecolor="white")
    ax.set_title("d  Critical beta distribution")
    ax.set_xlabel("Critical beta")
    ax.set_ylabel("Cities")

    fig.savefig(out / "fig5_whitebox_mechanism_audit.png", bbox_inches="tight")
    plt.close(fig)


def fig6_heterogeneity(out: Path) -> None:
    rates = pd.read_csv(ANALYSIS / "all79_city_period_update_rates.csv")
    rates = rates[rates["period"].isin(PERIODS)].copy()
    rates["period"] = pd.Categorical(rates["period"], PERIODS, ordered=True)
    rates["period_short"] = rates["period"].astype(str).map(SHORT)
    rates["renewal_dominance"] = rates["update_area_km2"] / (rates["update_area_km2"] + rates["expansion_area_km2"])

    rates["city_label"] = rates["city_name_en"].fillna(rates["city_name"])
    pivot_order = (
        rates.groupby("city_label")["renewal_dominance"].mean().sort_values(ascending=False).index.tolist()
    )
    heat = rates.pivot(index="city_label", columns="period", values="renewal_dominance").reindex(pivot_order).reindex(columns=PERIODS)
    save_df(heat.reset_index(), out / "fig6_city_period_renewal_dominance_matrix")

    trans = pd.read_csv(ANALYSIS / "all79_city_period_transition_counts.csv")
    trans = trans[trans["period"].isin(PERIODS)].copy()
    total = trans.groupby("period")["count"].sum()
    muni = trans[trans["to_class"] == 5].groupby("period")["count"].sum()
    muni_share = (muni / total).reindex(PERIODS).fillna(0).reset_index()
    muni_share.columns = ["period", "road_municipal_inflow_share"]
    muni_share["period_short"] = muni_share["period"].map(SHORT)
    save_df(muni_share, out / "fig6_road_municipal_inflow_share")

    size_map = {
        "超大城市": "Megacity",
        "特大城市": "Supercity",
        "I型大城市": "Large I",
        "II型大城市": "Large II",
        "中等城市": "Medium",
        "I型小城市": "Small I",
        "II型小城市": "Small II",
    }
    rates["city_size_label"] = rates["city_size_class"].map(size_map).fillna(rates["city_size_class"])
    size_order = ["Megacity", "Supercity", "Large I", "Large II", "Medium", "Small I", "Small II"]
    size = rates.groupby(["city_size_label", "period"], observed=True)["renewal_dominance"].mean().reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6), constrained_layout=True)
    ax = axes[0]
    im = ax.imshow(heat.values, aspect="auto", cmap="BrBG", vmin=0, vmax=1)
    ax.set_title("a  Renewal dominance by city")
    ax.set_xticks(np.arange(len(PERIODS)), [SHORT[p] for p in PERIODS], rotation=45, ha="right")
    step = max(1, len(heat.index) // 12)
    yticks = np.arange(0, len(heat.index), step)
    ax.set_yticks(yticks, [heat.index[i] for i in yticks])
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Renewal dominance")

    ax = axes[1]
    for cls in size_order:
        sub = size[size["city_size_label"] == cls].sort_values("period")
        if len(sub):
            ax.plot(sub["period"].astype(str).map(SHORT), sub["renewal_dominance"], marker="o", lw=1.8, label=cls)
    ax.axhline(0.5, color="#888888", ls=":", lw=1.0)
    ax.set_title("b  Renewal regimes by city size")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean renewal dominance")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[2]
    ax.plot(muni_share["period_short"], muni_share["road_municipal_inflow_share"], marker="o", color="#777777", lw=2.2)
    ax.set_title("c  Road/municipal as renewal sink")
    ax.set_ylabel("Share of renewal transitions into road/municipal")
    ax.set_ylim(0, max(0.5, muni_share["road_municipal_inflow_share"].max() + 0.08))

    fig.savefig(out / "fig6_city_heterogeneity_renewal_regimes.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)
    style()
    write_inventory(out)
    fig1_predictability(out)
    fig2_expansion_renewal(out)
    fig3_ablation(out)
    fig4_importance(out)
    fig5_whitebox(out)
    fig6_heterogeneity(out)
    print(f"Generated figures and tables in: {out}")


if __name__ == "__main__":
    main()
