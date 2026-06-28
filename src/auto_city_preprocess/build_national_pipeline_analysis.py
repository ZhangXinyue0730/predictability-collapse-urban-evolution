#!/usr/bin/env python3
"""Build national cross-city summaries from completed TCULU city pipeline runs."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PERIOD_ORDER = [
    "1984->1990",
    "1990->1995",
    "1995->2000",
    "2000->2005",
    "2005->2010",
    "2010->2015",
    "2015->2020",
    "2020->2024",
]
YEARS = [1984, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2024]

CLASS_ORDER = ["Cell class", "Neighborhood", "Syntax", "Functional"]


def setup_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Arial Unicode MS",
        "Songti SC",
        "Heiti SC",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def safe_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan


def period_rate_col(df: pd.DataFrame) -> str:
    if "update_rate" in df.columns:
        return "update_rate"
    return "update_rate_percent"


def normalize_rate(series: pd.Series, col: str) -> pd.Series:
    out = series.astype(float)
    if col.endswith("percent"):
        out = out / 100.0
    return out


def iter_complete_runs(summary: pd.DataFrame) -> Iterable[dict]:
    for _, row in summary.iterrows():
        run_dir = Path(row["run_dir"])
        data = run_dir / "data"
        required = [
            data / "update_period_rates.csv",
            data / "update_period_transition_counts.csv",
            data / "period_shap_group_importance.csv",
            data / "period_shap_top_features.csv",
        ]
        if not all(p.exists() for p in required):
            continue
        yield row.to_dict()


def load_period_rates(summary: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for rec in iter_complete_runs(summary):
        data = Path(rec["run_dir"]) / "data"
        df = read_csv(data / "update_period_rates.csv")
        rcol = period_rate_col(df)
        df["update_rate"] = normalize_rate(df[rcol], rcol)
        keep = ["period", "year_t1", "year_t2", "samples", "updates", "update_rate"]
        df = df[keep].copy()
        for key in ["province", "city_name", "city_name_en", "city_short_name", "region", "city_size_class", "sample_group", "sample_type", "run_dir"]:
            df[key] = rec.get(key, "")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_transition_counts(summary: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for rec in iter_complete_runs(summary):
        data = Path(rec["run_dir"]) / "data"
        df = read_csv(data / "update_period_transition_counts.csv")
        if "share_percent" in df.columns:
            df["city_period_share"] = df["share_percent"].astype(float) / 100.0
        for key in ["province", "city_name", "city_name_en", "city_short_name", "region", "city_size_class", "sample_group", "sample_type", "run_dir"]:
            df[key] = rec.get(key, "")
        df["transition"] = df["from_name"].astype(str) + "→" + df["to_name"].astype(str)
        df["transition_en"] = df["from_name_en"].astype(str) + "->" + df["to_name_en"].astype(str)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_period_shap(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_frames = []
    top_frames = []
    for rec in iter_complete_runs(summary):
        data = Path(rec["run_dir"]) / "data"
        g = read_csv(data / "period_shap_group_importance.csv")
        t = read_csv(data / "period_shap_top_features.csv")
        for df in [g, t]:
            for key in ["province", "city_name", "city_name_en", "city_short_name", "region", "city_size_class", "sample_group", "sample_type", "run_dir"]:
                df[key] = rec.get(key, "")
        group_frames.append(g)
        top_frames.append(t)
    group = pd.concat(group_frames, ignore_index=True) if group_frames else pd.DataFrame()
    top = pd.concat(top_frames, ignore_index=True) if top_frames else pd.DataFrame()
    return group, top


def load_ablation(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, rec in summary.iterrows():
        data = Path(rec["run_dir"]) / "data"
        path = data / "ablation_results.pkl"
        if not path.exists():
            continue
        try:
            obj = pickle.load(open(path, "rb"))
            items = obj.get("results") if isinstance(obj, dict) and "results" in obj else obj
            if isinstance(items, dict):
                items = [{"name": k, **v} for k, v in items.items() if isinstance(v, dict)]
            for item in items:
                if not isinstance(item, dict):
                    continue
                rows.append(
                    {
                        "province": rec.get("province", ""),
                        "city_name": rec.get("city_name", ""),
                        "city_name_en": rec.get("city_name_en", ""),
                        "sample_type": rec.get("sample_type", ""),
                        "model": item.get("name") or item.get("model") or item.get("config") or "",
                        "auc": item.get("auc", item.get("test_auc", item.get("AUC", np.nan))),
                        "ap": item.get("ap", item.get("test_ap", item.get("AP", np.nan))),
                    }
                )
        except Exception as exc:
            rows.append({"city_name": rec.get("city_name", ""), "model": "ERROR", "auc": np.nan, "ap": np.nan, "error": str(exc)})
    return pd.DataFrame(rows)


def load_syntax_qc(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, rec in summary.iterrows():
        run_dir = Path(rec["run_dir"])
        data = run_dir / "data"
        if not data.exists():
            continue
        for year in YEARS:
            road_path = data / f"road_{year}.npz"
            graph_path = data / f"syntax_graph_{year}.pkl"
            syntax_path = data / f"syntax_results_{year}.npz"
            if not road_path.exists() or not graph_path.exists():
                continue
            row = {
                "province": rec.get("province", ""),
                "city_name": rec.get("city_name", ""),
                "city_name_en": rec.get("city_name_en", ""),
                "sample_type": rec.get("sample_type", ""),
                "year": year,
            }
            try:
                road = np.load(road_path, allow_pickle=True)
                row["road_polylines"] = int(len(road["polys"])) if "polys" in road.files else np.nan
            except Exception:
                row["road_polylines"] = np.nan
            try:
                graph = pickle.load(open(graph_path, "rb"))
                row["syntax_segments"] = int(graph.get("n_segs", np.nan))
                row["syntax_edges"] = int(len(graph.get("edges", [])))
                lengths = np.asarray(graph.get("seg_length_m", []), dtype=float)
                row["syntax_total_length_km"] = float(np.nansum(lengths) / 1000.0) if len(lengths) else np.nan
                row["syntax_mean_degree"] = float((2 * row["syntax_edges"] / row["syntax_segments"]) if row["syntax_segments"] else np.nan)
            except Exception:
                row["syntax_segments"] = np.nan
                row["syntax_edges"] = np.nan
                row["syntax_total_length_km"] = np.nan
                row["syntax_mean_degree"] = np.nan
            row["syntax_results_exists"] = syntax_path.exists()
            rows.append(row)
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def plot_period_updates(period_summary: pd.DataFrame, out: Path, title: str) -> None:
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(period_summary))
    ax1.bar(x, period_summary["updates"], color="#3B82F6", label="更新数")
    ax1.set_ylabel("更新数")
    ax1.set_xticks(x)
    ax1.set_xticklabels(period_summary["period"], rotation=30, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, period_summary["update_rate"] * 100, color="#EF4444", marker="o", linewidth=2.5, label="更新率")
    ax2.set_ylabel("更新率 (%)")
    ax1.set_title(title)
    ax1.grid(axis="y", alpha=0.25)
    for i, v in enumerate(period_summary["updates"]):
        ax1.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_transition_stacked(top_period: pd.DataFrame, out: Path, title: str, top_n: int = 10) -> None:
    top_transitions = top_period.groupby("transition", as_index=False)["count"].sum().sort_values("count", ascending=False).head(top_n)["transition"].tolist()
    mat = top_period[top_period["transition"].isin(top_transitions)].pivot_table(index="period", columns="transition", values="count", aggfunc="sum", fill_value=0)
    mat = mat.reindex(PERIOD_ORDER).fillna(0)
    colors = ["#2563EB", "#DC2626", "#16A34A", "#F59E0B", "#7C3AED", "#0891B2", "#DB2777", "#65A30D", "#EA580C", "#475569"]
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(mat))
    for i, col in enumerate(mat.columns):
        vals = mat[col].to_numpy()
        ax.bar(mat.index, vals, bottom=bottom, label=col, color=colors[i % len(colors)])
        bottom += vals
    ax.set_title(title)
    ax.set_ylabel("更新数")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_city_top10(city_top: pd.DataFrame, out: Path, title: str) -> None:
    periods = PERIOD_ORDER
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.ravel()
    for ax, period in zip(axes, periods):
        sub = city_top[city_top["period"] == period].sort_values("updates", ascending=True)
        labels = sub["city_name"].astype(str).tolist()
        vals = sub["updates"].astype(float).to_numpy()
        ax.barh(labels, vals, color="#0F766E")
        ax.set_title(period)
        ax.grid(axis="x", alpha=0.25)
        for y, v in enumerate(vals):
            ax.text(v, y, f" {int(v):,}", va="center", fontsize=8)
    fig.suptitle(title, y=0.995, fontsize=15)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_shap_group(shap_group: pd.DataFrame, out: Path, title: str) -> None:
    if shap_group.empty:
        return
    grouped = shap_group.groupby(["period", "group"], as_index=False).agg(
        mean_delta_auc=("delta_auc", "mean"),
        median_delta_auc=("delta_auc", "median"),
        city_count=("city_name", "nunique"),
    )
    mat = grouped.pivot(index="period", columns="group", values="mean_delta_auc").reindex(PERIOD_ORDER)
    cols = [c for c in CLASS_ORDER if c in mat.columns] + [c for c in mat.columns if c not in CLASS_ORDER]
    mat = mat[cols].fillna(0)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(mat))
    width = 0.18 if len(cols) <= 4 else 0.12
    colors = {"Cell class": "#2563EB", "Neighborhood": "#F59E0B", "Syntax": "#7C3AED", "Functional": "#16A34A"}
    offset = -(len(cols) - 1) * width / 2
    for i, col in enumerate(cols):
        ax.bar(x + offset + i * width, mat[col].to_numpy(), width=width, label=col, color=colors.get(col))
    ax.axhline(0, color="#222", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(mat.index, rotation=30, ha="right")
    ax.set_ylabel("平均 ΔAUC")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def build_scope_outputs(scope_name: str, scope_df: pd.DataFrame, out_dir: Path) -> None:
    period_rates = load_period_rates(scope_df)
    transitions = load_transition_counts(scope_df)
    shap_group, shap_top = load_period_shap(scope_df)
    ablation = load_ablation(scope_df)
    syntax_qc = load_syntax_qc(scope_df)

    write_csv(period_rates, out_dir / f"{scope_name}_city_period_update_rates.csv")
    write_csv(transitions, out_dir / f"{scope_name}_city_period_transition_counts.csv")
    write_csv(shap_group, out_dir / f"{scope_name}_city_period_shap_group.csv")
    write_csv(shap_top, out_dir / f"{scope_name}_city_period_shap_top_features.csv")
    write_csv(ablation, out_dir / f"{scope_name}_city_ablation_results.csv")
    write_csv(syntax_qc, out_dir / f"{scope_name}_syntax_qc_by_city_year.csv")

    period_summary = period_rates.groupby("period", as_index=False).agg(
        city_count=("city_name", "nunique"),
        samples=("samples", "sum"),
        updates=("updates", "sum"),
        median_city_update_rate=("update_rate", "median"),
        mean_city_update_rate=("update_rate", "mean"),
    )
    period_summary["update_rate"] = period_summary["updates"] / period_summary["samples"]
    period_summary["period"] = pd.Categorical(period_summary["period"], PERIOD_ORDER, ordered=True)
    period_summary = period_summary.sort_values("period")
    write_csv(period_summary, out_dir / f"{scope_name}_national_period_update_summary.csv")

    positive_transitions = transitions[transitions["count"] > 0].copy()
    transition_period = transitions.groupby(["period", "transition", "transition_en", "from_name", "to_name"], as_index=False).agg(
        count=("count", "sum"),
    )
    positive_city_count = positive_transitions.groupby(["period", "transition"], as_index=False).agg(
        city_count=("city_name", "nunique"),
    )
    transition_period = transition_period.merge(positive_city_count, on=["period", "transition"], how="left")
    transition_period["city_count"] = transition_period["city_count"].fillna(0).astype(int)
    total_by_period = transition_period.groupby("period")["count"].transform("sum")
    transition_period["share"] = transition_period["count"] / total_by_period
    transition_period["rank"] = transition_period.groupby("period")["count"].rank(method="first", ascending=False).astype(int)
    transition_top10 = transition_period[transition_period["rank"] <= 10].sort_values(["period", "rank"])
    write_csv(transition_period, out_dir / f"{scope_name}_national_period_transition_all.csv")
    write_csv(transition_top10, out_dir / f"{scope_name}_national_period_transition_top10.csv")

    city_period_main = positive_transitions.sort_values("count", ascending=False).groupby(["city_name", "period"], as_index=False).first()[
        ["city_name", "period", "transition", "count"]
    ].rename(columns={"transition": "main_transition", "count": "main_transition_count"})
    city_top = period_rates.merge(city_period_main, on=["city_name", "period"], how="left")
    city_top["update_rank"] = city_top.groupby("period")["updates"].rank(method="first", ascending=False).astype(int)
    city_top10 = city_top[city_top["update_rank"] <= 10].sort_values(["period", "update_rank"])
    write_csv(city_top10, out_dir / f"{scope_name}_national_period_city_top10_by_updates.csv")

    city_top_rate = city_top[city_top["updates"] >= 10].copy()
    city_top_rate["update_rate_rank"] = city_top_rate.groupby("period")["update_rate"].rank(method="first", ascending=False).astype(int)
    city_top_rate10 = city_top_rate[city_top_rate["update_rate_rank"] <= 10].sort_values(["period", "update_rate_rank"])
    write_csv(city_top_rate10, out_dir / f"{scope_name}_national_period_city_top10_by_update_rate.csv")

    shap_group_summary = shap_group.groupby(["period", "group"], as_index=False).agg(
        city_count=("city_name", "nunique"),
        mean_delta_auc=("delta_auc", "mean"),
        median_delta_auc=("delta_auc", "median"),
        mean_share_positive=("share_positive", "mean"),
        mean_test_auc=("test_auc", "mean"),
        total_updates=("n_updates", "sum"),
    )
    shap_group_summary["rank"] = shap_group_summary.groupby("period")["mean_delta_auc"].rank(method="first", ascending=False).astype(int)
    write_csv(shap_group_summary.sort_values(["period", "rank"]), out_dir / f"{scope_name}_national_period_shap_group_summary.csv")

    shap_top1 = shap_top[shap_top["rank"] == 1].copy() if "rank" in shap_top.columns else shap_top.groupby(["city_name", "period"]).head(1).copy()
    shap_feature_summary = shap_top1.groupby(["period", "feature", "group"], as_index=False).agg(
        city_count=("city_name", "nunique"),
        mean_delta_auc=("delta_auc", "mean"),
        median_delta_auc=("delta_auc", "median"),
        total_updates=("n_updates", "sum"),
    )
    shap_feature_summary["rank"] = shap_feature_summary.groupby("period")["city_count"].rank(method="first", ascending=False).astype(int)
    shap_feature_top10 = shap_feature_summary[shap_feature_summary["rank"] <= 10].sort_values(["period", "rank", "mean_delta_auc"], ascending=[True, True, False])
    write_csv(shap_feature_top10, out_dir / f"{scope_name}_national_period_shap_top_feature_frequency.csv")

    syntax_summary = syntax_qc.groupby("city_name", as_index=False).agg(
        province=("province", "first"),
        sample_type=("sample_type", "first"),
        min_segments=("syntax_segments", "min"),
        max_segments=("syntax_segments", "max"),
        segments_2024=("syntax_segments", lambda s: s.iloc[-1] if len(s) else np.nan),
        min_total_length_km=("syntax_total_length_km", "min"),
        max_total_length_km=("syntax_total_length_km", "max"),
        mean_degree_2024=("syntax_mean_degree", lambda s: s.iloc[-1] if len(s) else np.nan),
        syntax_years=("syntax_results_exists", "sum"),
    )
    syntax_summary["syntax_complete"] = syntax_summary["syntax_years"] == len(YEARS)
    write_csv(syntax_summary, out_dir / f"{scope_name}_syntax_qc_summary_by_city.csv")

    # figures
    fig_dir = out_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_period_updates(period_summary, fig_dir / f"{scope_name}_period_update_total.png", f"{scope_name}: 每时期城市更新总量与更新率")
    plot_transition_stacked(transition_top10, fig_dir / f"{scope_name}_period_transition_top10_stacked.png", f"{scope_name}: 每时期主要用地转移类型")
    plot_city_top10(city_top10, fig_dir / f"{scope_name}_period_city_top10_updates.png", f"{scope_name}: 每时期更新数 Top 10 城市")
    plot_shap_group(shap_group, fig_dir / f"{scope_name}_period_shap_group_mean.png", f"{scope_name}: 每时期 SHAP 特征族平均贡献")

    # compact notes for reporting
    notes = []
    for _, r in period_summary.iterrows():
        p = str(r["period"])
        tt = transition_top10[transition_top10["period"].astype(str) == p].head(1)
        ct = city_top10[city_top10["period"].astype(str) == p].head(1)
        sg = shap_group_summary[shap_group_summary["period"].astype(str) == p].sort_values("rank").head(1)
        notes.append(
            {
                "period": p,
                "updates": int(r["updates"]),
                "update_rate": float(r["update_rate"]),
                "top_transition": tt.iloc[0]["transition"] if len(tt) else "",
                "top_transition_count": int(tt.iloc[0]["count"]) if len(tt) else np.nan,
                "top_city_by_updates": ct.iloc[0]["city_name"] if len(ct) else "",
                "top_city_updates": int(ct.iloc[0]["updates"]) if len(ct) else np.nan,
                "top_shap_group": sg.iloc[0]["group"] if len(sg) else "",
                "top_shap_mean_delta_auc": float(sg.iloc[0]["mean_delta_auc"]) if len(sg) else np.nan,
            }
        )
    write_csv(pd.DataFrame(notes), out_dir / f"{scope_name}_reporting_period_key_findings.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--summary", default="city_extract_national/pipeline_runs/sample_city_pipeline_summary_auto_updated.csv")
    ap.add_argument("--out-dir", default="city_extract_national/national_analysis")
    args = ap.parse_args()

    setup_plot_style()
    root = Path(args.root).resolve()
    summary = read_csv(root / args.summary)
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep a copy of the city-level summary alongside national outputs.
    write_csv(summary, out_dir / "city_pipeline_summary_used.csv")

    scopes = {
        "formal72": summary[summary["sample_type"] == "正式样本"].copy(),
        "all79": summary.copy(),
    }
    for scope_name, scope_df in scopes.items():
        build_scope_outputs(scope_name, scope_df, out_dir)

    print(f"Output directory: {out_dir}")
    print(f"Formal cities: {len(scopes['formal72'])}; All cities: {len(scopes['all79'])}")


if __name__ == "__main__":
    main()
