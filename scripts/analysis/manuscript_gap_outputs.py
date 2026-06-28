from __future__ import annotations

import math
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, kruskal
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("PROJECT_ROOT", ".")).resolve()
RUNS_DIR = ROOT / "city_extract_national/pipeline_runs_dynamic_t1"
ANALYSIS_DIR = ROOT / "city_extract_national/national_analysis_dynamic_t1"
FIG_DIR = ROOT / "论文/顶刊/我的/generated_figures"
MASTER = ANALYSIS_DIR / "all79_city_period_master_table_dynamic_t1_with_yearbook.csv"
TRANSITIONS = ANALYSIS_DIR / "all79_city_period_transition_counts.csv"
TOP_FEATURES = ANALYSIS_DIR / "all79_city_period_shap_top_features.csv"

RANDOM_STATE = 42
PERIODS = [
    (1984, 1990, "1984->1990", "84-90"),
    (1990, 1995, "1990->1995", "90-95"),
    (1995, 2000, "1995->2000", "95-00"),
    (2000, 2005, "2000->2005", "00-05"),
    (2005, 2010, "2005->2010", "05-10"),
    (2010, 2015, "2010->2015", "10-15"),
    (2015, 2020, "2015->2020", "15-20"),
    (2020, 2024, "2020->2024", "20-24"),
]
PERIOD_MID = {p: (a + b) / 2 for a, b, p, _ in PERIODS}
PERIOD_SHORT = {p: s for _, _, p, s in PERIODS}
HISTGBM_PARAMS = dict(max_iter=80, max_depth=5, learning_rate=0.05, random_state=RANDOM_STATE)


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_x(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)


def stratified_cap(indices: np.ndarray, y: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    if len(indices) <= max_n:
        return indices
    rng = np.random.RandomState(seed)
    yy = y[indices]
    pos = indices[yy == 1]
    neg = indices[yy == 0]
    if len(pos) == 0 or len(neg) == 0:
        return rng.choice(indices, size=max_n, replace=False)
    pos_n = max(1, round(max_n * len(pos) / len(indices)))
    pos_n = min(pos_n, len(pos))
    neg_n = max_n - pos_n
    if neg_n > len(neg):
        neg_n = len(neg)
        pos_n = max_n - neg_n
    chosen = np.concatenate(
        [
            rng.choice(pos, size=pos_n, replace=False),
            rng.choice(neg, size=neg_n, replace=False),
        ]
    )
    rng.shuffle(chosen)
    return chosen


def make_models():
    return {
        "GBDT": HistGradientBoostingClassifier(**HISTGBM_PARAMS),
        "Random forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "L2 logistic regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=800,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def rename_figure_numbers() -> pd.DataFrame:
    mapping = [
        ("fig5_whitebox_mechanism_audit.png", "fig6_whitebox_mechanism_audit.png"),
        ("fig5_whitebox_source_subset.csv", "fig6_whitebox_source_subset.csv"),
        ("fig6_city_heterogeneity_renewal_regimes.png", "fig7_city_heterogeneity_renewal_regimes.png"),
        ("fig6_city_period_renewal_dominance_matrix.csv", "fig7_city_period_renewal_dominance_matrix.csv"),
        ("fig6_road_municipal_inflow_share.csv", "fig7_road_municipal_inflow_share.csv"),
    ]
    rows = []
    for old, new in mapping:
        src = FIG_DIR / old
        dst = FIG_DIR / new
        status = "missing_source"
        if src.exists():
            dst.write_bytes(src.read_bytes())
            status = "copied_new_number"
        rows.append({"old_file": old, "new_file": new, "status": status})
    out = pd.DataFrame(rows)
    out.to_csv(FIG_DIR / "figure_renumbering_log.csv", index=False, encoding="utf-8-sig")
    return out


def period_ap_robustness(max_train=50_000, max_test=50_000, test_size=0.35) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_path = ANALYSIS_DIR / "appendix_A_multimodel_ap_city_period.csv"
    table_path = ANALYSIS_DIR / "appendix_A_table_A2_period_ap_by_classifier.csv"
    fig_path = FIG_DIR / "table_period_ap_multimodel_robustness.csv"
    if detail_path.exists() and table_path.exists():
        detail = pd.read_csv(detail_path)
        table = pd.read_csv(table_path)
        table.to_csv(fig_path, index=False, encoding="utf-8-sig")
        return detail, table

    meta = pd.read_csv(MASTER)[
        ["province", "city_name", "city_name_en", "region", "city_size_class", "period", "year_t1", "year_t2"]
    ].drop_duplicates(["city_name", "period"])
    rows = []
    run_dirs = sorted(p for p in RUNS_DIR.iterdir() if p.is_dir() and p.name.endswith("全流程"))
    for ci, run_dir in enumerate(run_dirs, 1):
        city = run_dir.name.replace("全流程", "")
        dataset = run_dir / "tculu_pipeline_v9_final/data/dataset.npz"
        if not dataset.exists():
            continue
        d = np.load(dataset, allow_pickle=True)
        X = d["X"]
        y = d["is_update"].astype(np.int8)
        years = d["year_t1"].astype(int)
        print(f"[AP {ci}/{len(run_dirs)}] {city}: n={len(y):,}", flush=True)
        for year1, year2, period, _ in PERIODS:
            mask = years == year1
            if not mask.any():
                continue
            yp = y[mask]
            n = int(len(yp))
            n_pos = int(yp.sum())
            if n_pos < 5 or n - n_pos < 5:
                for clf in make_models():
                    rows.append(
                        dict(city_name=city, period=period, year_t1=year1, year_t2=year2, samples=n, updates=n_pos,
                             classifier=clf, average_precision=np.nan, train_n_used=0, test_n_used=0,
                             status="skipped_too_few_class_samples")
                    )
                continue
            local = np.arange(n)
            try:
                tr, te = train_test_split(local, test_size=test_size, stratify=yp, random_state=RANDOM_STATE)
            except ValueError:
                continue
            tr = stratified_cap(tr, yp, max_train, RANDOM_STATE)
            te = stratified_cap(te, yp, max_test, RANDOM_STATE + 1)
            Xp = X[np.flatnonzero(mask)]
            Xtr = sanitize_x(Xp[tr])
            Xte = sanitize_x(Xp[te])
            ytr = yp[tr]
            yte = yp[te]
            for clf_name, model in make_models().items():
                t0 = time.time()
                try:
                    model.fit(Xtr, ytr)
                    pred = model.predict_proba(Xte)[:, 1]
                    ap = float(average_precision_score(yte, pred))
                    status = "ok"
                except Exception as exc:
                    ap = np.nan
                    status = f"failed:{type(exc).__name__}"
                print(f"  {period} {clf_name:<22} AP={ap if np.isfinite(ap) else np.nan:.4f} {time.time()-t0:.1f}s", flush=True)
                rows.append(
                    dict(city_name=city, period=period, year_t1=year1, year_t2=year2, samples=n, updates=n_pos,
                         classifier=clf_name, average_precision=ap, train_n_used=len(ytr), test_n_used=len(yte),
                         status=status)
                )
    detail = pd.DataFrame(rows).merge(meta, on=["city_name", "period", "year_t1", "year_t2"], how="left")
    detail["period_short"] = detail["period"].map(PERIOD_SHORT)
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

    ok = detail[detail["status"].astype(str).str.startswith("ok") & detail["average_precision"].notna()].copy()
    table = (
        ok.groupby(["classifier", "period", "period_short"], as_index=False)
        .agg(mean_ap=("average_precision", "mean"), median_ap=("average_precision", "median"), n_cities=("city_name", "nunique"))
    )
    table["period_order"] = table["period"].map(PERIOD_MID)
    table = table.sort_values(["classifier", "period_order"])
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    table.to_csv(fig_path, index=False, encoding="utf-8-sig")
    return detail, table


def twfe_residualize(y: np.ndarray, x: np.ndarray, city: np.ndarray, period: np.ndarray, max_iter=1000, tol=1e-12):
    yr = y.astype(float).copy()
    xr = x.astype(float).copy()
    for _ in range(max_iter):
        old = yr.copy()
        for labels in (city, period):
            for lab in np.unique(labels):
                m = labels == lab
                yr[m] -= yr[m].mean()
                xr[m] -= xr[m].mean(axis=0)
        if np.nanmax(np.abs(yr - old)) < tol:
            break
    return yr, xr


def ols_hc1(y: np.ndarray, x: np.ndarray):
    x = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ beta
    n, k = x.shape
    xtx_inv = np.linalg.pinv(x.T @ x)
    meat = x.T @ ((resid ** 2)[:, None] * x)
    cov = (n / max(n - k, 1)) * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = chi2.sf(t ** 2, 1)
    return beta, se, t, p, resid


def twfe_table() -> pd.DataFrame:
    df = pd.read_csv(MASTER)
    df = df[df["period_test_auc"].notna()].copy()
    df["tertiary_10pp"] = df["yb_t2_tertiary_share_city_pct"] / 10.0
    df["ln_gdp_pc"] = np.log(df["yb_t2_gdp_pc_city_yuan"].replace(0, np.nan))
    df["ln_builtup_stock"] = np.log(df["t1_extent_cells"].replace(0, np.nan))
    specs = [
        ("M1 tertiary only", ["tertiary_10pp"]),
        ("M2 tertiary + wealth", ["tertiary_10pp", "ln_gdp_pc"]),
        ("M3 tertiary + stock", ["tertiary_10pp", "ln_builtup_stock"]),
        ("M4 full", ["tertiary_10pp", "ln_gdp_pc", "ln_builtup_stock"]),
    ]
    rows = []
    for spec_name, xs in specs:
        sub = df[["period_test_auc", "city_name", "period"] + xs].dropna().copy()
        y = sub["period_test_auc"].to_numpy(float)
        x = sub[xs].to_numpy(float)
        city = sub["city_name"].astype(str).to_numpy()
        period = sub["period"].astype(str).to_numpy()
        yr, xr = twfe_residualize(y, x, city, period)
        beta, se, t, p, resid = ols_hc1(yr, xr)
        ssr = float((resid ** 2).sum())
        sst = float((yr ** 2).sum())
        r2_within = 1 - ssr / sst if sst else np.nan
        for j, var in enumerate(xs, start=1):
            rows.append(
                {
                    "model": spec_name,
                    "dependent_variable": "period_test_auc",
                    "fixed_effects": "city + period",
                    "variable": var,
                    "coefficient": beta[j],
                    "std_error_hc1": se[j],
                    "t_value": t[j],
                    "p_value": p[j],
                    "n": len(sub),
                    "within_r2": r2_within,
                    "interpretation": {
                        "tertiary_10pp": "AUC change associated with a 10 percentage-point higher tertiary-industry GDP share",
                        "ln_gdp_pc": "AUC change associated with a one-log-point higher GDP per capita",
                        "ln_builtup_stock": "AUC change associated with a one-log-point larger t1 built-up stock proxy",
                    }[var],
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(FIG_DIR / "table2_twfe_auc_economic_covariates.csv", index=False, encoding="utf-8-sig")
    return out


def feature_set_table() -> pd.DataFrame:
    rows = [
        ("M0", "cell only", "地块自身用地类别", "7", "仅包含当前地块的 one-hot 用地身份。"),
        ("M1", "cell + Neighborhood", "自身用地 + 周边组成", "73", "在多个半径内统计邻里用地比例、主导类型、多样性、建成/道路等周边形态。"),
        ("M2", "cell + SyntaxDecay", "自身用地 + 空间句法", "87", "加入多半径 Integration、NAIN、NACH/Choice 等街道网络可达性与选择性指标。"),
        ("M3", "cell + Functional", "自身用地 + 功能形态", "21", "加入功能团块尺度、纯度、边界、熵、多样性、簇形态等用地功能组织特征。"),
        ("M4", "cell + Neighborhood + Syntax", "自身用地 + 周边 + 句法", "153", "检验周边环境与街道网络共同作用。"),
        ("M5", "cell + Neighborhood + Functional", "自身用地 + 周边 + 功能", "87", "检验局部邻里与功能团块组织共同作用。"),
        ("M6", "cell + Syntax + Functional", "自身用地 + 句法 + 功能", "101", "检验街道网络与功能组织共同作用。"),
        ("M7", "AllFeatures", "全特征", "167", "包含自身用地、邻里、空间句法、功能形态全部特征。"),
    ]
    out = pd.DataFrame(rows, columns=["model_id", "short_name", "feature_combination_cn", "n_features", "description"])
    out.to_csv(FIG_DIR / "table_M0_M7_feature_sets.csv", index=False, encoding="utf-8-sig")
    with open(FIG_DIR / "table_M0_M7_feature_sets_insertion_note.md", "w", encoding="utf-8") as f:
        f.write(
            "建议放置位置：第四章第一段中首次介绍消融实验、图3之前。可在“我们构建了八组嵌套特征组合”之后接入本表，"
            "正文只需概括 M0-M7 从地块自身属性逐步叠加邻里、空间句法与功能形态，完整定义见表。\\n"
        )
    return out


def build_pdp_figure(max_per_city=3000, period="2020->2024") -> tuple[pd.DataFrame, pd.DataFrame]:
    out_png = FIG_DIR / "fig5_pdp_contemporary_top6.png"
    out_csv = FIG_DIR / "fig5_pdp_contemporary_top6_curves.csv"
    out_feat = FIG_DIR / "fig5_pdp_contemporary_top6_features.csv"
    if out_png.exists() and out_csv.exists():
        return pd.read_csv(out_csv), pd.read_csv(out_feat)
    top = pd.read_csv(TOP_FEATURES)
    top_period = top[top["period"] == period].copy()
    feature_rank = (
        top_period.groupby("feature", as_index=False)
        .agg(mean_delta_auc=("delta_auc", "mean"), n_cities=("city_name", "nunique"))
        .sort_values("mean_delta_auc", ascending=False)
    )
    candidates = feature_rank["feature"].tolist()

    rng = np.random.RandomState(RANDOM_STATE)
    xs, ys, names = [], [], None
    for run_dir in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir() and p.name.endswith("全流程")):
        dpath = run_dir / "tculu_pipeline_v9_final/data/dataset.npz"
        if not dpath.exists():
            continue
        d = np.load(dpath, allow_pickle=True)
        yr = d["year_t1"].astype(int)
        mask = yr == 2020
        if not mask.any():
            continue
        y = d["is_update"].astype(np.int8)[mask]
        idx = np.arange(len(y))
        chosen = stratified_cap(idx, y, max_per_city, RANDOM_STATE)
        xs.append(sanitize_x(d["X"][np.flatnonzero(mask)][chosen]))
        ys.append(y[chosen])
        if names is None:
            names = [str(v) for v in d["col_names"]]
    X = np.vstack(xs)
    y = np.concatenate(ys)
    feat_to_idx = {n: i for i, n in enumerate(names)}
    selected = [f for f in candidates if f in feat_to_idx][:6]
    if len(selected) < 6:
        extra = ["cell_class_1", "cluster_size_log", "neighbor_diversity_8", "R500_entropy", "mean_R2000_Integration", "max_R1000_NAIN"]
        selected += [f for f in extra if f in feat_to_idx and f not in selected]
        selected = selected[:6]
    Xtr_i, Xte_i, ytr, yte = train_test_split(np.arange(len(y)), y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    Xtr = X[Xtr_i]
    Xte = X[Xte_i]
    model = HistGradientBoostingClassifier(**HISTGBM_PARAMS)
    model.fit(Xtr, ytr)

    curve_rows = []
    for feat in selected:
        j = feat_to_idx[feat]
        vals = Xte[:, j]
        uniq = np.unique(vals[np.isfinite(vals)])
        if len(uniq) <= 3:
            grid = uniq
        else:
            grid = np.unique(np.quantile(vals[np.isfinite(vals)], np.linspace(0.05, 0.95, 12)))
        for g in grid:
            xx = Xte.copy()
            xx[:, j] = g
            pred = model.predict_proba(xx)[:, 1].mean()
            curve_rows.append({"feature": feat, "grid_value": float(g), "mean_pred_update_prob": float(pred)})
    curves = pd.DataFrame(curve_rows)
    feat_meta = feature_rank[feature_rank["feature"].isin(selected)].copy()
    feat_meta.to_csv(out_feat, index=False, encoding="utf-8-sig")
    curves.to_csv(out_csv, index=False, encoding="utf-8-sig")

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
    for ax, feat in zip(axes.ravel(), selected):
        g = curves[curves["feature"] == feat]
        ax.plot(g["grid_value"], g["mean_pred_update_prob"], color="#bd3f45", marker="o", lw=2)
        ax.fill_between(g["grid_value"], g["mean_pred_update_prob"], 0, color="#bd3f45", alpha=0.13)
        ax.set_title(feat, fontsize=11, fontweight="bold")
        ax.set_ylabel("avg P(update)")
        ax.grid(alpha=0.22)
    fig.suptitle("Figure 5. Marginal effects of contemporary top renewal predictors (2020-2024)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return curves, feat_meta


def kruskal_wallis_tests() -> pd.DataFrame:
    tr = pd.read_csv(TRANSITIONS)
    totals = tr.groupby(["city_name", "period", "region"], as_index=False).agg(total_updates=("count", "sum"))
    metrics = []
    ir = tr[((tr["from_class"] == 1) & (tr["to_class"] == 4)) | ((tr["from_class"] == 4) & (tr["to_class"] == 1))]
    metrics.append(("industrial_residential_reciprocal_share", ir))
    upgrade = tr[tr["to_class"].isin([2, 3])]
    metrics.append(("to_commercial_public_share", upgrade))
    ind_upgrade = tr[(tr["from_class"] == 4) & (tr["to_class"].isin([2, 3]))]
    metrics.append(("industrial_to_commercial_public_share", ind_upgrade))
    rows = []
    for metric, sub in metrics:
        m = totals.copy()
        num = sub.groupby(["city_name", "period"], as_index=False).agg(numerator=("count", "sum"))
        m = m.merge(num, on=["city_name", "period"], how="left")
        m["numerator"] = m["numerator"].fillna(0)
        m["value"] = m["numerator"] / m["total_updates"].replace(0, np.nan)
        for period_scope, ss in [("all_periods", m), ("2015->2024", m[m["period"].isin(["2015->2020", "2020->2024"])])]:
            groups = [g["value"].dropna().to_numpy() for _, g in ss.groupby("region") if len(g["value"].dropna()) >= 3]
            if len(groups) < 2:
                continue
            H, p = kruskal(*groups)
            med = ss.groupby("region")["value"].median().to_dict()
            rows.append(
                {
                    "metric": metric,
                    "period_scope": period_scope,
                    "test": "Kruskal-Wallis across regions",
                    "H": H,
                    "p_value": p,
                    "n_city_periods": int(ss["value"].notna().sum()),
                    "median_east": med.get("东部", np.nan),
                    "median_central": med.get("中部", np.nan),
                    "median_west": med.get("西部", np.nan),
                    "median_northeast": med.get("东北", np.nan),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(FIG_DIR / "table_kruskal_wallis_industrial_city_patterns.csv", index=False, encoding="utf-8-sig")
    return out


def main():
    ensure_dirs()
    print("1/6 renumbering figures")
    ren = rename_figure_numbers()
    print(ren.to_string(index=False))
    print("2/6 AP robustness")
    _, ap_table = period_ap_robustness()
    print(ap_table.head().to_string(index=False))
    print("3/6 TWFE table")
    twfe = twfe_table()
    print(twfe.to_string(index=False))
    print("4/6 M0-M7 table")
    feature_set_table()
    print("5/6 PDP figure")
    build_pdp_figure()
    print("6/6 Kruskal-Wallis tests")
    kw = kruskal_wallis_tests()
    print(kw.to_string(index=False))
    print("Done. Outputs:", FIG_DIR)


if __name__ == "__main__":
    main()
