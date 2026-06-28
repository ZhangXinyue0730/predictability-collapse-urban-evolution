from __future__ import annotations

import re
import os
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("PROJECT_ROOT", ".")).resolve()
NAT = ROOT / "city_extract_national" / "national_analysis_dynamic_t1"
RUNS = ROOT / "city_extract_national" / "pipeline_runs_dynamic_t1"
GEN = ROOT / "论文" / "顶刊" / "我的" / "generated_figures"
AUDIT = ROOT / "全国城市更新研究辅助文件" / "分析脚本与核验" / "核验结果"
OUT = Path(os.environ.get("SUPPLEMENTARY_WORKBOOK_INTERMEDIATE", "outputs/supplementary_data_workbook/intermediate")).resolve()


def public_path(value: str) -> str:
    """Convert local absolute paths to reproducible project-relative references."""
    text = str(value)
    root_text = str(ROOT)
    text = text.replace(root_text, "[project_root]")
    return text


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove machine-specific absolute paths from string/object columns."""
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].map(lambda x: public_path(x) if isinstance(x, str) else x)
    return out


def write_csv(df: pd.DataFrame, name: str, source: str, sources: list[dict]) -> None:
    path = OUT / f"{name}.csv"
    df = sanitize_dataframe(df)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    sources.append(
        {
            "sheet_name": name,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "source_file_or_rule": public_path(source),
        }
    )


def normalize_period(value: str) -> str:
    return str(value).replace("-", "->")


def city_from_run_path(path: Path) -> str:
    for part in path.parts:
        if part.endswith("全流程"):
            return part.removesuffix("全流程")
    return ""


def concat_city_csv(filename: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(RUNS.glob(f"*/tculu_pipeline_v9_final/data/{filename}")):
        df = pd.read_csv(path)
        city_cn = city_from_run_path(path)
        df.insert(0, "city_folder", city_cn)
        df["source_run_dir"] = public_path(str(path.parent.parent))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def infer_feature_dictionary(feature_importance: pd.DataFrame) -> pd.DataFrame:
    features = (
        feature_importance[["feature", "group"]]
        .drop_duplicates()
        .sort_values(["group", "feature"])
        .reset_index(drop=True)
    )

    def radius(name: str):
        match = re.search(r"R(\d+)", name)
        return int(match.group(1)) if match else None

    def family(name: str, group: str) -> str:
        if group:
            return group
        if name.startswith("cell_class_"):
            return "Cell class"
        if name.startswith(("nb_", "neighbor_")) or name.endswith(("_entropy", "_purity", "_boundary")):
            return "Neighborhood/Functional"
        if any(k in name for k in ["Integration", "NAIN", "NACH", "Choice", "_MD", "_TD", "_TL", "_NC"]):
            return "Syntax"
        return "Unclassified"

    def definition(name: str, group: str) -> str:
        r = radius(name)
        suffix = f" at {r} m radius" if r else ""
        if name.startswith("cell_class_"):
            return "One-hot focal-cell land-use class indicator."
        if name.startswith("nb_") and "_class_" in name:
            return "Neighbourhood share of a land-use class" + suffix + "."
        if "shannon" in name:
            return "Neighbourhood land-use Shannon diversity" + suffix + "."
        if "dominant" in name:
            return "Dominant neighbourhood land-use category" + suffix + "."
        if "builtup" in name:
            return "Built-up composition or intensity descriptor" + suffix + "."
        if "road" in name:
            return "Road/municipal composition descriptor" + suffix + "."
        if any(k in name for k in ["Integration", "NAIN", "NACH", "Choice", "_MD", "_TD", "_TL", "_NC"]):
            return "Street-network syntax metric projected to cells" + suffix + "."
        if "cluster" in name:
            return "Functional-cluster morphology descriptor."
        if "entropy" in name:
            return "Functional or neighbourhood entropy descriptor" + suffix + "."
        if "diversity" in name:
            return "Functional or neighbourhood diversity descriptor."
        if "purity" in name:
            return "Functional-cluster purity descriptor" + suffix + "."
        if "boundary" in name:
            return "Functional-cluster boundary descriptor" + suffix + "."
        return f"Model feature from the {group or 'unknown'} feature family."

    rows = []
    for _, row in features.iterrows():
        name = row["feature"]
        grp = row.get("group", "")
        rows.append(
            {
                "feature_name": name,
                "family": family(name, grp),
                "radius_m": radius(name),
                "definition": definition(name, grp),
                "source": "Derived from period_shap_feature_importance.csv feature names.",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sources: list[dict] = []

    master = pd.read_csv(NAT / "all79_city_period_master_table_dynamic_t1_with_yearbook.csv")
    write_csv(master, "SD1_Master_79x8", str(NAT / "all79_city_period_master_table_dynamic_t1_with_yearbook.csv"), sources)

    auc = pd.read_csv(NAT / "appendix_A_multimodel_auc_city_period.csv")
    ap = pd.read_csv(NAT / "appendix_A_multimodel_ap_city_period.csv")
    auc["period"] = auc["period"].map(normalize_period)
    ap["period"] = ap["period"].map(normalize_period)
    classifier_map = {"GBDT": "Gradient boosting"}
    ap["classifier"] = ap["classifier"].replace(classifier_map)
    auc_long = auc.rename(columns={"auc": "value"})
    auc_long["metric"] = "AUC"
    ap_long = ap.rename(columns={"average_precision": "value"})
    ap_long["metric"] = "AP"
    common_cols = [
        "province",
        "city_name",
        "city_name_en",
        "region",
        "city_size_class",
        "period",
        "period_short",
        "year_t1",
        "year_t2",
        "samples",
        "updates",
        "classifier",
        "metric",
        "value",
        "train_n_used",
        "test_n_used",
        "status",
    ]
    for col in common_cols:
        if col not in auc_long.columns:
            auc_long[col] = pd.NA
        if col not in ap_long.columns:
            ap_long[col] = pd.NA
    multimodel = pd.concat([auc_long[common_cols], ap_long[common_cols]], ignore_index=True)
    write_csv(
        multimodel,
        "SD2_Multimodel_AUC_AP",
        f"{NAT / 'appendix_A_multimodel_auc_city_period.csv'} + {NAT / 'appendix_A_multimodel_ap_city_period.csv'}",
        sources,
    )

    ablation = pd.read_csv(NAT / "all79_city_ablation_results.csv")
    write_csv(ablation, "SD3_Ablation_M0_M7", str(NAT / "all79_city_ablation_results.csv"), sources)

    importance = concat_city_csv("period_shap_feature_importance.csv")
    write_csv(
        importance,
        "SD4_Permutation_Importance",
        "Concatenated current-scope city files: pipeline_runs_dynamic_t1/*/tculu_pipeline_v9_final/data/period_shap_feature_importance.csv",
        sources,
    )

    transitions = concat_city_csv("update_period_transition_counts.csv")
    write_csv(
        transitions,
        "SD5_Transitions_All",
        "Concatenated current-scope city files: pipeline_runs_dynamic_t1/*/tculu_pipeline_v9_final/data/update_period_transition_counts.csv",
        sources,
    )

    expansion = pd.read_csv(AUDIT / "建成区扩张用地分类" / "city_period_expansion_landuse_long.csv")
    update_expansion = pd.merge(
        master[
            [
                "province",
                "region",
                "city_name",
                "city_name_en",
                "city_size_class",
                "period",
                "year_t1",
                "year_t2",
                "samples",
                "updates",
                "update_rate",
                "update_area_km2",
                "t1_extent_cells",
                "t2_extent_cells",
                "expansion_cells",
                "expansion_area_km2",
                "contraction_cells",
                "contraction_area_km2",
                "net_extent_change_cells",
                "net_extent_change_area_km2",
                "expansion_rate_vs_t1",
                "expansion_share_of_t2",
            ]
        ],
        expansion,
        on=["province", "region", "city_name", "city_name_en", "city_size_class", "period", "year_t1", "year_t2"],
        how="left",
        suffixes=("", "_by_landuse"),
    )
    write_csv(
        update_expansion,
        "SD6_Update_Expansion_Area",
        f"{NAT / 'all79_city_period_master_table_dynamic_t1_with_yearbook.csv'} + {AUDIT / '建成区扩张用地分类' / 'city_period_expansion_landuse_long.csv'}",
        sources,
    )

    regional = pd.read_csv(GEN / "table_kruskal_wallis_industrial_city_patterns.csv")
    write_csv(regional, "SD7_Regional_Tests", str(GEN / "table_kruskal_wallis_industrial_city_patterns.csv"), sources)

    syntax = pd.read_csv(NAT / "all79_syntax_metrics_city_year_wide.csv")
    write_csv(syntax, "SD8_Syntax_Summary", str(NAT / "all79_syntax_metrics_city_year_wide.csv"), sources)

    feature_dict = infer_feature_dictionary(importance)
    write_csv(feature_dict, "SD9_Feature_Dictionary_167", "Derived from SD4 feature names and groups.", sources)

    # Add compact supporting source index, including figure/table source files.
    extra_sources = [
        ("main_table_excel", NAT / "79城市新统计口径完整汇总大表_接入年鉴经济人口.xlsx"),
        ("master_dictionary", NAT / "all79_city_period_master_table_dynamic_t1_dictionary.csv"),
        ("table1_multimodel_summary", GEN / "table1_multimodel_auc_robustness.csv"),
        ("table2_twfe", GEN / "table2_twfe_auc_economic_covariates.csv"),
        ("table_m0_m7_feature_sets", GEN / "table_M0_M7_feature_sets.csv"),
        ("fig3_ablation_summary", GEN / "fig3_feature_ablation_summary.csv"),
        ("fig4_group_importance", GEN / "fig4_group_permutation_importance_summary.csv"),
        ("fig6_pdp_features", GEN / "fig6_pdp_contemporary_top6_features.csv"),
        ("fig6_pdp_curves", GEN / "fig6_pdp_contemporary_top6_curves.csv"),
        ("fig7_growth_auc", GEN / "fig7_built2built_growth_auc_audit_summary.csv"),
    ]
    source_df = pd.DataFrame(sources)
    extras = pd.DataFrame(
        [
            {
                "sheet_name": "",
                "rows": "",
                "columns": "",
                "source_file_or_rule": str(path),
                "source_label": label,
                "exists": path.exists(),
            }
            for label, path in extra_sources
        ]
    )
    source_df = pd.concat([source_df, extras], ignore_index=True)
    write_csv(source_df, "SD10_Source_Index", "Generated source index for Supplementary_Data.xlsx.", sources=[])

    summary = pd.DataFrame(sources)
    summary.to_csv(OUT / "_sheet_build_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"intermediate_dir={OUT}")


if __name__ == "__main__":
    main()
