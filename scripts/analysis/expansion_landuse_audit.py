#!/usr/bin/env python3
"""Classify built-up expansion by land-use type for the 79-city dynamic-t1 runs."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("PROJECT_ROOT", ".")).resolve()
MASTER_CSV = ROOT / "city_extract_national/national_analysis_dynamic_t1/all79_city_period_master_table_dynamic_t1_with_yearbook.csv"
OUT_DIR = ROOT / "全国城市更新研究辅助文件/分析脚本与核验/核验结果/建成区扩张用地分类"

CLASS_NAME = {
    1: "居住",
    2: "商业",
    3: "公共服务",
    4: "工业仓储",
    5: "道路市政",
    6: "绿地水域",
}
CLASS_NAME_EN = {
    1: "residential",
    2: "commercial",
    3: "public_service",
    4: "industrial_warehouse",
    5: "road_municipal",
    6: "green_water",
}
PIXEL_AREA_KM2 = 0.01


def read_master() -> pd.DataFrame:
    df = pd.read_csv(MASTER_CSV)
    required = {"province", "city_name", "period", "year_t1", "year_t2", "run_dir", "expansion_cells"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in master CSV: {missing}")
    return df


def compute_city_period_expansion(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide_rows: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []

    for _, meta in master.iterrows():
        run_dir = Path(str(meta["run_dir"]))
        data = run_dir / "data"
        y1 = int(meta["year_t1"])
        y2 = int(meta["year_t2"])

        extent1 = np.load(data / f"builtup_extent_dynamic_{y1}.npy").astype(bool)
        extent2 = np.load(data / f"builtup_extent_dynamic_{y2}.npy").astype(bool)
        lu2 = np.load(data / f"lu_{y2}.npy")

        expansion = extent2 & ~extent1
        expansion_cells = int(expansion.sum())
        class_counts = {cls: int((expansion & (lu2 == cls)).sum()) for cls in CLASS_NAME}
        classified_cells = int(sum(class_counts.values()))
        vacant_or_other_cells = expansion_cells - classified_cells

        row = {
            "province": meta.get("province", ""),
            "region": meta.get("region", ""),
            "city_name": meta.get("city_name", ""),
            "city_name_en": meta.get("city_name_en", ""),
            "city_size_class": meta.get("city_size_class", ""),
            "period": meta["period"],
            "year_t1": y1,
            "year_t2": y2,
            "expansion_cells_recomputed": expansion_cells,
            "expansion_area_km2_recomputed": expansion_cells * PIXEL_AREA_KM2,
            "expansion_cells_master": int(meta["expansion_cells"]) if pd.notna(meta["expansion_cells"]) else np.nan,
            "classified_expansion_cells_1_6": classified_cells,
            "classified_expansion_area_km2_1_6": classified_cells * PIXEL_AREA_KM2,
            "vacant_or_other_cells_in_expansion_extent": vacant_or_other_cells,
            "vacant_or_other_share_in_expansion_extent": vacant_or_other_cells / expansion_cells if expansion_cells else 0.0,
            "run_dir": str(run_dir),
        }

        for cls, name in CLASS_NAME.items():
            key = CLASS_NAME_EN[cls]
            count = class_counts[cls]
            row[f"expansion_{key}_cells"] = count
            row[f"expansion_{key}_area_km2"] = count * PIXEL_AREA_KM2
            row[f"expansion_{key}_share"] = count / classified_cells if classified_cells else 0.0
            long_rows.append({
                "province": row["province"],
                "region": row["region"],
                "city_name": row["city_name"],
                "city_name_en": row["city_name_en"],
                "city_size_class": row["city_size_class"],
                "period": row["period"],
                "year_t1": y1,
                "year_t2": y2,
                "landuse_class": cls,
                "landuse_name": name,
                "landuse_name_en": key,
                "expansion_cells": count,
                "expansion_area_km2": count * PIXEL_AREA_KM2,
                "share_of_classified_expansion": count / classified_cells if classified_cells else 0.0,
                "share_of_total_expansion_extent": count / expansion_cells if expansion_cells else 0.0,
                "city_period_classified_expansion_cells": classified_cells,
                "city_period_total_expansion_cells": expansion_cells,
                "run_dir": str(run_dir),
            })
        wide_rows.append(row)

    return pd.DataFrame(wide_rows), pd.DataFrame(long_rows)


def build_period_summary(long_df: pd.DataFrame, wide_df: pd.DataFrame) -> pd.DataFrame:
    grouped = long_df.groupby(["period", "year_t1", "year_t2", "landuse_class", "landuse_name"], as_index=False).agg(
        expansion_cells=("expansion_cells", "sum"),
        expansion_area_km2=("expansion_area_km2", "sum"),
    )
    period_total = wide_df.groupby(["period", "year_t1", "year_t2"], as_index=False).agg(
        total_expansion_cells=("expansion_cells_recomputed", "sum"),
        classified_expansion_cells_1_6=("classified_expansion_cells_1_6", "sum"),
        vacant_or_other_cells=("vacant_or_other_cells_in_expansion_extent", "sum"),
    )
    summary = grouped.merge(period_total, on=["period", "year_t1", "year_t2"], how="left")
    summary["share_of_classified_expansion"] = summary["expansion_cells"] / summary["classified_expansion_cells_1_6"]
    summary["share_of_total_expansion_extent"] = summary["expansion_cells"] / summary["total_expansion_cells"]
    return summary.sort_values(["year_t1", "landuse_class"])


def build_period_wide(summary: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for metric in ["expansion_cells", "expansion_area_km2", "share_of_classified_expansion"]:
        piv = summary.pivot_table(
            index=["period", "year_t1", "year_t2"],
            columns="landuse_name",
            values=metric,
            aggfunc="sum",
        ).reset_index()
        piv.columns = [
            f"{name}_{metric}" if name not in {"period", "year_t1", "year_t2"} else name
            for name in piv.columns
        ]
        parts.append(piv)
    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on=["period", "year_t1", "year_t2"], how="left")
    totals = summary[["period", "year_t1", "year_t2", "total_expansion_cells", "classified_expansion_cells_1_6", "vacant_or_other_cells"]].drop_duplicates()
    out = totals.merge(out, on=["period", "year_t1", "year_t2"], how="left")
    out["total_expansion_area_km2"] = out["total_expansion_cells"] * PIXEL_AREA_KM2
    return out.sort_values("year_t1")


def write_outputs(wide_df: pd.DataFrame, long_df: pd.DataFrame, summary: pd.DataFrame, period_wide: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    city_total = long_df.groupby(
        ["province", "region", "city_name", "city_name_en", "city_size_class", "landuse_class", "landuse_name"],
        as_index=False,
    ).agg(
        expansion_cells=("expansion_cells", "sum"),
        expansion_area_km2=("expansion_area_km2", "sum"),
    )
    city_den = wide_df.groupby(["province", "region", "city_name", "city_name_en", "city_size_class"], as_index=False).agg(
        classified_expansion_cells_1_6=("classified_expansion_cells_1_6", "sum"),
        total_expansion_cells=("expansion_cells_recomputed", "sum"),
    )
    city_total = city_total.merge(city_den, on=["province", "region", "city_name", "city_name_en", "city_size_class"], how="left")
    city_total["share_of_classified_expansion"] = city_total["expansion_cells"] / city_total["classified_expansion_cells_1_6"]

    municipal_rank = long_df[long_df["landuse_class"] == 5].sort_values("expansion_cells", ascending=False)
    commercial_rank = long_df[long_df["landuse_class"] == 2].sort_values("expansion_cells", ascending=False)

    files = {
        "city_period_expansion_landuse_wide.csv": wide_df,
        "city_period_expansion_landuse_long.csv": long_df,
        "period_expansion_landuse_summary_long.csv": summary,
        "period_expansion_landuse_summary_wide.csv": period_wide,
        "city_total_expansion_landuse_summary.csv": city_total,
        "municipal_expansion_city_period_rank.csv": municipal_rank,
        "commercial_expansion_city_period_rank.csv": commercial_rank,
    }
    for name, df in files.items():
        df.to_csv(OUT_DIR / name, index=False, encoding="utf-8-sig")

    xlsx = OUT_DIR / "建成区扩张用地分类统计.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        period_wide.to_excel(writer, sheet_name="各时期扩张分类汇总_宽表", index=False)
        summary.to_excel(writer, sheet_name="各时期扩张分类汇总_长表", index=False)
        wide_df.to_excel(writer, sheet_name="城市时期扩张分类_宽表", index=False)
        long_df.to_excel(writer, sheet_name="城市时期扩张分类_长表", index=False)
        municipal_rank.head(120).to_excel(writer, sheet_name="市政扩张城市时期排名", index=False)
        commercial_rank.head(120).to_excel(writer, sheet_name="商业扩张城市时期排名", index=False)
        city_total.to_excel(writer, sheet_name="城市总扩张分类", index=False)


def main() -> None:
    master = read_master()
    wide_df, long_df = compute_city_period_expansion(master)
    summary = build_period_summary(long_df, wide_df)
    period_wide = build_period_wide(summary)
    write_outputs(wide_df, long_df, summary, period_wide)
    print(OUT_DIR)
    print(period_wide[[
        "period",
        "total_expansion_cells",
        "居住_expansion_area_km2",
        "商业_expansion_area_km2",
        "公共服务_expansion_area_km2",
        "工业仓储_expansion_area_km2",
        "道路市政_expansion_area_km2",
        "绿地水域_expansion_area_km2",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
