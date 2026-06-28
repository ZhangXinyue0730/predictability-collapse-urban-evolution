#!/usr/bin/env python3
"""Audit transitions into Road/Municipal land use for the 79-city study."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import rasterio
except Exception as exc:  # pragma: no cover
    raise SystemExit("This script needs rasterio. Run it with the project venv.") from exc


ROOT = Path(os.environ.get("PROJECT_ROOT", ".")).resolve()
MASTER = ROOT / "city_extract_national/national_analysis_dynamic_t1/79城市新统计口径完整汇总大表_接入年鉴经济人口.xlsx"
MASTER_CSV = ROOT / "city_extract_national/national_analysis_dynamic_t1/all79_city_period_master_table_dynamic_t1_with_yearbook.csv"
OUT_DIR = ROOT / "全国城市更新研究辅助文件/分析脚本与核验/核验结果/道路市政转入核验"
YEARS = [1984, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2024]
RECENT_PERIODS = ["2015->2020", "2020->2024"]

CLASS_NAME = {
    0: "空地/未利用",
    1: "居住",
    2: "商业",
    3: "公共服务",
    4: "工业仓储",
    5: "道路市政",
    6: "绿地水域",
}

CLASS_NAME_EN = {
    0: "Vacant/unused",
    1: "Residential",
    2: "Commercial",
    3: "Public service",
    4: "Industrial/warehouse",
    5: "Road/municipal",
    6: "Green/water",
}

LANDUSE_COLORS = {
    0: (250, 247, 224, 255),  # vacant inside the valid city window
    1: (245, 166, 35, 255),
    2: (230, 0, 18, 255),
    3: (0, 105, 130, 255),
    4: (190, 157, 125, 255),
    5: (175, 175, 175, 255),
    6: (45, 135, 60, 255),
    255: (0, 0, 0, 0),
}

MUNICIPAL_SOURCE_COLORS = {
    0: (0, 0, 0, 0),
    1: (255, 140, 0, 255),
    2: (255, 0, 0, 255),
    3: (0, 115, 190, 255),
    4: (120, 80, 65, 255),
    6: (40, 160, 70, 255),
}


def read_master() -> pd.DataFrame:
    if MASTER_CSV.exists():
        df = pd.read_csv(MASTER_CSV)
    else:
        df = pd.read_excel(MASTER, sheet_name="master_with_yearbook")
    required = {"city_name", "period", "year_t1", "year_t2", "updates", "samples", "run_dir"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Master table missing columns: {missing}")
    return df


def rank_value(value: int, values: Iterable[int]) -> int | None:
    sorted_values = sorted((int(v) for v in values if int(v) > 0), reverse=True)
    if value <= 0:
        return None
    return sorted_values.index(int(value)) + 1 if int(value) in sorted_values else None


def load_transition_counts(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "data/update_period_transition_counts.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build_municipal_tables(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    source_rows = []
    for (city, run_dir_text), city_periods in master.groupby(["city_name", "run_dir"], dropna=False):
        run_dir = Path(str(run_dir_text))
        trans = load_transition_counts(run_dir)
        for _, meta in city_periods.iterrows():
            period = str(meta["period"])
            sub = trans[trans["period"].astype(str) == period].copy()
            if sub.empty:
                continue

            sub["count"] = sub["count"].fillna(0).astype(int)
            all_updates = int(meta["updates"])
            all_samples = int(meta["samples"])
            update_rate = float(meta["update_rate"]) if "update_rate" in meta and pd.notna(meta["update_rate"]) else np.nan

            municipal_in = sub[(sub["to_class"] == 5) & (sub["from_class"] != 5)]
            municipal_out = sub[(sub["from_class"] == 5) & (sub["to_class"] != 5)]
            municipal_in_count = int(municipal_in["count"].sum())
            municipal_out_count = int(municipal_out["count"].sum())

            by_target = sub.groupby("to_class", as_index=False)["count"].sum()
            municipal_target_count = int(by_target.loc[by_target["to_class"] == 5, "count"].sum())
            target_rank = rank_value(municipal_target_count, by_target["count"])

            top_row = sub.sort_values("count", ascending=False).head(1)
            top_transition = ""
            top_transition_count = 0
            top_transition_to_municipal = False
            if not top_row.empty:
                r = top_row.iloc[0]
                top_transition = f"{r['from_name']}→{r['to_name']}"
                top_transition_count = int(r["count"])
                top_transition_to_municipal = int(r["to_class"]) == 5 and int(r["from_class"]) != 5

            if municipal_in_count:
                main = municipal_in.sort_values("count", ascending=False).iloc[0]
                main_source = f"{main['from_name']}→道路市政"
                main_source_count = int(main["count"])
                main_source_share = main_source_count / municipal_in_count
            else:
                main_source = ""
                main_source_count = 0
                main_source_share = np.nan

            source_counts = {
                f"from_{CLASS_NAME_EN[i].lower().replace('/', '_').replace(' ', '_')}_to_municipal_count": int(
                    municipal_in.loc[municipal_in["from_class"] == i, "count"].sum()
                )
                for i in [1, 2, 3, 4, 6]
            }

            rows.append({
                "province": meta.get("province", ""),
                "region": meta.get("region", ""),
                "city_name": city,
                "city_name_en": meta.get("city_name_en", ""),
                "city_size_class": meta.get("city_size_class", ""),
                "period": period,
                "year_t1": int(meta["year_t1"]),
                "year_t2": int(meta["year_t2"]),
                "samples": all_samples,
                "updates": all_updates,
                "update_rate": update_rate,
                "municipal_in_count": municipal_in_count,
                "municipal_in_area_km2": municipal_in_count * 0.01,
                "municipal_in_share_of_updates": municipal_in_count / all_updates if all_updates else np.nan,
                "municipal_out_count": municipal_out_count,
                "municipal_net_count": municipal_in_count - municipal_out_count,
                "municipal_target_rank_among_to_classes": target_rank,
                "is_top1_transition_to_municipal": top_transition_to_municipal,
                "top_overall_transition": top_transition,
                "top_overall_transition_count": top_transition_count,
                "main_municipal_source": main_source,
                "main_municipal_source_count": main_source_count,
                "main_municipal_source_share": main_source_share,
                "period_test_auc": meta.get("period_test_auc", np.nan),
                "shap_top1_group": meta.get("shap_top1_group", ""),
                "shap_top1_feature": meta.get("shap_top1_feature", ""),
                "run_dir": str(run_dir),
                **source_counts,
            })

            for _, src in municipal_in.sort_values("count", ascending=False).iterrows():
                cnt = int(src["count"])
                source_rows.append({
                    "province": meta.get("province", ""),
                    "region": meta.get("region", ""),
                    "city_name": city,
                    "period": period,
                    "from_class": int(src["from_class"]),
                    "from_name": src["from_name"],
                    "to_class": 5,
                    "to_name": "道路市政",
                    "count": cnt,
                    "area_km2": cnt * 0.01,
                    "share_of_city_period_updates": cnt / all_updates if all_updates else np.nan,
                    "share_of_city_period_municipal_in": cnt / municipal_in_count if municipal_in_count else np.nan,
                    "run_dir": str(run_dir),
                })

    municipal = pd.DataFrame(rows)
    sources = pd.DataFrame(source_rows)
    period_summary = municipal.groupby("period", as_index=False).agg(
        year_t1=("year_t1", "first"),
        year_t2=("year_t2", "first"),
        samples=("samples", "sum"),
        updates=("updates", "sum"),
        municipal_in_count=("municipal_in_count", "sum"),
        municipal_in_area_km2=("municipal_in_area_km2", "sum"),
        municipal_out_count=("municipal_out_count", "sum"),
        municipal_net_count=("municipal_net_count", "sum"),
    )
    period_summary["municipal_in_share_of_updates"] = (
        period_summary["municipal_in_count"] / period_summary["updates"]
    )
    period_summary = period_summary.sort_values("year_t1")
    return municipal, sources, period_summary


def select_case_rows(municipal: pd.DataFrame) -> pd.DataFrame:
    recent = municipal[municipal["period"].isin(RECENT_PERIODS)].copy()
    recent = recent.sort_values(["municipal_in_count", "municipal_in_share_of_updates"], ascending=False)

    selected: list[pd.Series] = []
    seen_cities = set()
    for _, row in recent.iterrows():
        if row["city_name"] in seen_cities:
            continue
        selected.append(row)
        seen_cities.add(row["city_name"])
        if len(selected) >= 2:
            break

    share_rank = recent[recent["updates"] >= 1000].sort_values(
        ["municipal_in_share_of_updates", "municipal_in_count"],
        ascending=False,
    )
    for _, row in share_rank.iterrows():
        if row["city_name"] in seen_cities:
            continue
        selected.append(row)
        seen_cities.add(row["city_name"])
        break

    wuxi = recent[
        recent["city_name"].astype(str).str.contains("无锡", na=False)
        | recent.get("city_name_en", pd.Series("", index=recent.index)).astype(str).str.lower().eq("wuxi")
    ].sort_values("municipal_in_count", ascending=False)
    if not wuxi.empty and "无锡" not in seen_cities:
        selected.append(wuxi.iloc[0])

    cases = pd.DataFrame(selected)
    reasons = []
    for i, row in cases.reset_index(drop=True).iterrows():
        if "无锡" in str(row["city_name"]):
            reasons.append("teacher_flagged_reference_city_for_recent_municipal_land_validation")
        elif i == 2:
            reasons.append("top_recent_city_period_by_municipal_in_share_among_periods_with_updates_ge_1000")
        else:
            reasons.append(f"top_{i + 1}_recent_city_period_by_municipal_in_count")
    cases = cases.copy()
    cases.insert(0, "case_reason", reasons)
    return cases


def tif_profile(run_dir: Path, year: int) -> tuple[dict, np.ndarray | None]:
    ref = run_dir / f"cropped_years/{year}/landuse_raw.tif"
    mask_path = run_dir / f"cropped_years/{year}/mask.tif"
    if not ref.exists():
        raise FileNotFoundError(ref)
    with rasterio.open(ref) as src:
        profile = src.profile.copy()
    mask = None
    if mask_path.exists():
        with rasterio.open(mask_path) as src:
            mask = src.read(1) > 0
    profile.update(count=1, dtype="uint8", nodata=255, compress="lzw")
    return profile, mask


def write_tif(path: Path, array: np.ndarray, profile: dict, colormap: dict[int, tuple[int, int, int, int]], nodata: int = 255) -> None:
    p = profile.copy()
    p.update(dtype="uint8", count=1, nodata=nodata, compress="lzw")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(array.astype("uint8"), 1)
        dst.write_colormap(1, colormap)


def export_case_gis(cases: pd.DataFrame) -> None:
    for _, case in cases.iterrows():
        city = str(case["city_name"])
        period = str(case["period"])
        y1 = int(case["year_t1"])
        y2 = int(case["year_t2"])
        run_dir = Path(str(case["run_dir"]))
        data = run_dir / "data"
        case_dir = OUT_DIR / "gis核验" / f"{city}_{period.replace('->', '_')}"
        case_dir.mkdir(parents=True, exist_ok=True)

        lu1 = np.load(data / f"lu_{y1}.npy")
        lu2 = np.load(data / f"lu_{y2}.npy")
        extent = np.load(data / f"builtup_extent_dynamic_{y1}.npy").astype(bool)
        profile, mask = tif_profile(run_dir, y1)
        if mask is None:
            mask = np.ones(lu1.shape, dtype=bool)

        valid = extent & np.isin(lu1, [1, 2, 3, 4, 5, 6]) & np.isin(lu2, [1, 2, 3, 4, 5, 6])
        muni_in = valid & (lu2 == 5) & (lu1 != 5)

        source_arr = np.zeros(lu1.shape, dtype=np.uint8)
        source_arr[muni_in] = lu1[muni_in].astype(np.uint8)
        write_tif(
            case_dir / f"{city}_municipal_in_sources_{period.replace('->', '_')}.tif",
            source_arr,
            profile,
            {0: (0, 0, 0, 0), **MUNICIPAL_SOURCE_COLORS},
            nodata=0,
        )

        binary = np.zeros(lu1.shape, dtype=np.uint8)
        binary[muni_in] = 1
        write_tif(
            case_dir / f"{city}_municipal_in_binary_{period.replace('->', '_')}.tif",
            binary,
            profile,
            {0: (0, 0, 0, 0), 1: (255, 0, 255, 255)},
            nodata=0,
        )

        extent_arr = np.zeros(lu1.shape, dtype=np.uint8)
        extent_arr[extent] = 1
        write_tif(
            case_dir / f"{city}_builtup_extent_t1_{y1}.tif",
            extent_arr,
            profile,
            {0: (0, 0, 0, 0), 1: (255, 255, 255, 180)},
            nodata=0,
        )

        for year, lu in [(y1, lu1), (y2, lu2)]:
            out = np.full(lu.shape, 255, dtype=np.uint8)
            in_window = mask & np.isin(lu, [0, 1, 2, 3, 4, 5, 6])
            out[in_window] = lu[in_window].astype(np.uint8)
            write_tif(
                case_dir / f"{city}_landuse_{year}_colored.tif",
                out,
                profile,
                LANDUSE_COLORS,
                nodata=255,
            )

        # A compact CSV beside the rasters makes QGIS layer interpretation auditable.
        breakdown = []
        for src in [1, 2, 3, 4, 6]:
            cnt = int((source_arr == src).sum())
            if cnt:
                breakdown.append({
                    "city": city,
                    "period": period,
                    "source_class": src,
                    "source_name": CLASS_NAME[src],
                    "transition": f"{CLASS_NAME[src]}→道路市政",
                    "count": cnt,
                    "area_km2": cnt * 0.01,
                    "share_of_municipal_in": cnt / int(muni_in.sum()),
                })
        pd.DataFrame(breakdown).to_csv(case_dir / f"{city}_municipal_in_source_breakdown_{period.replace('->', '_')}.csv", index=False, encoding="utf-8-sig")

        readme = {
            "city": city,
            "period": period,
            "definition": "Within previous-period dynamic built-up extent; valid update requires t1 and t2 land-use classes both in 1-6; municipal-in means t2 class is 5 and t1 class is not 5.",
            "source_values": {str(k): CLASS_NAME[k] for k in [1, 2, 3, 4, 6]},
            "raster_files": {
                "municipal_in_sources": f"{city}_municipal_in_sources_{period.replace('->', '_')}.tif",
                "municipal_in_binary": f"{city}_municipal_in_binary_{period.replace('->', '_')}.tif",
                "builtup_extent_t1": f"{city}_builtup_extent_t1_{y1}.tif",
                "landuse_t1": f"{city}_landuse_{y1}_colored.tif",
                "landuse_t2": f"{city}_landuse_{y2}_colored.tif",
            },
        }
        (case_dir / "README.json").write_text(json.dumps(readme, ensure_ascii=False, indent=2), encoding="utf-8")


def write_outputs(municipal: pd.DataFrame, sources: pd.DataFrame, period_summary: pd.DataFrame, cases: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    recent = municipal[municipal["period"].isin(RECENT_PERIODS)].copy()
    recent_rank_count = recent.sort_values("municipal_in_count", ascending=False)
    recent_rank_share = recent[recent["updates"] >= 500].sort_values("municipal_in_share_of_updates", ascending=False)
    recent_city_total = recent.groupby(["province", "region", "city_name", "city_name_en", "city_size_class"], as_index=False).agg(
        updates=("updates", "sum"),
        municipal_in_count=("municipal_in_count", "sum"),
        municipal_in_area_km2=("municipal_in_area_km2", "sum"),
        municipal_out_count=("municipal_out_count", "sum"),
    )
    recent_city_total["municipal_in_share_of_updates"] = recent_city_total["municipal_in_count"] / recent_city_total["updates"]
    recent_city_total = recent_city_total.sort_values("municipal_in_count", ascending=False)

    for name, df in {
        "municipal_city_period_all": municipal,
        "municipal_source_breakdown_all": sources,
        "municipal_period_summary": period_summary,
        "municipal_recent_rank_by_count": recent_rank_count,
        "municipal_recent_rank_by_share": recent_rank_share,
        "municipal_recent_city_total": recent_city_total,
        "municipal_recommended_cases": cases,
    }.items():
        df.to_csv(OUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    try:
        xlsx = OUT_DIR / "道路市政转入核验汇总.xlsx"
        with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
            period_summary.to_excel(writer, sheet_name="各时期市政转入总量", index=False)
            recent_rank_count.head(80).to_excel(writer, sheet_name="近年城市时期排名_按数量", index=False)
            recent_rank_share.head(80).to_excel(writer, sheet_name="近年城市时期排名_按占比", index=False)
            recent_city_total.to_excel(writer, sheet_name="近年城市合计排名", index=False)
            sources[sources["period"].isin(RECENT_PERIODS)].to_excel(writer, sheet_name="近年来源拆解_全量", index=False)
            cases.to_excel(writer, sheet_name="建议GIS核验案例", index=False)
    except ImportError:
        (OUT_DIR / "README_Excel未生成.txt").write_text(
            "项目虚拟环境缺少 openpyxl，因此本脚本已先输出同内容 CSV；"
            "可用 bundled Python 或安装 openpyxl 后补生成 xlsx。\n",
            encoding="utf-8",
        )


def main() -> None:
    master = read_master()
    municipal, sources, period_summary = build_municipal_tables(master)
    cases = select_case_rows(municipal)
    write_outputs(municipal, sources, period_summary, cases)
    export_case_gis(cases)

    print(f"Output directory: {OUT_DIR}")
    print("Recommended GIS cases:")
    cols = ["case_reason", "city_name", "period", "municipal_in_count", "municipal_in_share_of_updates", "main_municipal_source"]
    print(cases[cols].to_string(index=False))


if __name__ == "__main__":
    main()
