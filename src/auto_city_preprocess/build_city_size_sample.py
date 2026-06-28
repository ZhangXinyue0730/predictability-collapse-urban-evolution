"""
build_city_size_sample.py
=========================

Build city size classes and a sample-city plan from the 2022 China Urban
Construction Statistical Yearbook CSV.

Input CSV should be exported from Sheet2:
  2-2 2022年全国城市人口和建设用地（按城市分列）

Classification basis:
  Urban resident population ~= 城区人口 + 城区暂住人口, unit: 10,000 persons.

Outputs:
  统计年鉴/city_size_classification_2022.csv
  统计年鉴/city_size_summary_2022.csv
  统计年鉴/city_sample_plan_2022.csv
  统计年鉴/city_sample_summary_2022.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


MUNICIPALITIES = {"北京", "天津", "上海", "重庆"}

PROVINCES = [
    "北京", "天津", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江",
    "上海", "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南",
    "湖北", "湖南", "广东", "广西", "海南", "重庆", "四川", "贵州",
    "云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆",
]

REGION_BY_PROVINCE = {
    # 国家统计局四大区域
    "北京": "东部", "天津": "东部", "河北": "东部", "上海": "东部",
    "江苏": "东部", "浙江": "东部", "福建": "东部", "山东": "东部",
    "广东": "东部", "海南": "东部",
    "山西": "中部", "安徽": "中部", "江西": "中部",
    "河南": "中部", "湖北": "中部", "湖南": "中部",
    "内蒙古": "西部", "广西": "西部", "重庆": "西部", "四川": "西部",
    "贵州": "西部", "云南": "西部", "西藏": "西部", "陕西": "西部",
    "甘肃": "西部", "青海": "西部", "宁夏": "西部", "新疆": "西部",
    "辽宁": "东北", "吉林": "东北", "黑龙江": "东北",
}


def parse_args():
    root = Path(__file__).resolve().parent.parent
    default_csv = root / "统计年鉴" / "2022年城市建设统计年鉴.csv"
    p = argparse.ArgumentParser(description="Build 2022 city size classification and sample plan.")
    p.add_argument("--input", type=Path, default=default_csv)
    p.add_argument("--output-dir", type=Path, default=root / "统计年鉴")
    p.add_argument("--sample-per-region", type=int, default=5)
    return p.parse_args()


def clean_name(value: str) -> str:
    return value.replace("\n", "").replace("\r", "").strip()


def to_float(value: str) -> float:
    value = str(value).replace(",", "").strip()
    if not value:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def classify_city_size(pop_10k: float) -> str:
    if pop_10k >= 1000:
        return "超大城市"
    if pop_10k >= 500:
        return "特大城市"
    if pop_10k >= 300:
        return "I型大城市"
    if pop_10k >= 100:
        return "II型大城市"
    if pop_10k >= 50:
        return "中等城市"
    if pop_10k >= 20:
        return "I型小城市"
    return "II型小城市"


def sample_group(city_size_class: str) -> str:
    if city_size_class in {"超大城市", "特大城市", "I型大城市", "II型大城市"}:
        return city_size_class
    return "中小城市"


def is_city_row(name: str) -> bool:
    if name in MUNICIPALITIES:
        return True
    return name.endswith("市")


def read_yearbook_csv(path: Path):
    rows = []
    current_province = None

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for raw in reader:
            if not raw:
                continue
            name = clean_name(raw[0]) if len(raw) > 0 else ""
            if not name:
                continue

            if name in {"全国", "Name ofCities", "Name of Cities"}:
                continue
            if name in PROVINCES and name not in MUNICIPALITIES:
                current_province = name
                continue
            if name in MUNICIPALITIES:
                current_province = name
            if current_province in MUNICIPALITIES and name == f"{current_province}市":
                continue
            if not is_city_row(name):
                continue

            # Sheet2 columns:
            # 0 城市名称, 2 市区人口, 3 市区暂住人口, 5 城区人口, 6 城区暂住人口
            urban_district_pop = to_float(raw[2]) if len(raw) > 2 else 0.0
            urban_district_temp = to_float(raw[3]) if len(raw) > 3 else 0.0
            urban_area_pop = to_float(raw[5]) if len(raw) > 5 else 0.0
            urban_area_temp = to_float(raw[6]) if len(raw) > 6 else 0.0
            urban_resident_pop = urban_area_pop + urban_area_temp

            if urban_resident_pop <= 0:
                continue

            province = name if name in MUNICIPALITIES else current_province
            region = REGION_BY_PROVINCE.get(province, "")
            city_class = classify_city_size(urban_resident_pop)

            rows.append({
                "city_name": name,
                "province": province,
                "region": region,
                "urban_population_10k": round(urban_resident_pop, 4),
                "urban_area_population_10k": round(urban_area_pop, 4),
                "urban_area_temp_population_10k": round(urban_area_temp, 4),
                "urban_district_population_10k": round(urban_district_pop, 4),
                "urban_district_temp_population_10k": round(urban_district_temp, 4),
                "city_size_class": city_class,
                "sample_group": sample_group(city_class),
            })

    rows.sort(key=lambda r: r["urban_population_10k"], reverse=True)
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {path}")


def build_size_summary(rows: list[dict]):
    order = ["超大城市", "特大城市", "I型大城市", "II型大城市", "中等城市", "I型小城市", "II型小城市"]
    out = []
    total = len(rows)
    for cls in order:
        subset = [r for r in rows if r["city_size_class"] == cls]
        out.append({
            "city_size_class": cls,
            "count": len(subset),
            "share": round(len(subset) / total, 6) if total else 0,
            "mean_urban_population_10k": round(
                sum(r["urban_population_10k"] for r in subset) / len(subset), 4
            ) if subset else 0,
            "max_urban_population_10k": max([r["urban_population_10k"] for r in subset], default=0),
            "min_urban_population_10k": min([r["urban_population_10k"] for r in subset], default=0),
        })
    return out


def build_sample_plan(rows: list[dict], sample_per_region: int):
    selected = []

    for r in rows:
        if r["city_size_class"] in {"超大城市", "特大城市", "I型大城市"}:
            rr = dict(r)
            rr["selection_rule"] = "全选"
            selected.append(rr)

    for group in ["II型大城市", "中小城市"]:
        for region in ["东部", "中部", "西部", "东北"]:
            subset = [
                r for r in rows
                if r["sample_group"] == group and r["region"] == region
            ]
            subset.sort(key=lambda r: r["urban_population_10k"], reverse=True)
            for r in subset[:sample_per_region]:
                rr = dict(r)
                rr["selection_rule"] = f"{group}-{region}前{sample_per_region}个"
                selected.append(rr)

    # Remove duplicates while preserving first selection rule.
    seen = set()
    unique = []
    for r in selected:
        key = (r["city_name"], r["province"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    unique.sort(key=lambda r: (
        ["超大城市", "特大城市", "I型大城市", "II型大城市", "中小城市"].index(r["sample_group"]),
        ["东部", "中部", "西部", "东北", ""].index(r["region"] if r["region"] in {"东部", "中部", "西部", "东北"} else ""),
        -r["urban_population_10k"],
    ))
    return unique


def build_sample_summary(sample_rows: list[dict]):
    d = defaultdict(int)
    for r in sample_rows:
        d[(r["sample_group"], r["region"])] += 1
    out = []
    for group in ["超大城市", "特大城市", "I型大城市", "II型大城市", "中小城市"]:
        for region in ["东部", "中部", "西部", "东北"]:
            out.append({
                "sample_group": group,
                "region": region,
                "count": d[(group, region)],
            })
    return out


def main():
    args = parse_args()
    rows = read_yearbook_csv(args.input)
    sample_rows = build_sample_plan(rows, args.sample_per_region)

    city_fields = [
        "city_name", "province", "region",
        "urban_population_10k", "urban_area_population_10k", "urban_area_temp_population_10k",
        "urban_district_population_10k", "urban_district_temp_population_10k",
        "city_size_class", "sample_group",
    ]
    sample_fields = city_fields + ["selection_rule"]

    print("=" * 60)
    print("Build city size classification and sample plan")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Cities kept: {len(rows):,}")

    summary = build_size_summary(rows)
    for item in summary:
        print(f"  {item['city_size_class']:<8} {item['count']:>4} cities")

    print(f"Sample cities selected: {len(sample_rows):,}")

    write_csv(args.output_dir / "city_size_classification_2022.csv", rows, city_fields)
    write_csv(args.output_dir / "city_size_summary_2022.csv", summary, [
        "city_size_class", "count", "share",
        "mean_urban_population_10k", "max_urban_population_10k", "min_urban_population_10k",
    ])
    write_csv(args.output_dir / "city_sample_plan_2022.csv", sample_rows, sample_fields)
    write_csv(args.output_dir / "city_sample_summary_2022.csv", build_sample_summary(sample_rows), [
        "sample_group", "region", "count",
    ])
    print("Done.")


if __name__ == "__main__":
    main()
