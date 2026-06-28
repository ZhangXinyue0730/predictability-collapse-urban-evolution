"""
batch_split_sample_cities.py
============================

Batch-run "2 km buffer grouping + administrative boundary split" for sample cities.

This script reads:
  统计年鉴/city_sample_plan_2022.csv

Then calls:
  auto_city_preprocess/split_builtup_by_admin.py

Only provinces with available adminareas shapefiles can be processed. At present,
the local workspace has Jiangsu and Beijing adminareas.

Examples:
  python auto_city_preprocess/batch_split_sample_cities.py --province 江苏
  python auto_city_preprocess/batch_split_sample_cities.py --province 江苏 --overwrite
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


ADMIN_SHP_BY_PROVINCE = {
    "江苏": "city_extract_national/osm_extracts/jiangsu-260504-free/gis_osm_adminareas_a_free_1.shp",
    "北京": "city_extract_national/osm_extracts/beijing-260427-free.shp/gis_osm_adminareas_a_free_1.shp",
}

SLUG_OVERRIDES = {
    "北京": "beijing",
    "南京市": "nanjing",
    "苏州市": "suzhou",
    "无锡市": "wuxi",
    "泰州市": "taizhou_js",
}


def parse_args():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Batch split sample cities by administrative boundaries.")
    p.add_argument("--root", type=Path, default=root)
    p.add_argument("--sample-csv", type=Path, default=root / "统计年鉴" / "city_sample_plan_2022.csv")
    p.add_argument("--province", default=None, help="Only process one province, e.g. 江苏")
    p.add_argument("--city-name", default=None, help="Only process one city, e.g. 苏州市")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def make_slug(city_name: str):
    if city_name in SLUG_OVERRIDES:
        return SLUG_OVERRIDES[city_name]
    return city_name.replace("市", "").replace(" ", "_")


def read_samples(path: Path, province: str | None, city_name: str | None):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if province and row["province"] != province:
                continue
            if city_name and row["city_name"] != city_name:
                continue
            rows.append(row)
    return rows


def read_summary(path: Path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def read_existing_index(path: Path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return {row.get("slug") or row.get("city_name"): row for row in rows}


def main():
    args = parse_args()
    split_script = args.root / "auto_city_preprocess" / "split_builtup_by_admin.py"
    samples = read_samples(args.sample_csv, args.province, args.city_name)
    if args.limit:
        samples = samples[:args.limit]

    print("=" * 60)
    print("Batch split sample cities by admin boundary")
    print("=" * 60)
    print(f"Sample cities to check: {len(samples)}")

    aggregate = []
    for row in samples:
        city = row["city_name"]
        province = row["province"]
        admin_rel = ADMIN_SHP_BY_PROVINCE.get(province)
        if not admin_rel:
            print(f"[SKIP] {city}: no adminareas shapefile configured for province {province}")
            continue

        admin_shp = args.root / admin_rel
        if not admin_shp.exists():
            print(f"[SKIP] {city}: missing adminareas shapefile: {admin_shp}")
            continue

        slug = make_slug(city)
        cmd = [
            sys.executable,
            str(split_script),
            "--root", str(args.root),
            "--city-name", city,
            "--slug", slug,
            "--admin-shp", str(admin_shp),
        ]
        if args.overwrite:
            cmd.append("--overwrite")

        print(f"\n[RUN] {city} ({province}) -> {slug}")
        proc = subprocess.run(cmd, cwd=str(args.root), text=True, capture_output=True)
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr)
            aggregate.append({
                **row,
                "slug": slug,
                "status": "failed",
                "error": proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "unknown",
            })
            continue

        summary_path = args.root / "city_extract_national" / "admin_split" / slug / f"{slug}_admin_split_summary.csv"
        summary = read_summary(summary_path)
        aggregate.append({
            **row,
            "slug": slug,
            "status": "ok",
            "summary_path": str(summary_path),
            "split_area_km2": summary.get("split_area_km2", ""),
            "split_components": summary.get("split_components", ""),
            "split_main_component_km2": summary.get("split_main_component_km2", ""),
            "seed_components_in_admin": summary.get("seed_components_in_admin", ""),
        })

    out = args.root / "city_extract_national" / "admin_split" / "sample_admin_split_index.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "city_name", "province", "region", "city_size_class", "sample_group",
        "urban_population_10k", "selection_rule", "slug", "status",
        "split_area_km2", "split_components", "split_main_component_km2",
        "seed_components_in_admin", "summary_path", "error",
    ]
    merged = read_existing_index(out)
    for row in aggregate:
        merged[row.get("slug") or row.get("city_name")] = row
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged.values())
    print(f"\nAggregate index: {out}")
    print("Done.")


if __name__ == "__main__":
    main()
