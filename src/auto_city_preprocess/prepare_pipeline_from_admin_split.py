"""
prepare_pipeline_from_admin_split.py
====================================

Bridge an automatically admin-split city into the teacher's TCULU pipeline.

This is the handoff step after:
  city_extract_national/admin_split/<slug>/

It creates a city-level pipeline folder, crops all national TCULU years to the
admin-mask grid, writes lu_{year}.npy / lu_clean_{year}.npy, updates config.py,
clips province OSM roads by the city admin boundary, and runs 01_extract_osm.py.

Example:
  python auto_city_preprocess/prepare_pipeline_from_admin_split.py \
    --city-name 南京市 --slug nanjing --province 江苏 --overwrite
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


YEARS = [1984, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2024]

RAW_TO_PIPE = {
    0: 0,
    1: 6,  # water -> green/water
    2: 6,  # green -> green/water
    3: 0,  # cropland -> vacant/unused
    4: 0,  # bare land -> vacant/unused
    5: 1,  # residential
    6: 2,  # commercial
    7: 3,  # public service
    8: 4,  # industrial/warehouse
    9: 5,  # road/municipal
}

OSM_BY_PROVINCE = {
    "江苏": {
        "roads": "city_extract_national/osm_extracts/jiangsu-260504-free/gis_osm_roads_free_1.shp",
        "admin": "city_extract_national/osm_extracts/jiangsu-260504-free/gis_osm_adminareas_a_free_1.shp",
        "suffix": "260504",
    },
    "北京": {
        "roads": "city_extract_national/osm_extracts/beijing-260427-free.shp/gis_osm_roads_free_1.shp",
        "admin": "city_extract_national/osm_extracts/beijing-260427-free.shp/gis_osm_adminareas_a_free_1.shp",
        "suffix": "260427",
    },
}


def parse_args():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Prepare a full pipeline folder from admin_split outputs.")
    p.add_argument("--root", type=Path, default=root)
    p.add_argument("--city-name", required=True, help="行政区名称，例如 南京市")
    p.add_argument("--city-label", default=None, help="写入 config 的城市名，例如 Nanjing")
    p.add_argument("--slug", required=True, help="ASCII slug，例如 nanjing")
    p.add_argument("--province", required=True, help="省份，例如 江苏")
    p.add_argument("--roads-shp", type=Path, default=None, help="Override province road shapefile.")
    p.add_argument("--admin-shp", type=Path, default=None, help="Override province/city admin shapefile.")
    p.add_argument("--osm-suffix", default=None, help="Suffix for clipped road output folder, e.g. latest")
    p.add_argument("--template", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def normalize_crs(crs):
    if crs is None:
        return "EPSG:3857"
    text = str(crs)
    if "Pseudo-Mercator" in text or "EngineeringCRS" in text:
        return "EPSG:3857"
    return crs


def remap_array(arr: np.ndarray) -> np.ndarray:
    out = np.zeros(arr.shape, dtype=np.uint8)
    for src_val, dst_val in RAW_TO_PIPE.items():
        out[arr == src_val] = dst_val
    return out


def copy_template(template: Path, output_dir: Path):
    def ignore(_dir, names):
        return {
            name for name in names
            if name in {"data", "figs", "__pycache__", ".DS_Store"}
        }

    if not output_dir.exists():
        shutil.copytree(template, output_dir, ignore=ignore)
    else:
        for name in ["README.md", "docs", "pipeline"]:
            src = template / name
            dst = output_dir / name
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore)
            elif src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    (output_dir / "figs").mkdir(parents=True, exist_ok=True)


def save_tif(path: Path, arr: np.ndarray, transform, crs, dtype="uint8"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(arr.astype(dtype), 1)


def crop_years_to_admin_grid(args, output_dir: Path, admin_split_dir: Path):
    mask_tif = admin_split_dir / f"{args.slug}_admin_mask_2024.tif"
    if not mask_tif.exists():
        raise FileNotFoundError(f"Missing admin mask: {mask_tif}")

    data_dir = output_dir / "data"
    cropped_dir = output_dir / "cropped_years"
    national_dir = args.root / "首套中国40年城市土地利用数据"

    with rasterio.open(mask_tif) as mask_src:
        admin_mask = (mask_src.read(1) > 0).astype(np.uint8)
        dst_transform = mask_src.transform
        dst_crs = normalize_crs(mask_src.crs)
        height, width = admin_mask.shape
        bounds = mask_src.bounds
        px_m = float(abs(mask_src.transform.a))
        x0_m = float(mask_src.transform.c)
        y0_m = float(mask_src.transform.f)

    for year in YEARS:
        src_tif = national_dir / f"china_land_use{year}.tif"
        if not src_tif.exists():
            raise FileNotFoundError(f"Missing national TCULU: {src_tif}")

        raw_aligned = np.zeros((height, width), dtype=np.uint8)
        with rasterio.open(src_tif) as src:
            window = rasterio.windows.from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
            src_arr = src.read(1, window=window).astype(np.uint8)
            src_transform = src.window_transform(window)
            reproject(
                source=src_arr,
                destination=raw_aligned,
                src_transform=src_transform,
                src_crs=normalize_crs(src.crs),
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )

        raw_clipped = np.where(admin_mask > 0, raw_aligned, 0).astype(np.uint8)
        mapped = remap_array(raw_clipped)

        year_dir = cropped_dir / str(year)
        save_tif(year_dir / "landuse_raw.tif", raw_clipped, dst_transform, dst_crs)
        save_tif(year_dir / "mask.tif", admin_mask, dst_transform, dst_crs)
        np.save(data_dir / f"lu_{year}.npy", mapped)
        np.save(data_dir / f"lu_clean_{year}.npy", mapped)

        vals, counts = np.unique(mapped, return_counts=True)
        print(f"  {year}: {height}x{width}, classes={dict(zip(vals.tolist(), counts.tolist()))}")

    return {
        "H_R": int(height),
        "W_R": int(width),
        "PX_M": px_m,
        "X0_M": x0_m,
        "Y0_M": y0_m,
        "cropped_years": str(cropped_dir),
    }


def update_config(config_path: Path, city_label: str, grid_meta: dict):
    text = config_path.read_text(encoding="utf-8")
    replacements = {
        "CITY_NAME": repr(city_label),
        "H_R": str(grid_meta["H_R"]),
        "W_R": str(grid_meta["W_R"]),
        "PX_M": f"{grid_meta['PX_M']:.6f}",
        "X0_M": f"{grid_meta['X0_M']:.6f}",
        "Y0_M": f"{grid_meta['Y0_M']:.6f}",
    }
    for key, value in replacements.items():
        text = re.sub(rf"^{key}\s*=.*$", f"{key} = {value}", text, flags=re.MULTILINE)
    config_path.write_text(text, encoding="utf-8")
    print(f"Updated config: {config_path}")


def clip_roads_and_extract_osm(args, output_dir: Path):
    osm = OSM_BY_PROVINCE.get(args.province)
    if not osm and not (args.roads_shp and args.admin_shp):
        raise SystemExit(f"No OSM config for province: {args.province}")

    roads_shp = args.roads_shp or Path(osm["roads"])
    admin_shp = args.admin_shp or Path(osm["admin"])
    if not roads_shp.is_absolute():
        roads_shp = args.root / roads_shp
    if not admin_shp.is_absolute():
        admin_shp = args.root / admin_shp
    suffix = args.osm_suffix or (osm["suffix"] if osm else "latest")
    clipped_dir = args.root / "city_extract_national" / "osm_extracts" / f"{args.slug}-{suffix}-free.shp"
    clip_script = args.root / "auto_city_preprocess" / "auto_clip_osm_roads_by_admin.py"
    extract_script = output_dir / "pipeline" / "01_extract_osm.py"

    if not roads_shp.exists():
        raise FileNotFoundError(f"Missing roads shapefile: {roads_shp}")
    if not admin_shp.exists():
        raise FileNotFoundError(f"Missing admin shapefile: {admin_shp}")

    cmd = [
        sys.executable, str(clip_script),
        "--city-name", args.city_name,
        "--roads-shp", str(roads_shp),
        "--admin-shp", str(admin_shp),
        "--output-dir", str(clipped_dir),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    subprocess.run(cmd, cwd=str(args.root), check=True)

    clipped_roads = clipped_dir / "gis_osm_roads_free_1.shp"
    subprocess.run([sys.executable, str(extract_script), str(clipped_roads)], cwd=str(output_dir), check=True)
    return clipped_roads


def main():
    args = parse_args()
    pipeline_runs = args.root / "city_extract_national" / "pipeline_runs"
    template = args.template or pipeline_runs / "苏州全流程" / "tculu_pipeline_v9_final"
    city_label = args.city_label or args.city_name.replace("市", "")
    output_dir = args.output_dir or pipeline_runs / f"{city_label}全流程" / "tculu_pipeline_v9_final"
    if not output_dir.is_absolute():
        output_dir = args.root / output_dir
    admin_split_dir = args.root / "city_extract_national" / "admin_split" / args.slug

    if not template.exists():
        raise FileNotFoundError(f"Missing template pipeline: {template}")
    if not admin_split_dir.exists():
        raise FileNotFoundError(f"Missing admin split dir: {admin_split_dir}")

    print("=" * 60)
    print("Prepare city pipeline from admin_split")
    print("=" * 60)
    print(f"City: {args.city_name} -> {city_label}")
    print(f"Output: {output_dir}")

    copy_template(template, output_dir)
    grid_meta = crop_years_to_admin_grid(args, output_dir, admin_split_dir)
    update_config(output_dir / "pipeline" / "config.py", city_label, grid_meta)
    clipped_roads = clip_roads_and_extract_osm(args, output_dir)

    metadata = {
        "city_name": args.city_name,
        "city_label": city_label,
        "slug": args.slug,
        "province": args.province,
        "source_admin_split_dir": str(admin_split_dir),
        "pipeline_dir": str(output_dir),
        "clipped_roads": str(clipped_roads),
        "grid": grid_meta,
    }
    meta_path = output_dir / "pipeline_city_metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Metadata: {meta_path}")
    print("Done. Next: run pipeline/run_all.py --from-step 3 with the clipped OSM path.")


if __name__ == "__main__":
    main()
