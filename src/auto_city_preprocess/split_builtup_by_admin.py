"""
split_builtup_by_admin.py
=========================

Validate the teacher's "2 km buffer + administrative boundary split" workflow.

Workflow:
1. Select one city administrative boundary from an adminareas shapefile.
2. Crop national TCULU 2024 around that boundary.
3. Extract urban core land-use classes: residential/commercial/public/industrial.
4. Use a 2 km raster buffer to decide which groups should be merged.
5. Split the original built-up core cells by the city administrative boundary.

Example for Suzhou:
  python auto_city_preprocess/split_builtup_by_admin.py \\
    --city-name 苏州市 \\
    --slug suzhou \\
    --admin-shp jiangsu-260504-free/gis_osm_adminareas_a_free_1.shp

Outputs:
  city_extract_national/admin_split/<slug>/
    <slug>_seed_core_2024.tif
    <slug>_merged_2km_2024.tif
    <slug>_admin_mask_2024.tif
    <slug>_admin_split_builtup_2024.tif
    <slug>_admin_split_summary.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from scipy import ndimage as ndi


CORE_CLASSES = [5, 6, 7, 8]


def normalize_crs(crs):
    if crs is None:
        return "EPSG:3857"
    text = str(crs)
    if "Pseudo-Mercator" in text or "EngineeringCRS" in text:
        return "EPSG:3857"
    return crs


def parse_args():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Split 2km-merged TCULU built-up mask by city admin boundary.")
    p.add_argument("--root", type=Path, default=root)
    p.add_argument("--city-name", required=True, help="Admin boundary name, e.g. 苏州市")
    p.add_argument("--slug", required=True, help="ASCII output slug, e.g. suzhou")
    p.add_argument("--admin-shp", type=Path, required=True)
    p.add_argument("--national-tif", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--name-field", default="name")
    p.add_argument("--target-resolution-m", type=float, default=100.0)
    p.add_argument("--merge-distance-m", type=float, default=2000.0)
    p.add_argument("--crop-padding-m", type=float, default=5000.0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def disk_kernel(radius_cells: int):
    yy, xx = np.ogrid[-radius_cells: radius_cells + 1, -radius_cells: radius_cells + 1]
    return (xx * xx + yy * yy <= radius_cells * radius_cells)


def build_buffer_mask(mask: np.ndarray, merge_distance_m: float, resolution_m: float):
    radius = max(1, int(round(merge_distance_m / resolution_m)))
    kernel = disk_kernel(radius)
    # The buffer is only used to group nearby patches. Do not fill holes here:
    # filling holes can turn lakes/farmland into false built-up area.
    return ndi.binary_dilation(mask.astype(bool), structure=kernel).astype(np.uint8)


def label_component_stats(mask: np.ndarray, resolution_m: float):
    labels, n_labels = ndi.label(mask > 0, structure=np.ones((3, 3), dtype=np.uint8))
    if n_labels == 0:
        return 0, 0.0, 0.0
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    pixel_area_km2 = resolution_m * resolution_m / 1_000_000
    return int(n_labels), float(mask.sum() * pixel_area_km2), float(counts.max() * pixel_area_km2)


def buffered_group_stats(seed: np.ndarray, buffer_mask: np.ndarray, admin_mask: np.ndarray, resolution_m: float):
    """Count 2km-merged groups, while measuring area from original seed cells."""
    buffer_labels, _ = ndi.label(buffer_mask > 0, structure=np.ones((3, 3), dtype=np.uint8))
    core_in_admin = (seed > 0) & (admin_mask > 0)
    ids = np.unique(buffer_labels[core_in_admin])
    ids = ids[ids > 0]
    if len(ids) == 0:
        return 0, 0.0, 0.0

    pixel_area_km2 = resolution_m * resolution_m / 1_000_000
    main_area = 0.0
    for lab in ids:
        area = np.count_nonzero(core_in_admin & (buffer_labels == lab)) * pixel_area_km2
        main_area = max(main_area, area)
    return int(len(ids)), float(np.count_nonzero(core_in_admin) * pixel_area_km2), float(main_area)


def save_tif(path: Path, arr: np.ndarray, transform, crs, dtype=np.uint8):
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
    print(f"  -> {path}")


def read_admin_boundary(admin_shp: Path, city_name: str, name_field: str):
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise SystemExit("需要 geopandas/pyogrio。请在项目 .venv 中运行。") from exc

    admin = gpd.read_file(admin_shp)
    if name_field not in admin.columns:
        raise SystemExit(f"行政边界字段不存在: {name_field}")

    # Municipality extracts can contain many lower-level polygons and only a few
    # named special zones. Prefer merging the regional extract instead of
    # accidentally matching a small zone such as "上海海港综合开发区".
    if city_name in {"北京", "北京市", "天津", "天津市", "上海", "上海市", "重庆", "重庆市"}:
        admin_3857 = admin.to_crs(3857)
        if hasattr(admin_3857.geometry, "union_all"):
            union_geom = admin_3857.geometry.union_all()
        else:
            union_geom = admin_3857.geometry.unary_union
        boundary = gpd.GeoDataFrame(
            [{name_field: city_name, "area_m2": union_geom.area}],
            geometry=[union_geom],
            crs=admin_3857.crs,
        )
        print(f"Selected admin boundary by union: {city_name}, area={union_geom.area/1e6:.1f} km2")
        return boundary

    names = admin[name_field].astype(str)
    selected = admin[names == city_name]
    if selected.empty:
        selected = admin[names.str.contains(city_name, regex=False, na=False)]
    if selected.empty:
        nearby = sorted(names[names.str.contains(city_name[:2], regex=False, na=False)].unique())
        raise SystemExit(f"找不到行政边界: {city_name}. 相近名称: {nearby[:20]}")

    selected = selected.to_crs(3857)
    selected["area_m2"] = selected.geometry.area
    boundary = selected.sort_values("area_m2", ascending=False).head(1)
    print(f"Selected admin boundary: {boundary.iloc[0][name_field]}, area={boundary.iloc[0]['area_m2']/1e6:.1f} km2")
    return boundary


def crop_tculu_to_boundary(national_tif: Path, boundary_3857, resolution_m: float, padding_m: float):
    with rasterio.open(national_tif) as src:
        raster_crs = normalize_crs(src.crs)
        boundary_src = boundary_3857.to_crs(raster_crs)
        left, bottom, right, top = boundary_src.total_bounds
        bounds = (left - padding_m, bottom - padding_m, right + padding_m, top + padding_m)
        window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()

        src_res_x = abs(src.transform.a)
        src_res_y = abs(src.transform.e)
        out_width = max(1, int(round(window.width * src_res_x / resolution_m)))
        out_height = max(1, int(round(window.height * src_res_y / resolution_m)))
        arr = src.read(1, window=window, out_shape=(out_height, out_width), resampling=rasterio.enums.Resampling.nearest)
        transform = src.window_transform(window) * Affine.scale(window.width / out_width, window.height / out_height)
        crs = raster_crs

    return arr.astype(np.uint8), transform, crs


def main():
    args = parse_args()
    national_tif = args.national_tif or args.root / "首套中国40年城市土地利用数据" / "china_land_use2024.tif"
    out_dir = args.output_dir or args.root / "city_extract_national" / "admin_split" / args.slug
    out_mask = out_dir / f"{args.slug}_admin_split_builtup_2024.tif"
    if out_mask.exists() and not args.overwrite:
        print(f"Output exists, skip: {out_mask}")
        return

    print("=" * 60)
    print("Split TCULU built-up mask by administrative boundary")
    print("=" * 60)
    print(f"City: {args.city_name}")
    print(f"National TCULU: {national_tif}")
    print(f"Admin shp: {args.admin_shp}")

    boundary = read_admin_boundary(args.admin_shp, args.city_name, args.name_field)
    arr, transform, crs = crop_tculu_to_boundary(
        national_tif, boundary, args.target_resolution_m, args.crop_padding_m
    )
    print(f"Local TCULU crop: {arr.shape}, resolution={args.target_resolution_m}m")

    seed = np.isin(arr, CORE_CLASSES).astype(np.uint8)
    buffer_mask = build_buffer_mask(seed, args.merge_distance_m, args.target_resolution_m)

    boundary_src = boundary.to_crs(crs)
    admin_mask = rasterize(
        [(geom, 1) for geom in boundary_src.geometry],
        out_shape=arr.shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )
    # Formal city built-up mask: original TCULU core cells inside the admin boundary.
    # The 2km buffer only groups nearby patches; it should not create new built-up cells.
    split_mask = (seed & admin_mask).astype(np.uint8)

    seed_components, seed_area, seed_main = label_component_stats(seed & admin_mask, args.target_resolution_m)
    split_components, split_area, split_main = buffered_group_stats(
        seed, buffer_mask, admin_mask, args.target_resolution_m
    )

    save_tif(out_dir / f"{args.slug}_seed_core_2024.tif", seed, transform, crs)
    save_tif(out_dir / f"{args.slug}_merged_2km_2024.tif", buffer_mask, transform, crs)
    save_tif(out_dir / f"{args.slug}_admin_mask_2024.tif", admin_mask, transform, crs)
    save_tif(out_mask, split_mask, transform, crs)

    summary_path = out_dir / f"{args.slug}_admin_split_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "city_name", "slug", "target_resolution_m", "merge_distance_m",
            "seed_components_in_admin", "seed_area_km2_in_admin", "seed_main_component_km2_in_admin",
            "split_components", "split_area_km2", "split_main_component_km2",
        ])
        writer.writeheader()
        writer.writerow({
            "city_name": args.city_name,
            "slug": args.slug,
            "target_resolution_m": args.target_resolution_m,
            "merge_distance_m": args.merge_distance_m,
            "seed_components_in_admin": seed_components,
            "seed_area_km2_in_admin": round(seed_area, 4),
            "seed_main_component_km2_in_admin": round(seed_main, 4),
            "split_components": split_components,
            "split_area_km2": round(split_area, 4),
            "split_main_component_km2": round(split_main, 4),
        })
    print(f"  -> {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
