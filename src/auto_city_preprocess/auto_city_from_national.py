"""
auto_city_from_national.py
==========================

Generic TCULU city extractor from national rasters.

The script follows the Suzhou-style workflow:
1. Use city lon/lat as an anchor.
2. Locate the national candidate component containing or near the anchor.
3. Crop a local window around the anchor from national 2024 TCULU.
4. Extract urban core seeds and keep the component nearest the anchor.
5. Use the 2024 main mask to crop all TCULU years.
6. Save metadata needed by the downstream pipeline config.

Example:
  python auto_city_from_national.py \\
    --city-name Suzhou --slug suzhou --lon 120.5853 --lat 31.2989

Outputs:
  city_extract_national/<slug>_candidate/
    <slug>_candidate_2024_raw.tif
    <slug>_core_main_mask_2024.tif
    cropped_years/<year>/landuse_raw.tif
    cropped_years/<year>/mask.tif
    pipeline_city_metadata.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.warp import reproject, transform as warp_transform
from rasterio.enums import Resampling
from scipy import ndimage as ndi


YEARS = [1984, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2024]
CORE_CLASSES = [5, 6, 7, 8]


def parse_args():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(
        description="Extract one city from national TCULU rasters by city anchor."
    )
    p.add_argument("--city-name", required=True, help="Display city name, e.g. Suzhou")
    p.add_argument("--slug", required=True, help="ASCII output slug, e.g. suzhou")
    p.add_argument("--lon", type=float, required=True, help="City center longitude")
    p.add_argument("--lat", type=float, required=True, help="City center latitude")
    p.add_argument("--root", type=Path, default=root, help="Dataset root directory")
    p.add_argument("--national-tif-dir", type=Path, default=None)
    p.add_argument("--candidate-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--local-half-size-m", type=float, default=45000.0)
    p.add_argument("--anchor-search-radius-cells", type=int, default=120)
    p.add_argument("--min-component-pixels", type=int, default=50)
    p.add_argument("--merge-radius-cells", type=int, default=10)
    p.add_argument("--pipeline-resolution-m", type=float, default=100.0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def normalize_crs(crs):
    if crs is None:
        return "EPSG:3857"
    text = str(crs)
    if "Pseudo-Mercator" in text or "EngineeringCRS" in text:
        return "EPSG:3857"
    return crs


def save_array_as_tif(arr, transform, crs, out_path: Path, dtype=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if dtype is None:
        dtype = arr.dtype
    with rasterio.open(
        out_path,
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


def label_components(mask: np.ndarray):
    structure = np.ones((3, 3), dtype=np.uint8)
    return ndi.label(mask, structure=structure)


def remove_small_components(labels: np.ndarray, min_pixels: int):
    counts = np.bincount(labels.ravel())
    keep_ids = np.where(counts >= min_pixels)[0]
    keep_ids = keep_ids[keep_ids > 0]
    remap = np.zeros(len(counts), dtype=np.int32)
    remap[keep_ids] = np.arange(1, len(keep_ids) + 1, dtype=np.int32)
    return remap[labels], len(keep_ids)


def disk_kernel(radius_cells: int):
    yy, xx = np.ogrid[-radius_cells: radius_cells + 1, -radius_cells: radius_cells + 1]
    return (xx * xx + yy * yy <= radius_cells * radius_cells)


def merge_local_clusters(mask: np.ndarray, radius_cells: int):
    kernel = disk_kernel(radius_cells)
    merged = ndi.binary_closing(mask.astype(bool), structure=kernel)
    merged = ndi.binary_fill_holes(merged)
    return merged.astype(np.uint8)


def read_candidate_table(path: Path):
    rows = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cid = int(row["city_id"])
            rows[cid] = {
                "city_id": cid,
                "area_pixels": int(row["area_pixels"]),
                "row_min": int(row["row_min"]),
                "col_min": int(row["col_min"]),
                "row_max": int(row["row_max"]),
                "col_max": int(row["col_max"]),
            }
    return rows


def anchor_pixel(transform, crs, lon, lat):
    x, y = warp_transform("EPSG:4326", normalize_crs(crs), [lon], [lat])
    col_f, row_f = ~transform * (x[0], y[0])
    return int(round(row_f)), int(round(col_f)), float(x[0]), float(y[0])


def find_candidate_id(candidate_dir: Path, lon: float, lat: float, radius_cells: int):
    labels_path = candidate_dir / "urban_merged_labels_2024.npy"
    mask_path = candidate_dir / "urban_merged_mask_2024.tif"
    csv_path = candidate_dir / "candidate_city_components_2024.csv"

    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing merged mask tif: {mask_path}")

    labels = np.load(labels_path)
    with rasterio.open(mask_path) as src:
        row, col, _, _ = anchor_pixel(src.transform, src.crs, lon, lat)

    row = int(np.clip(row, 0, labels.shape[0] - 1))
    col = int(np.clip(col, 0, labels.shape[1] - 1))
    cid = int(labels[row, col])

    if cid == 0:
        r0 = max(0, row - radius_cells)
        r1 = min(labels.shape[0], row + radius_cells + 1)
        c0 = max(0, col - radius_cells)
        c1 = min(labels.shape[1], col + radius_cells + 1)
        local = labels[r0:r1, c0:c1]
        ids, counts = np.unique(local[local > 0], return_counts=True)
        if len(ids) == 0:
            raise RuntimeError("No candidate component found near city anchor.")
        cid = int(ids[np.argmax(counts)])

    table = read_candidate_table(csv_path) if csv_path.exists() else {}
    return cid, table.get(cid), (row, col)


def local_anchor_bounds(crs, lon, lat, half_size_m):
    x, y = warp_transform("EPSG:4326", normalize_crs(crs), [lon], [lat])
    cx, cy = x[0], y[0]
    return cx - half_size_m, cy - half_size_m, cx + half_size_m, cy + half_size_m


def crop_local_candidate(args, paths):
    slug = args.slug
    coarse_out = paths["output_dir"] / f"{slug}_candidate_coarse_mask.tif"
    raw_out = paths["output_dir"] / f"{slug}_candidate_2024_raw.tif"
    if not args.overwrite and coarse_out.exists() and raw_out.exists():
        print("Local candidate already exists, skip crop.")
        return raw_out

    with rasterio.open(paths["coarse_mask"]) as src:
        bounds = local_anchor_bounds(src.crs, args.lon, args.lat, args.local_half_size_m)
        window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        coarse = src.read(1, window=window)
        coarse_transform = src.window_transform(window)
        coarse_crs = src.crs
        save_array_as_tif(coarse.astype(np.uint8), coarse_transform, coarse_crs, coarse_out, np.uint8)
        bounds = rasterio.windows.bounds(window, src.transform)

    with rasterio.open(paths["national_2024"]) as src:
        window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        arr = src.read(1, window=window).astype(np.uint8)
        transform = src.window_transform(window)
        save_array_as_tif(arr, transform, src.crs, raw_out, np.uint8)
        print(f"Local 2024 crop shape: {arr.shape}")
    return raw_out


def keep_component_near_anchor(mask: np.ndarray, transform, crs, lon, lat, radius_cells):
    labels, n_labels = label_components(mask > 0)
    if n_labels == 0:
        raise RuntimeError("No local urban core component found.")

    row, col, _, _ = anchor_pixel(transform, crs, lon, lat)
    row = int(np.clip(row, 0, labels.shape[0] - 1))
    col = int(np.clip(col, 0, labels.shape[1] - 1))
    lab = int(labels[row, col])

    if lab == 0:
        r0 = max(0, row - radius_cells)
        r1 = min(labels.shape[0], row + radius_cells + 1)
        c0 = max(0, col - radius_cells)
        c1 = min(labels.shape[1], col + radius_cells + 1)
        local = labels[r0:r1, c0:c1]
        ids, counts = np.unique(local[local > 0], return_counts=True)
        if len(ids) == 0:
            raise RuntimeError("No local component found near city anchor.")
        lab = int(ids[np.argmax(counts)])

    out = (labels == lab).astype(np.uint8)
    return out, lab, int(out.sum())


def extract_city_core(args, paths, raw_2024):
    slug = args.slug
    main_out = paths["output_dir"] / f"{slug}_core_main_mask_2024.tif"
    if not args.overwrite and main_out.exists():
        print("Main city mask already exists, skip extraction.")
        return main_out

    with rasterio.open(raw_2024) as src:
        arr = src.read(1)
        transform = src.transform
        crs = src.crs

    core = np.isin(arr, CORE_CLASSES)
    labels, n_labels = label_components(core)
    print(f"Local raw components: {n_labels}")
    clean_labels, n_clean = remove_small_components(labels, args.min_component_pixels)
    print(f"Local components after filtering: {n_clean}")
    clean_mask = (clean_labels > 0).astype(np.uint8)
    merged = merge_local_clusters(clean_mask, args.merge_radius_cells)
    main, main_id, main_area = keep_component_near_anchor(
        merged, transform, crs, args.lon, args.lat, args.anchor_search_radius_cells
    )
    print(f"Main component id={main_id}, pixels={main_area:,}")

    save_array_as_tif(clean_mask, transform, crs, paths["output_dir"] / f"{slug}_core_seed_mask_2024.tif", np.uint8)
    save_array_as_tif(merged, transform, crs, paths["output_dir"] / f"{slug}_core_merged_mask_2024.tif", np.uint8)
    save_array_as_tif(main, transform, crs, main_out, np.uint8)
    np.save(paths["output_dir"] / f"{slug}_core_main_labels_2024.npy", main)
    return main_out


def crop_all_years(args, paths, main_mask_tif):
    with rasterio.open(main_mask_tif) as src_mask:
        main_mask = src_mask.read(1).astype(np.uint8)
        main_transform = src_mask.transform
        main_crs = normalize_crs(src_mask.crs)
        left, bottom, right, top = src_mask.bounds

    out_root = paths["output_dir"] / "cropped_years"
    out_root.mkdir(parents=True, exist_ok=True)

    for year in YEARS:
        year_tif = paths["national_tif_dir"] / f"china_land_use{year}.tif"
        year_dir = out_root / str(year)
        landuse_out = year_dir / "landuse_raw.tif"
        mask_out = year_dir / "mask.tif"
        if not args.overwrite and landuse_out.exists() and mask_out.exists():
            print(f"  {year}: exists, skip")
            continue
        if not year_tif.exists():
            print(f"  {year}: missing {year_tif}")
            continue

        with rasterio.open(year_tif) as src:
            year_crs = normalize_crs(src.crs)
            window = from_bounds(left, bottom, right, top, transform=src.transform).round_offsets().round_lengths()
            arr = src.read(1, window=window).astype(np.uint8)
            year_transform = src.window_transform(window)

        year_mask = np.zeros(arr.shape, dtype=np.uint8)
        reproject(
            source=main_mask,
            destination=year_mask,
            src_transform=main_transform,
            src_crs=main_crs,
            dst_transform=year_transform,
            dst_crs=year_crs,
            resampling=Resampling.nearest,
        )
        year_mask = (year_mask > 0).astype(np.uint8)
        save_array_as_tif(np.where(year_mask > 0, arr, 0).astype(np.uint8), year_transform, year_crs, landuse_out, np.uint8)
        save_array_as_tif(year_mask, year_transform, year_crs, mask_out, np.uint8)
        print(f"  {year}: {arr.shape}")

    return out_root


def save_metadata(args, paths, main_mask_tif, city_id, candidate_row):
    with rasterio.open(main_mask_tif) as src:
        raw_px_m = float(abs(src.transform.a))
        width_m = float(src.width * raw_px_m)
        height_m = float(src.height * raw_px_m)
        pipeline_px_m = float(args.pipeline_resolution_m)
        meta = {
            "city_name": args.city_name,
            "slug": args.slug,
            "lon": args.lon,
            "lat": args.lat,
            "candidate_city_id": city_id,
            "candidate_row": candidate_row,
            "raw_crop_config": {
                "CITY_NAME": args.city_name,
                "H_R": int(src.height),
                "W_R": int(src.width),
                "PX_M": raw_px_m,
                "X0_M": float(src.transform.c),
                "Y0_M": float(src.transform.f),
            },
            "pipeline_config": {
                "CITY_NAME": args.city_name,
                "H_R": int(round(height_m / pipeline_px_m)),
                "W_R": int(round(width_m / pipeline_px_m)),
                "PX_M": pipeline_px_m,
                "X0_M": float(src.transform.c),
                "Y0_M": float(src.transform.f),
            },
            "bounds": {
                "left": float(src.bounds.left),
                "bottom": float(src.bounds.bottom),
                "right": float(src.bounds.right),
                "top": float(src.bounds.top),
            },
        }
    out = paths["output_dir"] / "pipeline_city_metadata.json"
    out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Metadata: {out}")


def main():
    args = parse_args()
    national_tif_dir = args.national_tif_dir or args.root / "首套中国40年城市土地利用数据"
    candidate_dir = args.candidate_dir or args.root / "city_extract_national" / "2024"
    output_dir = args.output_dir or args.root / "city_extract_national" / f"{args.slug}_candidate"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "national_tif_dir": national_tif_dir,
        "national_2024": national_tif_dir / "china_land_use2024.tif",
        "candidate_dir": candidate_dir,
        "coarse_mask": candidate_dir / "urban_merged_mask_2024.tif",
        "output_dir": output_dir,
    }

    print(f"=== Extract {args.city_name} from national TCULU ===")
    city_id, candidate_row, anchor_rc = find_candidate_id(
        candidate_dir, args.lon, args.lat, args.anchor_search_radius_cells
    )
    print(f"Anchor coarse pixel: {anchor_rc}, candidate_city_id={city_id}")
    if candidate_row:
        print(f"Candidate row: {candidate_row}")

    raw_2024 = crop_local_candidate(args, paths)
    main_mask = extract_city_core(args, paths, raw_2024)
    crop_all_years(args, paths, main_mask)
    save_metadata(args, paths, main_mask, city_id, candidate_row)
    print("=== Done ===")


if __name__ == "__main__":
    main()
