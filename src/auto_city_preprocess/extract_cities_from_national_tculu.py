import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from affine import Affine
from scipy import ndimage as ndi


# =========================
# 输入输出路径
# =========================
INPUT_TIF = Path(os.environ.get("TCULU_2024_TIF", "data/china_land_use2024.tif")).resolve()

OUTPUT_DIR = Path(os.environ.get("CITY_EXTRACT_2024_DIR", "outputs/city_extract_national/2024")).resolve()

# 原始 TCULU 九类中，作为“城市核心种子”的类别
# 5=居住, 6=商业, 7=公共服务, 8=工业
URBAN_SEED_CLASSES = [5, 6, 7, 8]

# 全国图先降到 100m 再识别城市
TARGET_RESOLUTION_M = 100.0

# 太小的团块先过滤掉
MIN_COMPONENT_PIXELS = 100
MERGE_DISTANCE_M = 2000.0


def load_raster_downsampled(path: Path, target_resolution_m: float = 100.0):
    """
    读取全国栅格时直接降采样，避免全国 10m 图把内存打爆。
    用 nearest 保持分类值不被平均污染。
    """
    with rasterio.open(path) as src:
        src_res_x = abs(src.transform.a)
        src_res_y = abs(src.transform.e)

        scale_x = target_resolution_m / src_res_x
        scale_y = target_resolution_m / src_res_y

        out_height = max(1, int(src.height / scale_y))
        out_width = max(1, int(src.width / scale_x))

        arr = src.read(
            1,
            out_shape=(out_height, out_width),
            resampling=Resampling.nearest,
        )

        transform = src.transform * Affine.scale(
            src.width / out_width,
            src.height / out_height,
        )
        crs = src.crs

    return arr, transform, crs


def build_seed_mask(landuse: np.ndarray) -> np.ndarray:
    """提取城市核心种子：居住/商业/公共服务/工业"""
    return np.isin(landuse, URBAN_SEED_CLASSES)


def label_components(mask: np.ndarray):
    """8 邻域连通域标记"""
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, n = ndi.label(mask, structure=structure)
    return labels, n


def remove_small_components(labels: np.ndarray, n_labels: int, min_pixels: int):
    """去掉太小的噪声团块（向量化快速版）"""
    counts = np.bincount(labels.ravel())

    # 0 是背景，不算团块
    keep_ids = np.where(counts >= min_pixels)[0]
    keep_ids = keep_ids[keep_ids > 0]

    out = np.zeros_like(labels, dtype=np.int32)
    if len(keep_ids) == 0:
        return out, 0

    # 把保留的旧 label 重新映射成连续的新 label
    remap = np.zeros(len(counts), dtype=np.int32)
    remap[keep_ids] = np.arange(1, len(keep_ids) + 1, dtype=np.int32)

    out = remap[labels]
    return out, len(keep_ids)

def extract_component_bboxes(labels: np.ndarray, n_labels: int):
    """
    提取每个团块的面积和包围框（快速版）
    """
    from scipy import ndimage as ndi

    objects = ndi.find_objects(labels)
    counts = np.bincount(labels.ravel())

    components = []

    for lab, slc in enumerate(objects, start=1):
        if slc is None:
            continue

        row_slice, col_slice = slc
        comp = {
            "city_id": int(lab),
            "area_pixels": int(counts[lab]),
            "row_min": int(row_slice.start),
            "col_min": int(col_slice.start),
            "row_max": int(row_slice.stop - 1),
            "col_max": int(col_slice.stop - 1),
        }
        components.append(comp)

    components.sort(key=lambda x: x["area_pixels"], reverse=True)
    return components


def save_component_table(components, out_path: Path):
    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["city_id", "area_pixels", "row_min", "col_min", "row_max", "col_max"],
        )
        writer.writeheader()
        writer.writerows(components)

def disk_kernel(radius_cells: int):
    """生成圆形结构元素"""
    yy, xx = np.ogrid[-radius_cells: radius_cells + 1, -radius_cells: radius_cells + 1]
    return (xx * xx + yy * yy <= radius_cells * radius_cells)


def merge_nearby_clusters(mask: np.ndarray, merge_distance_m: float, resolution_m: float):
    """
    轻量版相邻团块合并。
    先做膨胀让近邻团块连上，再做一次轻量腐蚀恢复边界。
    比大半径 binary_closing 更稳一些。
    """
    radius_cells = max(1, int(round(merge_distance_m / resolution_m)))

    # 先别真用满 2km，先保守一点，不然后面在全国图上太重
    radius_cells = min(radius_cells, 8)   # 8 cells ≈ 800m（100m 栅格）

    kernel = disk_kernel(radius_cells)

    expanded = ndi.binary_dilation(mask.astype(bool), structure=kernel)
    merged = ndi.binary_erosion(expanded, structure=kernel)
    merged = ndi.binary_fill_holes(merged)

    return merged.astype(np.uint8)


def save_uint8_tif(arr: np.ndarray, transform, crs, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(arr.astype(np.uint8), 1)


def main():
    print("读取全国 2024 土地利用图（降采样到 100m）...")
    landuse, transform, crs = load_raster_downsampled(
        INPUT_TIF, target_resolution_m=TARGET_RESOLUTION_M
    )

    print("降采样后栅格大小:", landuse.shape)
    print("唯一类别值:", np.unique(landuse))

    print("提取城市核心种子类别...")
    seed_mask = build_seed_mask(landuse)

    print("连通域标记...")
    labels, n_labels = label_components(seed_mask)
    print(f"原始团块数: {n_labels}")

    print("去掉太小团块...")
    clean_labels, n_clean = remove_small_components(
        labels, n_labels, MIN_COMPONENT_PIXELS
    )
    print(f"过滤后团块数: {n_clean}")

    clean_mask = (clean_labels > 0).astype(np.uint8)

    print("合并 2km 内相邻团块...")
    merged_mask = merge_nearby_clusters(
        clean_mask,
        merge_distance_m=MERGE_DISTANCE_M,
        resolution_m=TARGET_RESOLUTION_M,
    )

    print("对合并后的团块重新标记...")
    merged_labels, merged_n = label_components(merged_mask > 0)
    print(f"合并后城市团块数: {merged_n}")

    print("提取候选城市团块包围框...")
    components = extract_component_bboxes(merged_labels, merged_n)

    print("面积最大的前 20 个候选城市团块：")
    for comp in components[:20]:
        print(
            f"city_id={comp['city_id']}, "
            f"area_pixels={comp['area_pixels']}, "
            f"bbox=({comp['row_min']},{comp['col_min']})-({comp['row_max']},{comp['col_max']})"
        )

    save_component_table(
        components,
        OUTPUT_DIR / "candidate_city_components_2024.csv"
    )

    save_uint8_tif(
        clean_mask,
        transform,
        crs,
        OUTPUT_DIR / "urban_seed_mask_2024.tif",
    )

    save_uint8_tif(
        merged_mask,
        transform,
        crs,
        OUTPUT_DIR / "urban_merged_mask_2024.tif",
    )

    np.save(OUTPUT_DIR / "urban_seed_labels_2024.npy", clean_labels)
    np.save(OUTPUT_DIR / "urban_merged_labels_2024.npy", merged_labels)

    print("完成")
    print(f"种子 mask: {OUTPUT_DIR / 'urban_seed_mask_2024.tif'}")
    print(f"种子 labels: {OUTPUT_DIR / 'urban_seed_labels_2024.npy'}")
    print(f"候选城市表: {OUTPUT_DIR / 'candidate_city_components_2024.csv'}")


if __name__ == "__main__":
    main()
