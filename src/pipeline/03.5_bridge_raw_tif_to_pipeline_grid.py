import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject


# =========================
# 你的自动裁剪结果目录
# =========================
CROPPED_YEARS_DIR = Path(os.environ.get("CROPPED_YEARS_DIR", "data/cropped_years")).resolve()

# 老师 pipeline 的 data 目录
PIPELINE_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

YEARS = [1984, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2024]

# 老师 config.py 里的标准北京网格
H_R = 612
W_R = 668
PX_M = 99.65
X0_M = 12924263.80111564
Y0_M = 4889246.50836138
DST_CRS = "EPSG:3857"

# TCULU 原始九类 -> 老师 pipeline 统一编码
RAW_TO_PIPE = {
    0: 0,
    1: 6,  # 水体
    2: 6,  # 绿地
    3: 0,  # 农田
    4: 0,  # 裸地
    5: 1,  # 居住
    6: 2,  # 商业
    7: 3,  # 公共服务
    8: 4,  # 工业
    9: 5,  # 交通
}


def normalize_crs(crs):
    if crs is None:
        return "EPSG:3857"
    crs_text = str(crs)
    if ("Pseudo-Mercator" in crs_text) or ("EngineeringCRS" in crs_text):
        return "EPSG:3857"
    return crs


def remap_array(arr: np.ndarray, mapping: dict) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.uint8)
    for src_val, dst_val in mapping.items():
        out[arr == src_val] = dst_val
    return out


def main():
    PIPELINE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 老师 config.py 对应的目标 transform
    dst_transform = from_origin(
        X0_M,              # 左上角 x
        Y0_M,              # 左上角 y
        PX_M,              # 像元宽
        PX_M,              # 像元高
    )

    print("开始桥接到老师 pipeline 标准网格...")
    print(f"目标网格: {H_R} x {W_R}, PX_M={PX_M}")

    for year in YEARS:
        tif_path = CROPPED_YEARS_DIR / str(year) / "landuse_raw.tif"
        out_lu = PIPELINE_DATA_DIR / f"lu_{year}.npy"
        out_clean = PIPELINE_DATA_DIR / f"lu_clean_{year}.npy"

        if not tif_path.exists():
            print(f"  {year}: 找不到 {tif_path}，跳过")
            continue

        print(f"  处理 {year}...")

        with rasterio.open(tif_path) as src:
            src_arr = src.read(1).astype(np.uint8)
            src_transform = src.transform
            src_crs = normalize_crs(src.crs)

        # 先做类别映射，再对齐到老师网格
        mapped_src = remap_array(src_arr, RAW_TO_PIPE)

        dst_arr = np.zeros((H_R, W_R), dtype=np.uint8)

        reproject(
            source=mapped_src,
            destination=dst_arr,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=DST_CRS,
            resampling=Resampling.nearest,
        )

        np.save(out_lu, dst_arr)
        np.save(out_clean, dst_arr)

        unique, counts = np.unique(dst_arr, return_counts=True)
        print(f"    已保存 -> {out_lu.name}, {out_clean.name}")
        print("    类别分布:", dict(zip(unique.tolist(), counts.tolist())))

    print("全部完成")


if __name__ == "__main__":
    main()
