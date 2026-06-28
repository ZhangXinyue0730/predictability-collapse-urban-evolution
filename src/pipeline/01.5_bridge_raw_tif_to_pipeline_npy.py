import os
from pathlib import Path

import numpy as np
import rasterio


# =========================
# 你的自动裁剪结果目录
# =========================
CROPPED_YEARS_DIR = Path(os.environ.get("CROPPED_YEARS_DIR", "data/cropped_years")).resolve()

# 老师 pipeline 的 data 目录
PIPELINE_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

YEARS = [1984, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2024]

# TCULU 原始九类 -> 老师 pipeline 统一编码
RAW_TO_PIPE = {
    0: 0,  # 背景/无效
    1: 6,  # 水体 -> 绿地水域
    2: 6,  # 绿地 -> 绿地水域
    3: 0,  # 农田 -> 空地/未利用
    4: 0,  # 裸地 -> 空地/未利用
    5: 1,  # 居住 -> 居住
    6: 2,  # 商业 -> 商业
    7: 3,  # 公共服务 -> 公共服务
    8: 4,  # 工业 -> 工业仓储
    9: 5,  # 交通 -> 道路市政
}


def remap_array(arr: np.ndarray, mapping: dict) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.uint8)
    for src_val, dst_val in mapping.items():
        out[arr == src_val] = dst_val
    return out


def main():
    PIPELINE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("开始桥接 landuse_raw.tif -> pipeline lu_{year}.npy")

    for year in YEARS:
        tif_path = CROPPED_YEARS_DIR / str(year) / "landuse_raw.tif"
        out_lu = PIPELINE_DATA_DIR / f"lu_{year}.npy"
        out_clean = PIPELINE_DATA_DIR / f"lu_clean_{year}.npy"

        if not tif_path.exists():
            print(f"  {year}: 找不到 {tif_path}，跳过")
            continue

        with rasterio.open(tif_path) as src:
            arr = src.read(1).astype(np.uint8)

        mapped = remap_array(arr, RAW_TO_PIPE)

        np.save(out_lu, mapped)
        np.save(out_clean, mapped)

        unique, counts = np.unique(mapped, return_counts=True)
        print(f"  {year}: 已保存 -> {out_lu.name}, {out_clean.name}")
        print("    类别分布:", dict(zip(unique.tolist(), counts.tolist())))

    print("全部完成")


if __name__ == "__main__":
    main()
