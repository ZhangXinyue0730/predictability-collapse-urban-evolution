"""
06_neighborhood_features.py — 多半径周边用地特征 (9 年统一版)
=============================================================

每个 cell 在不同半径下的"周边用地特征":
  - 各类比例 (7 维)
  - 建成区比例
  - 道路比例
  - Shannon 熵 (混合度)
  - 主导类别 (argmax)

半径: 1000, 1500, 2000, 3000, 4000, 5000 米 (栅格步数 ≈ R / 100)
实现: scipy.ndimage.uniform_filter (盒型滑动平均).

输出 (per year):
  data/cell_features_{year}.npz
    'feat'      : (N_cell, 66) float16   6 半径 × (7 ratios + 4) = 66
    'col_names' : (66,) str

注: 使用未经道路连通清理的 lu_{year}.npy。道路连通性不应删除真实
城市组团；255 (AOI 外) 仅作为非建设用地 0 处理。
"""
import numpy as np
import time
from pathlib import Path
from scipy.ndimage import uniform_filter
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import DATA, YEARS, H_R, W_R, BUILTUP_CLASSES, NB_RADII_M


PX_M_GRID = 100.0


def load_lu(year):
    """读标准化原始用地，把 255 (AOI 外) 设成 0."""
    a = np.load(DATA / f'lu_{year}.npy')
    lu = a.copy()
    lu[lu == 255] = 0
    return lu.astype(np.int8)


def compute_features(year):
    lu = load_lu(year)
    H, W = lu.shape
    n_classes = 7
    onehot = np.zeros((n_classes, H, W), dtype=np.float32)
    for c in range(n_classes):
        onehot[c] = (lu == c).astype(np.float32)
    builtup = np.zeros((H, W), dtype=np.float32)
    for c in BUILTUP_CLASSES:
        builtup += (lu == c).astype(np.float32)
    
    cols_list, col_names = [], []
    for R_m in NB_RADII_M:
        size = int(round(R_m / PX_M_GRID))
        win = 2 * size + 1
        ratios = np.stack([uniform_filter(onehot[c], size=win, mode='constant', cval=0)
                           for c in range(n_classes)], axis=0)
        builtup_ratio = uniform_filter(builtup, size=win, mode='constant', cval=0)
        road_ratio = uniform_filter(onehot[5], size=win, mode='constant', cval=0)
        eps = 1e-9
        sum_r = ratios.sum(axis=0) + eps
        p = ratios / sum_r[None, :, :]
        H_shannon = -(p * np.log(p + eps)).sum(axis=0)
        dominant = ratios.argmax(axis=0).astype(np.int8)
        
        cols_list.append(ratios.reshape(7, -1).T.astype(np.float32))
        for c in range(7):
            col_names.append(f'R{R_m}_ratio_c{c}')
        cols_list.append(builtup_ratio.flatten().astype(np.float32)[:, None])
        col_names.append(f'R{R_m}_builtup')
        cols_list.append(road_ratio.flatten().astype(np.float32)[:, None])
        col_names.append(f'R{R_m}_road')
        cols_list.append(H_shannon.flatten().astype(np.float32)[:, None])
        col_names.append(f'R{R_m}_shannon')
        cols_list.append(dominant.flatten().astype(np.float32)[:, None])
        col_names.append(f'R{R_m}_dominant')
    feat_matrix = np.concatenate(cols_list, axis=1).astype(np.float16)
    return feat_matrix, col_names


def main():
    print("=" * 60)
    print(f"步骤 06: 多半径周边用地特征 ({len(YEARS)} 年)")
    print("=" * 60)
    for year in YEARS:
        out = DATA / f'cell_features_{year}.npz'
        if out.exists():
            print(f"  {year}: 已存在, 跳过")
            continue
        t0 = time.time()
        feat, cols = compute_features(year)
        np.savez_compressed(out, feat=feat, col_names=np.array(cols))
        print(f"  {year}: shape={feat.shape}  [{time.time()-t0:.1f}s]")
    print("\n✅ 完成")


if __name__ == '__main__':
    main()
