"""
08_functional_features.py — "功能整体" 形态学特征 (9 年统一版)
============================================================

用户洞察: "方格彼此组合起来, 构成一个有机功能整体"

每个 cell 在多个半径 R (500m, 1000m, 2000m) 内, 提取:
  1) 同类簇规模: 与 c 同类(8-邻接连通)的最大 cluster 像素数 (log)
  2) 局部熵: 多半径 Shannon entropy
  3) 主导纯度 (max - 2nd_max), 大 = 该范围用地纯净
  4) 建成区比例
  5) 边界比例 (1 - 主导比例): 高 = 异质
  6) 邻接异类多样度: 与该 cell 相邻 (8-邻接) 的不同类别数

这些 14 维特征代表"功能整体"的概念:
  - 大簇 / 紧凑 / 边界清晰 = 完整工业园区
  - 小簇 / 高混合度 / 多边界 = 居住-商业混合
  - 多种相邻 = CBD 综合体

输出 (per year):
  data/cell_functional_{year}.npz
    feat: (N_cell, 14) float32
    col_names: (14,) str

注: 使用未经道路连通清理的 lu_{year}.npy。道路连通性不应删除真实
城市组团；255 (AOI 外) 仅作为非建设用地 0 处理。
"""
import numpy as np
import time
from pathlib import Path
from scipy.ndimage import label as cc_label, uniform_filter
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import DATA, YEARS, H_R, W_R, BUILTUP_CLASSES, FUNCTIONAL_RADII


def load_lu(year):
    a = np.load(DATA / f'lu_{year}.npy')
    lu = a.copy()
    lu[lu == 255] = 0
    return lu.astype(np.int8)


def compute_year(year):
    lu = load_lu(year)
    H, W = lu.shape
    
    # 1) 同类簇规模 (per class, then assign to each cell)
    cluster_size = np.zeros((H, W), dtype=np.float32)
    for c in range(7):
        mask = (lu == c)
        if not mask.any(): continue
        lbl, n_cc = cc_label(mask, structure=np.ones((3, 3)))
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0
        my_size = sizes[lbl]
        cluster_size[mask] = my_size[mask]
    
    # 2) 多半径形态特征
    onehot = np.zeros((7, H, W), dtype=np.float32)
    for c in range(7):
        onehot[c] = (lu == c)
    
    feats, feat_names = [], []
    for R_m in FUNCTIONAL_RADII:
        size = int(round(R_m / 100.0))
        win = 2 * size + 1
        ratios = np.stack([uniform_filter(onehot[c], size=win, mode='constant')
                           for c in range(7)], axis=0)
        eps = 1e-9
        ratios_norm = ratios / (ratios.sum(axis=0, keepdims=True) + eps)
        H_ent = -(ratios_norm * np.log(ratios_norm + eps)).sum(axis=0)
        feats.append(H_ent.flatten()); feat_names.append(f'R{R_m}_entropy')
        
        sorted_ratios = np.sort(ratios, axis=0)
        purity = sorted_ratios[-1] - sorted_ratios[-2]
        feats.append(purity.flatten()); feat_names.append(f'R{R_m}_purity')
        
        builtup_ratio = ratios[1] + ratios[2] + ratios[3] + ratios[4]
        feats.append(builtup_ratio.flatten()); feat_names.append(f'R{R_m}_builtup')
        
        boundary = 1 - sorted_ratios[-1]
        feats.append(boundary.flatten()); feat_names.append(f'R{R_m}_boundary')
    
    # 3) cluster size (log)
    feats.append(np.log1p(cluster_size).flatten())
    feat_names.append('cluster_size_log')
    
    # 4) 邻接异类数
    diversity_3x3 = np.zeros((H, W), dtype=np.int8)
    pad = np.pad(lu, 1, mode='edge')
    for di in range(-1, 2):
        for dj in range(-1, 2):
            if di == 0 and dj == 0: continue
            shifted = pad[1+di:1+di+H, 1+dj:1+dj+W]
            diversity_3x3 += (shifted != lu).astype(np.int8)
    feats.append(diversity_3x3.flatten().astype(np.float32))
    feat_names.append('neighbor_diversity_8')
    
    feat_matrix = np.stack(feats, axis=1).astype(np.float32)
    return feat_matrix, feat_names


def main():
    print("=" * 60)
    print(f"步骤 08: '功能整体' 形态学特征 ({len(YEARS)} 年)")
    print("=" * 60)
    for year in YEARS:
        out = DATA / f'cell_functional_{year}.npz'
        if out.exists():
            print(f"  {year}: 已存在, 跳过")
            continue
        t0 = time.time()
        feat, cols = compute_year(year)
        np.savez_compressed(out, feat=feat, col_names=np.array(cols))
        print(f"  {year}: shape={feat.shape}  [{time.time()-t0:.1f}s]")
    print("\n✅ 完成")


if __name__ == '__main__':
    main()
