"""
07_syntax_kde_to_cell.py — 句法 KDE 扩散赋值到 100m 格
======================================================

把 segment-level 的句法值, 通过 KDE (Kernel Density Estimation)
平滑扩散到所有 100m × 100m 的栅格 cell.

重要: 这是关键改进 — 取代了之前的 "硬切 ±100m 内取均值" 旧法.
  - 旧法: 离道路 100m 内的 cell 拿到值, 100m 外 = 0 (硬切, 50% 覆盖率)
  - 新法: 高斯模糊扩散到 1500m, 平滑过渡 (76-91% 覆盖率)

算法:
  1) 把每个 segment 的句法值"撒"到沿路像素上 (取 max 重叠)
  2) 用 gaussian_filter (σ ≈ 500m) 加权扩散到整个空间
  3) 用 maximum_filter (size ≈ 5px) 也算一份"局部 max"

输出: data/cell_syntax_decay_{year}.npz
  feat: (N_cell, 80) — 40 metrics × {mean (gaussian), max (filter)}
"""
import numpy as np
import pickle
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import DATA, YEARS, H_R, W_R, KDE_SIGMA_PX
from scipy.ndimage import gaussian_filter, maximum_filter
import time

SIGMA_PX = KDE_SIGMA_PX


def diffuse_year(year):
    t0 = time.time()
    
    z = np.load(DATA / f'road_{year}.npz', allow_pickle=True)
    polys = list(z['polys'])
    n_segs = len(polys)
    
    syn = np.load(DATA / f'syntax_results_{year}.npz')
    keys = sorted(syn.files)
    n_metrics = len(keys)
    M = np.stack([syn[k] for k in keys], axis=1).astype(np.float32)
    M_log = np.log1p(np.maximum(M, 0))   # (n_segs, 40)
    
    # 1) 一次性收集所有道路像素及其对应 segment idx
    # 对每条 segment 沿 polyline 找像素
    px_rs = []; px_cs = []; px_seg = []
    for si, p in enumerate(polys):
        if len(p) < 2: continue
        for j in range(len(p) - 1):
            xa, ya = p[j]; xb, yb = p[j+1]
            n_pts = max(int(np.hypot(xb-xa, yb-ya)) * 2, 3)
            ts = np.linspace(0, 1, n_pts)
            xs = (xa + ts*(xb-xa)).round().astype(np.int32)
            ys = (ya + ts*(yb-ya)).round().astype(np.int32)
            valid = (xs >= 0) & (xs < W_R) & (ys >= 0) & (ys < H_R)
            xs = xs[valid]; ys = ys[valid]
            px_rs.append(ys)
            px_cs.append(xs)
            px_seg.append(np.full(len(xs), si, dtype=np.int32))
    px_rs = np.concatenate(px_rs)
    px_cs = np.concatenate(px_cs)
    px_seg = np.concatenate(px_seg)
    print(f"  {year}: {len(px_rs):,} 道路像素", flush=True)
    
    # 2) 对每像素取该 segment 的句法值, 多 segment 重叠时取 max
    # 用 sort + reduce 加速: 把 (row, col) 编码成 int 排序
    pix_id = px_rs * W_R + px_cs   # (M,)
    
    # 对每个 metric 算 max
    feat_mean = np.zeros((H_R, W_R, n_metrics), dtype=np.float32)
    feat_max_diff = np.zeros((H_R, W_R, n_metrics), dtype=np.float32)
    
    presence = np.zeros((H_R, W_R), dtype=np.float32)
    presence_flat = presence.ravel()
    np.add.at(presence_flat, pix_id, 1.0)
    presence = (presence > 0).astype(np.float32)
    weight = gaussian_filter(presence, sigma=SIGMA_PX)
    
    for k_idx in range(n_metrics):
        v_arr = M_log[:, k_idx]   # (n_segs,)
        # 每像素的值 = max over its covering segments
        # 用 np.maximum.at (但慢) 或 sort-based
        val_map = np.zeros((H_R, W_R), dtype=np.float32)
        val_map_flat = val_map.ravel()
        # 先用 fill: 每个像素位置遍历, 如果 v_arr[seg] > 现值, 更新
        # 用 np.maximum.at (无序但正确):
        np.maximum.at(val_map_flat, pix_id, v_arr[px_seg])
        
        # 加权平均扩散
        weighted_sum = gaussian_filter(val_map * presence, sigma=SIGMA_PX)
        diffused = np.zeros_like(weighted_sum)
        ok = weight > 1e-9
        diffused[ok] = weighted_sum[ok] / weight[ok]
        feat_mean[:, :, k_idx] = diffused
        
        # max 扩散
        feat_max_diff[:, :, k_idx] = maximum_filter(val_map, size=2*SIGMA_PX+1)
        
        if (k_idx + 1) % 10 == 0:
            print(f"    {k_idx+1}/{n_metrics}", flush=True)
    
    feat = np.concatenate([
        feat_mean.reshape(-1, n_metrics),
        feat_max_diff.reshape(-1, n_metrics)
    ], axis=1).astype(np.float32)
    
    cols = []
    for prefix in ['mean', 'max']:
        for k in keys:
            cols.append(f'{prefix}_{k}')
    
    cov = (feat[:, 0] > 0).mean()
    print(f"  {year}: feat {feat.shape}, 覆盖率 {cov*100:.1f}%, 耗时 {time.time()-t0:.0f}s")
    
    np.savez_compressed(DATA / f'cell_syntax_decay_{year}.npz',
                         feat=feat, col_names=np.array(cols))


def main():
    print("=" * 60)
    print("高效 KDE 扩散赋值")
    print("=" * 60)
    for year in YEARS:
        diffuse_year(year)
    print("\n✅ 完成")


if __name__ == '__main__':
    main()
