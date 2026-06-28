"""
09_cross_links.py
---------------------
跨层连接: 用地 cell ↔ 道路 segment

连接规则:
  对每个道路 segment:
    沿 polyline 每隔 1 像素采样, 找两侧 100m × 100m 的方格
    把这些方格 → segment 加入边 (双向)

最终我们得到一个超图:
  - 道路层: segments + angular edges (来自 syntax_graph)
  - 用地层: cells + 4-邻接 (来自 landuse_graph)
  - 跨层边: cell ↔ segment (本步)
  - 跨年边: 同一 cell 在 year_t 与 year_{t+1} (用地图节点 id 一致, 直接对应)

输出 (per year):
  data/cross_links_{year}.npz
    'cell_road_edges'  : (M, 2) cell_idx ↔ seg_idx 跨层边
    'seg_centroid_cell' : (n_segs,) 每条 segment 中点对应的 cell idx
"""
import numpy as np
import pickle
from pathlib import Path
from skimage.morphology import dilation, disk
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import DATA, YEARS, H_R, W_R


def cell_idx_of(col, row):
    """像素坐标 → cell idx (cell 网格 == 像素网格)"""
    if col < 0 or col >= W_R or row < 0 or row >= H_R:
        return -1
    return int(row) * W_R + int(col)


def main():
    print("阶段 C-3: 用地 cell ↔ 道路 segment 跨层边")
    for year in YEARS:
        print(f"\n{year}: ...")
        # 加载 segment graph
        with open(DATA / f'syntax_graph_{year}.pkl', 'rb') as f:
            sg = pickle.load(f)
        # 加载道路 polylines
        z = np.load(DATA / f'road_{year}.npz', allow_pickle=True)
        polys = z['polys']
        n_segs = len(polys)

        # 每条 segment 沿 polyline 采样, 每个采样点的 8 邻域内的 cell 都连接
        cell_road_set = set()  # (cell_idx, seg_idx)
        for si, p in enumerate(polys):
            if len(p) < 2: continue
            # 沿弧长每像素采样
            cum = np.cumsum(np.concatenate([[0],
                            np.linalg.norm(np.diff(p, axis=0), axis=1)]))
            L = cum[-1]
            n_samp = max(int(L * 2), 2)   # 每 0.5 px 一个点 (高密度)
            ts = np.linspace(0, L, n_samp)
            idxs = np.searchsorted(cum, ts); idxs = np.clip(idxs, 1, len(p)-1)
            prev = idxs - 1
            frac = (ts - cum[prev]) / np.maximum(cum[idxs] - cum[prev], 1e-9)
            xs = p[prev, 0] + frac * (p[idxs, 0] - p[prev, 0])
            ys = p[prev, 1] + frac * (p[idxs, 1] - p[prev, 1])
            for x, y in zip(xs, ys):
                # 该点的 8 邻域 (即 3x3 cell 范围: 自身 + 上下左右 + 4 对角)
                cx, cy = int(round(x)), int(round(y))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        cidx = cell_idx_of(cx + dx, cy + dy)
                        if cidx >= 0:
                            cell_road_set.add((cidx, si))
        cell_road_edges = np.array(list(cell_road_set), dtype=np.int32)
        # segment 中点对应 cell
        seg_centroid_cell = np.zeros(n_segs, dtype=np.int32)
        for si in range(n_segs):
            mid = sg['seg_xy_mid'][si]
            seg_centroid_cell[si] = cell_idx_of(int(round(mid[0])), int(round(mid[1])))

        n_cells_unique = len(np.unique(cell_road_edges[:, 0])) if len(cell_road_edges) else 0
        print(f"  cells with road link: {n_cells_unique}, edges: {len(cell_road_edges)}")
        np.savez_compressed(DATA / f'cross_links_{year}.npz',
                             cell_road_edges=cell_road_edges,
                             seg_centroid_cell=seg_centroid_cell)
        print(f"  → cross_links_{year}.npz")
    print("✅ 完成")


if __name__ == '__main__':
    main()
