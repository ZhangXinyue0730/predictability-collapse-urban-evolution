"""
04_segment_graph.py
-------------------------
构造 segment-based angular graph (针对每年).

定义 (标准 Space Syntax angular segment analysis):
  - 节点 = polyline (一个道路段, 一条 fclass+name+几何 单位)
  - 边 = 两个 polyline 在共享端点 (交叉口) 处相连
  - 边权 = 两 polyline 在该端点处的转角 (angular cost),
           完全直行 = 0,  90 度转弯 = 1.0,  180 度掉头 = 2.0
           (标准化: cost = angle_deg / 90)

输出 (per year):
  data/syntax_graph_{year}.pkl
    {
      'n_segs': N,
      'seg_xy_mid': (N, 2)  # 每个 segment 的中点
      'seg_length_m': (N,)
      'seg_fclass': (N,) str
      'seg_name': (N,) str
      'edges': [(seg_i, seg_j, ang_cost), ...]  无向边 (双向都存)
    }
"""
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import DATA, YEARS, PX_M


def end_directions(poly):
    """返回 polyline 两端的"朝外向量"(单位向量).
    a 端朝外 = poly[0] → poly[1] 反向; b 端朝外 = poly[-1] → poly[-2] 反向
    我们用 "朝向交叉口" 方向: a 端 = poly[1] - poly[0], b 端 = poly[-2] - poly[-1] 反向
    但 angular 分析需要 "在交叉口看向 segment 内部" 的方向:
      在 a 节点看, segment 走的方向 = poly[0] → poly[k] 取若干步后
      在 b 节点看, segment 走的方向 = poly[-1] → poly[-k]
    """
    if len(poly) < 2:
        return np.array([1.0, 0.0]), np.array([1.0, 0.0])
    look = min(3, len(poly) - 1)
    # 在 a 端: 从 a 朝内看的方向 (a 是 poly[0])
    da = poly[look] - poly[0]
    Lda = np.linalg.norm(da)
    da = da / Lda if Lda > 1e-9 else np.array([1.0, 0.0])
    # 在 b 端: 从 b 朝内看的方向 (b 是 poly[-1])
    db = poly[-1 - look] - poly[-1]
    Ldb = np.linalg.norm(db)
    db = db / Ldb if Ldb > 1e-9 else np.array([1.0, 0.0])
    return da, db


def angular_cost(d1, d2):
    """两条 segment 在同一节点处的转角 cost.
    d1, d2 都是 "从节点朝内看 segment 的方向" 单位向量.
    完全直行 = 两段方向相反 (角度 180 度) → cost 0
    完全 U 转 = 两段方向相同 (角度 0 度) → cost 2.0
    我们: cost = (180 - angle_between(d1, d2)) / 90
    """
    cos = float(np.clip(np.dot(d1, d2), -1, 1))
    angle_deg = float(np.degrees(np.arccos(cos)))
    # 完全反向 (直行) → angle = 180 → cost = 0
    # 90 度 → cost = 1
    # 0 度 (U 掉头) → cost = 2
    cost = (180.0 - angle_deg) / 90.0
    return cost


def build_segment_graph(year):
    """读取 road_{year} 构造 segment-based graph"""
    z = np.load(DATA / f'road_{year}.npz', allow_pickle=True)
    polys = z['polys']
    fcs = z['fcs']
    names = z['names']
    node_a = z['node_a']
    node_b = z['node_b']

    n_segs = len(polys)
    # 每个 segment 的中点和长度 (米)
    seg_xy_mid = np.zeros((n_segs, 2), dtype=np.float32)
    seg_length_m = np.zeros(n_segs, dtype=np.float32)
    seg_dir_at_a = np.zeros((n_segs, 2), dtype=np.float32)
    seg_dir_at_b = np.zeros((n_segs, 2), dtype=np.float32)
    for i, p in enumerate(polys):
        if len(p) < 2: continue
        # 中点 (按弧长一半)
        cum = np.cumsum(np.concatenate([[0], np.linalg.norm(np.diff(p, axis=0), axis=1)]))
        L = cum[-1]
        seg_length_m[i] = L * PX_M
        half = L / 2
        idx = np.searchsorted(cum, half)
        idx = max(1, min(idx, len(p) - 1))
        prev = idx - 1
        frac = (half - cum[prev]) / max(cum[idx] - cum[prev], 1e-9)
        seg_xy_mid[i] = (p[prev] + frac * (p[idx] - p[prev])).astype(np.float32)
        # 端点方向
        da, db = end_directions(p)
        seg_dir_at_a[i] = da
        seg_dir_at_b[i] = db

    # 构造边: 把所有共享一个节点的 segment 两两连接, 边权 = angular cost
    # 节点 → list of (seg_idx, side) 其中 side 是 'a' 或 'b'
    node_to_seg = defaultdict(list)
    for i in range(n_segs):
        if node_a[i] == node_b[i]:
            # 自环 (圆形封闭路) — 跳过
            continue
        node_to_seg[int(node_a[i])].append((i, 'a'))
        node_to_seg[int(node_b[i])].append((i, 'b'))

    edges = []  # (seg_i, seg_j, cost)
    for nid, segs_at in node_to_seg.items():
        for ii in range(len(segs_at)):
            si, side_i = segs_at[ii]
            di = seg_dir_at_a[si] if side_i == 'a' else seg_dir_at_b[si]
            for jj in range(ii + 1, len(segs_at)):
                sj, side_j = segs_at[jj]
                dj = seg_dir_at_a[sj] if side_j == 'a' else seg_dir_at_b[sj]
                cost = angular_cost(di, dj)
                edges.append((si, sj, cost))

    return {
        'n_segs': n_segs,
        'seg_xy_mid': seg_xy_mid,
        'seg_length_m': seg_length_m,
        'seg_fclass': fcs,
        'seg_name': names,
        'seg_node_a': node_a,
        'seg_node_b': node_b,
        'edges': edges,
    }


def main():
    print("=" * 60)
    print("阶段 B-1: 构造每年的 angular segment graph")
    print("=" * 60)
    for year in YEARS:
        g = build_segment_graph(year)
        print(f"\n{year}: {g['n_segs']:,} segments, {len(g['edges']):,} angular edges")
        # 边数除以 segment 数 = 平均每个 segment 在多少个交叉口
        avg = 2 * len(g['edges']) / max(g['n_segs'], 1)
        print(f"  平均每 seg 邻接边数: {avg:.2f}")
        # 转角分布
        costs = np.array([e[2] for e in g['edges']])
        if len(costs) > 0:
            print(f"  cost 分布: 直行 (<0.3) {(costs<0.3).sum():,} "
                  f"| 拐弯 (0.3-1.3) {((costs>=0.3)&(costs<1.3)).sum():,} "
                  f"| U转 (>1.3) {(costs>=1.3).sum():,}")
        with open(DATA / f'syntax_graph_{year}.pkl', 'wb') as f:
            pickle.dump(g, f)
        print(f"  → data/syntax_graph_{year}.pkl")
    print("\n✅ 完成")


if __name__ == '__main__':
    main()
