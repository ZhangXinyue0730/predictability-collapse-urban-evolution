"""
03_road_and_landuse.py — 一站式生成每年道路+用地连通图

集成所有最佳实践 (来自 v18-v21 迭代):
  1) OSM 全图拓扑构建 (端点 KD 合并)
  2) 时间单调修复 (用 8 年 mask 联合判定, 修假阴性)
  3) 主干闭合 (端点拓扑, 同等级或相邻级 + 优先同名)
  4) 最大连通分量 (抛弃孤岛 polyline)
  5) 用地双层连通 (建成区必须能经相邻用地→道路)

输出:
  data/road_{year}.npz   — 道路 polyline + fclass + name + node_a/b
  data/lu_clean_{year}.npy     — 清洁版用地 (孤岛建成区 → 0)
  data/road_mask_{year}.npy
  data/global_topo.pkl   — 全图拓扑信息 (节点位置/边/邻接)
"""
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict, deque
from scipy.spatial import cKDTree
from scipy.ndimage import label
from skimage.morphology import dilation, disk
import time

import sys
sys.path.insert(0, str(Path(__file__).parent))

from common import rasterize_polylines, hit_ratio_along
from config import (DATA, YEARS, H_R, W_R, GRADE_RANK, MAIN_GRADE,
                    BUILTUP_CLASSES, threshold_for_fclass)


# ============================================================
# 一次性准备: OSM 全图拓扑
# ============================================================
def build_osm_topology(polys, ep_merge_radius_px=0.3):
    """端点 KD 合并构建图拓扑.
    返回:
      poly_nodes: list of (na, nb) 每条 polyline 的两个节点 id
      node_adj: dict node_id → list of polyline_idx
      node_xy: dict node_id → (x, y)
      node_degree: dict node_id → 该节点的度
    """
    end_pts = []
    for pi, p in enumerate(polys):
        if len(p) < 2: continue
        end_pts.append((pi, 'a', float(p[0, 0]), float(p[0, 1])))
        end_pts.append((pi, 'b', float(p[-1, 0]), float(p[-1, 1])))
    ep_xy = np.array([[e[2], e[3]] for e in end_pts])
    tree = cKDTree(ep_xy)
    groups = tree.query_ball_tree(tree, r=ep_merge_radius_px)
    parent = list(range(len(end_pts)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry: parent[rx] = ry
    for i, g in enumerate(groups):
        for j in g: union(i, j)
    ep_to_node = {}
    for i in range(len(end_pts)):
        r = find(i)
        if r not in ep_to_node:
            ep_to_node[r] = len(ep_to_node)
        ep_to_node[i] = ep_to_node[r]

    poly_nodes = [(-1, -1)] * len(polys)
    ep_idx = 0
    for pi in range(len(polys)):
        if len(polys[pi]) < 2: continue
        poly_nodes[pi] = (ep_to_node[ep_idx], ep_to_node[ep_idx + 1])
        ep_idx += 2

    node_adj = defaultdict(list)
    for pi, (na, nb) in enumerate(poly_nodes):
        if na < 0: continue
        node_adj[na].append(pi)
        if nb != na: node_adj[nb].append(pi)
    node_degree = {n: len(v) for n, v in node_adj.items()}

    node_xy = {}
    for i, (pi, side, x, y) in enumerate(end_pts):
        nid = ep_to_node[i]
        node_xy.setdefault(nid, []).append((x, y))
    node_xy = {n: (float(np.mean([p[0] for p in pts])),
                    float(np.mean([p[1] for p in pts])))
               for n, pts in node_xy.items()}

    return {
        'poly_nodes': poly_nodes,
        'node_adj': node_adj,
        'node_xy': node_xy,
        'node_degree': node_degree,
    }


# ============================================================
# 时间单调修复
# ============================================================
def temporal_monotonic_fix(hit_matrix, attrs):
    """
    输入: hit_matrix (n_poly, 8) — 每条 polyline 在 8 年的命中比例
         attrs: list of dict (含 fclass)
    输出: alive_matrix (n_poly, 8) bool — 时间一致后的存活
    
    规则: 对每条 polyline
      1) 用 fclass 阈值判定 raw_alive (n, 8) bool
      2) 假阴性修复: 若中间年是 F 但两侧都有 T → 改 T
    """
    n_poly = hit_matrix.shape[0]
    thresholds = np.array([threshold_for_fclass(a['fclass']) for a in attrs],
                           dtype=np.float32)
    raw_alive = (hit_matrix >= thresholds[:, None])

    # 假阴性修复: 中间 F 被 T 包围 → T
    fixed = raw_alive.copy()
    n_years = raw_alive.shape[1]
    for i in range(1, n_years - 1):
        # 看 i 这一年是 F, 但 [<i] 有 T 且 [>i] 有 T
        is_F = ~raw_alive[:, i]
        has_left = raw_alive[:, :i].any(axis=1)
        has_right = raw_alive[:, i+1:].any(axis=1)
        flip = is_F & has_left & has_right
        fixed[flip, i] = True
    return fixed, raw_alive


# ============================================================
# 主干闭合
# ============================================================
def main_closure(initial_set, all_polys, attrs, topo, max_iter=15):
    """对主干道做端点闭合: 子图端点 deg=1 但全图 deg≥2 且不在边界 + 主干等级 → 加邻接的同级 polyline"""
    poly_nodes = topo['poly_nodes']
    node_adj = topo['node_adj']
    node_xy = topo['node_xy']
    node_degree = topo['node_degree']

    def is_boundary(nid, m=2):
        if nid not in node_xy: return False
        x, y = node_xy[nid]
        return x < m or x > W_R - m or y < m or y > H_R - m

    def is_dead_end(nid):
        return node_degree.get(nid, 0) == 1

    closed = set(initial_set)
    n_added = 0
    for it in range(max_iter):
        ep_count = defaultdict(int)
        ep_fcs_at = defaultdict(list)
        for pi in closed:
            na, nb = poly_nodes[pi]
            fc = attrs[pi]['fclass']
            if na >= 0:
                ep_count[na] += 1
                ep_fcs_at[na].append((pi, fc))
            if nb >= 0 and nb != na:
                ep_count[nb] += 1
                ep_fcs_at[nb].append((pi, fc))

        broken = []
        for nid, c in ep_count.items():
            if c >= 2: continue
            if is_boundary(nid): continue
            if is_dead_end(nid): continue
            fcs_here = [fc for _, fc in ep_fcs_at[nid]]
            if not any(fc in MAIN_GRADE for fc in fcs_here):
                continue
            broken.append(nid)
        if not broken: break
        added_this = 0
        for nid in broken:
            ref_fcs = [fc for _, fc in ep_fcs_at[nid]]
            ref_names = [attrs[pi].get('name', '') for pi, _ in ep_fcs_at[nid]]
            ref_grade = min(GRADE_RANK.get(fc, 9) for fc in ref_fcs)
            cand = []
            for pi in node_adj[nid]:
                if pi in closed: continue
                fc = attrs[pi]['fclass']
                if fc not in MAIN_GRADE: continue
                g = GRADE_RANK.get(fc, 9)
                if g > ref_grade + 1: continue
                same_name = (attrs[pi].get('name', '') in ref_names and
                             attrs[pi].get('name', ''))
                p = all_polys[pi]
                L = np.linalg.norm(np.diff(p, axis=0), axis=1).sum() if len(p) > 1 else 0
                cand.append((g, -int(bool(same_name)), L, pi))
            cand.sort()
            if cand:
                _, _, _, pi = cand[0]
                closed.add(pi); added_this += 1
        n_added += added_this
        if added_this == 0: break
    return closed, n_added


# ============================================================
# 主连通分量
# ============================================================
def find_main_component(polys_set, all_polys, kd_radius=1.5):
    """端点 KD 合并 (放宽到 1.5px), 找最大连通分量"""
    if not polys_set: return set(), 0
    pis_in = list(polys_set)
    pi_to_local = {pi: i for i, pi in enumerate(pis_in)}
    pts = []
    for pi in pis_in:
        p = all_polys[pi]
        if len(p) < 2:
            pts.append((0, 0)); pts.append((0, 0)); continue
        pts.append((float(p[0, 0]), float(p[0, 1])))
        pts.append((float(p[-1, 0]), float(p[-1, 1])))
    pts_arr = np.array(pts)
    tree = cKDTree(pts_arr)
    groups = tree.query_ball_tree(tree, r=kd_radius)
    parent = list(range(len(pts)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry: parent[rx] = ry
    for i, g in enumerate(groups):
        for j in g: union(i, j)
    poly_node_a = [find(2 * i) for i in range(len(pis_in))]
    poly_node_b = [find(2 * i + 1) for i in range(len(pis_in))]

    node_adj_local = defaultdict(list)
    for i in range(len(pis_in)):
        node_adj_local[poly_node_a[i]].append(i)
        if poly_node_b[i] != poly_node_a[i]:
            node_adj_local[poly_node_b[i]].append(i)

    visited = set()
    components = []
    for start in range(len(pis_in)):
        if start in visited: continue
        comp = set()
        q = deque([start])
        while q:
            u = q.popleft()
            if u in comp: continue
            comp.add(u); visited.add(u)
            for n in (poly_node_a[u], poly_node_b[u]):
                for v in node_adj_local[n]:
                    if v not in comp: q.append(v)
        components.append(comp)
    components.sort(key=len, reverse=True)
    main = components[0] if components else set()
    main_pis = set(pis_in[i] for i in main)
    return main_pis, len(components)


# ============================================================
# 用地连通
# ============================================================
def land_use_connectivity(lu, road_mask, builtup_classes=BUILTUP_CLASSES):
    """用地必须通过相邻用地 (8-邻接) 到达道路.
    返回: lu_clean (孤岛设 0), isolated mask (bool)
    """
    builtup = np.zeros_like(lu, dtype=bool)
    for c in builtup_classes:
        builtup |= (lu == c)
    seeds = builtup & dilation(road_mask, disk(1))
    lbl, _ = label(builtup, structure=np.ones((3, 3)))
    seeded_labels = set(np.unique(lbl[seeds]).tolist()) - {0}
    connected = np.isin(lbl, list(seeded_labels))
    isolated = builtup & ~connected
    lu_clean = lu.copy()
    lu_clean[isolated] = 0
    return lu_clean, isolated


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("Pipeline 02: 一站式生成每年道路+用地连通图 (v22)")
    print("=" * 60)

    # 1) 加载 OSM
    print("\n[1] 加载 OSM ...")
    with open(DATA / 'osm_named.pkl', 'rb') as f:
        d = pickle.load(f)
    all_polys = d['polys']
    all_attrs = d['attrs']
    print(f"  {len(all_polys):,} polylines")

    # 2) 全图拓扑
    print("\n[2] 建 OSM 全图拓扑 ...")
    t0 = time.time()
    topo = build_osm_topology(all_polys, ep_merge_radius_px=0.3)
    print(f"  完成 ({time.time()-t0:.0f}s), 节点 {len(topo['node_xy']):,}, "
          f"度≥3 真交叉 {sum(1 for d in topo['node_degree'].values() if d>=3):,}")
    # 保存全图拓扑
    with open(DATA / 'global_topo.pkl', 'wb') as f:
        pickle.dump(topo, f)

    # 3) 算 hit_matrix (复用已有的)
    print("\n[3] 加载/计算 hit_matrix ...")
    if (DATA / 'hit_matrix.npy').exists():
        hit_matrix = np.load(DATA / 'hit_matrix.npy')
        print(f"  已存在, shape {hit_matrix.shape}")
    else:
        print("  重新计算 ...")
        hit_matrix = np.zeros((len(all_polys), len(YEARS)), dtype=np.float32)
        for yi, year in enumerate(YEARS):
            # 从用地图自动提取道路 mask (类别 5)
            mask_path = DATA / f'mask_road_{year}.npy'
            if mask_path.exists():
                mask = np.load(mask_path)
            else:
                lu = np.load(DATA / f'lu_{year}.npy')
                mask = (lu == 5)   # 类别 5 = 道路市政
                # 适当膨胀 (1 px ≈ 100m), 让命中更宽容
                from skimage.morphology import dilation
                mask = dilation(mask, disk(1))
                np.save(mask_path, mask)
            for pi in range(len(all_polys)):
                p = all_polys[pi]
                if len(p) < 2: continue
                hit_matrix[pi, yi] = hit_ratio_along(p, mask)
        np.save(DATA / 'hit_matrix.npy', hit_matrix)

    # 4) 时间单调修复
    print("\n[4] 时间单调修复 (假阴性) ...")
    fixed_alive, raw_alive = temporal_monotonic_fix(hit_matrix, all_attrs)
    n_repair = ((fixed_alive ^ raw_alive)).sum()
    print(f"  修复 (poly, year) 对: {n_repair:,}")
    np.save(DATA / 'temporal_alive.npy', fixed_alive)

    # 5) 对每年: 主干闭合 + 主连通 + 用地连通
    print("\n[5] 每年生成最终连通图 ...")
    summary = {}
    for yi, year in enumerate(YEARS):
        print(f"\n  --- {year} ---")
        initial = set(np.where(fixed_alive[:, yi])[0])
        print(f"    时间修复后初始: {len(initial):,}")

        # 主干闭合
        closed, n_clo = main_closure(initial, all_polys, all_attrs, topo)
        print(f"    主干闭合: +{n_clo}")

        # 主连通分量
        main_set, n_comps = find_main_component(closed, all_polys, kd_radius=1.5)
        print(f"    连通分量 {n_comps}, 主分量 {len(main_set):,} "
              f"({100*len(main_set)/max(len(closed),1):.0f}%)")

        # 输出 polyline + 节点关联
        sorted_pis = sorted(main_set)
        out_polys = [all_polys[pi] for pi in sorted_pis]
        out_fcs = [all_attrs[pi]['fclass'] for pi in sorted_pis]
        out_names = [all_attrs[pi].get('name', '') for pi in sorted_pis]
        # 用全图拓扑的节点 id (供下游 angular 分析用)
        out_na = [topo['poly_nodes'][pi][0] for pi in sorted_pis]
        out_nb = [topo['poly_nodes'][pi][1] for pi in sorted_pis]

        L = sum(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in out_polys)
        L_km = L * 11.0725 * 9 / 1000

        # 道路栅格化
        road_mask = rasterize_polylines(out_polys, (H_R, W_R))

        # 用地连通
        lu = np.load(DATA / f'lu_{year}.npy')
        lu_clean, iso = land_use_connectivity(lu, road_mask)
        print(f"    总长 {L_km:.0f} km, 孤岛用地 {iso.sum():,} px")

        # 保存
        np.savez_compressed(DATA / f'road_{year}.npz',
                             polys=np.array(out_polys, dtype=object),
                             fcs=np.array(out_fcs),
                             names=np.array(out_names),
                             node_a=np.array(out_na, dtype=np.int32),
                             node_b=np.array(out_nb, dtype=np.int32))
        np.save(DATA / f'lu_clean_{year}.npy', lu_clean)
        np.save(DATA / f'road_mask_{year}.npy', road_mask)
        np.save(DATA / f'land_iso_{year}.npy', iso)

        summary[year] = {'n': len(out_polys), 'L_km': L_km, 'iso_lu': int(iso.sum())}

    print(f"\n{'年':<6} {'polylines':>10} {'总长 km':>10} {'孤岛用地 px':>12}")
    for y, s in summary.items():
        print(f"{y:<6} {s['n']:>10,} {s['L_km']:>10.0f} {s['iso_lu']:>12,}")

    print("\n✅ Pipeline 02 完成")


if __name__ == '__main__':
    main()
