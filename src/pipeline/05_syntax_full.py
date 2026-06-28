"""
05_syntax_full.py
--------------------------
严格 angular 空间句法分析 (含 NACH).

关键优化:
  - NC, TD, TL, MD, Integration, NAIN: 标准 Dijkstra (O(V·(V+E)·logV))
    对每个 segment 跑 truncated Dijkstra → 已经是 O(V) 内
  - Choice / NACH: 用 sampling-based betweenness (Riondato 2014)
    从 K 个随机 source 出发, 累积 partial betweenness
    最终 Choice 估计 = (V/K) × sum(partial)
    用 K=2000 → 速度提升 ~25× vs 全 Brandes

参考: 用户建议 "随机最短路径" 加快 — 这正是采样 betweenness.

NACH: Normalized Angular Choice (Hillier 2012)
  NACH = log(Choice + 1) / log(TD + 3)

输出 (per year per radius):
  data/syntax_{year}_R{R}.npz  含 NC, TD, TL, MD, Integration, NAIN, Choice, NACH
增量: 即使中断也保留已完成的 (年, 半径) 结果

每个 (年, R) 完成后, 同时合并到 syntax_results_{year}.npz
"""
import numpy as np
import pickle
from pathlib import Path
import heapq
import time
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (DATA, YEARS, SYNTAX_RADII, SYNTAX_NACH_RADII,
                    SYNTAX_BETWEENNESS_K)

# 半径配置: 算 NC/TD/TL/MD/Integ/NAIN 的所有半径; NACH 只在 1000 和 2000 算 (用户要求)
RADII_BASIC = SYNTAX_RADII
RADII_NACH = SYNTAX_NACH_RADII
N_SAMPLE_NACH = SYNTAX_BETWEENNESS_K   # K = 3000 sources for sampling betweenness (速度 vs 精度平衡)


def build_csr_adj(g):
    n = g['n_segs']
    seg_len = g['seg_length_m']
    src = []; dst = []; ang = []; geo = []
    for i, j, c in g['edges']:
        d = 0.5 * (seg_len[i] + seg_len[j])
        src.append(i); dst.append(j); ang.append(c); geo.append(d)
        src.append(j); dst.append(i); ang.append(c); geo.append(d)
    src = np.array(src); dst = np.array(dst)
    ang = np.array(ang, dtype=np.float32); geo = np.array(geo, dtype=np.float32)
    order = np.argsort(src, kind='stable')
    src = src[order]; dst = dst[order]; ang = ang[order]; geo = geo[order]
    adj_idx = np.zeros(n + 1, dtype=np.int32)
    counts = np.bincount(src, minlength=n)
    adj_idx[1:] = np.cumsum(counts)
    return adj_idx, dst.astype(np.int32), ang, geo


def truncated_dijkstra(start, adj_idx, adj_dst, adj_ang, adj_geo, R):
    """返回 visited list (按 angular_dist 升序), angular_dist dict, sigma, pred"""
    angular_dist = {start: 0.0}
    pq = [(0.0, 0.0, start)]
    visited = []
    in_visited = set()
    while pq:
        d_a, d_g, u = heapq.heappop(pq)
        if u in in_visited: continue
        if angular_dist.get(u, float('inf')) < d_a - 1e-9: continue
        if d_g > R + 1e-9: continue
        in_visited.add(u)
        visited.append(u)
        s = adj_idx[u]; e = adj_idx[u + 1]
        for k in range(s, e):
            v = int(adj_dst[k])
            if v in in_visited: continue
            new_g = d_g + adj_geo[k]
            if new_g > R + 1e-9: continue
            new_a = d_a + adj_ang[k]
            old = angular_dist.get(v, float('inf'))
            if new_a < old - 1e-9:
                angular_dist[v] = new_a
                heapq.heappush(pq, (new_a, new_g, v))
    return visited, angular_dist


def brandes_truncated(start, adj_idx, adj_dst, adj_ang, adj_geo, R):
    """truncated Brandes: 返回 visited, angular_dist, delta (partial betweenness from this source)"""
    angular_dist = {start: 0.0}
    geo_dist = {start: 0.0}
    sigma = {start: 1.0}
    pred = {start: []}
    pq = [(0.0, 0.0, start)]
    visited = []
    in_v = set()
    while pq:
        d_a, d_g, u = heapq.heappop(pq)
        if u in in_v: continue
        if angular_dist.get(u, float('inf')) < d_a - 1e-9: continue
        if d_g > R + 1e-9: continue
        in_v.add(u); visited.append(u)
        s = adj_idx[u]; e = adj_idx[u + 1]
        for k in range(s, e):
            v = int(adj_dst[k])
            if v in in_v: continue
            new_g = d_g + adj_geo[k]
            if new_g > R + 1e-9: continue
            new_a = d_a + adj_ang[k]
            old_a = angular_dist.get(v, float('inf'))
            if new_a < old_a - 1e-9:
                angular_dist[v] = new_a
                geo_dist[v] = new_g
                sigma[v] = sigma[u]
                pred[v] = [u]
                heapq.heappush(pq, (new_a, new_g, v))
            elif abs(new_a - old_a) < 1e-9:
                sigma[v] += sigma[u]
                pred[v].append(u)
    # 反向: 计算 delta
    delta = {v: 0.0 for v in visited}
    # 按 angular_dist 降序
    for v in sorted(visited, key=lambda x: -angular_dist[x]):
        for u in pred.get(v, []):
            if sigma[v] > 0:
                delta[u] += (sigma[u] / sigma[v]) * (1.0 + delta[v])
    delta.pop(start, None)
    return visited, angular_dist, delta


def compute_year_radius(year, R, force=False):
    """对一年一个半径计算所有指标 (含 NACH 如果 R 在 RADII_NACH)"""
    out_path = DATA / f'syntax_{year}_R{R}.npz'
    if out_path.exists() and not force:
        # 检查是否已含 NACH (如果用户要求 NACH 但旧文件没有)
        z = np.load(out_path)
        if R not in RADII_NACH:
            print(f"  R={R}m: 已存在 (无需 NACH), 跳过")
            return
        if 'NACH' in z.files:
            print(f"  R={R}m: 已存在含 NACH, 跳过")
            return
        # 没 NACH, 需要补算 NACH

    print(f"  R={R}m 开始 ...", flush=True)
    with open(DATA / f'syntax_graph_{year}.pkl', 'rb') as f:
        g = pickle.load(f)
    n = g['n_segs']
    seg_len = g['seg_length_m']
    adj_idx, adj_dst, adj_ang, adj_geo = build_csr_adj(g)
    
    t0 = time.time()
    # === 基础指标 (Dijkstra)===
    TD = np.zeros(n, dtype=np.float64)
    NC = np.zeros(n, dtype=np.float64)
    TL = np.zeros(n, dtype=np.float64)
    for s in range(n):
        visited, ang_d = truncated_dijkstra(s, adj_idx, adj_dst, adj_ang, adj_geo, R)
        NC[s] = len(visited)
        td = 0.0; tl = 0.0
        for v in visited:
            td += ang_d[v]
            tl += seg_len[v]
        TD[s] = td
        TL[s] = tl
    
    MD = np.where(NC > 1, TD / np.maximum(NC - 1, 1), 0)
    Integration = np.where(TD > 0, NC * NC / np.maximum(TD, 1e-9), 0)
    NAIN = np.where(TD > 0, np.power(NC, 1.2) / np.maximum(TD, 1e-9), 0)
    print(f"    Dijkstra 完成 ({time.time()-t0:.1f}s), NC max {NC.max():.0f}", flush=True)

    save_dict = {
        'TD': TD.astype(np.float32),
        'NC': NC.astype(np.float32),
        'TL': TL.astype(np.float32),
        'MD': MD.astype(np.float32),
        'Integration': Integration.astype(np.float32),
        'NAIN': NAIN.astype(np.float32),
    }

    # === NACH 仅 R=1000, 2000 ===
    if R in RADII_NACH:
        t1 = time.time()
        # 采样 sources (固定种子)
        rng = np.random.default_rng(42)
        K = min(N_SAMPLE_NACH, n)
        sources = rng.choice(n, size=K, replace=False)
        Choice_partial = np.zeros(n, dtype=np.float64)
        for si, s in enumerate(sources):
            if si % 500 == 0 and si > 0:
                print(f"    Brandes sampling {si}/{K} ({time.time()-t1:.0f}s)", flush=True)
            visited, ang_d, delta = brandes_truncated(s, adj_idx, adj_dst, adj_ang, adj_geo, R)
            for v, d in delta.items():
                Choice_partial[v] += d
        # 缩放: 估计 = (V/K) × sum(partial)
        Choice = Choice_partial * (n / K)
        # NACH = log(Choice + 1) / log(TD + 3)
        NACH = np.log(Choice + 1) / np.maximum(np.log(TD + 3), 1e-9)
        print(f"    NACH 完成 ({time.time()-t1:.1f}s), Choice max {Choice.max():.0f}, NACH max {NACH.max():.3f}", flush=True)
        save_dict['Choice'] = Choice.astype(np.float32)
        save_dict['NACH'] = NACH.astype(np.float32)

    np.savez_compressed(out_path, **save_dict)
    print(f"  R={R}m: 保存 → syntax_{year}_R{R}.npz", flush=True)


def merge_year_results(year):
    """合并所有 syntax_{year}_R*.npz → syntax_results_{year}.npz"""
    all_data = {}
    for R in RADII_BASIC:
        f = DATA / f'syntax_{year}_R{R}.npz'
        if not f.exists():
            print(f"  ⚠ {year} R={R} 缺失")
            continue
        z = np.load(f)
        for key in z.files:
            all_data[f'R{R}_{key}'] = z[key]
    out = DATA / f'syntax_results_{year}.npz'
    np.savez_compressed(out, **all_data)
    print(f"  {year}: 合并 → {len(all_data)} keys")


def main():
    print("=" * 60)
    print("阶段 B-2 严格版: 含 Choice/NACH 计算")
    print("=" * 60)
    print(f"基础半径: {RADII_BASIC}")
    print(f"NACH 半径: {RADII_NACH} (sampling K={N_SAMPLE_NACH})")
    
    for year in YEARS:
        print(f"\n=== {year} ===")
        for R in RADII_BASIC:
            compute_year_radius(year, R)
        merge_year_results(year)
    print("\n✅ 完成")


if __name__ == '__main__':
    main()
