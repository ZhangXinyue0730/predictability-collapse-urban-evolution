"""
common.py — 通用工具函数

包括: 几何辅助、用地颜色渲染、栅格化、KDTree 邻接等城市无关函数.
"""
import numpy as np
from config import H_R, W_R, LU_COLORS, PX_M


def hit_ratio_along(poly, mask, max_samp=200):
    """polyline 沿弧长在 mask 上命中比例 (0~1).
    
    用于判定一条 OSM polyline 在某年是否实际"存在"
    (历史卫星图栅格化的 mask).
    """
    if len(poly) < 2:
        return 0
    cum = np.zeros(len(poly))
    for i in range(1, len(poly)):
        cum[i] = cum[i-1] + np.linalg.norm(poly[i] - poly[i-1])
    if cum[-1] < 1:
        return 0
    n_samp = max(int(cum[-1]), 5); n_samp = min(n_samp, max_samp)
    ts = np.linspace(0, cum[-1], n_samp)
    idxs = np.searchsorted(cum, ts); idxs = np.clip(idxs, 1, len(poly) - 1)
    prev = idxs - 1
    frac = (ts - cum[prev]) / np.maximum(cum[idxs] - cum[prev], 1e-9)
    cs = (poly[prev, 0] + frac * (poly[idxs, 0] - poly[prev, 0])).round().astype(int)
    rs = (poly[prev, 1] + frac * (poly[idxs, 1] - poly[prev, 1])).round().astype(int)
    H, W = mask.shape
    cs = np.clip(cs, 0, W - 1); rs = np.clip(rs, 0, H - 1)
    return mask[rs, cs].mean()


def render_lu(lu):
    """用地图 → RGB 图像 (用于可视化)"""
    H, W = lu.shape
    rgb = np.full((H, W, 3), 250, dtype=np.uint8)
    for c, h in LU_COLORS.items():
        m = (lu == c)
        r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
        rgb[m] = (r, g, b)
    return rgb


def rasterize_polylines(polys, shape):
    """polyline 列表 → 栅格 mask (二值)"""
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    for p in polys:
        for i in range(len(p) - 1):
            xa, ya = p[i]; xb, yb = p[i+1]
            n_pts = max(int(np.hypot(xb - xa, yb - ya)) * 3, 3)
            ts = np.linspace(0, 1, n_pts)
            xs = (xa + ts * (xb - xa)).round().astype(int)
            ys = (ya + ts * (yb - ya)).round().astype(int)
            valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
            mask[ys[valid], xs[valid]] = True
    return mask


def cell_grid_coords():
    """返回每个 cell 的 (col, row) 像素坐标, shape (H*W, 2)"""
    rows = np.arange(H_R).reshape(-1, 1).repeat(W_R, axis=1)
    cols = np.arange(W_R).reshape(1, -1).repeat(H_R, axis=0)
    return np.stack([cols.flatten(), rows.flatten()], axis=1).astype(np.float32)
