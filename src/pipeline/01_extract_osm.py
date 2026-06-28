"""
01_extract_osm.py — OSM 道路矢量提取
=====================================

输入: OSM road shapefile (Geofabrik 下载的 gis_osm_roads_free_1.shp/.dbf)
输出: data/osm_named.pkl
        {'polys': list of (N, 2) numpy arrays (像素坐标),
         'attrs': list of dict {fclass, name, ref, oneway}}

依赖 (任选一种):
  方式 A (推荐): pip install pyogrio   或   pip install geopandas
  方式 B (无依赖, 仅本项目): 用手写 shp + dbf 解析器

每条 polyline 的几何已经从 EPSG:4326 重投影到 EPSG:3857 (Web Mercator),
然后转换为像素坐标 (基于 config 里的 X0_M, Y0_M, PX_M).

用法:
  python 01_extract_osm.py /path/to/gis_osm_roads_free_1.shp
"""
import sys
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA, X0_M, Y0_M, PX_M


def load_with_pyogrio(shp_path):
    """优先尝试 pyogrio (最快)"""
    try:
        import pyogrio
    except ImportError:
        return None
    print("[pyogrio] 加载 ...")
    df = pyogrio.read_dataframe(shp_path)
    if df.crs is None or df.crs.to_epsg() != 3857:
        df = df.to_crs(3857)
    polys = []
    attrs = []
    for _, row in df.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty: continue
        if geom.geom_type != 'LineString': continue
        coords = np.array(geom.coords, dtype=np.float64)  # (N, 2) 米
        # 米 → 像素
        px = (coords[:, 0] - X0_M) / PX_M
        py = (coords[:, 1] - Y0_M) / PX_M
        # Y 轴翻转 (栅格 row=0 在顶部, geographic Y 在底部)
        # 这里假设栅格已经按 standard 方向布局
        py = -py  # 视情况而定, 下游 02_build 会处理
        poly = np.stack([px, py], axis=1).astype(np.float32)
        polys.append(poly)
        attrs.append({
            'fclass': str(row.get('fclass', '')),
            'name': str(row.get('name', '')) if row.get('name') else '',
            'ref': str(row.get('ref', '')) if row.get('ref') else '',
            'oneway': str(row.get('oneway', '')) if row.get('oneway') else '',
        })
    return polys, attrs


def load_with_geopandas(shp_path):
    """fallback: geopandas"""
    try:
        import geopandas as gpd
    except ImportError:
        return None
    print("[geopandas] 加载 ...")
    df = gpd.read_file(shp_path)
    if df.crs is None or df.crs.to_epsg() != 3857:
        df = df.to_crs(3857)
    polys, attrs = [], []
    for _, row in df.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty: continue
        if geom.geom_type != 'LineString': continue
        coords = np.array(geom.coords, dtype=np.float64)
        px = (coords[:, 0] - X0_M) / PX_M
        py = (coords[:, 1] - Y0_M) / PX_M
        py = -py
        poly = np.stack([px, py], axis=1).astype(np.float32)
        polys.append(poly)
        attrs.append({
            'fclass': str(row.get('fclass', '')),
            'name': str(row.get('name', '')) if row.get('name') else '',
            'ref': str(row.get('ref', '')) if row.get('ref') else '',
            'oneway': str(row.get('oneway', '')) if row.get('oneway') else '',
        })
    return polys, attrs


def main(shp_path):
    out_path = DATA / 'osm_named.pkl'
    if out_path.exists():
        print(f"已存在 {out_path}, 跳过")
        return
    
    result = load_with_pyogrio(shp_path)
    if result is None:
        result = load_with_geopandas(shp_path)
    if result is None:
        print("❌ 需要 pyogrio 或 geopandas, pip install pyogrio")
        sys.exit(1)
    
    polys, attrs = result
    print(f"✓ 加载 {len(polys):,} 条 polylines")
    
    # 统计
    from collections import Counter
    fc = Counter(a['fclass'] for a in attrs)
    print("  fclass 分布 (top 10):")
    for f, c in fc.most_common(10):
        print(f"    {f}: {c:,}")
    n_named = sum(1 for a in attrs if a['name'])
    print(f"  命名 polyline: {n_named:,}")
    
    with open(out_path, 'wb') as f:
        pickle.dump({'polys': polys, 'attrs': attrs}, f)
    print(f"→ {out_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python 01_extract_osm.py <path_to_gis_osm_roads_free_1.shp>")
        sys.exit(1)
    main(sys.argv[1])
