"""
TCULU 城市提取脚本
================
功能：从全国 TCULU 大图中，自动识别并切出每一个城市的用地数据。

老师的需求：
1. 读取 TCULU GeoTIFF（全国范围）
2. 判断哪些像素是"城市建成区"（含绿地；裸地按规则判断）
3. 用空间聚类把连片的城市用地分成一组一组（每组=一个城市）
4. 去掉距主体超过 2km 的"飞地"
5. 把每个城市切出来，保存成工具箱期望的格式

使用方法：
  python tculu_city_extractor.py \
      --tculu-dir /path/to/tculu_tifs \
      --output-dir /path/to/output \
      --years 1984 1995 2005 2015 \
      --min-area-km2 5 \
      --flyover-km 2.0

依赖：
  pip install rasterio numpy scipy scikit-learn tqdm
  （tifffile 作为 rasterio 不可用时的备选）
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# ============================================================
# 常量定义（和老师的 config.py 保持一致）
# ============================================================

# 用地类别编码（TCULU 原始编码）
TCULU_TO_UNIFIED = {
    1: 6,   # 水体 -> 绿地水
    2: 6,   # 绿地 -> 绿地水
    3: 0,   # 农田 -> 非城市
    4: 0,   # 裸地 -> 先非城市，后续规则再说
    5: 1,   # 居住
    6: 2,   # 商业
    7: 3,   # 公共服务
    8: 4,   # 工业
    9: 5,   # 交通
}


# 统一编码下，哪些算"明确城市用地"（用于聚类种子）
# 居住(1)、商业(2)、公共服务(3)、工业仓储(4)、交通市政(5)
URBAN_SEED_CLASSES = [1, 2, 3, 4, 5]

# 绿地(6) 也算建成区（老师要求）
URBAN_WITH_GREEN_CLASSES = [1, 2, 3, 4, 5, 6]

# 裸地判断用：如果一个格子四周都是城市，或者时间序列前后都是城市，也算城市
# （裸地在 TCULU 里通常是 0 = 背景，或者是某些未分类区域）
BARE_LAND_CLASSES = [0]


# ============================================================
# 第一步：读取 TCULU GeoTIFF
# ============================================================

def read_tculu_tif(path):
    """
    读取一张 TCULU GeoTIFF 文件。
    优先用 rasterio（能获取地理信息），备用 tifffile。
    
    返回：
        raster  : [H, W] uint8，原始类别值
        meta    : 地理信息字典（transform、crs），没有 rasterio 时为空
    """
    path = str(path)
    try:
        import rasterio
        with rasterio.open(path) as src:
            raster = src.read(1).astype(np.uint8)
            meta = {
                'transform': src.transform,
                'crs': src.crs,
                'height': src.height,
                'width': src.width,
                'pixel_size_m': abs(src.transform.a),  # 像素大小（米）
            }
        print(f"  [rasterio] 读取成功: {raster.shape}, 像素={meta['pixel_size_m']:.0f}m")
        return raster, meta
    except ImportError:
        print("  注意: 未安装 rasterio，改用 tifffile（无地理信息）")
    except Exception as e:
        print(f"  rasterio 读取失败: {e}，改用 tifffile")
    
    try:
        import tifffile
        raster = tifffile.imread(path).astype(np.uint8)
        meta = {'pixel_size_m': 10.0}  # TCULU 默认 10m
        print(f"  [tifffile] 读取成功: {raster.shape}")
        return raster, meta
    except ImportError:
        raise ImportError("请安装 rasterio 或 tifffile：pip install rasterio tifffile")


# ============================================================
# 第二步：类别重映射（TCULU原始编码 → 统一6类）
# ============================================================

def remap_to_unified(raster, mapping=None):
    """
    把 TCULU 原始类别映射到统一 6+1 类编码。
    """
    if mapping is None:
        mapping = TCULU_TO_UNIFIED
    out = np.zeros_like(raster, dtype=np.uint8)
    for src_cls, dst_cls in mapping.items():
        out[raster == src_cls] = dst_cls
    return out


# ============================================================
# 第三步：判断"建成区"像素
# ============================================================

def detect_urban_pixels(rasters_by_year, pixel_size_m=10.0):
    """
    综合多年份数据，判断每个像素是否属于城市建成区。
    
    规则（按老师要求）：
      1. 居住/商业/工业/公共服务/交通  → 直接是城市
      2. 绿地 → 也算城市（老师明确说绿地视为建设用地）
      3. 裸地/背景(0)：
         - 如果在任意一年变成了城市用地   → 算城市
         - 如果四周全是城市用地           → 算城市（空洞填充）
    
    参数：
        rasters_by_year : dict {year_str: [H, W] uint8}，已映射到统一类别
    
    返回：
        urban_mask : [H, W] bool，True=城市建成区
    """
    years = sorted(rasters_by_year.keys())
    shape = list(rasters_by_year.values())[0].shape
    
    # --- 规则1+2：在任意年份是明确城市用地（含绿地）---
    ever_urban = np.zeros(shape, dtype=bool)
    for y in years:
        r = rasters_by_year[y]
        ever_urban |= np.isin(r, URBAN_WITH_GREEN_CLASSES)
    
    # --- 规则3a：裸地/背景在其他年份变成了城市 ---
    # ever_urban 已经覆盖了这一条（只要有一年是城市就为True）
    
    # --- 规则3b：裸地四周都是城市（填充城市内部空洞）---
    # 用形态学填充：把被城市包围的非城市格子也纳入
    from scipy.ndimage import binary_fill_holes, binary_dilation
    
    # 先对 ever_urban 做一次空洞填充（填充城市内部的孤立非城市格子）
    filled = binary_fill_holes(ever_urban)
    
    # 额外：膨胀一点点，把裸地小斑块也纳入
    # 2km / pixel_size_m = 扩展的像素数（取较小值避免过度膨胀）
    dilate_px = max(1, int(100 / pixel_size_m))  # 约 100m 缓冲
    kernel = np.ones((3, 3), dtype=bool)
    dilated = binary_dilation(filled, structure=kernel, iterations=dilate_px)
    
    # 裸地规则：原本是裸地 AND 被膨胀后的城市覆盖 → 算城市
    bare_land = ~ever_urban  # 从未是城市的格子
    bare_becomes_urban = bare_land & dilated
    
    urban_mask = ever_urban | bare_becomes_urban
    
    print(f"  建成区像素数: {urban_mask.sum():,}  "
          f"({urban_mask.sum() * pixel_size_m**2 / 1e6:.1f} km²)")
    return urban_mask


# ============================================================
# 第四步：空间聚类，把连片城市用地分成独立城市
# ============================================================

def cluster_cities(urban_mask, seed_rasters_by_year, pixel_size_m=10.0,
                   flyover_km=2.0, min_area_km2=5.0):
    """
    用连通区域标记（等价于空间聚类）把城市建成区分成一组一组。
    
    老师的方法：
    - 用居住/商业等"明确城市用地"作为聚类种子
    - 连片的聚为一组
    - 同一组内，距主体超过 flyover_km 的部分算"飞地"，剔除
    - 面积太小（< min_area_km2）的组直接忽略
    
    参数：
        urban_mask      : [H, W] bool，全部建成区像素
        seed_rasters    : dict {year: raster}，用于找聚类种子
        pixel_size_m    : 像素大小（米）
        flyover_km      : 飞地距离阈值（千米）
        min_area_km2    : 最小城市面积（平方千米）
    
    返回：
        city_labels : [H, W] int，0=非城市，1/2/3...=城市编号
        city_info   : list of dict，每个城市的基本信息
    """
    import cv2
    from scipy.ndimage import label as ndlabel
    
    # --- Step 4a：找"明确城市用地"作为种子 ---
    seed_mask = np.zeros_like(urban_mask, dtype=bool)
    for y, r in seed_rasters_by_year.items():
        seed_mask |= np.isin(r, URBAN_SEED_CLASSES)
    
    print(f"  聚类种子像素数: {seed_mask.sum():,}")
    
    # --- Step 4b：对整个建成区做连通区域标记 ---
    # connectivity=8 表示 8 邻接（包含斜角方向）
    n_labels, label_map = cv2.connectedComponents(
        urban_mask.astype(np.uint8), connectivity=8
    )
    print(f"  初始连通区域数: {n_labels - 1}")
    
    # --- Step 4c：计算每个连通区域的面积，过滤太小的 ---
    pixel_area_km2 = (pixel_size_m ** 2) / 1e6
    flyover_px = int(flyover_km * 1000 / pixel_size_m)  # 飞地距离转像素
    
    city_labels = np.zeros_like(label_map, dtype=np.int32)
    city_info = []
    city_id = 0
    
    for lbl in range(1, n_labels):
        region = (label_map == lbl)
        area_km2 = region.sum() * pixel_area_km2
        
        # 面积太小，跳过
        if area_km2 < min_area_km2:
            continue
        
        # 检查这个区域是否有"明确城市用地"种子
        has_seed = (region & seed_mask).sum() > 0
        if not has_seed:
            continue  # 全是绿地/裸地，没有明确城市用地，跳过
        
        # --- Step 4d：处理飞地 ---
        # 找到区域内的"主体"（最大的连片种子区域）
        seed_in_region = region & seed_mask
        n_sub, sub_labels = cv2.connectedComponents(
            seed_in_region.astype(np.uint8), connectivity=8
        )
        if n_sub <= 1:
            # 只有一块，不需要处理飞地
            final_region = region
        else:
            # 找最大子区域作为主体
            sub_sizes = [(sub_labels == i).sum() for i in range(1, n_sub)]
            main_sub_id = np.argmax(sub_sizes) + 1
            main_body = (sub_labels == main_sub_id)
            
            # 计算每个像素到主体的距离
            from scipy.ndimage import distance_transform_edt
            dist_to_main = distance_transform_edt(~main_body) * pixel_size_m / 1000.0
            
            # 只保留距主体 flyover_km 以内的部分
            final_region = region & (dist_to_main <= flyover_km)
        
        # 重新检查面积
        final_area_km2 = final_region.sum() * pixel_area_km2
        if final_area_km2 < min_area_km2:
            continue
        
        city_id += 1
        city_labels[final_region] = city_id
        
        # 记录城市信息（边界框）
        ys, xs = np.where(final_region)
        city_info.append({
            'city_id': city_id,
            'area_km2': round(final_area_km2, 2),
            'bbox': (int(ys.min()), int(xs.min()),
                     int(ys.max()), int(xs.max())),  # (row_min, col_min, row_max, col_max)
            'center_yx': (int(ys.mean()), int(xs.mean())),
            'pixel_count': int(final_region.sum()),
        })
    
    print(f"  提取城市数量: {city_id}  （最小面积={min_area_km2}km²，飞地阈值={flyover_km}km）")
    return city_labels, city_info


# ============================================================
# 第五步：聚合到 100m 并保存每个城市
# ============================================================

def aggregate_to_100m(raster_10m):
    """
    把 10m 栅格聚合到 100m（取 10×10 块内众数）。
    """
    H, W = raster_10m.shape
    H_out = H // 10
    W_out = W // 10
    raster = raster_10m[:H_out * 10, :W_out * 10]
    flat = raster.reshape(H_out, 10, W_out, 10).transpose(0, 2, 1, 3).reshape(H_out, W_out, 100)
    
    try:
        from scipy.stats import mode as scipy_mode
        result = scipy_mode(flat, axis=2, keepdims=False)
        return result.mode.astype(np.uint8)
    except Exception:
        # 手动众数（较慢但稳定）
        out = np.zeros((H_out, W_out), dtype=np.uint8)
        for i in range(H_out):
            for j in range(W_out):
                vals, cnts = np.unique(flat[i, j], return_counts=True)
                out[i, j] = vals[cnts.argmax()]
        return out


def save_city_data(city_info, city_labels, rasters_by_year,
                   output_dir, pixel_size_m=10.0, aggregate=True):
    """
    把每个城市的数据切出来，保存成工具箱期望的格式。
    
    输出结构（与老师的 UrbanAnalysis 完全兼容）：
        output_dir/
          city_001/
            1984/
              landuse.npy   [H, W] uint8
              roads.npy     [H, W] uint8（用交通类代替）
              mask.npy      [H, W] bool
            1995/
              ...
          city_002/
            ...
          cities_index.txt  （城市编号和大小汇总）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    target_pixel_size = 100.0  # 目标分辨率（米）
    do_aggregate = aggregate and (pixel_size_m < target_pixel_size)
    scale = int(target_pixel_size / pixel_size_m) if do_aggregate else 1
    
    index_lines = ["city_id,area_km2,bbox_row_min,bbox_col_min,bbox_row_max,bbox_col_max"]
    
    for info in city_info:
        cid = info['city_id']
        r0, c0, r1, c1 = info['bbox']
        city_dir = output_dir / f"city_{cid:04d}"
        city_dir.mkdir(exist_ok=True)
        
        # 建城市的 mask（只保留属于该城市的像素）
        city_mask_full = (city_labels == cid)
        
        for year, raster in rasters_by_year.items():
            year_dir = city_dir / str(year)
            year_dir.mkdir(exist_ok=True)
            
            # 按边界框切出这个城市的区域
            lu_crop = raster[r0:r1+1, c0:c1+1]
            mask_crop = city_mask_full[r0:r1+1, c0:c1+1]
            
            # 把不属于该城市的格子置为 0
            lu_masked = lu_crop.copy()
            lu_masked[~mask_crop] = 0
            
            if do_aggregate:
                lu_final = aggregate_to_100m(lu_masked)
                # mask 也聚合：10x10 块内超过一半是城市才算城市
                mask_float = mask_crop.astype(np.float32)
                H_out = mask_float.shape[0] // 10
                W_out = mask_float.shape[1] // 10
                mask_blocks = mask_float[:H_out*10, :W_out*10].reshape(
                    H_out, 10, W_out, 10).transpose(0, 2, 1, 3).reshape(H_out, W_out, 100)
                mask_final = (mask_blocks.mean(axis=2) >= 0.3)
            else:
                lu_final = lu_masked
                mask_final = mask_crop
            
            # 道路：用"交通市政"类（编码=5）作为道路代理（和老师代码一致）
            roads_final = (lu_final == 5).astype(np.uint8)
            
            # 保存
            np.save(year_dir / 'landuse.npy', lu_final.astype(np.uint8))
            np.save(year_dir / 'roads.npy', roads_final)
            np.save(year_dir / 'mask.npy', mask_final)
        
        index_lines.append(f"{cid},{info['area_km2']},{r0},{c0},{r1},{c1}")
        print(f"  城市 {cid:04d} 已保存: {info['area_km2']:.1f} km²  → {city_dir}")
    
    # 写索引文件
    with open(output_dir / 'cities_index.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(index_lines))
    print(f"\n城市索引已保存: {output_dir / 'cities_index.txt'}")


# ============================================================
# 第六步（可选）：针对北京，直接按经纬度裁剪，跑现有分析
# ============================================================

def extract_beijing(tculu_path, output_dir,
                    bbox_latlon=(39.75, 116.15, 40.15, 116.65)):
    """
    专门提取北京区域（按经纬度裁剪），同时保存为 NPY 和 GeoTIFF。
    bbox_latlon: (min_lat, min_lon, max_lat, max_lon)
    """
    print(f"\n提取北京区域: {bbox_latlon}")

    import numpy as np
    from pathlib import Path
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds, transform as window_transform
    from rasterio.transform import Affine

    with rasterio.open(str(tculu_path)) as src:
        # 经纬度 bbox -> 栅格坐标系 bbox
        bbox_proj = transform_bounds(
            'EPSG:4326', src.crs,
            bbox_latlon[1], bbox_latlon[0],
            bbox_latlon[3], bbox_latlon[2]
        )

        window = from_bounds(*bbox_proj, transform=src.transform)
        clipped = src.read(1, window=window).astype(np.uint8)

        # 裁剪后的地理参考
        clipped_transform = window_transform(window, src.transform)
        clipped_crs = src.crs
        pixel_size_m = abs(src.transform.a)

    print(f"  按窗口读取成功: {clipped.shape}, 像素={pixel_size_m:.1f}m")

    # 类别重映射
    remapped = remap_to_unified(clipped)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 如果原始是 10m，则聚合到 100m
    if pixel_size_m < 100:
        lu_100m = aggregate_to_100m(remapped)

        # 100m 的 transform = 原 transform 基础上放大 10 倍
        scale_factor = int(round(100 / pixel_size_m))   # 通常是 10
        out_transform = clipped_transform * Affine.scale(scale_factor, scale_factor)
    else:
        lu_100m = remapped
        out_transform = clipped_transform

    roads_100m = (lu_100m == 5).astype(np.uint8)
    mask_100m = (lu_100m > 0).astype(np.uint8)

    # 保存 NPY
    np.save(out / 'landuse.npy', lu_100m.astype(np.uint8))
    np.save(out / 'roads.npy', roads_100m.astype(np.uint8))
    np.save(out / 'mask.npy', mask_100m.astype(bool))

    # 保存 GeoTIFF
    height, width = lu_100m.shape

    with rasterio.open(
        out / 'landuse.tif',
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=lu_100m.dtype,
        crs=clipped_crs,
        transform=out_transform,
        compress='lzw'
    ) as dst:
        dst.write(lu_100m, 1)

    with rasterio.open(
        out / 'roads.tif',
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=roads_100m.dtype,
        crs=clipped_crs,
        transform=out_transform,
        compress='lzw'
    ) as dst:
        dst.write(roads_100m, 1)

    with rasterio.open(
        out / 'mask.tif',
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=mask_100m.dtype,
        crs=clipped_crs,
        transform=out_transform,
        compress='lzw'
    ) as dst:
        dst.write(mask_100m, 1)

    print(f"  北京数据已保存: {out}")
    print(f"  生成文件: landuse.npy / roads.npy / mask.npy / landuse.tif / roads.tif / mask.tif")
    print(f"  shape={lu_100m.shape}")
    return lu_100m


# ============================================================
# 主函数：完整的全国城市提取流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='从全国 TCULU 数据中提取各城市用地数据'
    )
    parser.add_argument('--tculu-dir', required=True,
                        help='TCULU GeoTIFF 目录，包含 TCULU_{year}.tif')
    parser.add_argument('--output-dir', required=True,
                        help='输出目录，每个城市一个子文件夹')
    parser.add_argument('--years', nargs='+', type=int,
                        default=[1984, 1995, 2005, 2015],
                        help='要处理的年份，例如 1984 1995 2005 2015')
    parser.add_argument('--flyover-km', type=float, default=2.0,
                        help='飞地距离阈值（千米），超过此距离视为飞地，默认 2.0')
    parser.add_argument('--min-area-km2', type=float, default=5.0,
                        help='最小城市面积（平方千米），默认 5.0')
    parser.add_argument('--no-aggregate', action='store_true',
                        help='不做 10m→100m 聚合（数据已经是 100m 时使用）')
    parser.add_argument('--beijing-only', action='store_true',
                        help='只提取北京（用于测试，直接按经纬度裁剪）')
    parser.add_argument('--beijing-bbox', nargs=4, type=float,
                        default=[39.75, 116.15, 40.15, 116.65],
                        help='北京裁剪范围 min_lat min_lon max_lat max_lon')
    args = parser.parse_args()
    
    tculu_dir = Path(args.tculu_dir)
    output_dir = Path(args.output_dir)
    
    print("="*70)
    print("TCULU 城市提取工具")
    print("="*70)
    print(f"TCULU 目录: {tculu_dir}")
    print(f"输出目录:   {output_dir}")
    print(f"年份:       {args.years}")
    print(f"飞地阈值:   {args.flyover_km} km")
    print(f"最小面积:   {args.min_area_km2} km²")
    print()
    
    # ---- 只跑北京模式 ----
    if args.beijing_only:
        print("【北京专项提取模式】")
        for year in args.years:
            tif_path = tculu_dir / f'china_land_use{year}.tif'
            if not tif_path.exists():
                print(f"  跳过 {year}：文件不存在 {tif_path}")
                continue
            print(f"\n处理 {year}...")
            out_year_dir = output_dir / 'beijing' / str(year)
            extract_beijing(tif_path, out_year_dir,
                            bbox_latlon=tuple(args.beijing_bbox))
        print("\n✅ 北京数据提取完成")
        print(f"   数据在: {output_dir / 'beijing'}")
        print("   接下来可以用老师的 UrbanAnalysis 直接分析这个目录")
        return
    
    # ---- 全国城市提取模式 ----
    # 1. 读取所有年份的数据
    print("【第1步】读取所有年份的 TCULU 数据")
    rasters_raw = {}     # 原始编码
    rasters_unified = {} # 统一编码
    pixel_size_m = 10.0
    
    for year in args.years:
        tif_path = tculu_dir / f'china_land_use{year}.tif'
        if not tif_path.exists():
            print(f"  跳过 {year}：{tif_path} 不存在")
            continue
        print(f"\n  读取 {year}: {tif_path}")
        raw, meta = read_tculu_tif(tif_path)
        pixel_size_m = meta.get('pixel_size_m', 10.0)
        rasters_raw[str(year)] = raw
        rasters_unified[str(year)] = remap_to_unified(raw)
    
    if not rasters_unified:
        print("错误：没有找到任何 TCULU 文件，请检查 --tculu-dir 路径")
        sys.exit(1)
    
    print(f"\n已加载 {len(rasters_unified)} 个年份: {list(rasters_unified.keys())}")
    print(f"像素大小: {pixel_size_m} m")
    
    # 2. 判断建成区
    print("\n【第2步】识别城市建成区像素")
    urban_mask = detect_urban_pixels(rasters_unified, pixel_size_m)
    
    # 3. 聚类，分出各城市
    print("\n【第3步】空间聚类，分割城市")
    city_labels, city_info = cluster_cities(
        urban_mask,
        seed_rasters_by_year=rasters_unified,
        pixel_size_m=pixel_size_m,
        flyover_km=args.flyover_km,
        min_area_km2=args.min_area_km2,
    )
    
    print(f"\n共识别出 {len(city_info)} 个城市")
    print(f"前10个城市面积：")
    for info in sorted(city_info, key=lambda x: -x['area_km2'])[:10]:
        print(f"  城市{info['city_id']:04d}: {info['area_km2']:.1f} km²")
    
    # 4. 保存每个城市的数据
    print("\n【第4步】保存各城市数据")
    save_city_data(
        city_info=city_info,
        city_labels=city_labels,
        rasters_by_year=rasters_unified,
        output_dir=output_dir,
        pixel_size_m=pixel_size_m,
        aggregate=not args.no_aggregate,
    )
    
    # 5. 打印完成信息
    print("\n" + "="*70)
    print("✅ 全部完成！")
    print(f"   共提取 {len(city_info)} 个城市")
    print(f"   数据保存在: {output_dir}")
    print()
    print("接下来使用老师的 UrbanAnalysis 分析某个城市：")
    print("  from urban_update_toolkit import UrbanAnalysis")
    print(f"  ua = UrbanAnalysis(")
    print(f"      city='city_0001',")
    print(f"      data_dir='{output_dir}/city_0001',")
    print(f"      years={[str(y) for y in args.years]},")
    print(f"      cell_size_m=100,")
    print(f"      output_dir='./outputs/city_0001'")
    print(f"  )")
    print(f"  ua.run_all(...)")
    print("="*70)


if __name__ == '__main__':
    main()
