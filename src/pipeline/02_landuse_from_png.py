"""
02_landuse_from_png.py — 用地 PNG → 100m 栅格 npy (9 年统一版)
==============================================================

输入: 每年一张 .png (TCULU/EULUC 等用地分类成果)
      支持的命名 (按优先级):
        {CITY_NAME.lower()}_{year}.png      # 例: beijing_1984.png
        北京_{year}.png                      # 中文名 (CITY_NAME=='Beijing' 时映射)
        {year}.png                          # 仅年份
      可选: 同名 .pgw (世界文件: 像素分辨率 + 左上角米坐标)

输出 (per year):
  data/lu_{year}.npy        分类后栅格 (H_R, W_R), 值 0~6 (按 LU_NAMES) + 255 (AOI 外)
  data/lu_clean_{year}.npy  默认副本; 如果之后跑步骤 03, 会被覆盖为去孤岛版

降采样:
  - 若有 .pgw, 按 target_m / px_m_orig 比例
  - 若无 .pgw, 按 PNG 原 shape / target shape 推断 (取较大整数比)

用法:
  python 02_landuse_from_png.py <input_dir>           # 批处理
  python 02_landuse_from_png.py <single_png_path>     # 单文件 (年份从文件名解析)
"""
import sys
import re
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA, YEARS, CITY_NAME, H_R, W_R


# ============================================================
# 颜色锚点 (适配 TCULU 配色)
# ============================================================
ANCHORS = [
    # RGB                cls   label
    ((240, 240, 240),    0,   'blank'),
    ((238, 238, 238),    0,   'blank'),
    ((255, 255, 255),    0,   'blank'),       # 纯白也算 blank
    ((240, 240, 208),   -1,   'outside'),     # AOI 外
    ((245, 248, 220),   -1,   'outside'),
    ((245, 248, 221),   -1,   'outside'),
    ((249, 168,  37),    1,   'residential'),
    ((255, 255,  45),    1,   'residential'),
    ((227,  26,  28),    2,   'commercial'),
    ((240,   0,   0),    2,   'commercial'),
    ((254,   0,   0),    2,   'commercial'),
    ((  0,  99, 128),    3,   'public_service'),
    (( 54,  98, 123),    3,   'public_service'),
    ((109,  76,  65),    4,   'industrial'),
    ((187, 150, 116),    4,   'industrial'),
    ((186, 150, 116),    4,   'industrial'),
    (( 97,  97,  97),    5,   'road'),
    ((183, 183, 183),    5,   'road'),
    (( 56, 142,  60),    6,   'green'),
    ((  0, 255,   0),    6,   'green'),       # 亮绿 (公园)
    (( 51, 142, 192),    6,   'water'),
    (( 50, 142, 191),    6,   'water'),
]
ANC_RGB = np.array([a[0] for a in ANCHORS], dtype=np.int32)
ANC_CLS = np.array([a[1] for a in ANCHORS], dtype=np.int32)


CITY_ALIASES = {
    'beijing': ['beijing', '北京'],
    'shanghai': ['shanghai', '上海'],
    'shenzhen': ['shenzhen', '深圳'],
    'guangzhou': ['guangzhou', '广州'],
}


def _city_prefixes():
    aliases = CITY_ALIASES.get(CITY_NAME.lower(), [CITY_NAME.lower()])
    return aliases


def find_png_for_year(input_dir, year):
    """按多个命名模式找 PNG."""
    inp = Path(input_dir)
    candidates = []
    for prefix in _city_prefixes():
        candidates.append(inp / f'{prefix}_{year}.png')
    candidates.append(inp / f'{year}.png')
    for c in candidates:
        if c.exists():
            return c
    return None


def parse_year_from_filename(p: Path):
    """从单文件名提取年份 (4 位数字)."""
    m = re.search(r'(?<!\d)(19\d{2}|20\d{2})(?!\d)', p.stem)
    return int(m.group(1)) if m else None


def read_pgw(pgw_path):
    """世界文件 → 像素分辨率 (米)"""
    with open(pgw_path) as f:
        v = [float(x.strip()) for x in f.readlines()[:6]]
    return v[0]


def png_to_landuse(png_path, target_h=H_R, target_w=W_R, target_m=100.0):
    """PNG → (H_R, W_R) 分类栅格 (值 0~6, AOI 外标 255)."""
    pgw_path = png_path.with_suffix('.pgw')
    img = Image.open(png_path).convert('RGB')
    W, H = img.size
    
    if pgw_path.exists():
        px_m_orig = read_pgw(pgw_path)
        ds = max(int(round(target_m / px_m_orig)), 1)
    else:
        # 无 pgw, 按目标尺寸反推
        ds_h = max(int(round(H / target_h)), 1)
        ds_w = max(int(round(W / target_w)), 1)
        ds = max(ds_h, ds_w)
    
    Wd, Hd = W // ds, H // ds
    print(f"    源 ({H},{W}) → 降采样 ds={ds} → ({Hd},{Wd})")
    small = img.resize((Wd, Hd), resample=Image.NEAREST)
    arr = np.array(small).astype(np.int32)
    flat = arr.reshape(-1, 3)
    
    # 找最近的颜色锚点
    dists = np.sum((flat[:, None, :] - ANC_RGB[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(dists, axis=1)
    cls = ANC_CLS[nearest].reshape(Hd, Wd)
    # -1 (outside AOI) → 255
    out = np.where(cls == -1, 255, cls).astype(np.int16)
    
    # 裁剪/填充到 (target_h, target_w)
    if out.shape != (target_h, target_w):
        out2 = np.full((target_h, target_w), 255, dtype=np.int16)
        h, w = min(out.shape[0], target_h), min(out.shape[1], target_w)
        out2[:h, :w] = out[:h, :w]
        out = out2
    
    return out.astype(np.uint8)


def align_aoi_with_existing(lu, year):
    """把 lu 的 AOI 边界跟其它年份对齐 (用 union 'outside' 区域作 AOI 外掩码)."""
    refs = []
    for y in YEARS:
        if y == year: continue
        p = DATA / f'lu_clean_{y}.npy'
        if p.exists():
            refs.append(np.load(p))
    if not refs:
        return lu
    # 所有参考年都 ==255 的位置 → 一定是 AOI 外
    all_outside = np.ones_like(lu, dtype=bool)
    for r in refs:
        all_outside &= (r == 255)
    out = lu.copy()
    out[all_outside] = 255
    return out


def process_one_year(png_path, year, force=False):
    out_lu = DATA / f'lu_{year}.npy'
    out_clean = DATA / f'lu_clean_{year}.npy'
    if out_clean.exists() and not force:
        print(f"  {year}: lu_clean_{year}.npy 已存在, 跳过 (--force 强制重算)")
        return False
    print(f"  {year}: 处理 {png_path.name} ...")
    cls = png_to_landuse(png_path)
    cls = align_aoi_with_existing(cls, year)
    np.save(out_lu, cls)
    # 默认把 lu 当作 lu_clean (步骤 03 跑过会覆盖); 这样下游 06+ 可以直接用
    np.save(out_clean, cls)
    
    # 类别分布
    LU_NAMES = {0:'空地', 1:'居住', 2:'商业', 3:'公共服务',
                4:'工业仓储', 5:'道路市政', 6:'绿地水域', 255:'AOI 外'}
    unique, counts = np.unique(cls, return_counts=True)
    total = cls.size
    print(f"    类别分布:")
    for u, c in zip(unique, counts):
        n = LU_NAMES.get(int(u), f'unknown {u}')
        print(f"      [{u:>3}] {n:<8}: {c:>7,} ({100*c/total:5.2f}%)")
    print(f"    → {out_lu}, {out_clean}")
    return True


def main(arg, force=False):
    arg = Path(arg)
    print("=" * 60)
    print("步骤 02: 用地 PNG → 100m 栅格")
    print(f"  CITY_NAME = {CITY_NAME}")
    print(f"  YEARS = {YEARS}")
    print("=" * 60)
    
    if arg.is_dir():
        print(f"\n输入目录: {arg}")
        n_done = 0
        for y in YEARS:
            png = find_png_for_year(arg, y)
            if png is None:
                print(f"  {y}: ⚠ 未找到 PNG (尝试了 {_city_prefixes()}_{y}.png 和 {y}.png)")
                continue
            if process_one_year(png, y, force=force):
                n_done += 1
        print(f"\n共处理 {n_done} 年.")
    elif arg.is_file():
        # 单文件: 从文件名解析年份
        y = parse_year_from_filename(arg)
        if y is None:
            print(f"❌ 无法从文件名 '{arg.name}' 解析年份")
            sys.exit(1)
        print(f"\n单文件模式, 年份 = {y}")
        process_one_year(arg, y, force=force)
    else:
        print(f"❌ 输入路径不存在: {arg}")
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法:")
        print("  python 02_landuse_from_png.py <input_dir>           # 批处理")
        print("  python 02_landuse_from_png.py <single_png_path>     # 单文件")
        print("  添加 --force 强制重算")
        sys.exit(1)
    force = '--force' in sys.argv
    args = [a for a in sys.argv[1:] if a != '--force']
    main(args[0], force=force)
