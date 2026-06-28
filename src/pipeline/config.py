"""
config.py — 城市更新预测 (TCULU) Pipeline 配置

所有城市相关参数集中在这里. 移植到其他城市时, 只需修改本文件.
"""
from pathlib import Path

# ============================================================
# 路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # tculu/ 目录
DATA = PROJECT_ROOT / 'data'
FIG = PROJECT_ROOT / 'figs'
DATA.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


# ============================================================
# 城市 — 修改这里以适配新城市
# ============================================================
CITY_NAME = 'Beijing'

# 时间序列 (年份列表, 至少需要 2 个时间点才能算更新)
# v9 年份版本: 加入 2020
YEARS = [1984, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2024]

# 用地栅格尺寸 (像素)
H_R = 612
W_R = 668

# 地理参考: 每像素多少米
PX_M = 99.65   # ≈ 100m × 100m / 格

# 投影坐标参考点 (像素 (0,0) 对应的米坐标, Web Mercator EPSG:3857)
X0_M = 12924263.80111564
Y0_M = 4889246.50836138


# ============================================================
# 用地分类 (TCULU 标准, 也可改为 EULUC / OpenStreetMap landuse)
# ============================================================
LU_NAMES = {
    0: '空地/未利用',
    1: '居住',
    2: '商业',
    3: '公共服务',
    4: '工业仓储',
    5: '道路市政',
    6: '绿地水域',
}
LU_COLORS = {
    0: '#fafafa', 1: '#F9A825', 2: '#E31A1C', 3: '#006380',
    4: '#BB9674', 5: '#B7B7B7', 6: '#388E3C',
}

# 建成区类别 (用于"严格 redev"定义的过滤)
BUILTUP_CLASSES = {1, 2, 3, 4}
# 可被开发/更新的类别 (含空地, 不含道路绿水)
CHANGEABLE_CLASSES = {0, 1, 2, 3, 4}


# ============================================================
# OSM 道路等级
# ============================================================
GRADE_RANK = {
    'motorway': 0, 'motorway_link': 0,
    'trunk': 1, 'trunk_link': 1,
    'primary': 2, 'primary_link': 2,
    'secondary': 3, 'secondary_link': 3,
    'tertiary': 4, 'tertiary_link': 4,
    'unclassified': 5, 'residential': 5,
    'service': 6, 'living_street': 6,
}
MAIN_GRADE = {'motorway', 'motorway_link', 'trunk', 'trunk_link',
              'primary', 'primary_link', 'secondary', 'secondary_link'}
DRIVABLE = set(GRADE_RANK.keys())

# OSM 风格配色
OSM_COLORS = {
    'motorway': '#7f3f00', 'motorway_link': '#7f3f00',
    'trunk': '#cc0000', 'trunk_link': '#cc0000',
    'primary': '#ff8000', 'primary_link': '#ff8000',
    'secondary': '#f4d75e', 'secondary_link': '#f4d75e',
    'tertiary': '#9b59b6', 'tertiary_link': '#9b59b6',
    'unclassified': '#999', 'residential': '#999',
    'service': '#bbb', 'living_street': '#bbb',
}


# ============================================================
# 道路命中阈值 (按等级)
# ============================================================
def threshold_for_fclass(fc):
    """polyline 在历史用地图上 mask 命中比例阈值, 按 fclass 分级"""
    if fc in ('motorway', 'motorway_link', 'trunk', 'trunk_link'): return 0.30
    elif fc in ('primary', 'primary_link', 'secondary', 'secondary_link'): return 0.35
    elif fc in ('tertiary', 'tertiary_link'): return 0.40
    elif fc in ('unclassified', 'residential'): return 0.45
    return 0.50


# ============================================================
# 空间句法参数
# ============================================================
# 计算半径 (米)
SYNTAX_RADII = [1000, 1500, 2000, 3000, 4000, 5000]
# 哪些半径计算 NACH/Choice (耗时较多, 通常只需 1000 和 2000)
SYNTAX_NACH_RADII = [1000, 2000]
# Riondato 2014 采样 betweenness 的 source 数量
SYNTAX_BETWEENNESS_K = 3000


# ============================================================
# 用地周边特征参数
# ============================================================
# 多半径周边特征
NB_RADII_M = [1000, 1500, 2000, 3000, 4000, 5000]


# ============================================================
# KDE 句法扩散参数 (替代旧的硬切赋值)
# ============================================================
# 高斯模糊 sigma (像素), 5px ≈ 500m
KDE_SIGMA_PX = 5


# ============================================================
# 功能整体特征参数
# ============================================================
FUNCTIONAL_RADII = [500, 1000, 2000]  # 簇分析半径


# ============================================================
# 训练划分 (时间外推, 修改这里以改变 train/val/test)
# ============================================================
# v9 年份版本: 8 对 (1984→1990, ..., 2015→2020, 2020→2024)
TRAIN_YEARS = [1984, 1990, 1995, 2000]   # t1 in 这些年 → 训练 (4 对)
VAL_YEARS = [2005]                       # → 验证 (1 对)
TEST_YEARS = [2010, 2015, 2020]          # → 测试 (3 对, 最近三对)

# 更新定义: 'strict_redev' (建成区→建成区, 类别变了)
#           'any_change' (任何变化, 含新建)
UPDATE_DEFINITION = 'dynamic_t1_extent'

# Dynamic t1 built-up extent definition
DYNAMIC_CORE_CLASSES = {1, 2, 3, 4}
DYNAMIC_LINK_DISTANCE_M = 2000.0
DYNAMIC_MAX_HOLE_AREA_KM2 = 4.0



# ============================================================
# 模型参数
# ============================================================
HISTGBM_PARAMS = dict(
    max_iter=80,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
)
