"""
10_assemble_dataset.py — 训练数据集组装 (统一版, 9 年支持)
==========================================================

把以下特征拼成最终特征矩阵:
  A. cell 自身用地 one-hot         (7 dim)
  B. 周边用地 (6 半径 × 11)        (66 dim)
  C. KDE 衰减句法                  (80 dim)
  D. 功能整体                      (14 dim)
  -------------------------------------
  总计                             (167 dim)

句法特征源 (按优先级自动选择):
  1) data/cell_syntax_decay_{year}.npz       (步骤 07 输出, OSM 路网衰减句法)
  2) data/syntax_lut.npz                     (步骤 07b 输出, 代理 LUT)
  → 若两者都缺失, 报错并停止.

样本筛选 (动态前期建成区口径):
  1) 步骤 09.5 以 t1 年居住、商业、公共服务、工业仓储为核心，
     连接约 2 km 内片区，识别 t1 动态建成区边界。
  2) 位于 t1 建成区边界内、且 t1 与 t2 均为非空地类别 1-6 的
     cell 才进入比较样本。
  3) 上述有效样本中，t1 与 t2 用地类别不同即标记为更新。

建成区边界之外、属于 t2 新建成区的部分由步骤 13 统计为扩张。

输出:
  data/dataset.npz
    X         (N, 167) float16/32   特征
    y         (N,) int8             t2 类别 (0~6)
    is_update (N,) bool             是否更新
    year_t1   (N,) int16
    cell_idx  (N,) int32
    cell_xy   (N, 2) float32
    col_names (167,) str
"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config import DATA, YEARS, H_R, W_R, NB_RADII_M

N_CELL = H_R * W_R


# ============================================================
# 句法源选择
# ============================================================
def have_real_syntax(year):
    return (DATA / f'cell_syntax_decay_{year}.npz').exists()


def have_lut():
    return (DATA / 'syntax_lut.npz').exists()


def detect_syntax_source(verbose=True):
    """返回 ('real', None) 或 ('lut', lut_data)."""
    real_years = [y for y in YEARS if have_real_syntax(y)]
    if len(real_years) == len(YEARS):
        if verbose:
            print(f"  ✓ 句法源: REAL (cell_syntax_decay_*.npz, 全 {len(YEARS)} 年都有)")
        return ('real', None)
    if have_lut():
        missing = [y for y in YEARS if y not in real_years]
        if verbose:
            if real_years:
                print(f"  ⚠ 句法源: 部分 REAL ({real_years}) + LUT 代理 (缺 {missing})")
            else:
                print(f"  ⚠ 句法源: LUT 代理 (无任何 cell_syntax_decay_*.npz)")
        lut = np.load(DATA / 'syntax_lut.npz')
        return ('lut', lut)
    raise FileNotFoundError(
        "未找到任何句法源:\n"
        "  - data/cell_syntax_decay_{year}.npz (步骤 07 输出)\n"
        "  - data/syntax_lut.npz (步骤 07b 输出)\n"
        "请先跑 03→04→05→07 (含 OSM) 或 07b (代理 LUT)."
    )


# ============================================================
# 句法加载
# ============================================================
def load_syntax_for_year(year, mode, lut):
    """统一返回 (N_CELL, 80) 句法矩阵."""
    if mode == 'real':
        z = np.load(DATA / f'cell_syntax_decay_{year}.npz')
        return z['feat'].astype(np.float32)
    # lut 模式: 优先用 real (若该年存在), 其次用 LUT
    if have_real_syntax(year):
        z = np.load(DATA / f'cell_syntax_decay_{year}.npz')
        return z['feat'].astype(np.float32)
    return lut['syntax_lut'].astype(np.float32)


def load_syntax_col_names(mode, lut):
    """80 维句法的列名."""
    if mode == 'real':
        # 从任意一年读 col_names
        for y in YEARS:
            p = DATA / f'cell_syntax_decay_{y}.npz'
            if p.exists():
                return list(np.load(p)['col_names'])
        raise RuntimeError("real 模式但找不到任何 cell_syntax_decay")
    # lut 模式
    return list(lut['syntax_cols'])


def load_syntax_mask(year, mode, lut):
    """每个 cell 是否有句法值. shape (N_CELL,) bool."""
    if mode == 'real':
        return np.ones(N_CELL, dtype=bool)
    if have_real_syntax(year):
        return np.ones(N_CELL, dtype=bool)
    return lut['has_syntax']


# ============================================================
# 特征装配
# ============================================================
def build_features(year, mode, lut):
    """组装一个年份所有 cell 的 167 维特征."""
    # A. cell one-hot (255 → 0)
    lu_raw = np.load(DATA / f'lu_{year}.npy')
    lu = lu_raw.copy()
    lu[lu == 255] = 0
    lu = lu.flatten().astype(np.int8)
    cur_oh = np.eye(7, dtype=np.float32)[lu.clip(0, 6)]
    
    # B. 周边
    nb = np.load(DATA / f'cell_features_{year}.npz')['feat'].astype(np.float32)
    nb = np.nan_to_num(nb, nan=0.0, posinf=0.0, neginf=0.0)
    
    # C. 句法
    syn = load_syntax_for_year(year, mode, lut)
    syn = np.nan_to_num(syn, nan=0.0, posinf=0.0, neginf=0.0)
    
    # D. 功能整体
    fn = np.load(DATA / f'cell_functional_{year}.npz')['feat'].astype(np.float32)
    fn = np.nan_to_num(fn, nan=0.0, posinf=0.0, neginf=0.0)
    
    return np.concatenate([cur_oh, nb, syn, fn], axis=1).astype(np.float32)


def make_col_names(mode, lut):
    cols = []
    cols += [f'cell_class_{i}' for i in range(7)]
    for R in NB_RADII_M:
        for c in range(7):
            cols.append(f'nb_R{R}_class_{c}')
        cols.append(f'nb_R{R}_builtup')
        cols.append(f'nb_R{R}_road')
        cols.append(f'nb_R{R}_shannon')
        cols.append(f'nb_R{R}_dominant')
    cols += load_syntax_col_names(mode, lut)
    fn_z = np.load(DATA / f'cell_functional_{YEARS[-1]}.npz')
    cols += list(fn_z['col_names'])
    return cols


# ============================================================
# 主流程
# ============================================================
def main(save_float16=True):
    print("=" * 60)
    print("步骤 10: 数据集组装 (9 年统一版)")
    print(f"  YEARS = {YEARS}  ({len(YEARS)-1} 对)")
    print("  更新定义: 前一期动态建成区内、前后均非空地的类别变化")
    print("=" * 60)
    
    mode, lut = detect_syntax_source()
    
    col_names = make_col_names(mode, lut)
    print(f"  特征列总数: {len(col_names)}")
    
    pairs = list(zip(YEARS[:-1], YEARS[1:]))
    X_all, y_all, upd_all, year_all, idx_all, xy_all = [], [], [], [], [], []
    
    rows = np.arange(H_R).reshape(-1, 1).repeat(W_R, axis=1).flatten()
    cols_grid = np.arange(W_R).reshape(1, -1).repeat(H_R, axis=0).flatten()
    cell_xy_full = np.stack([cols_grid, rows], axis=1).astype(np.float32)
    
    for t1, t2 in pairs:
        print(f"\n  {t1}→{t2}: ...", flush=True)
        X_t1 = build_features(t1, mode, lut)
        lu1 = np.load(DATA / f'lu_{t1}.npy').copy()
        lu2 = np.load(DATA / f'lu_{t2}.npy').copy()
        lu1[lu1 == 255] = 0; lu2[lu2 == 255] = 0
        lu1, lu2 = lu1.flatten(), lu2.flatten()
        
        extent_path = DATA / f'builtup_extent_dynamic_{t1}.npy'
        if not extent_path.exists():
            raise FileNotFoundError(
                f"缺少动态建成区边界: {extent_path}\n"
                "请先运行 pipeline/09.5_dynamic_builtup_extent.py"
            )
        extent = np.load(extent_path).astype(bool).flatten()

        # 空地/未利用地属于非建设用地。涉及类别 0 的像元不进入更新
        # 样本，避免把开发、清退或分类缺失误计为存量城市更新。
        nonvacant_pair = (lu1 >= 1) & (lu1 <= 6) & (lu2 >= 1) & (lu2 <= 6)
        mk = extent & nonvacant_pair
        excluded_vacant = int(extent.sum() - mk.sum())
        print(f"    涉及空地排除: {excluded_vacant:,} cells")
        
        # 句法掩码 (LUT 模式下可能有缺值的 cell)
        syn_mask = load_syntax_mask(t1, mode, lut)
        mk_with_syn = mk & syn_mask
        n_drop = mk.sum() - mk_with_syn.sum()
        if n_drop > 0:
            print(f"    缺句法丢弃: {n_drop:,} cells "
                  f"({100*n_drop/max(1, mk.sum()):.2f}% of mask)")
        mk = mk_with_syn
        
        upd = (lu1 != lu2) & mk
        upd_in_mk = upd[mk]
        
        n = mk.sum()
        n_upd = upd_in_mk.sum()
        print(f"    样本 {n:,} | 更新 {n_upd:,} ({100*n_upd/max(n,1):.1f}%)")
        
        idx = np.where(mk)[0]
        X_all.append(X_t1[mk])
        y_all.append(lu2[mk].astype(np.int8))
        upd_all.append(upd_in_mk.astype(bool))
        year_all.append(np.full(n, t1, dtype=np.int16))
        idx_all.append(idx.astype(np.int32))
        xy_all.append(cell_xy_full[mk])
    
    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    upd = np.concatenate(upd_all, axis=0)
    year = np.concatenate(year_all, axis=0)
    idx = np.concatenate(idx_all, axis=0)
    xy = np.concatenate(xy_all, axis=0)
    
    print(f"\n总样本: {len(X):,}")
    print(f"  X shape: {X.shape}, dtype: {X.dtype}")
    print(f"  更新事件: {upd.sum():,} ({100*upd.sum()/len(X):.2f}%)")
    print(f"\n  按 year_t1 分布:")
    for y_t1 in YEARS[:-1]:
        m = (year == y_t1)
        n_m = m.sum()
        n_u = upd[m].sum()
        print(f"    {y_t1}: 样本 {n_m:>7,} | 更新 {n_u:>6,} "
              f"({100*n_u/max(1, n_m):.2f}%)")
    
    if save_float16:
        X = X.astype(np.float16)
    
    np.savez_compressed(
        DATA / 'dataset.npz',
        X=X, y=y, is_update=upd,
        year_t1=year, cell_idx=idx, cell_xy=xy,
        col_names=np.array(col_names),
        scope_definition=np.array('dynamic_t1_builtup_extent'),
        update_definition=np.array(
            'nonvacant_class_change_inside_dynamic_t1_extent'
        ),
    )
    print(f"\n→ data/dataset.npz")


if __name__ == '__main__':
    main()
