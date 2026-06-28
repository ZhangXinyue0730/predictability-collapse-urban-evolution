"""
07_syntax_lut_proxy.py — 句法代理表 (无 OSM 时使用)
====================================================

构建一个 (cell_idx → 80 维句法) 查找表, 用作"代理"句法 — 当步骤 03-07
(OSM 路径) 没法跑时, 这是补救方案.

输入:
  必须存在以下任一文件 (会自动选第一个找到的):
    data/dataset_orig_8year.npz    建议: 原 8 年版数据集 (含真实句法)
    data/dataset.npz               已组装好的当前数据集 (从中提取句法)

  注: 输入数据集必须含 80 维句法列 (mean_R*_*, max_R*_*).

输出:
  data/syntax_lut.npz
    syntax_lut: (H_R*W_R, 80) float32   每个 cell 的 80 维句法
    has_syntax: (H_R*W_R,) bool          该 cell 是否有句法数据
    syntax_cols: (80,) str               列名

策略: 对每个 cell_idx, 取 year_t1 最大 (即最近) 的那条记录的 80 维句法值.
原因: OSM 路网逐年变化很慢, 各年同 cell 的句法值差异 < 5%.

⚠ 局限: 这是代理, 不是真实 2020 句法. 要做完整版需要 OSM shp + 步骤 03-07.
"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config import DATA, H_R, W_R


SYN_START = 7 + 66    # 73 (cell + nb)
SYN_END = SYN_START + 80   # 153
N_CELL = H_R * W_R


def find_input_dataset():
    """优先用原 8 年版, 没就用当前."""
    candidates = [DATA / 'dataset_orig_8year.npz', DATA / 'dataset.npz']
    for c in candidates:
        if c.exists():
            return c
    return None


def main():
    print("=" * 60)
    print("步骤 07_proxy: 提取句法 LUT (代理路径)")
    print("=" * 60)
    
    inp = find_input_dataset()
    if inp is None:
        print("❌ 未找到输入数据集 (dataset_orig_8year.npz / dataset.npz)")
        print("   请先把原 8 年版 dataset.npz 拷贝到 data/dataset_orig_8year.npz")
        sys.exit(1)
    
    print(f"  输入: {inp}")
    d = np.load(inp, allow_pickle=True)
    X = d['X']
    cell_idx = d['cell_idx']
    year_t1 = d['year_t1']
    col_names = d['col_names']
    
    print(f"  数据集: X={X.shape}, 样本={len(cell_idx):,}")
    
    # 检测句法列位置 (默认 [73:153], 但 dataset 列结构可能不同)
    # 假定 cell(7) + nb(66) + syntax(80) + fn(余下)
    if X.shape[1] < SYN_END:
        print(f"❌ 数据集仅 {X.shape[1]} 维, 不足 {SYN_END} 维 (期望 cell+nb+syntax)")
        sys.exit(1)
    
    syn_cols = col_names[SYN_START:SYN_END]
    print(f"  句法列范围 [{SYN_START}:{SYN_END}], 共 {len(syn_cols)} 维")
    print(f"  前 3 句法列名: {list(syn_cols[:3])}")
    
    # 验证这些列名看起来像句法
    syn_keywords = ['NC', 'TD', 'TL', 'MD', 'NACH', 'NAIN', 'Choice', 'Integration']
    matched = sum(1 for c in syn_cols
                  if any(k in str(c) for k in syn_keywords))
    if matched < 50:
        print(f"⚠ 仅 {matched}/80 列名匹配句法关键字, 可能位置错误")
    else:
        print(f"  ✓ {matched}/80 列名包含句法关键字")
    
    # 初始化 LUT
    syntax_lut = np.full((N_CELL, 80), np.nan, dtype=np.float32)
    
    # 按 year_t1 从大到小填充
    years_seen = sorted(np.unique(year_t1).tolist(), reverse=True)
    print(f"\n  按 year_t1 顺序填充 (latest first): {years_seen}")
    
    for y in years_seen:
        m = (year_t1 == y)
        idx_y = cell_idx[m]
        syn_y = X[m, SYN_START:SYN_END].astype(np.float32)
        new_cells = np.isnan(syntax_lut[idx_y, 0])
        idx_to_fill = idx_y[new_cells]
        syntax_lut[idx_to_fill] = syn_y[new_cells]
        n_fill = new_cells.sum()
        n_cum = (~np.isnan(syntax_lut[:, 0])).sum()
        print(f"    {y}: 该年记录 {m.sum():>7,}, 新填 {n_fill:>7,}, 累计 {n_cum:>7,}")
    
    has_syntax = ~np.isnan(syntax_lut[:, 0])
    print(f"\n  最终覆盖: {has_syntax.sum():,} / {N_CELL:,} ({100*has_syntax.mean():.2f}%)")
    
    syntax_lut[~has_syntax] = 0
    
    np.savez_compressed(
        DATA / 'syntax_lut.npz',
        syntax_lut=syntax_lut,
        has_syntax=has_syntax,
        syntax_cols=np.array(syn_cols),
    )
    print(f"\n  → data/syntax_lut.npz")


if __name__ == '__main__':
    main()
