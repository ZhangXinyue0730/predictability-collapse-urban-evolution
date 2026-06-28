"""
12_shap_analysis.py — SHAP 风格特征重要性 + PDP (统一版, 9 年支持)
==================================================================

由于 sklearn HistGradientBoostingClassifier 不直接支持 shap 库, 我们用
permutation importance (经典 SHAP 替代):
  对每个特征 j: shuffle X[:,j] → 重新预测 → ΔAUC = base_AUC − permuted_AUC

支持断点续算 (data/shap_checkpoint.pkl), 默认 8000 子样本, 167 维全跑.

输出:
  data/shap_importance.pkl
    {best_config, cols_idx, col_names, importances, test_auc}
  figs/shap_top25.png       Top 25 特征条形图
  figs/shap_group.png       按特征族分组总贡献
  figs/shap_pdp_top6.png    Top 6 PDP 边际效应

用法:
  python 12_shap_analysis.py            # 默认: M7 (全特征)
  python 12_shap_analysis.py M3         # 指定配置
  python 12_shap_analysis.py --replot   # 仅重绘 (从已有 pkl)
"""
import numpy as np
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (DATA, FIG, TRAIN_YEARS, TEST_YEARS, HISTGBM_PARAMS, NB_RADII_M)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Zen Hei',
                                            'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


N_PERM_SAMPLE = 8000
CKPT_EVERY = 20


# ============================================================
# 颜色编码
# ============================================================
def feat_color(name):
    if name.startswith('cell_'): return '#0d6efd'
    if 'NACH' in name or 'Choice' in name: return '#dc3545'
    if 'Integration' in name: return '#e83e8c'
    if 'NAIN' in name: return '#9c27b0'
    if any(k in name for k in ['_NC', '_TD', '_TL', '_MD']): return '#ff9800'
    if 'cluster' in name: return '#198754'
    if name.startswith('R') and any(k in name for k in ['entropy', 'purity', 'boundary', 'builtup']):
        return '#20c997'
    if 'neighbor_diversity' in name: return '#20c997'
    if name.startswith('nb_') and 'shannon' in name: return '#fd7e14'
    if name.startswith('nb_'): return '#6f42c1'
    return '#6c757d'


# ============================================================
# 配置选择
# ============================================================
def get_config_cols(config_name, n_total):
    n_cell = 7
    n_nb = 11 * len(NB_RADII_M)
    n_syn = 80
    fn_dim = n_total - n_cell - n_nb - n_syn
    
    COLS_CELL = list(range(n_cell))
    COLS_NB = list(range(n_cell, n_cell + n_nb))
    COLS_SYN = list(range(n_cell + n_nb, n_cell + n_nb + n_syn))
    COLS_FN = list(range(n_cell + n_nb + n_syn, n_total))
    
    CONFIGS = {
        'M0 cell only':         COLS_CELL,
        'M1 cell+Neighborhood':         COLS_CELL + COLS_NB,
        'M2 cell+SyntaxDecay':      COLS_CELL + COLS_SYN,
        'M3 cell+Functional':      COLS_CELL + COLS_FN,
        'M4 cell+Neighborhood+Syntax':     COLS_CELL + COLS_NB + COLS_SYN,
        'M5 cell+Neighborhood+Functional':     COLS_CELL + COLS_NB + COLS_FN,
        'M6 cell+Syntax+Functional':     COLS_CELL + COLS_SYN + COLS_FN,
        'M7 AllFeatures':            COLS_CELL + COLS_NB + COLS_SYN + COLS_FN,
    }
    return CONFIGS[config_name]


def select_best_config():
    """从 ablation_results.pkl 选最佳 (默认 M7 全特征, 因为含全部特征族)."""
    p = DATA / 'ablation_results.pkl'
    if not p.exists():
        return 'M7 AllFeatures'
    r = pickle.load(open(p, 'rb'))
    if 'M7 AllFeatures' in r:
        return 'M7 AllFeatures'
    return max(r.keys(), key=lambda k: r[k]['auc'])


# ============================================================
# Permutation importance (含 checkpoint)
# ============================================================
def compute_permutation_importance(best_name=None):
    if best_name is None:
        best_name = select_best_config()
    
    print("=" * 60)
    print(f"SHAP (Permutation Importance) — {best_name}")
    print("=" * 60)
    
    d = np.load(DATA / 'dataset.npz')
    X = d['X'].astype(np.float32)
    y = d['is_update'].astype(np.int8)
    yr = d['year_t1']
    col_names_full = list(d['col_names'])
    
    cols = get_config_cols(best_name, X.shape[1])
    cols_names = [col_names_full[i] for i in cols]
    print(f"  特征数: {len(cols)}")
    
    tr = np.isin(yr, TRAIN_YEARS); te = np.isin(yr, TEST_YEARS)
    Xtr = X[tr][:, cols]; ytr = y[tr]
    Xte = X[te][:, cols]; yte = y[te]
    print(f"  Train {tr.sum():,}, Test {te.sum():,}")
    
    # 训练
    print(f"  训练 ...")
    t0 = time.time()
    clf = HistGradientBoostingClassifier(**HISTGBM_PARAMS)
    clf.fit(Xtr, ytr)
    p_te = clf.predict_proba(Xte)[:, 1]
    base_auc_te = roc_auc_score(yte, p_te)
    print(f"  Test AUC = {base_auc_te:.4f} ({time.time()-t0:.0f}s)")
    
    # 子样本
    np.random.seed(42)
    n_sub = min(N_PERM_SAMPLE, len(Xte))
    sub_idx = np.random.choice(len(Xte), size=n_sub, replace=False)
    Xsub = Xte[sub_idx].copy(); ysub = yte[sub_idx]
    base_auc = roc_auc_score(ysub, clf.predict_proba(Xsub)[:, 1])
    print(f"  子样本 AUC = {base_auc:.4f} (N={n_sub})")
    
    # Checkpoint
    ckpt_path = DATA / 'shap_checkpoint.pkl'
    if ckpt_path.exists():
        ck = pickle.load(open(ckpt_path, 'rb'))
        if ck.get('config') == best_name and ck.get('n_features') == len(cols):
            importances = ck['importances']
            start_j = ck['next_j']
            print(f"  从 checkpoint 恢复: {start_j}/{len(cols)}")
        else:
            importances = np.zeros(len(cols))
            start_j = 0
    else:
        importances = np.zeros(len(cols))
        start_j = 0
    
    # Permutation
    print(f"  开始 permutation (j={start_j}..{len(cols)})")
    rng = np.random.RandomState(42)
    t0 = time.time()
    for j in range(start_j, len(cols)):
        Xperm = Xsub.copy()
        rng.shuffle(Xperm[:, j])
        perm_auc = roc_auc_score(ysub, clf.predict_proba(Xperm)[:, 1])
        importances[j] = base_auc - perm_auc
        
        if (j + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (j + 1 - start_j) / max(elapsed, 1e-6)
            eta = (len(cols) - j - 1) / max(rate, 1e-9)
            print(f"  {j+1}/{len(cols)} | {elapsed:.0f}s | rate={rate:.2f}/s | ETA={eta:.0f}s",
                  flush=True)
        if (j + 1) % CKPT_EVERY == 0:
            with open(ckpt_path, 'wb') as f:
                pickle.dump({'importances': importances, 'next_j': j + 1,
                             'config': best_name, 'n_features': len(cols)}, f)
    
    # 最终保存
    with open(DATA / 'shap_importance.pkl', 'wb') as f:
        pickle.dump({
            'best_config': best_name,
            'cols_idx': cols,
            'col_names': cols_names,
            'importances': importances,
            'test_auc': base_auc_te,
        }, f)
    if ckpt_path.exists():
        ckpt_path.unlink()
    print(f"  → data/shap_importance.pkl")
    
    return importances, cols_names, base_auc_te, best_name


# ============================================================
# 绘图
# ============================================================
def plot_top25(importances, names, auc, best_name):
    n_top = min(25, len(importances))
    top = np.argsort(importances)[::-1][:n_top]
    colors = [feat_color(names[i]) for i in top]
    
    fig, ax = plt.subplots(figsize=(12, max(5, n_top * 0.35)))
    y_pos = np.arange(n_top)
    ax.barh(y_pos, importances[top], color=colors)
    ax.set_yticks(y_pos); ax.set_yticklabels([names[i] for i in top], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('ΔAUC (Permutation Importance)')
    ax.set_title(f'Top {n_top} Feature Importance — {best_name} (Test AUC={auc:.4f})')
    ax.grid(alpha=0.3, axis='x')
    legend = [
        Patch(color='#198754', label='Functional cluster'),
        Patch(color='#20c997', label='Functional morphology'),
        Patch(color='#0d6efd', label='Cell class'),
        Patch(color='#ff9800', label='Syntax NC/TD/TL/MD'),
        Patch(color='#dc3545', label='Syntax NACH/Choice'),
        Patch(color='#e83e8c', label='Syntax Integration'),
        Patch(color='#9c27b0', label='Syntax NAIN'),
        Patch(color='#fd7e14', label='Neighborhood Shannon'),
        Patch(color='#6f42c1', label='Neighborhood ratio/dominant'),
        Patch(color='#6c757d', label='其他'),
    ]
    ax.legend(handles=legend, loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / 'shap_top25.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  → figs/shap_top25.png")


def plot_groups(importances, names):
    g = {'Cell class': 0, 'Neighborhood-ratio': 0, 'Neighborhood-shannon': 0,
         'Neighborhood-builtup/road': 0, 'Neighborhood-dominant': 0,
         'Syntax-NACH/Choice': 0, 'Syntax-Integration': 0,
         'Syntax-NAIN': 0, 'Syntax-NC/TD/TL/MD': 0,
         'Functional-cluster': 0, 'Functional-morphology': 0}
    for i, name in enumerate(names):
        v = importances[i]
        if name.startswith('cell_'): g['Cell class'] += v
        elif 'NACH' in name or 'Choice' in name: g['Syntax-NACH/Choice'] += v
        elif 'Integration' in name: g['Syntax-Integration'] += v
        elif 'NAIN' in name: g['Syntax-NAIN'] += v
        elif any(k in name for k in ['_NC', '_TD', '_TL', '_MD']):
            g['Syntax-NC/TD/TL/MD'] += v
        elif 'cluster' in name: g['Functional-cluster'] += v
        elif name.startswith('R') and any(k in name for k in ['entropy', 'purity',
                                                                'boundary', 'builtup']):
            g['Functional-morphology'] += v
        elif 'neighbor_diversity' in name: g['Functional-morphology'] += v
        elif name.startswith('nb_') and '_class_' in name: g['Neighborhood-ratio'] += v
        elif 'shannon' in name: g['Neighborhood-shannon'] += v
        elif 'builtup' in name or '_road' in name: g['Neighborhood-builtup/road'] += v
        elif 'dominant' in name: g['Neighborhood-dominant'] += v
    
    items = sorted(g.items(), key=lambda x: -x[1])
    g_names = [k for k, _ in items]
    vals = [v for _, v in items]
    bar_colors = ['#198754' if 'cluster' in n
                  else '#20c997' if '形态' in n
                  else '#0d6efd' if 'cell' in n
                  else '#ff9800' if 'NC/TD' in n
                  else '#dc3545' if 'NACH' in n
                  else '#e83e8c' if 'Integration' in n
                  else '#9c27b0' if 'NAIN' in n
                  else '#fd7e14' if 'shannon' in n
                  else '#6f42c1' for n in g_names]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(g_names))
    ax.barh(y_pos, vals, color=bar_colors)
    for i, v in enumerate(vals):
        ax.text(v + (0.0005 if v > 0 else -0.0005), i,
                f'{v:+.4f}', va='center',
                ha='left' if v > 0 else 'right', fontsize=9)
    ax.set_yticks(y_pos); ax.set_yticklabels(g_names)
    ax.invert_yaxis()
    ax.set_xlabel('Σ ΔAUC (grouped by feature family)')
    ax.set_title('Feature-family importance summary')
    ax.axvline(0, color='k', lw=0.5)
    ax.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(FIG / 'shap_group.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  → figs/shap_group.png")
    print(f"\n  特征族总贡献:")
    for k, v in items:
        print(f"    {k:<25}: {v:+.4f}")


def plot_pdp_top6(importances, names, best_name):
    """重训 best_name 模型并算 PDP."""
    print(f"\n  训练 {best_name} 用于 PDP ...")
    d = np.load(DATA / 'dataset.npz')
    X = d['X'].astype(np.float32)
    y = d['is_update'].astype(np.int8)
    yr = d['year_t1']
    full_names = list(d['col_names'])
    
    cols = get_config_cols(best_name, X.shape[1])
    
    tr = np.isin(yr, TRAIN_YEARS); te = np.isin(yr, TEST_YEARS)
    Xtr = X[tr][:, cols]; ytr = y[tr]
    
    clf = HistGradientBoostingClassifier(**HISTGBM_PARAMS)
    clf.fit(Xtr, ytr)
    
    np.random.seed(42)
    n_sub = min(8000, te.sum())
    sub_pool = np.where(te)[0]
    sub_idx = np.random.choice(sub_pool, size=n_sub, replace=False)
    Xsub = X[sub_idx][:, cols].copy()
    
    # Top 6 unique features
    order = np.argsort(importances)[::-1]
    top6 = []
    seen = set()
    for i in order:
        n = names[i]
        if n in seen: continue
        seen.add(n); top6.append(i)
        if len(top6) == 6: break
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, fi in zip(axes.flat, top6):
        name = names[fi]
        # local index in cols (since we use Xsub[:, cols])
        local_idx = fi   # because names was built from cols, indices match local
        vals = Xsub[:, local_idx]
        grid = np.linspace(np.percentile(vals, 1), np.percentile(vals, 99), 20)
        pdp_vals = []
        Xtmp = Xsub.copy()
        for v in grid:
            Xtmp[:, local_idx] = v
            pdp_vals.append(clf.predict_proba(Xtmp)[:, 1].mean())
        ax.plot(grid, pdp_vals, lw=2.5, color='#dc3545', marker='o', markersize=5)
        ax.fill_between(grid, pdp_vals, alpha=0.25, color='#dc3545')
        ax.set_xlabel(name, fontsize=10)
        ax.set_ylabel('avg P(update)')
        ax.set_title(f'{name}\nΔAUC = {importances[fi]:+.4f}', fontsize=10)
        ax.grid(alpha=0.3)
    plt.suptitle('Top 6 Feature Marginal Effects (Partial Dependence)', fontsize=13, y=1.0)
    plt.tight_layout()
    plt.savefig(FIG / 'shap_pdp_top6.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  → figs/shap_pdp_top6.png")


# ============================================================
# 入口
# ============================================================
def main():
    args = [a for a in sys.argv[1:]]
    replot_only = '--replot' in args
    args = [a for a in args if a != '--replot']
    
    config_arg = args[0] if args else None  # e.g. "M3"
    if config_arg and not config_arg.startswith('M'):
        config_arg = None
    
    if replot_only:
        print("仅重绘模式 (从 shap_importance.pkl)")
        r = pickle.load(open(DATA / 'shap_importance.pkl', 'rb'))
        importances = r['importances']
        names = r['col_names']
        auc = r['test_auc']
        best = r['best_config']
    else:
        # 选 config
        if config_arg:
            best_name = next((c for c in [
                'M0 cell only', 'M1 cell+Neighborhood', 'M2 cell+SyntaxDecay',
                'M3 cell+Functional', 'M4 cell+Neighborhood+Syntax', 'M5 cell+Neighborhood+Functional',
                'M6 cell+Syntax+Functional', 'M7 AllFeatures']
                if c.startswith(config_arg + ' ')), None)
            if best_name is None:
                print(f"❌ 未知配置: {config_arg}, 用 M0..M7")
                sys.exit(1)
        else:
            best_name = select_best_config()
        importances, names, auc, best = compute_permutation_importance(best_name)
    
    # Top 25 表
    top25 = np.argsort(importances)[::-1][:25]
    print("\n=== Top 25 重要特征 ===")
    for r, i in enumerate(top25):
        print(f"  {r+1:>2}. {names[i]:<35}  ΔAUC = {importances[i]:+.5f}")
    
    importances = np.asarray(importances)
    plot_top25(importances, names, auc, best)
    plot_groups(importances, names)
    plot_pdp_top6(importances, names, best)


if __name__ == '__main__':
    main()
