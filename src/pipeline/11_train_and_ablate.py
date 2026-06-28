"""
11_train_and_ablate.py — 模型训练 + 特征消融
=============================================

跑 8 种特征配置, 对比各机制的贡献:
  M0: cell only                       (7 dim)
  M1: cell + 周边                     (73 dim)
  M2: cell + 衰减句法                 (87 dim)
  M3: cell + 功能整体                 (21 dim)
  M4: cell + 周边 + 句法              (153 dim)
  M5: cell + 周边 + 功能              (87 dim)
  M6: cell + 句法 + 功能              (101 dim)
  M7: 全特征                          (167 dim)

评估: 时间外推 (TRAIN_YEARS → TEST_YEARS, 见 config.py)

输出:
  data/ablation_results.pkl
  figs/ablation_comparison.png
"""
import numpy as np
import pickle
import sys
import gc
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (DATA, FIG, TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
                    HISTGBM_PARAMS, NB_RADII_M)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Zen Hei',
                                            'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt


def main():
    print("=" * 60)
    print("Feature ablation study")
    print("=" * 60)
    
    d = np.load(DATA / 'dataset.npz')
    X = d['X']; y = d['is_update'].astype(np.int8); yr = d['year_t1']
    
    tr = np.isin(yr, TRAIN_YEARS)
    te = np.isin(yr, TEST_YEARS)
    print(f"Train {tr.sum():,}, Test {te.sum():,}")
    print(f"Train pos rate: {y[tr].mean()*100:.2f}%")
    print(f"Test  pos rate: {y[te].mean()*100:.2f}%")
    
    # 特征列范围
    n_cell = 7
    n_nb = 11 * len(NB_RADII_M)        # 11 维 / 半径 × N 半径
    n_syn = 80                          # 40 metrics × {mean, max}
    # 自动找出功能特征维度
    fn_dim = X.shape[1] - n_cell - n_nb - n_syn
    
    COLS_CELL = list(range(n_cell))
    COLS_NB = list(range(n_cell, n_cell + n_nb))
    COLS_SYN = list(range(n_cell + n_nb, n_cell + n_nb + n_syn))
    COLS_FN = list(range(n_cell + n_nb + n_syn, X.shape[1]))
    
    print(f"\nFeature groups: cell={len(COLS_CELL)}, neighborhood={len(COLS_NB)}, "
          f"syntax={len(COLS_SYN)}, functional={len(COLS_FN)}")
    
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
    
    out_path = DATA / 'ablation_results.pkl'
    results = {}
    if out_path.exists():
        results = pickle.load(open(out_path, 'rb'))
    
    for name, cols in CONFIGS.items():
        if name in results:
            r = results[name]
            print(f"  ✓ {name:<25}: AUC={r['auc']:.4f}  AP={r['ap']:.4f} (cached)")
            continue
        Xtr = X[tr][:, cols].copy()
        Xte = X[te][:, cols].copy()
        t0 = time.time()
        clf = HistGradientBoostingClassifier(**HISTGBM_PARAMS)
        clf.fit(Xtr, y[tr])
        p_te = clf.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(y[te], p_te)
        ap = average_precision_score(y[te], p_te)
        results[name] = {'auc': auc, 'ap': ap, 'n_dim': len(cols),
                          'time': time.time() - t0,
                          'test_pred': p_te}
        print(f"  {name:<25}: AUC={auc:.4f}  AP={ap:.4f} dim={len(cols)} "
              f"({time.time()-t0:.0f}s)", flush=True)
        with open(out_path, 'wb') as f:
            pickle.dump(results, f)
        del Xtr, Xte, clf; gc.collect()
    
    # 汇总打印
    print("\n" + "="*70)
    print("Final summary (time extrapolation on test set)")
    print("="*70)
    print(f"{'配置':<25} {'维度':>6} {'Test AUC':>10} {'Test AP':>10}")
    print("-" * 55)
    for name, r in results.items():
        print(f"{name:<25} {r['n_dim']:>6} {r['auc']:>10.4f} {r['ap']:>10.4f}")
    
    # 增量贡献 (相对 M1 cell+周边)
    if 'M1 cell+Neighborhood' in results:
        base = results['M1 cell+Neighborhood']
        print(f"\nDelta vs M1 (cell+Neighborhood, AUC={base['auc']:.4f}):")
        for name, r in results.items():
            if name == 'M1 cell+Neighborhood': continue
            print(f"  {name:<25}: ΔAUC={r['auc']-base['auc']:+.4f} "
                  f"ΔAP={r['ap']-base['ap']:+.4f}")

    # 可视化
    plot_ablation(results)


def plot_ablation(results):
    order = ['M0 cell only', 'M1 cell+Neighborhood', 'M2 cell+SyntaxDecay',
             'M3 cell+Functional', 'M4 cell+Neighborhood+Syntax', 'M5 cell+Neighborhood+Functional',
             'M6 cell+Syntax+Functional', 'M7 AllFeatures']
    items = [(n, results[n]) for n in order if n in results]
    
    labels = ['cell only', '+Neighborhood', '+SyntaxDecay', '+Functional',
              '+Neighborhood+Syntax', '+Neighborhood+Functional', '+Syntax+Functional', 'AllFeatures']
    aucs = [it[1]['auc'] for it in items]
    aps = [it[1]['ap'] for it in items]
    
    colors = ['#888', '#198754', '#dc3545', '#fd7e14',
              '#6f42c1', '#0dcaf0', '#e83e8c', '#0d6efd']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, vals, ylabel in [(axes[0], aucs, 'Test AUC'),
                              (axes[1], aps, 'Test AP')]:
        bars = ax.bar(range(len(items)), vals, color=colors[:len(items)])
        for i, (b, v) in enumerate(zip(bars, vals)):
            ax.text(b.get_x()+b.get_width()/2, v+0.002, f'{v:.4f}',
                    ha='center', fontsize=9)
        ax.set_xticks(range(len(items)))
        ax.set_xticklabels(labels[:len(items)], rotation=20, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f'Feature Ablation — {ylabel}')
        ax.grid(alpha=0.3, axis='y')
    
    plt.suptitle('Feature Ablation Comparison (time extrapolation Train→Test)', fontsize=13, y=1.0)
    plt.tight_layout()
    plt.savefig(FIG / 'ablation_comparison.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n→ figs/ablation_comparison.png")


if __name__ == '__main__':
    main()
