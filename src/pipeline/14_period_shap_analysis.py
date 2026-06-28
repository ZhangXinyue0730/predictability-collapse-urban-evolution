"""
14_period_shap_analysis.py — period-wise feature importance
===========================================================

This script explains the dominant update-driving factors for each land-use
transition period, e.g. 1984->1990, 1990->1995, ... 2020->2024.

It uses the same SHAP-style proxy as 12_shap_analysis.py:
permutation importance = base AUC - AUC after shuffling one feature.

Inputs:
  data/dataset.npz

Outputs:
  data/period_shap_feature_importance.csv
  data/period_shap_top_features.csv
  data/period_shap_group_importance.csv
  figs/period_shap_group_stacked.png
  figs/period_shap_top_features_heatmap.png

Usage:
  python pipeline/14_period_shap_analysis.py       # default: M7 AllFeatures
  python pipeline/14_period_shap_analysis.py M6    # use M6 feature set
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA, FIG, YEARS, HISTGBM_PARAMS, NB_RADII_M, CITY_NAME

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = [
    'PingFang SC', 'Arial Unicode MS', 'Heiti TC', 'Songti SC', 'DejaVu Sans'
]
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt


RANDOM_STATE = 42
N_PERM_SAMPLE = 8000
TOP_N = 10


CONFIG_NAMES = [
    'M0 cell only',
    'M1 cell+Neighborhood',
    'M2 cell+SyntaxDecay',
    'M3 cell+Functional',
    'M4 cell+Neighborhood+Syntax',
    'M5 cell+Neighborhood+Functional',
    'M6 cell+Syntax+Functional',
    'M7 AllFeatures',
]


GROUP_ORDER = ['Cell class', 'Neighborhood', 'Syntax', 'Functional']
GROUP_COLORS = {
    'Cell class': '#0d6efd',
    'Neighborhood': '#6f42c1',
    'Syntax': '#e83e8c',
    'Functional': '#198754',
}


def get_config_cols(config_name, n_total):
    n_cell = 7
    n_nb = 11 * len(NB_RADII_M)
    n_syn = 80

    cols_cell = list(range(n_cell))
    cols_nb = list(range(n_cell, n_cell + n_nb))
    cols_syn = list(range(n_cell + n_nb, n_cell + n_nb + n_syn))
    cols_fn = list(range(n_cell + n_nb + n_syn, n_total))

    configs = {
        'M0 cell only': cols_cell,
        'M1 cell+Neighborhood': cols_cell + cols_nb,
        'M2 cell+SyntaxDecay': cols_cell + cols_syn,
        'M3 cell+Functional': cols_cell + cols_fn,
        'M4 cell+Neighborhood+Syntax': cols_cell + cols_nb + cols_syn,
        'M5 cell+Neighborhood+Functional': cols_cell + cols_nb + cols_fn,
        'M6 cell+Syntax+Functional': cols_cell + cols_syn + cols_fn,
        'M7 AllFeatures': cols_cell + cols_nb + cols_syn + cols_fn,
    }
    return configs[config_name]


def parse_config_arg():
    if len(sys.argv) <= 1:
        return 'M7 AllFeatures'
    arg = sys.argv[1].strip()
    for name in CONFIG_NAMES:
        if name.startswith(arg + ' ') or name == arg:
            return name
    raise SystemExit(f"Unknown model config: {arg}. Use M0..M7.")


def feature_group(name):
    if name.startswith('cell_'):
        return 'Cell class'
    if name.startswith('nb_'):
        return 'Neighborhood'
    if (
        'NACH' in name or 'Choice' in name or 'Integration' in name or
        'NAIN' in name or any(k in name for k in ['_NC', '_TD', '_TL', '_MD'])
    ):
        return 'Syntax'
    return 'Functional'


def safe_auc(y_true, pred):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, pred)


def compute_period_importance(X, y, cols, names, mask, period_label):
    Xp = X[mask][:, cols].astype(np.float32, copy=False)
    yp = y[mask].astype(np.int8, copy=False)

    n_samples = len(yp)
    n_updates = int(yp.sum())
    n_non_updates = int(n_samples - n_updates)
    if n_updates < 5 or n_non_updates < 5:
        print(f"  {period_label}: skipped, too few class samples")
        return None

    idx = np.arange(n_samples)
    tr_idx, te_idx = train_test_split(
        idx,
        test_size=0.35,
        random_state=RANDOM_STATE,
        stratify=yp,
    )

    Xtr, ytr = Xp[tr_idx], yp[tr_idx]
    Xte, yte = Xp[te_idx], yp[te_idx]

    clf = HistGradientBoostingClassifier(**HISTGBM_PARAMS)
    clf.fit(Xtr, ytr)

    pred = clf.predict_proba(Xte)[:, 1]
    test_auc = safe_auc(yte, pred)

    rng = np.random.RandomState(RANDOM_STATE)
    n_sub = min(N_PERM_SAMPLE, len(Xte))
    sub_idx = rng.choice(len(Xte), size=n_sub, replace=False)
    Xsub = Xte[sub_idx].copy()
    ysub = yte[sub_idx]

    base_auc = safe_auc(ysub, clf.predict_proba(Xsub)[:, 1])
    importances = np.zeros(len(cols), dtype=np.float32)

    if np.isnan(base_auc):
        return None

    for j in range(len(cols)):
        Xperm = Xsub.copy()
        rng.shuffle(Xperm[:, j])
        perm_auc = safe_auc(ysub, clf.predict_proba(Xperm)[:, 1])
        importances[j] = 0.0 if np.isnan(perm_auc) else base_auc - perm_auc

    group_scores = {g: 0.0 for g in GROUP_ORDER}
    for name, value in zip(names, importances):
        group_scores[feature_group(name)] += float(value)

    return {
        'period': period_label,
        'n_samples': n_samples,
        'n_updates': n_updates,
        'update_rate': n_updates / max(n_samples, 1),
        'test_auc': float(test_auc),
        'base_auc': float(base_auc),
        'importances': importances,
        'group_scores': group_scores,
    }


def write_csv_outputs(results, names):
    feature_path = DATA / 'period_shap_feature_importance.csv'
    top_path = DATA / 'period_shap_top_features.csv'
    group_path = DATA / 'period_shap_group_importance.csv'

    with open(feature_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['period', 'feature', 'group', 'delta_auc', 'rank',
                    'n_samples', 'n_updates', 'update_rate', 'test_auc'])
        for r in results:
            order = np.argsort(r['importances'])[::-1]
            ranks = np.empty(len(order), dtype=int)
            ranks[order] = np.arange(1, len(order) + 1)
            for i, name in enumerate(names):
                w.writerow([
                    r['period'], name, feature_group(name),
                    f"{float(r['importances'][i]):.8f}", int(ranks[i]),
                    r['n_samples'], r['n_updates'], f"{r['update_rate']:.8f}",
                    f"{r['test_auc']:.8f}",
                ])

    with open(top_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['period', 'rank', 'feature', 'group', 'delta_auc',
                    'n_samples', 'n_updates', 'update_rate', 'test_auc'])
        for r in results:
            order = np.argsort(r['importances'])[::-1][:TOP_N]
            for rank, i in enumerate(order, 1):
                w.writerow([
                    r['period'], rank, names[i], feature_group(names[i]),
                    f"{float(r['importances'][i]):.8f}",
                    r['n_samples'], r['n_updates'], f"{r['update_rate']:.8f}",
                    f"{r['test_auc']:.8f}",
                ])

    with open(group_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['period', 'group', 'delta_auc', 'share_positive',
                    'n_samples', 'n_updates', 'update_rate', 'test_auc'])
        for r in results:
            positives = {g: max(0.0, r['group_scores'][g]) for g in GROUP_ORDER}
            total_pos = sum(positives.values())
            for g in GROUP_ORDER:
                share = positives[g] / total_pos if total_pos > 0 else 0.0
                w.writerow([
                    r['period'], g, f"{r['group_scores'][g]:.8f}",
                    f"{share:.8f}", r['n_samples'], r['n_updates'],
                    f"{r['update_rate']:.8f}", f"{r['test_auc']:.8f}",
                ])

    print(f"  -> {feature_path}")
    print(f"  -> {top_path}")
    print(f"  -> {group_path}")


def plot_group_stacked(results):
    periods = [r['period'] for r in results]
    bottom = np.zeros(len(results), dtype=float)

    fig, ax = plt.subplots(figsize=(13, 6))
    for group in GROUP_ORDER:
        vals = np.array([max(0.0, r['group_scores'][group]) for r in results])
        ax.bar(periods, vals, bottom=bottom, label=group, color=GROUP_COLORS[group])
        bottom += vals

    for i, r in enumerate(results):
        ax.text(i, bottom[i] + max(bottom.max() * 0.015, 0.0002),
                f"n={r['n_updates']:,}\nAUC={r['test_auc']:.3f}",
                ha='center', va='bottom', fontsize=8)

    ax.set_title(f'{CITY_NAME} period-wise dominant factor groups')
    ax.set_ylabel('Positive summed permutation importance (Σ ΔAUC)')
    ax.set_xlabel('Transition period')
    ax.tick_params(axis='x', rotation=30)
    ax.grid(alpha=0.25, axis='y')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout()
    out = FIG / 'period_shap_group_stacked.png'
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  -> {out}")


def plot_top_feature_heatmap(results, names):
    selected = []
    seen = set()
    for r in results:
        order = np.argsort(r['importances'])[::-1]
        for i in order[:5]:
            if names[i] not in seen:
                seen.add(names[i])
                selected.append(i)
            if len(selected) >= 20:
                break
        if len(selected) >= 20:
            break

    mat = np.array([[r['importances'][i] for i in selected] for r in results], dtype=float)
    vmax = max(abs(mat.min()), abs(mat.max()), 1e-6)

    fig, ax = plt.subplots(figsize=(max(12, len(selected) * 0.55), 5.5))
    im = ax.imshow(mat, aspect='auto', cmap='RdYlGn', vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(len(results)))
    ax.set_yticklabels([r['period'] for r in results])
    ax.set_xticks(np.arange(len(selected)))
    ax.set_xticklabels([names[i] for i in selected], rotation=55, ha='right', fontsize=8)
    ax.set_title(f'{CITY_NAME} period-wise top feature importance heatmap')
    ax.set_xlabel('Features selected from period top lists')
    ax.set_ylabel('Transition period')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('ΔAUC')
    plt.tight_layout()
    out = FIG / 'period_shap_top_features_heatmap.png'
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  -> {out}")


def main():
    config_name = parse_config_arg()
    print("=" * 60)
    print(f"Step 14: period-wise SHAP-style analysis — {config_name}")
    print("=" * 60)

    d = np.load(DATA / 'dataset.npz')
    X = d['X'].astype(np.float32)
    y = d['is_update'].astype(np.int8)
    yr = d['year_t1']
    all_names = list(d['col_names'])

    cols = get_config_cols(config_name, X.shape[1])
    names = [all_names[i] for i in cols]

    results = []
    t0 = time.time()
    for t1, t2 in zip(YEARS[:-1], YEARS[1:]):
        period = f'{t1}->{t2}'
        mask = yr == t1
        print(f"\n  {period}: samples={int(mask.sum()):,}")
        r = compute_period_importance(X, y, cols, names, mask, period)
        if r is None:
            continue
        results.append(r)
        print(
            f"    updates={r['n_updates']:,} "
            f"rate={r['update_rate']*100:.2f}% "
            f"AUC={r['test_auc']:.4f}"
        )
        top = np.argsort(r['importances'])[::-1][:3]
        print("    top3: " + ", ".join(
            f"{names[i]}({r['importances'][i]:+.4f})" for i in top
        ))

    if not results:
        raise RuntimeError("No period has enough positive and negative samples.")

    write_csv_outputs(results, names)
    plot_group_stacked(results)
    plot_top_feature_heatmap(results, names)

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
