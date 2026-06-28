"""
run_all.py — 动态前期建成区 Pipeline 总入口
==========================================

步骤 01-09 沿用已有特征，09.5-14 按动态前期建成区口径运行。

两条路径:
  完整路径 (有 OSM):
    01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12
  代理路径 (无 OSM, 仍可生成 9 年数据集):
    02 → 06 → 07_proxy → 08 → 10 → 11 → 12

用法:
  # 完整路径 (有 OSM 路网 shp):
  python run_all.py --osm /path/to/gis_osm_roads_free_1.shp \\
                    --png /path/to/landuse_pngs/

  # 代理路径 (PNG 文件 + 已有原 dataset.npz 即可):
  python run_all.py --png /path/to/landuse_pngs/ --proxy

  # 从某步开始:
  python run_all.py --from-step 9.5 --to-step 14
"""
import sys
import argparse
import subprocess
from pathlib import Path

PIPE = Path(__file__).parent
PROJECT = PIPE.parent
DATA = PROJECT / 'data'


def run(script, *args):
    print(f"\n{'='*70}\n>>> 运行 {script}\n{'='*70}")
    cmd = [sys.executable, str(PIPE / script)] + list(args)
    print('Cmd:', ' '.join(str(x) for x in cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"❌ {script} 失败 (rc={rc})")
        sys.exit(rc)


def have_osm_outputs():
    """判断是否有 OSM 路径产物 (步骤 03 输出)."""
    return any((DATA / f'road_{y}.npz').exists() for y in [1984, 1990, 2024])


def have_real_syntax():
    """判断是否有完整句法产物 (步骤 07 输出)."""
    from config import YEARS
    return all((DATA / f'cell_syntax_decay_{y}.npz').exists() for y in YEARS)


def have_lut():
    return (DATA / 'syntax_lut.npz').exists()


def main():
    parser = argparse.ArgumentParser(description='TCULU dynamic-t1 pipeline runner')
    parser.add_argument('--osm', help='OSM road shapefile path (步骤 01 输入)')
    parser.add_argument('--png', help='Landuse PNG directory (步骤 02 输入)')
    parser.add_argument('--proxy', action='store_true',
                        help='强制使用代理路径 (跳过 03-09, 用 LUT)')
    parser.add_argument('--from-step', type=float, default=1, help='从第几步开始')
    parser.add_argument('--to-step', type=float, default=14, help='到第几步停 (含)')
    args = parser.parse_args()
    
    # 自动检测路径
    use_proxy = args.proxy
    if not use_proxy and not args.osm and not have_osm_outputs():
        # 没有 OSM 输入且没有 OSM 产物 → 默认走代理
        if (DATA / 'dataset.npz').exists() or have_lut():
            print("⚙ 自动检测: 使用代理路径 (没找到 OSM 数据/产物)")
            use_proxy = True
    
    full_steps = [
        (1,  '01_extract_osm.py',           [args.osm] if args.osm else []),
        (2,  '02_landuse_from_png.py',      [args.png] if args.png else []),
        (3,  '03_road_and_landuse.py',      []),
        (4,  '04_segment_graph.py',         []),
        (5,  '05_syntax_full.py',           []),
        (6,  '06_neighborhood_features.py', []),
        (7,  '07_syntax_kde_to_cell.py',    []),
        (8,  '08_functional_features.py',   []),
        (9,  '09_cross_links.py',           []),
        (9.5, '09.5_dynamic_builtup_extent.py', []),
        (10, '10_assemble_dataset.py',      []),
        (11, '11_train_and_ablate.py',      []),
        (12, '12_shap_analysis.py',         []),
        (13, '13_update_transition_stats.py', []),
        (14, '14_period_shap_analysis.py',  []),
    ]
    
    proxy_steps = [
        (1,  None, []),                                       # 跳过
        (2,  '02_landuse_from_png.py',      [args.png] if args.png else []),
        (3,  None, []),                                       # 跳过 (需 OSM)
        (4,  None, []),
        (5,  None, []),
        (6,  '06_neighborhood_features.py', []),
        (7,  '07_syntax_lut_proxy.py',      []),              # 代理 LUT
        (8,  '08_functional_features.py',   []),
        (9,  None, []),                                       # 跳过 (需 OSM)
        (9.5, '09.5_dynamic_builtup_extent.py', []),
        (10, '10_assemble_dataset.py',      []),
        (11, '11_train_and_ablate.py',      []),
        (12, '12_shap_analysis.py',         []),
        (13, '13_update_transition_stats.py', []),
        (14, '14_period_shap_analysis.py',  []),
    ]
    
    steps = proxy_steps if use_proxy else full_steps
    print(f"\n模式: {'代理路径 (无 OSM)' if use_proxy else '完整路径 (含 OSM)'}")
    print(f"范围: 步骤 {args.from_step} → {args.to_step}")
    
    for sid, script, sargs in steps:
        if sid < args.from_step or sid > args.to_step:
            continue
        if script is None:
            print(f"\n  ⊘ 跳过步骤 {sid} ({'代理路径不需要' if use_proxy else 'no input'})")
            continue
        if sid in (1, 2) and not sargs:
            # 1, 2 需输入路径; 没给就跳过 (假设已经有产物)
            print(f"\n  ⊘ 跳过步骤 {sid} ({script}, 无输入路径)")
            continue
        run(script, *sargs)
    
    print("\n" + "=" * 70)
    print("🎉 Pipeline 完成")
    print("=" * 70)
    print("查看结果:")
    print("  data/dataset.npz             9 年特征 (167 维)")
    print("  data/ablation_results.pkl    8 配置消融")
    print("  data/shap_importance.pkl     M7 SHAP")
    print("  figs/timeline_9years.png     9 年时序图")
    print("  figs/ablation_comparison.png 消融对比")
    print("  figs/shap_top25.png          Top 25 特征")
    print("  figs/shap_group.png          特征族总贡献")
    print("  figs/shap_pdp_top6.png       Top 6 PDP")
    print("  data/update_period_rates.csv 分时期更新与扩张")
    print("  data/period_shap_*.csv       分时期驱动力")


if __name__ == '__main__':
    main()
