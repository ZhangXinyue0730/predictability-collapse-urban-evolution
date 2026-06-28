#!/usr/bin/env python3
"""Deploy the corrected non-vacant dynamic-t1 pipeline to all city runs."""
from __future__ import annotations

import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path


SOURCE_FILES = {
    "06_neighborhood_features.py": "06_neighborhood_features_revised.py",
    "08_functional_features.py": "08_functional_features_revised.py",
    "09.5_dynamic_builtup_extent.py": "09.5_dynamic_builtup_extent_revised.py",
    "10_assemble_dataset.py": "10_assemble_dataset_revised.py",
    "13_update_transition_stats.py": "13_update_transition_stats_revised.py",
}

DATA_NAMES = [
    "dataset.npz",
    "ablation_results.pkl",
    "shap_importance.pkl",
    "shap_checkpoint.pkl",
    "period_shap_feature_importance.csv",
    "period_shap_group_importance.csv",
    "period_shap_top_features.csv",
    "update_period_rates.csv",
    "update_period_top_transitions.csv",
    "update_period_transition_counts.csv",
    "update_transition_matrix.csv",
    "update_transition_top.csv",
    "dynamic_builtup_extent_summary.csv",
]

FIG_NAMES = [
    "ablation_comparison.png",
    "shap_group.png",
    "shap_pdp_top6.png",
    "shap_top25.png",
    "period_shap_group_stacked.png",
    "period_shap_top_features_heatmap.png",
    "period_update_rates.png",
    "period_transition_top10_stacked.png",
    "period_transition_top5_stacked.png",
    "update_transition_matrix.png",
    "update_transition_top10.png",
    "dynamic_builtup_extent_9years.png",
    "period_update_and_expansion.png",
]


def archive_matches(folder: Path, archive: Path, patterns: list[str]) -> int:
    moved = 0
    archive.mkdir(parents=True, exist_ok=True)
    paths = []
    for pattern in patterns:
        paths.extend(folder.glob(pattern))
    for path in sorted(set(paths)):
        if not path.exists():
            continue
        target = archive / path.name
        if target.exists():
            target.unlink()
        shutil.move(str(path), str(target))
        moved += 1
    return moved


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: deploy_nonvacant_dynamic_t1_revision.py ROOT SOURCE_DIR"
        )
    root = Path(sys.argv[1]).resolve()
    source = Path(sys.argv[2]).resolve()
    runs = root / "city_extract_national" / "pipeline_runs_dynamic_t1"
    index = runs / "dynamic_t1_city_workspace_index.csv"
    with index.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 79:
        raise RuntimeError(f"Expected 79 cities, found {len(rows)}")

    revision = "invalid_lu_clean_scope_20260613"
    for i, row in enumerate(rows, 1):
        project = Path(row["target_project"])
        pipe = project / "pipeline"
        data = project / "data"
        figs = project / "figs"

        script_archive = pipe / "legacy_invalid_lu_clean_20260613"
        script_archive.mkdir(exist_ok=True)
        for target_name, source_name in SOURCE_FILES.items():
            target = pipe / target_name
            if target.exists():
                shutil.copy2(target, script_archive / target_name)
            shutil.copy2(source / source_name, target)

        data_archive = data / revision
        fig_archive = figs / revision
        moved_data = archive_matches(
            data,
            data_archive,
            DATA_NAMES
            + ["cell_features_*.npz", "cell_functional_*.npz",
               "builtup_extent_dynamic_*.npy"],
        )
        moved_figs = archive_matches(figs, fig_archive, FIG_NAMES)

        marker = project / "NONVACANT_DYNAMIC_T1_REVISION.txt"
        marker.write_text(
            "Revision deployed: "
            + datetime.now().isoformat(timespec="seconds")
            + "\nBuilt-up extent source: lu_{year}.npy\n"
            + "Renewal sample: inside t1 extent and both t1/t2 classes in 1-6\n"
            + "Affected steps rerun: 06, 08, 09.5, 10, 11, 12, 13, 14\n",
            encoding="utf-8",
        )
        print(
            f"[{i:02d}/79] {row['target_folder']}: "
            f"scripts deployed, archived data={moved_data}, figs={moved_figs}",
            flush=True,
        )


if __name__ == "__main__":
    main()
