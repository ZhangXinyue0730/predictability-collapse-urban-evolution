"""
Deploy the formal dynamic-t1 pipeline to all 79 cloned city projects.
"""
import csv
import re
import shutil
import sys
from pathlib import Path


DATA_OLD_PATTERNS = [
    "dataset.npz",
    "dataset_dynamic_t1.npz",
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
    "dynamic_scope_period_comparison.csv",
    "dynamic_scope_period_transition_counts.csv",
    "dynamic_builtup_extent_summary.csv",
]

FIG_OLD_PATTERNS = [
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
    "dynamic_scope_update_rate_comparison.png",
    "dynamic_scope_period_expansion.png",
    "period_update_and_expansion.png",
]


def backup_script(pipe, name):
    source = pipe / name
    backup = pipe / name.replace(".py", "_old_scope.py")
    if source.exists() and not backup.exists():
        shutil.copy2(source, backup)


def archive_outputs(folder, names):
    archive = folder / "legacy_old_scope"
    moved = 0
    for name in names:
        path = folder / name
        if path.exists():
            archive.mkdir(exist_ok=True)
            target = archive / name
            if target.exists():
                target.unlink()
            shutil.move(str(path), str(target))
            moved += 1
    return moved


def patch_config(path):
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"UPDATE_DEFINITION\s*=\s*['\"][^'\"]+['\"]",
        "UPDATE_DEFINITION = 'dynamic_t1_extent'",
        text,
    )
    marker = "UPDATE_DEFINITION = 'dynamic_t1_extent'"
    settings = """

# Dynamic t1 built-up extent definition
DYNAMIC_CORE_CLASSES = {1, 2, 3, 4}
DYNAMIC_LINK_DISTANCE_M = 2000.0
DYNAMIC_MAX_HOLE_AREA_KM2 = 4.0
"""
    if "DYNAMIC_LINK_DISTANCE_M" not in text:
        text = text.replace(marker, marker + settings)
    path.write_text(text, encoding="utf-8")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: deploy_dynamic_t1_pipeline.py PROJECT_ROOT SOURCE_DIR")

    root = Path(sys.argv[1]).resolve()
    source_dir = Path(sys.argv[2]).resolve()
    new_root = root / "city_extract_national" / "pipeline_runs_dynamic_t1"
    index_path = new_root / "dynamic_t1_city_workspace_index.csv"

    with index_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 79:
        raise RuntimeError(f"Expected 79 cities, found {len(rows)}")

    source_files = {
        "09.5_dynamic_builtup_extent.py": source_dir / "dynamic_builtup_extent.py",
        "10_assemble_dataset.py": source_dir / "10_assemble_dataset_dynamic_t1.py",
        "13_update_transition_stats.py": source_dir / "13_update_transition_stats_dynamic_t1.py",
        "run_all.py": source_dir / "run_all_dynamic_t1.py",
    }
    for path in source_files.values():
        if not path.exists():
            raise FileNotFoundError(path)

    for idx, row in enumerate(rows, 1):
        project = Path(row["target_project"])
        pipe = project / "pipeline"
        data = project / "data"
        figs = project / "figs"

        backup_script(pipe, "10_assemble_dataset.py")
        backup_script(pipe, "13_update_transition_stats.py")
        backup_script(pipe, "run_all.py")

        moved_data = archive_outputs(data, DATA_OLD_PATTERNS)
        moved_figs = archive_outputs(figs, FIG_OLD_PATTERNS)

        # Dynamic extent arrays inherited from Beijing/Suzhou trials are
        # regenerated, so archive them as well.
        dynamic_arrays = sorted(data.glob("builtup_extent_dynamic_*.npy"))
        if dynamic_arrays:
            archive = data / "legacy_old_scope"
            archive.mkdir(exist_ok=True)
            for path in dynamic_arrays:
                target = archive / path.name
                if target.exists():
                    target.unlink()
                shutil.move(str(path), str(target))

        for target_name, source in source_files.items():
            shutil.copy2(source, pipe / target_name)
        patch_config(pipe / "config.py")

        print(
            f"[{idx:02d}/79] {row['target_folder']}: "
            f"deployed, archived data={moved_data}, figs={moved_figs}",
            flush=True,
        )

    print("Deployment complete.")


if __name__ == "__main__":
    main()
