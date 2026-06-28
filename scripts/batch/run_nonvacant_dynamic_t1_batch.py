#!/usr/bin/env python3
"""Run only the steps affected by the corrected non-vacant definition."""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


STEPS = [
    "06_neighborhood_features.py",
    "08_functional_features.py",
    "09.5_dynamic_builtup_extent.py",
    "10_assemble_dataset.py",
    "11_train_and_ablate.py",
    "12_shap_analysis.py",
    "13_update_transition_stats.py",
    "14_period_shap_analysis.py",
]

REQUIRED = [
    "data/dataset.npz",
    "data/ablation_results.pkl",
    "data/shap_importance.pkl",
    "data/update_period_rates.csv",
    "data/period_shap_group_importance.csv",
]


def complete(project: Path) -> bool:
    return all((project / item).exists() for item in REQUIRED)


def write_status(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--cities", nargs="*")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    runs = root / "city_extract_national" / "pipeline_runs_dynamic_t1"
    python = root / ".venv" / "bin" / "python"
    logs = runs / "_batch_logs_nonvacant_revision"
    logs.mkdir(exist_ok=True)
    status_path = runs / "nonvacant_dynamic_t1_batch_status.csv"
    requested = set(args.cities or [])

    city_dirs = sorted(
        p for p in runs.glob("*全流程")
        if not requested or p.name.removesuffix("全流程") in requested
    )
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = "/tmp/matplotlib"
    env["PYTHONPYCACHEPREFIX"] = "/tmp"
    rows = []
    print(f"Cities found: {len(city_dirs)}", flush=True)

    for index, city_dir in enumerate(city_dirs, 1):
        project = city_dir / "tculu_pipeline_v9_final"
        log_path = logs / f"{city_dir.name}.log"
        if complete(project) and not args.force:
            print(f"[{index:02d}/{len(city_dirs)}] SKIP {city_dir.name}", flush=True)
            rows.append({
                "city_folder": city_dir.name,
                "status": "complete_existing",
                "return_code": 0,
                "started_at": "",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "failed_step": "",
                "log_path": str(log_path),
            })
            write_status(status_path, rows)
            continue

        started = datetime.now()
        rc = 0
        failed_step = ""
        print(f"[{index:02d}/{len(city_dirs)}] RUN {city_dir.name}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            for step in STEPS:
                log.write(f"\n{'=' * 70}\nRUN {step}\n{'=' * 70}\n")
                log.flush()
                result = subprocess.run(
                    [str(python), f"pipeline/{step}"],
                    cwd=project,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                if result.returncode != 0:
                    rc = result.returncode
                    failed_step = step
                    break
        finished = datetime.now()
        status = "complete" if rc == 0 and complete(project) else "failed"
        print(
            f"[{index:02d}/{len(city_dirs)}] {status.upper()} "
            f"{city_dir.name} ({(finished-started).total_seconds():.0f}s)"
            + (f" at {failed_step}" if failed_step else ""),
            flush=True,
        )
        rows.append({
            "city_folder": city_dir.name,
            "status": status,
            "return_code": rc,
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": finished.isoformat(timespec="seconds"),
            "failed_step": failed_step,
            "log_path": str(log_path),
        })
        write_status(status_path, rows)

    failures = [row for row in rows if row["status"] == "failed"]
    print(
        f"Batch finished: {len(rows)-len(failures)} OK, "
        f"{len(failures)} failed",
        flush=True,
    )
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
