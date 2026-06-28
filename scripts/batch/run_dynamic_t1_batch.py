"""Run the formal dynamic-t1 analysis pipeline city by city with resume support."""
import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REQUIRED_OUTPUTS = (
    "data/dataset.npz",
    "data/ablation_results.pkl",
    "data/shap_importance.pkl",
    "data/update_period_rates.csv",
    "data/period_shap_group_importance.csv",
)


def is_complete(project):
    return all((project / rel).exists() for rel in REQUIRED_OUTPUTS)


def write_status(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "city_folder",
                "status",
                "return_code",
                "started_at",
                "finished_at",
                "log_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--from-step", default="9.5")
    parser.add_argument("--to-step", default="14")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    runs_root = args.root / "city_extract_national" / "pipeline_runs_dynamic_t1"
    python = args.root / ".venv" / "bin" / "python"
    logs = runs_root / "_batch_logs"
    logs.mkdir(parents=True, exist_ok=True)
    status_path = runs_root / "dynamic_t1_batch_status.csv"

    city_dirs = sorted(
        p for p in runs_root.glob("*全流程")
        if (p / "tculu_pipeline_v9_final" / "pipeline" / "run_all.py").exists()
    )
    rows = []
    print(f"Cities found: {len(city_dirs)}")

    env = os.environ.copy()
    env["MPLCONFIGDIR"] = "/tmp/matplotlib"
    env["PYTHONPYCACHEPREFIX"] = "/tmp"

    for index, city_dir in enumerate(city_dirs, 1):
        project = city_dir / "tculu_pipeline_v9_final"
        city_name = city_dir.name
        log_path = logs / f"{city_name}.log"

        if not args.force and is_complete(project):
            print(f"[{index:02d}/{len(city_dirs)}] SKIP {city_name} (complete)")
            rows.append({
                "city_folder": city_name,
                "status": "complete_existing",
                "return_code": 0,
                "started_at": "",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "log_path": str(log_path),
            })
            write_status(status_path, rows)
            continue

        started = datetime.now()
        print(f"[{index:02d}/{len(city_dirs)}] RUN  {city_name}")
        cmd = [
            str(python),
            "pipeline/run_all.py",
            "--from-step",
            args.from_step,
            "--to-step",
            args.to_step,
        ]
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                cmd,
                cwd=project,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )

        finished = datetime.now()
        status = "complete" if result.returncode == 0 and is_complete(project) else "failed"
        print(
            f"[{index:02d}/{len(city_dirs)}] {status.upper()} {city_name} "
            f"({(finished - started).total_seconds():.0f}s)"
        )
        rows.append({
            "city_folder": city_name,
            "status": status,
            "return_code": result.returncode,
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": finished.isoformat(timespec="seconds"),
            "log_path": str(log_path),
        })
        write_status(status_path, rows)

    failures = [row for row in rows if row["status"] == "failed"]
    print(f"Batch finished: {len(rows) - len(failures)} OK, {len(failures)} failed")
    print(f"Status: {status_path}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
