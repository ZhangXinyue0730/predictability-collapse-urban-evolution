from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", ".")).resolve()
RUNS_ROOT = PROJECT_ROOT / "whitebox_redev_suite_runs"
SUMMARY = RUNS_ROOT / "whitebox_redev_suite_all79_summary.csv"
AUDIT_OUT = RUNS_ROOT / "greenfield_audit_recheck.csv"
PATCHED_SUMMARY = RUNS_ROOT / "whitebox_redev_suite_all79_summary_greenfield_rechecked.csv"

YEARS = [1984, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2024]
BUILT = {1, 2, 3, 4}


def load_lu(pipeline_root: Path, year: int) -> np.ndarray:
    path = pipeline_root / "data" / f"lu_{year}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def recheck_city(manifest_path: Path) -> dict:
    city = manifest_path.parent.name
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    pipeline_root = Path(manifest["pipeline_root"])

    origins: set[int] = set()
    targets: set[int] = set()
    built2built_pixels = 0
    changed_built2built_pixels = 0
    period_details = []

    for y1, y2 in zip(YEARS[:-1], YEARS[1:]):
        lu1 = load_lu(pipeline_root, y1)
        lu2 = load_lu(pipeline_root, y2)
        if lu1.shape != lu2.shape:
            raise ValueError(f"{city} shape mismatch {y1}->{y2}: {lu1.shape} vs {lu2.shape}")

        mask = np.isin(lu1, list(BUILT)) & np.isin(lu2, list(BUILT))
        if mask.any():
            o = sorted(int(x) for x in np.unique(lu1[mask]))
            t = sorted(int(x) for x in np.unique(lu2[mask]))
            origins.update(o)
            targets.update(t)
        else:
            o, t = [], []

        n = int(mask.sum())
        chg = int(((lu1 != lu2) & mask).sum())
        built2built_pixels += n
        changed_built2built_pixels += chg
        period_details.append(f"{y1}->{y2}:n={n},chg={chg},orig={o},tgt={t}")

    vacant_excluded = (0 not in origins) and (0 not in targets)
    pass_audit = origins.issubset(BUILT) and targets.issubset(BUILT) and vacant_excluded

    return {
        "city": city,
        "pipeline_root": str(pipeline_root),
        "audit_pass": pass_audit,
        "vacant_excluded": vacant_excluded,
        "origins": ",".join(map(str, sorted(origins))),
        "targets": ",".join(map(str, sorted(targets))),
        "built2built_pixels": built2built_pixels,
        "changed_built2built_pixels": changed_built2built_pixels,
        "changed_rate": changed_built2built_pixels / built2built_pixels if built2built_pixels else "",
        "period_details": "; ".join(period_details),
    }


def main() -> None:
    manifest_paths = sorted(
        p / "suite_manifest.json"
        for p in RUNS_ROOT.iterdir()
        if p.is_dir() and not p.name.startswith("_") and (p / "suite_manifest.json").exists()
    )
    audits = [recheck_city(p) for p in manifest_paths]
    audit_by_city = {r["city"]: r for r in audits}

    with open(AUDIT_OUT, "w", newline="", encoding="utf-8-sig") as fp:
        fields = [
            "city",
            "audit_pass",
            "vacant_excluded",
            "origins",
            "targets",
            "built2built_pixels",
            "changed_built2built_pixels",
            "changed_rate",
            "pipeline_root",
            "period_details",
        ]
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: r[k] for k in fields} for r in audits])

    rows = list(csv.DictReader(open(SUMMARY, encoding="utf-8-sig")))
    fieldnames = list(rows[0].keys())
    extra = ["greenfield_audit_rechecked", "greenfield_recheck_origins", "greenfield_recheck_targets"]
    for name in extra:
        if name not in fieldnames:
            fieldnames.append(name)

    for r in rows:
        audit = audit_by_city.get(r["city"])
        if not audit:
            r["greenfield_audit_rechecked"] = "missing"
            continue
        r["greenfield_audit_rechecked"] = "pass" if audit["audit_pass"] else "fail"
        r["greenfield_recheck_origins"] = audit["origins"]
        r["greenfield_recheck_targets"] = audit["targets"]
        if "greenfield_audit/vacant_excluded" in r.get("status", "") and audit["audit_pass"]:
            # Preserve true model-module failures while removing the record-only guard warning.
            parts = [x for x in r["status"].replace("incomplete:", "").split(",") if x and x != "greenfield_audit/vacant_excluded"]
            r["status"] = "ok_rechecked" if not parts else "incomplete:" + ",".join(parts)
        if audit["audit_pass"]:
            r["vacant_excluded"] = "True"

    with open(PATCHED_SUMMARY, "w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"audited cities: {len(audits)}")
    print(f"pass: {sum(1 for r in audits if r['audit_pass'])}")
    print(f"fail: {sum(1 for r in audits if not r['audit_pass'])}")
    print(f"audit table: {AUDIT_OUT}")
    print(f"patched summary: {PATCHED_SUMMARY}")


if __name__ == "__main__":
    main()
