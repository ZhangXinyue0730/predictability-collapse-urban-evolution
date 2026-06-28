"""
Step 13: dynamic built-up update, expansion, and transition statistics.

Update:
    Any class change among classes 1-6 inside the t1 dynamic built-up extent.
    Samples involving vacant/unused class 0 in either period are excluded.

Expansion:
    Cells belonging to the t2 dynamic built-up extent but outside the t1
    dynamic built-up extent.

The script summarizes all seven TCULU classes and keeps the standard output
filenames used by the national aggregation scripts.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = [
    "PingFang SC", "Arial Unicode MS", "Heiti TC", "Songti SC", "DejaVu Sans"
]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import CITY_NAME, DATA, FIG, LU_COLORS, LU_NAMES, PX_M, YEARS


# Class 0 is non-construction land and is excluded from the renewal sample.
CLASSES = list(range(1, 7))
NEXT_YEAR = dict(zip(YEARS[:-1], YEARS[1:]))
PIXEL_AREA_KM2 = PX_M * PX_M / 1_000_000.0
LU_NAMES_EN = {
    0: "Vacant/unused",
    1: "Residential",
    2: "Commercial",
    3: "Public service",
    4: "Industrial/warehouse",
    5: "Road/municipal",
    6: "Green/water",
}


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_dataset():
    path = DATA / "dataset.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    d = np.load(path, allow_pickle=True)
    x = d["X"]
    return {
        "y1": np.argmax(x[:, :7].astype(np.float32), axis=1).astype(np.int8),
        "y2": d["y"].astype(np.int8),
        "is_update": d["is_update"].astype(bool),
        "year_t1": d["year_t1"].astype(np.int16),
    }


def transition_counter(y1, y2, mask):
    return Counter(
        (int(a), int(b))
        for a, b in zip(y1[mask], y2[mask])
        if int(a) != int(b)
    )


def extent_stats(t1, t2):
    p1 = DATA / f"builtup_extent_dynamic_{t1}.npy"
    p2 = DATA / f"builtup_extent_dynamic_{t2}.npy"
    if not p1.exists() or not p2.exists():
        raise FileNotFoundError("Run 09.5_dynamic_builtup_extent.py first.")
    e1 = np.load(p1).astype(bool)
    e2 = np.load(p2).astype(bool)
    expansion = e2 & ~e1
    contraction = e1 & ~e2
    return {
        "t1_extent_cells": int(e1.sum()),
        "t2_extent_cells": int(e2.sum()),
        "expansion_cells": int(expansion.sum()),
        "expansion_area_km2": float(expansion.sum() * PIXEL_AREA_KM2),
        "expansion_rate_vs_t1_percent": (
            float(expansion.sum() / e1.sum() * 100) if e1.any() else 0.0
        ),
        "expansion_share_of_t2_percent": (
            float(expansion.sum() / e2.sum() * 100) if e2.any() else 0.0
        ),
        "contraction_cells": int(contraction.sum()),
        "contraction_area_km2": float(contraction.sum() * PIXEL_AREA_KM2),
        "net_extent_change_cells": int(e2.sum() - e1.sum()),
        "net_extent_change_area_km2": float((e2.sum() - e1.sum()) * PIXEL_AREA_KM2),
    }


def save_period_outputs(data):
    rate_rows = []
    count_rows = []
    top_rows = []

    for t1, t2 in zip(YEARS[:-1], YEARS[1:]):
        period = f"{t1}->{t2}"
        mask = data["year_t1"] == t1
        update_mask = mask & data["is_update"]
        n_samples = int(mask.sum())
        n_updates = int(update_mask.sum())
        ext = extent_stats(t1, t2)

        rate_rows.append({
            "period": period,
            "year_t1": t1,
            "year_t2": t2,
            "samples": n_samples,
            "updates": n_updates,
            "update_area_km2": n_updates * PIXEL_AREA_KM2,
            "update_rate_percent": n_updates / n_samples * 100 if n_samples else 0.0,
            **ext,
        })

        counter = transition_counter(data["y1"], data["y2"], update_mask)
        total = sum(counter.values())
        for a in CLASSES:
            for b in CLASSES:
                if a == b:
                    continue
                count = int(counter.get((a, b), 0))
                count_rows.append({
                    "period": period,
                    "year_t1": t1,
                    "year_t2": t2,
                    "from_class": a,
                    "from_name": LU_NAMES[a],
                    "from_name_en": LU_NAMES_EN[a],
                    "to_class": b,
                    "to_name": LU_NAMES[b],
                    "to_name_en": LU_NAMES_EN[b],
                    "count": count,
                    "area_km2": count * PIXEL_AREA_KM2,
                    "share_percent": count / total * 100 if total else 0.0,
                })
        for rank, ((a, b), count) in enumerate(counter.most_common(10), 1):
            top_rows.append({
                "period": period,
                "rank": rank,
                "from_class": a,
                "from_name": LU_NAMES[a],
                "from_name_en": LU_NAMES_EN[a],
                "to_class": b,
                "to_name": LU_NAMES[b],
                "to_name_en": LU_NAMES_EN[b],
                "count": int(count),
                "area_km2": count * PIXEL_AREA_KM2,
                "share_percent": count / total * 100 if total else 0.0,
            })

    rate_fields = list(rate_rows[0].keys())
    count_fields = list(count_rows[0].keys())
    top_fields = list(top_rows[0].keys())
    write_csv(DATA / "update_period_rates.csv", rate_fields, rate_rows)
    write_csv(DATA / "update_period_transition_counts.csv", count_fields, count_rows)
    write_csv(DATA / "update_period_top_transitions.csv", top_fields, top_rows)
    return rate_rows, count_rows


def save_overall_outputs(data):
    counter = transition_counter(data["y1"], data["y2"], data["is_update"])
    total = sum(counter.values())

    top_rows = []
    for rank, ((a, b), count) in enumerate(counter.most_common(), 1):
        top_rows.append({
            "rank": rank,
            "from_class": a,
            "from_name": LU_NAMES[a],
            "from_name_en": LU_NAMES_EN[a],
            "to_class": b,
            "to_name": LU_NAMES[b],
            "to_name_en": LU_NAMES_EN[b],
            "count": int(count),
            "area_km2": count * PIXEL_AREA_KM2,
            "share_percent": count / total * 100 if total else 0.0,
        })
    write_csv(DATA / "update_transition_top.csv", list(top_rows[0].keys()), top_rows)

    matrix_rows = []
    for a in CLASSES:
        row = {
            "from_class": a,
            "from_name": LU_NAMES[a],
            "from_name_en": LU_NAMES_EN[a],
        }
        for b in CLASSES:
            row[f"to_{b}_{LU_NAMES_EN[b]}"] = int(counter.get((a, b), 0))
        matrix_rows.append(row)
    write_csv(
        DATA / "update_transition_matrix.csv",
        list(matrix_rows[0].keys()),
        matrix_rows,
    )
    return counter


def plot_update_and_expansion(rate_rows):
    periods = [r["period"] for r in rate_rows]
    update_area = np.array([r["update_area_km2"] for r in rate_rows])
    expansion_area = np.array([r["expansion_area_km2"] for r in rate_rows])
    x = np.arange(len(periods))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, update_area, width, color="#2A9D8F", label="更新面积")
    ax.bar(x + width / 2, expansion_area, width, color="#D4A017", label="扩张面积")
    ax.set_xticks(x)
    ax.set_xticklabels(periods, rotation=25, ha="right")
    ax.set_ylabel("面积 (km²)")
    ax.set_title(f"{CITY_NAME} 分时期更新与扩张")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    out = FIG / "period_update_and_expansion.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def plot_transition_matrix(counter):
    matrix = np.array([
        [counter.get((a, b), 0) for b in CLASSES]
        for a in CLASSES
    ], dtype=np.int64)
    labels = [LU_NAMES_EN[c] for c in CLASSES]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="YlOrRd")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("To class")
    ax.set_ylabel("From class")
    ax.set_title(f"{CITY_NAME}: land-use update transitions within t1 extent")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")
    plt.tight_layout()
    out = FIG / "update_transition_matrix.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def main():
    print("=" * 68)
    print("Step 13: dynamic built-up update and expansion statistics")
    print(f"  city: {CITY_NAME}")
    print("=" * 68)
    data = load_dataset()
    rate_rows, _ = save_period_outputs(data)
    counter = save_overall_outputs(data)
    plot_update_and_expansion(rate_rows)
    plot_transition_matrix(counter)
    print(f"  samples: {len(data['y2']):,}")
    print(f"  updates: {int(data['is_update'].sum()):,}")
    print(f"  update rate: {data['is_update'].mean() * 100:.2f}%")
    print("Done.")


if __name__ == "__main__":
    main()
