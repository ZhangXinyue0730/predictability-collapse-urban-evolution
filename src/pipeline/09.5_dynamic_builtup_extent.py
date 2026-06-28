"""
Build a dynamic built-up extent for each TCULU year.

The extent is inferred from the four patch-like urban classes (1-4). Core
patches separated by at most roughly 2 km are connected. Small enclosed holes
are filled so roads, municipal land, parks, and small vacant parcels inside the
continuous urban fabric remain in the comparison footprint.

This trial script does not overwrite any existing pipeline output.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).parent))
from config import CITY_NAME, DATA, FIG, YEARS, PX_M

try:
    from config import (
        DYNAMIC_CORE_CLASSES,
        DYNAMIC_LINK_DISTANCE_M,
        DYNAMIC_MAX_HOLE_AREA_KM2,
    )
except ImportError:
    DYNAMIC_CORE_CLASSES = {1, 2, 3, 4}
    DYNAMIC_LINK_DISTANCE_M = 2000.0
    DYNAMIC_MAX_HOLE_AREA_KM2 = 4.0


CORE_CLASSES = tuple(sorted(DYNAMIC_CORE_CLASSES))
MAX_LINK_DISTANCE_M = float(DYNAMIC_LINK_DISTANCE_M)
MAX_HOLE_AREA_KM2 = float(DYNAMIC_MAX_HOLE_AREA_KM2)


def disk_kernel(radius_cells):
    yy, xx = np.ogrid[
        -radius_cells: radius_cells + 1,
        -radius_cells: radius_cells + 1,
    ]
    return (xx * xx + yy * yy) <= radius_cells * radius_cells


def fill_small_holes(mask, max_hole_cells):
    all_filled = ndi.binary_fill_holes(mask)
    holes = all_filled & ~mask
    labels, n_labels = ndi.label(holes, structure=np.ones((3, 3), dtype=np.uint8))
    if n_labels == 0:
        return mask, 0

    counts = np.bincount(labels.ravel())
    fill_ids = np.flatnonzero((counts <= max_hole_cells) & (np.arange(len(counts)) > 0))
    selected = np.isin(labels, fill_ids)
    return mask | selected, int(selected.sum())


def build_extent(lu):
    seed = np.isin(lu, CORE_CLASSES)

    # A closing radius of half the maximum link distance joins patches whose
    # edge-to-edge gap is approximately 2 km or less.
    radius_cells = max(1, int(round(MAX_LINK_DISTANCE_M / (2.0 * PX_M))))
    connected = ndi.binary_closing(
        seed,
        structure=disk_kernel(radius_cells),
        border_value=0,
    )
    connected |= seed

    max_hole_cells = int(round(
        MAX_HOLE_AREA_KM2 * 1_000_000.0 / (PX_M * PX_M)
    ))
    extent, filled_hole_cells = fill_small_holes(connected, max_hole_cells)
    return seed, extent, radius_cells, filled_hole_cells


def write_summary(rows):
    path = DATA / "dynamic_builtup_extent_summary.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {path}")


def plot_previews(previews):
    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    cmap = ListedColormap(["#F7F7F7", "#F28E2B", "#1F77B4"])

    for ax, (year, seed, extent) in zip(axes.ravel(), previews):
        view = np.zeros(seed.shape, dtype=np.uint8)
        view[extent] = 1
        view[seed] = 2
        ax.imshow(view, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        ax.set_title(str(year), fontsize=13, fontweight="bold")
        ax.axis("off")

    city = CITY_NAME or "City"
    fig.suptitle(
        f"{city}: dynamic built-up extents\n"
        "blue = core classes 1-4, orange = included internal/gap area",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIG / "dynamic_builtup_extent_9years.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def main():
    print("=" * 68)
    print("Dynamic built-up extent trial")
    print(f"  city: {CITY_NAME}")
    print(f"  core classes: {CORE_CLASSES}")
    print(f"  maximum link distance: {MAX_LINK_DISTANCE_M:.0f} m")
    print(f"  maximum filled hole: {MAX_HOLE_AREA_KM2:.1f} km2")
    print("=" * 68)

    rows = []
    previews = []
    pixel_area_km2 = PX_M * PX_M / 1_000_000.0

    for year in YEARS:
        # 建成区必须由未经道路连通清理的标准化用地识别。lu_clean
        # 会把未接入主路网的真实组团置为 0，导致多组团城市范围缺失。
        lu = np.load(DATA / f"lu_{year}.npy").copy()
        lu[lu == 255] = 0
        seed, extent, radius_cells, filled_hole_cells = build_extent(lu)

        np.save(DATA / f"builtup_extent_dynamic_{year}.npy", extent.astype(np.uint8))

        seed_cells = int(seed.sum())
        extent_cells = int(extent.sum())
        added = extent & ~seed
        rows.append({
            "year": year,
            "core_seed_cells": seed_cells,
            "core_seed_area_km2": seed_cells * pixel_area_km2,
            "dynamic_extent_cells": extent_cells,
            "dynamic_extent_area_km2": extent_cells * pixel_area_km2,
            "added_internal_cells": int(added.sum()),
            "added_internal_area_km2": int(added.sum()) * pixel_area_km2,
            "road_municipal_cells_in_extent": int(np.count_nonzero(extent & (lu == 5))),
            "green_water_cells_in_extent": int(np.count_nonzero(extent & (lu == 6))),
            "vacant_cells_in_extent": int(np.count_nonzero(extent & (lu == 0))),
            "closing_radius_cells": radius_cells,
            "filled_small_hole_cells": filled_hole_cells,
        })
        previews.append((year, seed, extent))
        print(
            f"  {year}: core={seed_cells:,}, extent={extent_cells:,}, "
            f"added={int(added.sum()):,} "
            f"({100 * added.sum() / max(extent_cells, 1):.1f}%)"
        )

    write_summary(rows)
    plot_previews(previews)
    print("Done.")


if __name__ == "__main__":
    main()
