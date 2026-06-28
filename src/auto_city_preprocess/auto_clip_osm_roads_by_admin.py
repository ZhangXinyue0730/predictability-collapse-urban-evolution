"""
auto_clip_osm_roads_by_admin.py
===============================

Clip a province-level Geofabrik OSM road shapefile by one city admin boundary.

Example:
  python auto_clip_osm_roads_by_admin.py \\
    --city-name 苏州市 \\
    --roads-shp jiangsu-260504-free/gis_osm_roads_free_1.shp \\
    --admin-shp jiangsu-260504-free/gis_osm_adminareas_a_free_1.shp \\
    --output-dir suzhou-260504-free

Output:
  <output-dir>/gis_osm_roads_free_1.shp
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Clip OSM roads by city admin boundary.")
    p.add_argument("--city-name", required=True, help="Admin name, e.g. 苏州市")
    p.add_argument("--roads-shp", type=Path, required=True)
    p.add_argument("--admin-shp", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--name-field", default="name")
    p.add_argument("--admin-level", default=None, help="Optional fclass/admin level filter")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = args.output_dir / "gis_osm_roads_free_1.shp"
    if out_path.exists() and not args.overwrite:
        print(f"Output exists, skip: {out_path}")
        return

    try:
        import geopandas as gpd
    except ImportError as exc:
        raise SystemExit("geopandas is required. Install geopandas or run in the project .venv.") from exc

    print("Loading admin boundaries...")
    admin = gpd.read_file(args.admin_shp)
    if args.name_field not in admin.columns:
        raise SystemExit(f"Name field not found: {args.name_field}")

    if args.city_name in {"北京", "北京市", "天津", "天津市", "上海", "上海市", "重庆", "重庆市"}:
        selected = admin.to_crs(3857)
        if hasattr(selected.geometry, "union_all"):
            union_geom = selected.geometry.union_all()
        else:
            union_geom = selected.geometry.unary_union
        boundary = gpd.GeoDataFrame(
            [{args.name_field: args.city_name, "area_m2": union_geom.area}],
            geometry=[union_geom],
            crs=selected.crs,
        )
        print(f"Selected boundary by union: {args.city_name}, area={union_geom.area/1e6:.1f} km2")
    else:
        name_series = admin[args.name_field].astype(str)
        selected = admin[name_series == args.city_name]
        if selected.empty:
            selected = admin[name_series.str.contains(args.city_name, regex=False, na=False)]
        if args.admin_level and "fclass" in selected.columns:
            selected = selected[selected["fclass"].astype(str) == str(args.admin_level)]
        if selected.empty:
            candidates = sorted(name_series[name_series.str.contains(args.city_name[:2], regex=False, na=False)].unique())
            raise SystemExit(f"No admin boundary matched {args.city_name}. Similar names: {candidates[:20]}")

        selected = selected.to_crs(3857)
        selected["area_m2"] = selected.geometry.area
        boundary = selected.sort_values("area_m2", ascending=False).head(1)
        print(f"Selected boundary: {boundary.iloc[0][args.name_field]}, area={boundary.iloc[0]['area_m2']/1e6:.1f} km2")

    print("Loading roads...")
    roads_crs = gpd.read_file(args.roads_shp, rows=1).crs
    bbox = tuple(boundary.to_crs(roads_crs).total_bounds)
    roads = gpd.read_file(args.roads_shp, bbox=bbox)
    if roads.empty:
        raise SystemExit("No roads loaded inside boundary bbox.")
    roads = roads.to_crs(3857)
    print(f"Roads in bbox: {len(roads):,}")

    print("Clipping roads...")
    clipped = gpd.clip(roads, boundary[["geometry"]])
    clipped = clipped[~clipped.geometry.is_empty & clipped.geometry.notna()]
    clipped = clipped.explode(index_parts=False).reset_index(drop=True)
    clipped = clipped[clipped.geometry.geom_type == "LineString"]
    clipped = clipped.to_crs(4326)
    print(f"Clipped roads: {len(clipped):,}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clipped.to_file(out_path, encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
