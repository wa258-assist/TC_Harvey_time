"""
scripts/h3_grid.py — H3 hexagonal grid utilities
H3 resolution 8: avg edge ~461m, avg area ~0.737 km²
"""
import json, logging, sys
from pathlib import Path
import h3
import numpy as np
from shapely.geometry import Polygon, mapping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import H3_RESOLUTION
from scripts.utils import get_logger

log = get_logger("h3_grid")


def generate_h3_cells(bbox, resolution=H3_RESOLUTION):
    """Return all H3 cells at `resolution` intersecting bbox=[W,S,E,N]."""
    w, s, e, n = bbox
    geojson = {"type": "Polygon", "coordinates": [[
        [w,s],[e,s],[e,n],[w,n],[w,s]
    ]]}
    cells = list(h3.geo_to_cells(geojson, res=resolution))
    log.info("H3 res=%d: %d cells over [W%.2f S%.2f E%.2f N%.2f]",
             resolution, len(cells), w, s, e, n)
    return cells


def h3_to_polygon(cell):
    """H3 cell → Shapely Polygon (lon, lat ordering)."""
    boundary = h3.cell_to_boundary(cell)   # returns [(lat,lon),...]
    return Polygon([(lon, lat) for lat, lon in boundary])


def h3_bounds(cell):
    """Return (minx, miny, maxx, maxy) for an H3 cell."""
    return h3_to_polygon(cell).bounds


def h3_centroid(cell):
    """Return (lat, lon) centroid."""
    return h3.cell_to_latlng(cell)


def support_radius_m(resolution=H3_RESOLUTION):
    """Inscribed circle radius in metres (edge × √3/2)."""
    edge = h3.average_hexagon_edge_length(resolution, unit="m")
    return int(edge * (3**0.5) / 2)


def cells_over_raster(raster_path, cells):
    """
    Filter cells to those whose bounding box intersects the raster extent.
    Fast extent-only check — no pixel reads.
    """
    import rasterio
    with rasterio.open(raster_path) as src:
        rb = src.bounds
    kept = []
    for cell in cells:
        minx, miny, maxx, maxy = h3_bounds(cell)
        if minx < rb.right and maxx > rb.left and miny < rb.top and maxy > rb.bottom:
            kept.append(cell)
    log.info("Extent filter: %d / %d cells overlap raster", len(kept), len(cells))
    return kept


def save_geojson(cells, out_path):
    """Save cell grid as GeoJSON for inspection in QGIS / Mapbox."""
    feats = []
    for cell in cells:
        lat, lon = h3_centroid(cell)
        feats.append({"type": "Feature",
                       "geometry": mapping(h3_to_polygon(cell)),
                       "properties": {"h3_index": cell, "lat": round(lat,6), "lon": round(lon,6)}})
    Path(out_path).write_text(json.dumps({"type":"FeatureCollection","features":feats}))
    log.info("GeoJSON saved: %s", out_path)
