"""
harvey_floodpin_pipeline.py
============================
Full pipeline for Tropical Cyclone Harvey (Aug 25-29, 2017)

Steps:
  1. Download VIIRS flood fraction tiles from NOAA S3 for Harvey dates
     (tiles covering Texas Gulf Coast: GLB038, GLB039)
  2. Mosaic + normalise to flood fraction [0,1] (from VIIRS_download.py)
  3. Download permanent water body mask from JRC Global Surface Water
  4. Run downscaling (from downscaling_classic.py) → fine flood_extent + water_depth
  5. Build H3 resolution-8 grid (~500m) over the AOI
  6. Zonal statistics per H3 cell:
       - hazard_probability_pct  (v26) = mean flood fraction
       - hazard_probability_p10  (v27) = p10 of flood fraction pixels
       - hazard_probability_p90  (v28) = p90 of flood fraction pixels
       - depth_bin_m             (v22) = mean depth → bin
       - depth_p10_m             (v23) = p10 depth
       - depth_p90_m             (v24) = p90 depth
       - confidence_overall      (v29) = valid pixel coverage fraction
  7. Mask permanent water bodies from flooded cells
  8. Emit a FloodPin for every flooded H3 cell (hazard_probability_pct > threshold)
     with all v1–v36 fields from the FloodPin schema

Usage:
  python harvey_floodpin_pipeline.py \
    --dem path/to/dem.tif \
    --output-dir ./harvey_output \
    [--perm-water path/to/copernicus_water.tif]  # optional: auto-downloaded if omitted
    [--flood-threshold 0.10]
    [--skip-download]  # if VIIRS tiles already downloaded

Dependencies:
  pip install boto3 botocore rasterio h3 numpy pandas scipy shapely pyproj requests tqdm
  pip install numba  # optional, speeds up downscaling
"""

import argparse
import datetime
import json
import logging
import os
import sys
import time
import traceback
import uuid
from pathlib import Path

import boto3
import botocore
import botocore.client
import h3
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.merge import merge
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from scipy import ndimage
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import ACTIVE_EVENT_ID, KNOWN_EVENTS, get_event_dates
from scripts.perm_water import get_permanent_water_mask as _jrc_perm_water
from scripts.viirs_download import find_tiles_by_bbox, mosaic_and_normalise as _mosaic_normalise
from scripts.notify import (
    notify_validation_errors, notify_pipeline_complete, notify_pipeline_failed,
)

# Import the downscaling engine from the project
# (downscaling_classic.py must be in the same directory or on PYTHONPATH)
try:
    from downscaling_classic import (
        Config as DownscaleConfig,
        compute_pixel_water_levels,
        cluster_water_polygons,
        correct_polygon_water_levels,
        downscale_to_fine,
        compute_water_depth,
        load_static_data,
    )
    HAS_DOWNSCALER = True
except ImportError:
    HAS_DOWNSCALER = False
    print("WARNING: downscaling_classic.py not found. "
          "Depth downscaling will be skipped.")

logging.basicConfig(
    level=logging.INFO,
    format="[Harvey][%(name)s] %(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("pipeline")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

EVENT_ID        = ACTIVE_EVENT_ID
EVENT_NAME      = "Harvey"
HAZARD_TYPE     = "flood"
PRODUCT         = "FloodSENS"
SOURCE_SYSTEM   = "FloodPin"
SOURCE_ORG      = "FloodPin"
SENSOR          = "VIIRS"
GSD_M           = 375
SPATIAL_RES_M   = 375
H3_RESOLUTION   = 8            # ~500m edge length
FLOOD_THRESHOLD = 0.10         # min hazard_probability_pct to emit a pin
_EVENT          = KNOWN_EVENTS[EVENT_ID]
VALID_START_UTC = _EVENT["valid_start"]
VALID_END_UTC   = _EVENT["valid_end"]
HARVEY_DATES    = get_event_dates(EVENT_ID)


# NOAA S3 bucket for VIIRS flood product
VIIRS_BUCKET     = "noaa-jpss"
VIIRS_BASE_PREFIX = "JPSS_Blended_Products/VFM_1day_GLB/TIF/"

# AOI bounding box for Harvey (Houston / Texas Gulf Coast)
# [lon_min, lat_min, lon_max, lat_max]
HARVEY_BBOX = [-98.0, 27.5, -93.0, 31.5]

# Copernicus Water Bodies product endpoint (seasonal, 100m global)
# Tile: 20°N–40°N, 100°W–80°W covers Texas
COPERNICUS_WATER_URL = (
    "https://s3-eu-west-1.amazonaws.com/vito.landcover.global/v3.0.1/"
    "2017/W100N40/W100N40_PROBAV_LC100_global_v3.0.1_2017-conso_Water-Seasonal-Coverfraction-layer_EPSG-4326.tif"
)

REQUIRED_FIELDS = {
    "v1_event_id", "v2_pin_id", "v5_obs_time_utc", "v6_hazard_type",
    "v14_latitude", "v15_longitude", "v18_H3_res8",
    "v26_hazard_probability_pct", "v29_confidence_overall",
    "v34_sensor", "v35_gsd_m", "v36_coverage_flag",
}


def validate_pins_v(pins):
    valid, errors = [], []
    for i, p in enumerate(pins):
        missing = [f for f in REQUIRED_FIELDS if p.get(f) is None]
        if missing:
            errors.append({"index": i, "pin_id": p.get("v2_pin_id"), "missing": missing})
        else:
            valid.append(p)
    return valid, errors


# Depth bins per v22 spec
DEPTH_BINS = [
    (0.0,   0.1,  "0.0-0.1"),
    (0.1,   0.3,  "0.1-0.3"),
    (0.3,   0.6,  "0.3-0.6"),
    (0.6,   1.0,  "0.6-1.0"),
    (1.0,   2.0,  "1.0-2.0"),
    (2.0, 999.0,  ">2.0"),
]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 + 2: VIIRS DOWNLOAD & MOSAIC
# (adapted directly from VIIRS_download.py)
# ─────────────────────────────────────────────────────────────────────────────

def validate_date(year, month, day):
    try:
        datetime.date(year, month, day)
        return True
    except ValueError:
        return False


def download_single_viirs_tile(
    bucket_name: str,
    base_prefix: str,
    year: int, month: int, day: int,
    glb_number: str,
    download_dir: Path,
) -> Path | None:
    """Download one VIIRS flood fraction tile from NOAA S3."""
    if not validate_date(year, month, day):
        log.warning("Invalid date %04d-%02d-%02d for tile GLB%s", year, month, day, glb_number)
        return None

    date_str = f"{year:04d}{month:02d}{day:02d}"
    date_prefix = f"{year:04d}/{month:02d}/{day:02d}/"
    full_prefix = (base_prefix + date_prefix).replace("\\", "/")
    if not full_prefix.endswith("/"):
        full_prefix += "/"

    s3_cfg = botocore.client.Config(signature_version=botocore.UNSIGNED)
    s3 = boto3.client("s3", region_name="us-east-1", config=s3_cfg)
    paginator = s3.get_paginator("list_objects_v2")

    expected_product = "VIIRS-Flood-1day-"
    expected_glb     = f"GLB{glb_number}"
    expected_date    = f"_s{date_str}"

    local_path = None
    try:
        for page in paginator.paginate(Bucket=bucket_name, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                key  = obj["Key"]
                name = os.path.basename(key)
                if not name.lower().endswith(".tif"):
                    continue
                if (expected_product in name and expected_glb in name and expected_date in name):
                    dest = download_dir / name
                    if dest.exists():
                        log.info("  Already exists: %s", name)
                        return dest
                    log.info("  Downloading %s ...", name)
                    s3.download_file(bucket_name, key, str(dest))
                    log.info("  Saved to %s", dest)
                    return dest
    except botocore.exceptions.ClientError as exc:
        log.error("S3 error for %s/%s: %s", date_str, glb_number, exc)
    return None


def mosaic_viirs_tiles(tile_paths: list, output_path: Path) -> bool:
    """
    Mosaic VIIRS tiles and normalise to flood fraction [0,1].
    Values 100-200 are flood class; (value-100)/100 gives fraction.
    Matches mosaic_rasters() in VIIRS_download.py exactly.
    """
    sources = []
    try:
        for p in tile_paths:
            if p and Path(p).exists():
                sources.append(rasterio.open(p))
            else:
                log.warning("  Missing tile %s — skipping", p)

        if not sources:
            log.error("No valid tiles to mosaic")
            return False

        mosaic, out_trans = merge(sources)
        mosaic = mosaic.astype(np.float32)

        # Keep only flood-class values (100–200); set everything else to NaN
        mosaic[mosaic < 100] = np.nan
        mosaic[mosaic > 200] = np.nan

        if np.any(~np.isnan(mosaic)):
            log.info("  Raw range: %.0f – %.0f",
                     float(np.nanmin(mosaic)), float(np.nanmax(mosaic)))
        else:
            log.warning("  No flood pixels in mosaic")

        # Normalise to [0,1] flood fraction
        with np.errstate(invalid="ignore"):
            mosaic = (mosaic - 100.0) / 100.0

        out_meta = sources[0].meta.copy()
        out_meta.update(
            driver="GTiff", height=mosaic.shape[1], width=mosaic.shape[2],
            transform=out_trans, crs=sources[0].crs,
            dtype=np.float32, nodata=np.nan,
            compress="lzw", predictor=3, tiled=True,
            blockxsize=512, blockysize=512,
        )
        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(mosaic)

        log.info("  Mosaic → %s", output_path.name)
        return True

    except Exception as exc:
        log.error("Mosaic error: %s", exc)
        return False
    finally:
        for src in sources:
            src.close()


def download_viirs_harvey(
    download_dir: Path,
    dates: list = HARVEY_DATES,
    bbox: list = HARVEY_BBOX,
) -> dict:
    """
    Download all VIIRS tiles for Harvey dates and mosaic each day.
    Tiles are discovered spatially — no hardcoded GLB numbers.
    Returns {date_str: mosaic_path} for days with valid mosaics.
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    daily_mosaics = {}

    for obs_date in dates:
        date_str    = obs_date.strftime("%Y%m%d")
        mosaic_path = download_dir / f"{date_str}_VIIRS_mosaic.tif"

        if mosaic_path.exists():
            with rasterio.open(mosaic_path) as src:
                rb = src.bounds
            w, s, e, n = bbox
            if rb.right > w and rb.left < e and rb.top > s and rb.bottom < n:
                log.info("[%s] Valid mosaic exists", date_str)
                daily_mosaics[date_str] = mosaic_path
                continue
            log.warning("[%s] Cached mosaic outside AOI — deleting", date_str)
            mosaic_path.unlink()

        log.info("=== Downloading VIIRS for %s ===", obs_date.isoformat())
        tile_paths = find_tiles_by_bbox(
            obs_date.year, obs_date.month, obs_date.day,
            bbox, download_dir / "tiles",
        )

        if tile_paths:
            ok = _mosaic_normalise(tile_paths, mosaic_path)
            if ok:
                daily_mosaics[date_str] = mosaic_path
                for p in tile_paths:
                    try: Path(p).unlink()
                    except Exception: pass
        else:
            log.warning("[%s] No tiles found overlapping AOI", date_str)

    return daily_mosaics


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: PERMANENT WATER MASK
# ─────────────────────────────────────────────────────────────────────────────

def download_copernicus_water(output_path: Path) -> bool:
    """
    Download the Copernicus Global Land Service water bodies layer.
    This is the Seasonal Water Cover Fraction from LC100 v3.0.1 (2017).
    URL: https://land.copernicus.eu/en/products/water-bodies

    The tile W100N40 covers 100°W–80°W, 20°N–40°N → includes all of Texas.
    """
    if output_path.exists():
        log.info("Permanent water mask already exists at %s", output_path)
        return True

    log.info("Downloading Copernicus water bodies mask...")
    log.info("  Source: %s", COPERNICUS_WATER_URL)

    try:
        resp = requests.get(COPERNICUS_WATER_URL, stream=True, timeout=300)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100 * downloaded / total
                    print(f"  Downloading... {pct:.0f}%", end="\r")
        print()
        log.info("  Saved to %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)
        return True
    except Exception as exc:
        log.error("Failed to download Copernicus water mask: %s", exc)
        log.info("  You can manually download from:")
        log.info("  https://land.copernicus.eu/en/products/water-bodies")
        log.info("  and pass the path via --perm-water")
        return False


def build_permanent_water_binary(
    water_fraction_path: Path,
    output_path: Path,
    threshold: float = 50.0,
) -> Path | None:
    """
    Convert the Copernicus water cover fraction (0–100%) to a binary mask.
    Pixels with seasonal water fraction >= threshold% are marked as permanent water.
    Default threshold = 50% (majority water in a pixel counts as permanent).
    """
    if output_path.exists():
        return output_path

    if not water_fraction_path.exists():
        log.warning("Water fraction source not found: %s", water_fraction_path)
        return None

    log.info("Building permanent water binary mask (threshold=%.0f%%)...", threshold)
    with rasterio.open(water_fraction_path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata or 255.0
        meta = src.meta.copy()

    data[data == nodata] = 0.0
    binary = (data >= threshold).astype(np.uint8)

    meta.update(dtype="uint8", nodata=255, compress="lzw")
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(binary, 1)

    n_water = int(np.sum(binary))
    log.info("  Permanent water pixels: %d", n_water)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: DOWNSCALING (wrapper around downscaling_classic.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_downscaling(
    mosaic_path: Path,
    dem_path: str,
    perm_water_path: str,
    output_dir: Path,
    date_str: str,
) -> tuple:
    """
    Wrap the downscaling_classic pipeline for one VIIRS mosaic.
    Returns (flood_extent_path, water_depth_path) or (None, None) on failure.
    """
    if not HAS_DOWNSCALER:
        log.warning("Downscaler not available — skipping depth computation for %s", date_str)
        return None, None

    extent_path = output_dir / f"{date_str}_flood_extent.tif"
    depth_path  = output_dir / f"{date_str}_water_depth.tif"

    if extent_path.exists() and depth_path.exists():
        log.info("[%s] Downscaled outputs already exist — skipping", date_str)
        return extent_path, depth_path

    log.info("[%s] Running downscaling...", date_str)

    cfg = DownscaleConfig()
    cfg.dem_path              = dem_path
    cfg.permanent_water_path  = perm_water_path or ""
    cfg.output_dir            = str(output_dir)
    cfg.scale_ratio           = 0          # auto-detect from raster resolution
    cfg.min_water_fraction    = 0.01
    cfg.elevation_step        = 0.1
    cfg.fraction_tolerance    = 0.02
    cfg.polygon_max_box       = 25
    cfg.polygon_min_size      = 3
    cfg.polygon_elev_threshold = 10.0
    cfg.dryland_inundation_threshold = 0.30
    cfg.enable_dryland_correction    = True
    cfg.enable_flood_extension       = True
    cfg.min_depth = 0.0
    cfg.max_depth = 50.0

    try:
        static = load_static_data(cfg, str(mosaic_path))

        Hc = static["Hc"]
        Wc = static["Wc"]
        wf_window = Window(
            col_off=static["wf_col_start"],
            row_off=static["wf_row_start"],
            width=Wc, height=Hc,
        )
        with rasterio.open(mosaic_path) as src:
            wf = src.read(1, window=wf_window).astype(np.float32)
            if src.nodata is not None:
                wf[wf == src.nodata] = np.nan

        n_water = int(np.count_nonzero((~np.isnan(wf)) & (wf >= cfg.min_water_fraction)))
        log.info("  [%s] Coarse water pixels: %d", date_str, n_water)
        if n_water == 0:
            log.warning("  [%s] No water pixels — skipping", date_str)
            return None, None

        pixel_wsl = compute_pixel_water_levels(wf, static["dem"], static["perm_water"], static["scale"], cfg)
        poly_labels, poly_wsl = cluster_water_polygons(wf, pixel_wsl, static["dem"], static["scale"], cfg)
        poly_wsl = correct_polygon_water_levels(poly_labels, poly_wsl, wf, static["dem"], static["scale"], cfg)
        flood_extent, wsl_map = downscale_to_fine(wf, static["dem"], static["perm_water"], poly_labels, poly_wsl, static["scale"], cfg)
        water_depth = compute_water_depth(flood_extent, wsl_map, static["dem"], cfg)

        # Write outputs
        output_dir.mkdir(parents=True, exist_ok=True)
        prof = static["dem_profile"].copy()

        prof.update(dtype="uint8", count=1, nodata=255, compress="lzw")
        with rasterio.open(extent_path, "w", **prof) as dst:
            dst.write(flood_extent, 1)

        prof.update(dtype="float32", nodata=np.nan)
        with rasterio.open(depth_path, "w", **prof) as dst:
            dst.write(water_depth, 1)

        log.info("  [%s] Flood cells: %d  → %s", date_str,
                 int(np.count_nonzero(flood_extent)), extent_path.name)
        return extent_path, depth_path

    except Exception as exc:
        log.error("[%s] Downscaling failed: %s", date_str, exc)
        traceback.print_exc()
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: H3 GRID GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def get_h3_cells_for_bbox(
    bbox: list,
    resolution: int = H3_RESOLUTION,
) -> list:
    """
    Return all H3 cells at `resolution` that intersect the bounding box.
    bbox = [lon_min, lat_min, lon_max, lat_max]
    """
    lon_min, lat_min, lon_max, lat_max = bbox

    # Build a GeoJSON polygon for the bbox and fill with H3
    bbox_geojson = {
        "type": "Polygon",
        "coordinates": [[
            [lon_min, lat_min],
            [lon_max, lat_min],
            [lon_max, lat_max],
            [lon_min, lat_max],
            [lon_min, lat_min],
        ]],
    }

    cells = list(h3.geo_to_cells(bbox_geojson, res=resolution))
    log.info("H3 res=%d: %d cells over bbox %s", resolution, len(cells), bbox)
    return cells


def h3_cell_to_shapely(cell: str) -> Polygon:
    """Convert an H3 cell index to a Shapely polygon (lon/lat)."""
    boundary = h3.cell_to_boundary(cell)  # returns list of (lat, lon)
    # H3 returns (lat, lon) — swap to (lon, lat) for Shapely
    coords = [(lon, lat) for lat, lon in boundary]
    return Polygon(coords)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: ZONAL STATISTICS PER H3 CELL
# ─────────────────────────────────────────────────────────────────────────────

def raster_pixels_in_h3(
    raster_src,
    cell_bounds: tuple,
    cell_polygon: Polygon,
    nodata=None,
) -> np.ndarray:
    """
    Extract valid pixel values from `raster_src` that fall within an H3 cell.
    Uses the cell bounds for windowed read, then masks to exact cell polygon.
    Returns 1-D array of valid pixel values.
    """
    from rasterio.mask import geometry_mask
    from shapely.geometry import mapping

    lon_min, lat_min, lon_max, lat_max = cell_bounds

    try:
        # Windowed read — much faster than reading the whole raster
        window = raster_src.window(lon_min, lat_min, lon_max, lat_max)
        window = window.round_lengths().round_offsets()

        if window.width <= 0 or window.height <= 0:
            return np.array([])

        data = raster_src.read(1, window=window)
        win_transform = raster_src.window_transform(window)

        # Mask to exact H3 polygon
        mask = geometry_mask(
            [mapping(cell_polygon)],
            transform=win_transform,
            invert=True,          # True = inside polygon
            out_shape=data.shape,
        )

        pixels = data[mask].astype(np.float32)

        # Remove nodata
        if nodata is not None and not np.isnan(nodata):
            pixels = pixels[pixels != nodata]
        pixels = pixels[~np.isnan(pixels)]
        pixels = pixels[np.isfinite(pixels)]

        return pixels

    except Exception:
        return np.array([])


def zonal_stats_h3(
    mosaic_path: Path,
    flood_extent_path: Path | None,
    water_depth_path: Path | None,
    perm_water_path: str | None,
    cells: list,
) -> dict:
    """
    Compute per-H3-cell zonal statistics from VIIRS mosaic + downscaled depth.

    Returns dict: {h3_index: {stat_name: value, ...}}
    """
    log.info("Computing H3 zonal statistics for %d cells...", len(cells))
    t0 = time.time()

    results = {}

    # Open all rasters once
    viirs_src   = rasterio.open(mosaic_path)
    extent_src  = rasterio.open(flood_extent_path) if flood_extent_path and Path(flood_extent_path).exists() else None
    depth_src   = rasterio.open(water_depth_path)  if water_depth_path  and Path(water_depth_path).exists()  else None
    pw_src      = rasterio.open(perm_water_path)   if perm_water_path   and Path(perm_water_path).exists()   else None

    viirs_nodata = viirs_src.nodata

    for i, cell in enumerate(cells):
        if i % 5000 == 0 and i > 0:
            log.info("  ... %d / %d cells processed", i, len(cells))

        try:
            lat, lon = h3.cell_to_latlng(cell)
            cell_poly = h3_cell_to_shapely(cell)
            bounds = cell_poly.bounds  # (minx, miny, maxx, maxy)

            # ── VIIRS flood fraction pixels ──────────────────────────────
            viirs_pixels = raster_pixels_in_h3(viirs_src, bounds, cell_poly, nodata=viirs_nodata)
            n_total_pixels = max(len(viirs_pixels), 1)

            # Filter to valid (non-NaN) pixels for confidence
            valid_fraction_pixels = viirs_pixels[viirs_pixels >= 0]
            n_valid = len(valid_fraction_pixels)

            if n_valid == 0:
                # No satellite data over this cell → skip
                continue

            confidence_overall = float(n_valid / n_total_pixels)

            # Hazard probability (v26, v27, v28)
            hazard_prob_pct = float(np.mean(valid_fraction_pixels))
            hazard_p10 = float(np.percentile(valid_fraction_pixels, 10)) if n_valid > 1 else hazard_prob_pct
            hazard_p90 = float(np.percentile(valid_fraction_pixels, 90)) if n_valid > 1 else hazard_prob_pct

            # ── Permanent water mask ─────────────────────────────────────
            is_perm_water = False
            if pw_src is not None:
                pw_pixels = raster_pixels_in_h3(pw_src, bounds, cell_poly, nodata=255)
                if len(pw_pixels) > 0:
                    perm_fraction = float(np.mean(pw_pixels > 0))
                    is_perm_water = (perm_fraction >= 0.5)  # majority permanent water

            # ── Depth statistics (v22, v23, v24) from downscaled depth ───
            depth_mean = np.nan
            depth_p10  = np.nan
            depth_p90  = np.nan
            depth_bin  = None

            if depth_src is not None:
                depth_pixels = raster_pixels_in_h3(depth_src, bounds, cell_poly, nodata=None)
                depth_valid  = depth_pixels[(depth_pixels > 0) & ~np.isnan(depth_pixels)]
                if len(depth_valid) > 0:
                    depth_mean = float(np.mean(depth_valid))
                    depth_p10  = float(np.percentile(depth_valid, 10))
                    depth_p90  = float(np.percentile(depth_valid, 90))
                    for lo, hi, label in DEPTH_BINS:
                        if lo <= depth_mean < hi:
                            depth_bin = label
                            break

            # ── Coverage / QA ────────────────────────────────────────────
            # Check for cloud cover (NaN pixels in VIIRS = cloud or no data)
            n_nan  = int(np.sum(np.isnan(viirs_pixels)))
            if n_nan / max(len(viirs_pixels), 1) > 0.8:
                qa_flag   = False
                qa_reason = "cloud"
                coverage  = "cloud"
            elif confidence_overall < 0.5:
                qa_flag   = False
                qa_reason = "partial coverage"
                coverage  = "partial"
            else:
                qa_flag   = True
                qa_reason = None
                coverage  = "good"

            results[cell] = {
                "lat": lat,
                "lon": lon,
                "hazard_probability_pct": round(hazard_prob_pct, 4),
                "hazard_probability_p10": round(hazard_p10, 4),
                "hazard_probability_p90": round(hazard_p90, 4),
                "confidence_overall":     round(confidence_overall, 4),
                "is_perm_water":          is_perm_water,
                "depth_mean_m":           None if np.isnan(depth_mean) else round(float(depth_mean), 3),
                "depth_p10_m":            None if np.isnan(depth_p10)  else round(float(depth_p10),  3),
                "depth_p90_m":            None if np.isnan(depth_p90)  else round(float(depth_p90),  3),
                "depth_bin_m":            depth_bin,
                "qa_flag":                qa_flag,
                "qa_reason":              qa_reason,
                "coverage_flag":          coverage,
                "n_pixels":               n_valid,
            }

        except Exception as exc:
            log.debug("Cell %s error: %s", cell, exc)
            continue

    # Close rasters
    viirs_src.close()
    if extent_src: extent_src.close()
    if depth_src:  depth_src.close()
    if pw_src:     pw_src.close()

    elapsed = time.time() - t0
    log.info("Zonal stats complete: %d cells with data in %.1fs", len(results), elapsed)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 + 8: PERMANENT WATER MASKING + PIN GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def assign_depth_bin(depth_m: float | None) -> str | None:
    if depth_m is None or np.isnan(depth_m):
        return None
    for lo, hi, label in DEPTH_BINS:
        if lo <= depth_m < hi:
            return label
    return ">2.0"


def generate_pins(
    date_str: str,
    stats: dict,
    flood_threshold: float = FLOOD_THRESHOLD,
) -> list:
    """
    Emit one FloodPin record per flooded H3 cell.
    Excludes permanent water bodies and cells below flood threshold.
    Maps all fields to v1–v36 FloodPin schema.
    """
    obs_date  = datetime.datetime.strptime(date_str, "%Y%m%d")
    obs_utc   = obs_date.strftime("%Y-%m-%dT00:00:00Z")
    issued_utc = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    pins = []
    pin_counter = 1

    for cell, s in stats.items():
        # Filter 1: flood threshold
        if s["hazard_probability_pct"] < flood_threshold:
            continue

        # Filter 2: permanent water body
        if s["is_perm_water"]:
            continue

        # Deterministic pin_id from event + date + H3 index
        pin_id = f"{EVENT_ID}_{date_str}_{str(pin_counter).zfill(4)}"
        pin_counter += 1

        # Support radius from H3 cell edge length (~500m at res 8 → radius ~300m)
        # H3 res-8 avg edge length ≈ 461m, area ≈ 0.737 km²
        support_radius_m = 300

        pin = {
            # Required schema fields v1–v36
            "v1_event_id":                  EVENT_ID,
            "v2_pin_id":                    pin_id,
            "v3_revision":                  1,
            "v4_issued_utc":                issued_utc,
            "v5_obs_time_utc":              obs_utc,
            "v6_hazard_type":               HAZARD_TYPE,
            "v7_data_version":              "1.0",
            "v8_evidence_type":             "Optical",
            "v9_product":                   PRODUCT,
            "v10_source_system":            SOURCE_SYSTEM,
            "v11_pin_mode":                 "Observed",
            "v12_valid_start_utc":          VALID_START_UTC,
            "v13_valid_end_utc":            VALID_END_UTC,
            "v14_latitude":                 round(s["lat"], 6),
            "v15_longitude":                round(s["lon"], 6),
            "v16_spatial_res_m":            SPATIAL_RES_M,
            "v17_support_radius_m":         support_radius_m,
            "v18_H3_res8":                  cell,
            "v19_source_org":               SOURCE_ORG,
            "v20_qa_flag":                  s["qa_flag"],
            "v21_qa_reason":                s.get("qa_reason"),
            "v22_depth_bin_m":              s.get("depth_bin_m"),
            "v23_depth_p10_m":              s.get("depth_p10_m"),
            "v24_depth_p90_m":              s.get("depth_p90_m"),
            "v25_depth_source":             "VIIRS_extent+DEM_30m",
            "v26_hazard_probability_pct":   round(s["hazard_probability_pct"], 4),
            "v27_hazard_probability_p10":   round(s["hazard_probability_p10"], 4),
            "v28_hazard_probability_p90":   round(s["hazard_probability_p90"], 4),
            "v29_confidence_overall":       round(s["confidence_overall"], 4),
            "v30_confidence_interval_basis": "heuristic",
            "v31_validation_metric":        None,
            "v32_independent_source":       None,
            "v33_validation_score":         None,
            "v34_sensor":                   SENSOR,
            "v35_gsd_m":                    GSD_M,
            "v36_coverage_flag":            s["coverage_flag"],
            # Extra operational fields
            "obs_date":                     obs_date.date().isoformat(),
            "h3_resolution":                H3_RESOLUTION,
            "n_pixels":                     s["n_pixels"],
            "perm_water_masked":            s["is_perm_water"],
        }
        pins.append(pin)

    log.info("[%s] Generated %d flood pins (threshold=%.2f, excl. perm water)",
             date_str, len(pins), flood_threshold)
    return pins


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_harvey_pipeline(
    dem_path: str,
    output_dir: Path,
    perm_water_path: str | None = None,
    flood_threshold: float = FLOOD_THRESHOLD,
    skip_download: bool = False,
    bbox: list = HARVEY_BBOX,
    notify_to: str = "",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    viirs_dir   = output_dir / "viirs_tiles"
    ds_dir      = output_dir / "downscaled"
    pins_dir    = output_dir / "pins"
    results_dir = output_dir / "results"
    for d in (pins_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("FloodPin Harvey Pipeline")
    log.info("Event: %s | Dates: %s – %s", EVENT_ID,
             HARVEY_DATES[0].isoformat(), HARVEY_DATES[-1].isoformat())
    log.info("AOI bbox: %s", bbox)
    log.info("H3 resolution: %d (~500m)", H3_RESOLUTION)
    log.info("Flood threshold: %.2f", flood_threshold)
    log.info("=" * 60)

    # ── Step 1+2: Download VIIRS ─────────────────────────────────────────
    if not skip_download:
        daily_mosaics = download_viirs_harvey(viirs_dir, bbox=bbox)
    else:
        log.info("Skipping download — scanning %s for existing mosaics", viirs_dir)
        daily_mosaics = {
            p.stem.split("_")[0]: p
            for p in sorted(viirs_dir.glob("*_VIIRS_mosaic.tif"))
        }

    if not daily_mosaics:
        log.error("No VIIRS mosaics available — aborting.")
        notify_pipeline_failed(notify_to, "No VIIRS mosaics available", EVENT_ID)
        sys.exit(1)
    log.info("Available mosaics: %s", list(daily_mosaics.keys()))

    # ── Step 3: Permanent water mask (JRC Global Surface Water) ─────────
    # JRC GSW occurrence >= 75% over 1984-2021 is the correct source:
    # it reflects stable open water, not transient flood water from Harvey itself.
    if perm_water_path is None:
        pw = _jrc_perm_water(
            aoi_name = "HARVEY",
            bbox     = bbox,
            out_dir  = output_dir / "perm_water",
        )
        perm_water_path = str(pw) if pw else None
        if perm_water_path is None:
            log.warning("Proceeding without permanent water mask")

    # ── Step 5: H3 grid ─────────────────────────────────────────────────
    log.info("Building H3 grid...")
    h3_cells = get_h3_cells_for_bbox(bbox, resolution=H3_RESOLUTION)
    log.info("H3 cells: %d", len(h3_cells))

    # ── Steps 4, 6, 7, 8: Per-day loop ──────────────────────────────────
    all_pins = []

    for date_str, mosaic_path in sorted(daily_mosaics.items()):
        log.info("\n%s\n[%s] Processing %s\n%s",
                 "─" * 50, date_str, mosaic_path.name, "─" * 50)

        # Step 4: Downscaling
        flood_extent_path, water_depth_path = None, None
        if dem_path:
            flood_extent_path, water_depth_path = run_downscaling(
                mosaic_path, dem_path, perm_water_path, ds_dir, date_str,
            )
        else:
            log.info("No DEM provided — depth computation skipped")

        # Step 6: Zonal statistics
        stats = zonal_stats_h3(
            mosaic_path, flood_extent_path, water_depth_path,
            perm_water_path, h3_cells,
        )

        # Steps 7+8: Mask perm water + generate pins
        daily_pins = generate_pins(date_str, stats, flood_threshold)
        valid_pins, errors = validate_pins_v(daily_pins)
        all_pins.extend(valid_pins)

        if errors:
            notify_validation_errors(notify_to, errors, EVENT_ID, date_str)

        # Save daily pins
        if valid_pins:
            day_df   = pd.DataFrame(valid_pins)
            day_csv  = pins_dir / f"{EVENT_ID}_{date_str}_pins.csv"
            day_json = pins_dir / f"{EVENT_ID}_{date_str}_pins.json"
            day_df.to_csv(day_csv, index=False)
            day_df.to_csv(results_dir / f"{EVENT_ID}_{date_str}_pins.csv", index=False)
            with open(day_json, "w") as f:
                json.dump(valid_pins, f, indent=2, default=str)
            log.info("[%s] Saved %d pins → %s", date_str, len(valid_pins), day_csv.name)
        else:
            log.warning("[%s] No pins generated", date_str)

    # ── Final outputs ────────────────────────────────────────────────────
    if all_pins:
        df = pd.DataFrame(all_pins)

        # Full event CSV
        event_csv = output_dir / f"{EVENT_ID}_Harvey_all_pins.csv"
        df.to_csv(event_csv, index=False)
        df.to_csv(results_dir / f"{EVENT_ID}_Harvey_all_pins.csv", index=False)
        log.info("\nTotal pins: %d across %d days", len(df), len(daily_mosaics))

        # Summary stats
        mean_conf = float(df["v29_confidence_overall"].mean())
        log.info("Flood probability stats:")
        log.info("  Mean hazard_prob:  %.3f", df["v26_hazard_probability_pct"].mean())
        log.info("  P10 hazard_prob:   %.3f", df["v27_hazard_probability_p10"].mean())
        log.info("  P90 hazard_prob:   %.3f", df["v28_hazard_probability_p90"].mean())
        log.info("  Mean confidence:   %.3f", mean_conf)
        if "v22_depth_bin_m" in df.columns:
            log.info("Depth bin distribution:\n%s",
                     df["v22_depth_bin_m"].value_counts().to_string())

        # Event-level JSON summary
        summary = {
            "event_id":        EVENT_ID,
            "event_name":      EVENT_NAME,
            "hazard_type":     HAZARD_TYPE,
            "valid_start":     VALID_START_UTC,
            "valid_end":       VALID_END_UTC,
            "h3_resolution":   H3_RESOLUTION,
            "flood_threshold": flood_threshold,
            "total_pins":      len(df),
            "mean_confidence": round(mean_conf, 4),
            "days_covered":    sorted(daily_mosaics.keys()),
            "bbox":            bbox,
            "output_csv":      str(event_csv),
            "generated_utc":   datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        summary_path = output_dir / f"{EVENT_ID}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        (results_dir / f"{EVENT_ID}_summary.json").write_text(json.dumps(summary, indent=2))
        log.info("Summary → %s", summary_path)
        log.info("Full CSV → %s", event_csv)

        result_csvs = sorted(results_dir.glob("*.csv"))
        notify_pipeline_complete(notify_to, summary, attachments=result_csvs)

    else:
        log.warning("No pins generated for Harvey event")
        notify_pipeline_failed(notify_to, "No pins generated after all processing", EVENT_ID)

    log.info("Pipeline complete.")
    return all_pins


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FloodPin pipeline for Tropical Cyclone Harvey (Aug 25-29, 2017)"
    )
    parser.add_argument(
        "--dem", default=None,
        help="Path to DEM GeoTIFF (e.g. FABDEM or SRTM 30m). "
             "Required for water depth (v22-v24). If omitted, depth fields will be null."
    )
    parser.add_argument(
        "--output-dir", default="./harvey_output",
        help="Root output directory (default: ./harvey_output)"
    )
    parser.add_argument(
        "--perm-water", default=None,
        help="Path to permanent water binary mask GeoTIFF. "
             "If omitted, Copernicus water bodies will be auto-downloaded."
    )
    parser.add_argument(
        "--flood-threshold", type=float, default=FLOOD_THRESHOLD,
        help=f"Minimum hazard_probability_pct to emit a pin (default: {FLOOD_THRESHOLD})"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip VIIRS download; use existing mosaics in <output-dir>/viirs_tiles/"
    )
    parser.add_argument(
        "--bbox", default=",".join(str(v) for v in HARVEY_BBOX),
        help="AOI bounding box lon_min,lat_min,lon_max,lat_max "
             f"(default: {HARVEY_BBOX})"
    )
    parser.add_argument(
        "--notify-email", default=os.environ.get("NOTIFY_EMAIL", ""),
        help="Engineer email for validation/completion notifications (overrides NOTIFY_EMAIL env var)"
    )
    args = parser.parse_args()

    bbox = [float(x) for x in args.bbox.split(",")]

    run_harvey_pipeline(
        dem_path        = args.dem,
        output_dir      = Path(args.output_dir),
        perm_water_path = args.perm_water,
        flood_threshold = args.flood_threshold,
        skip_download   = args.skip_download,
        bbox            = bbox,
        notify_to       = args.notify_email,
    )
