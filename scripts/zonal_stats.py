"""
scripts/zonal_stats.py
-----------------------
Raster → H3 zonal statistics.
Computes per-cell: hazard_probability p10/mean/p90, depth p10/mean/p90,
permanent water flag, QA, and confidence.
"""
import logging, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import geometry_mask
from shapely.geometry import mapping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    CLOUD_FRACTION_THRESHOLD, MIN_COVERAGE_FOR_PIN,
    PERM_WATER_CELL_THRESHOLD, depth_to_bin,
)
from scripts.h3_grid import h3_to_polygon, h3_bounds, h3_centroid
from scripts.utils import get_logger

log = get_logger("zonal_stats")


def _read_pixels(src, cell, nodata=None):
    """
    Read all pixels inside an H3 cell from an open rasterio dataset.
    Returns (all_pixels, valid_pixels) as 1-D float32 arrays.
    """
    bounds  = h3_bounds(cell)
    polygon = h3_to_polygon(cell)
    try:
        win = src.window(*bounds)
        win = win.round_lengths().round_offsets()
        if win.width <= 0 or win.height <= 0:
            return np.array([]), np.array([])
        data = src.read(1, window=win).astype(np.float32)
        tr   = src.window_transform(win)
        mask = geometry_mask([mapping(polygon)], transform=tr,
                             invert=True, out_shape=data.shape)
        all_px = data[mask]
        vmask  = np.ones(len(all_px), bool)
        if nodata is not None and not np.isnan(float(nodata)):
            vmask &= (all_px != nodata)
        vmask &= ~np.isnan(all_px) & np.isfinite(all_px)
        return all_px, all_px[vmask]
    except Exception:
        return np.array([]), np.array([])


def _flood_stats(valid_px, all_px):
    n_all  = max(len(all_px), 1)
    n_v    = len(valid_px)
    if n_v == 0:
        return dict(hazard_probability_pct=None,
                    hazard_probability_p10=None,
                    hazard_probability_p90=None,
                    confidence_overall=0.0, n_valid=0, n_total=n_all)
    fp  = valid_px.clip(0.0, 1.0)
    return dict(
        hazard_probability_pct = round(float(fp.mean()), 4),
        hazard_probability_p10 = round(float(np.percentile(fp, 10)), 4),
        hazard_probability_p90 = round(float(np.percentile(fp, 90)), 4),
        confidence_overall     = round(float(n_v/n_all), 4),
        n_valid                = n_v,
        n_total                = n_all,
    )


def _depth_stats(depth_src, cell):
    empty = dict(depth_bin_m=None, depth_p10_m=None, depth_p90_m=None, depth_mean_m=None)
    if depth_src is None:
        return empty
    _, vp = _read_pixels(depth_src, cell)
    fp    = vp[(vp > 0) & ~np.isnan(vp)]
    if len(fp) == 0:
        return empty
    mean_d = float(fp.mean())
    return dict(
        depth_bin_m  = depth_to_bin(mean_d),
        depth_p10_m  = round(float(np.percentile(fp, 10)), 3),
        depth_p90_m  = round(float(np.percentile(fp, 90)), 3),
        depth_mean_m = round(mean_d, 3),
    )


def _perm_water(pw_src, cell):
    if pw_src is None:
        return False
    _, vp = _read_pixels(pw_src, cell, nodata=255)
    if len(vp) == 0:
        return False
    return float((vp > 0).mean()) >= PERM_WATER_CELL_THRESHOLD


def _qa(flood_stats, all_px):
    conf      = flood_stats["confidence_overall"]
    n_all     = max(flood_stats["n_total"], 1)
    cloud_frac = float(np.isnan(all_px).sum()) / n_all if len(all_px) else 0.0
    if cloud_frac >= CLOUD_FRACTION_THRESHOLD:
        return dict(qa_flag=False, qa_reason="cloud",           coverage_flag="cloud")
    if conf < MIN_COVERAGE_FOR_PIN:
        return dict(qa_flag=False, qa_reason="partial coverage", coverage_flag="partial")
    if conf == 0.0:
        return dict(qa_flag=False, qa_reason="no valid data",   coverage_flag="unknown")
    return dict(qa_flag=True,  qa_reason=None,               coverage_flag="good")


def run_zonal_stats(viirs_path, depth_path, pw_path, cells, log_every=2000):
    """
    Compute zonal stats for all H3 cells.
    Returns DataFrame with one row per cell that has valid data.
    """
    log.info("Zonal stats: %d cells", len(cells))
    t0 = time.time()

    viirs_src = rasterio.open(viirs_path)
    depth_src = rasterio.open(depth_path) if depth_path and Path(depth_path).exists() else None
    pw_src    = rasterio.open(pw_path)    if pw_path    and Path(pw_path).exists()    else None
    viirs_nd  = viirs_src.nodata
    rows      = []

    for i, cell in enumerate(cells):
        if i and i % log_every == 0:
            elapsed = time.time()-t0
            log.info("  %d/%d  (%.0f/s ETA %.0fs)", i, len(cells),
                     i/elapsed, (len(cells)-i)/max(i/elapsed,0.001))
        try:
            lat, lon   = h3_centroid(cell)
            all_px, vp = _read_pixels(viirs_src, cell, nodata=viirs_nd)
            if len(all_px) == 0:
                continue
            fs = _flood_stats(vp, all_px)
            if fs["hazard_probability_pct"] is None:
                continue
            ds = _depth_stats(depth_src, cell)
            pw = _perm_water(pw_src, cell)
            qa = _qa(fs, all_px)
            rows.append({"h3_index": cell, "lat": round(lat,6), "lon": round(lon,6),
                          **fs, **ds, "is_perm_water": pw, **qa})
        except Exception as exc:
            log.debug("cell %s: %s", cell, exc)

    viirs_src.close()
    if depth_src: depth_src.close()
    if pw_src:    pw_src.close()

    df = pd.DataFrame(rows)
    log.info("Zonal stats done: %d cells with data in %.1fs", len(df), time.time()-t0)
    return df
