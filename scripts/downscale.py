"""
scripts/downscale.py
--------------------
Wrapper around downscaling_classic.py (Li et al. algorithm).
Handles import fallback gracefully and applies all settings from config.
"""
import logging, sys, time, traceback
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    DS_MIN_WATER_FRACTION, DS_ELEVATION_STEP, DS_FRACTION_TOLERANCE,
    DS_POLYGON_MAX_BOX, DS_POLYGON_MIN_SIZE, DS_POLYGON_ELEV_THRESHOLD,
    DS_DRYLAND_THRESHOLD, DS_ENABLE_DRYLAND_CORRECTION,
    DS_ENABLE_FLOOD_EXTENSION, DS_MIN_DEPTH, DS_MAX_DEPTH,
)
from scripts.utils import get_logger

log = get_logger("downscale")

# Try to import downscaling_classic — look in repo root and scripts/
_DS = None
for _search in [
    Path(__file__).resolve().parent.parent / "downscaling_classic.py",
    Path(__file__).resolve().parent / "downscaling_classic.py",
]:
    if _search.exists():
        import importlib.util, shutil
        spec = importlib.util.spec_from_file_location("downscaling_classic", str(_search))
        _DS  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_DS)
        log.info("Loaded downscaling_classic from %s", _search)
        break

if _DS is None:
    log.warning("downscaling_classic.py not found — depth will be skipped")


def _make_config():
    cfg = _DS.Config()
    cfg.min_water_fraction        = DS_MIN_WATER_FRACTION
    cfg.elevation_step            = DS_ELEVATION_STEP
    cfg.fraction_tolerance        = DS_FRACTION_TOLERANCE
    cfg.polygon_max_box           = DS_POLYGON_MAX_BOX
    cfg.polygon_min_size          = DS_POLYGON_MIN_SIZE
    cfg.polygon_elev_threshold    = DS_POLYGON_ELEV_THRESHOLD
    cfg.dryland_inundation_threshold = DS_DRYLAND_THRESHOLD
    cfg.enable_dryland_correction = DS_ENABLE_DRYLAND_CORRECTION
    cfg.enable_flood_extension    = DS_ENABLE_FLOOD_EXTENSION
    cfg.min_depth                 = DS_MIN_DEPTH
    cfg.max_depth                 = DS_MAX_DEPTH
    cfg.scale_ratio               = 0   # auto from raster resolution
    return cfg


def run_downscaling(mosaic_path, dem_path, perm_water_path, out_dir, date_str):
    """
    Run Li et al. downscaling for one VIIRS mosaic.
    Returns (flood_extent_path, water_depth_path) or (None, None).
    """
    extent_path = Path(out_dir) / f"{date_str}_flood_extent.tif"
    depth_path  = Path(out_dir) / f"{date_str}_water_depth.tif"

    if extent_path.exists() and depth_path.exists():
        log.info("[%s] downscaled outputs already exist", date_str)
        return extent_path, depth_path

    if _DS is None:
        log.warning("[%s] downscaling_classic not available — skipping", date_str)
        return None, None

    if not dem_path or not Path(dem_path).exists():
        log.warning("[%s] DEM not available — skipping downscaling", date_str)
        return None, None

    log.info("[%s] running downscaling...", date_str)
    t0 = time.time()
    try:
        cfg             = _make_config()
        cfg.dem_path    = str(dem_path)
        cfg.permanent_water_path = str(perm_water_path) if perm_water_path and Path(perm_water_path).exists() else ""
        cfg.output_dir  = str(out_dir)

        static = _DS.load_static_data(cfg, str(mosaic_path))
        wf_win = Window(col_off=static["wf_col_start"], row_off=static["wf_row_start"],
                        width=static["Wc"], height=static["Hc"])
        with rasterio.open(mosaic_path) as src:
            wf = src.read(1, window=wf_win).astype(np.float32)
            if src.nodata is not None:
                wf[wf == src.nodata] = np.nan

        n_water = int(np.count_nonzero((~np.isnan(wf)) & (wf >= cfg.min_water_fraction)))
        log.info("  [%s] coarse water pixels: %d", date_str, n_water)
        if n_water == 0:
            log.warning("  [%s] no water pixels — skipping", date_str)
            return None, None

        pixel_wsl      = _DS.compute_pixel_water_levels(wf, static["dem"], static["perm_water"], static["scale"], cfg)
        poly_labels, poly_wsl = _DS.cluster_water_polygons(wf, pixel_wsl, static["dem"], static["scale"], cfg)
        poly_wsl       = _DS.correct_polygon_water_levels(poly_labels, poly_wsl, wf, static["dem"], static["scale"], cfg)
        flood_extent, wsl_map = _DS.downscale_to_fine(wf, static["dem"], static["perm_water"], poly_labels, poly_wsl, static["scale"], cfg)
        water_depth    = _DS.compute_water_depth(flood_extent, wsl_map, static["dem"], cfg)

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        prof = static["dem_profile"].copy()

        prof.update(dtype="uint8", count=1, nodata=255, compress="lzw")
        with rasterio.open(extent_path, "w", **prof) as dst:
            dst.write(flood_extent, 1)

        prof.update(dtype="float32", nodata=np.nan)
        with rasterio.open(depth_path, "w", **prof) as dst:
            dst.write(water_depth, 1)

        n_flood = int(np.count_nonzero(flood_extent))
        log.info("  [%s] flooded cells: %d  (%.1fs)", date_str, n_flood, time.time()-t0)
        return extent_path, depth_path

    except Exception as exc:
        log.error("[%s] downscaling failed: %s", date_str, exc)
        traceback.print_exc()
        return None, None
