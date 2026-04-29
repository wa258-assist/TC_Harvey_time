"""
scripts/viirs_download.py
--------------------------
Download VIIRS 1-day flood fraction tiles from NOAA S3 and mosaic them.

Tile discovery is spatial: for each date all available TIFs are listed from S3,
downloaded, and checked against the AOI bbox — only overlapping tiles are kept.
This avoids the need to hardcode GLB tile numbers, whose geographic mapping in
the NOAA JPSS VFM product does not follow the expected SSEC 17×8 grid.
"""
import os, sys
from pathlib import Path

import numpy as np
import boto3, botocore, botocore.client
import rasterio
from rasterio.merge import merge

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import get_logger

log = get_logger("viirs_download")

BUCKET      = "noaa-jpss"
BASE_PREFIX = "JPSS_Blended_Products/VFM_1day_GLB/TIF/"


def _s3_client():
    cfg = botocore.client.Config(signature_version=botocore.UNSIGNED)
    return boto3.client("s3", region_name="us-east-1", config=cfg)


def _overlaps(raster_path, bbox):
    """Return True if raster bounds intersect bbox=[W, S, E, N]."""
    w, s, e, n = bbox
    with rasterio.open(raster_path) as src:
        rb = src.bounds
    return rb.right > w and rb.left < e and rb.top > s and rb.bottom < n


def find_tiles_by_bbox(year, month, day, bbox, tile_dir):
    """
    List every VIIRS-Flood-1day TIF in the S3 date prefix, download each one,
    keep only those whose bounds overlap bbox=[W, S, E, N], delete the rest.

    Returns list of local Paths for overlapping tiles.
    """
    date_str = f"{year:04d}{month:02d}{day:02d}"
    prefix   = f"{BASE_PREFIX}{year:04d}/{month:02d}/{day:02d}/"
    tile_dir = Path(tile_dir)
    tile_dir.mkdir(parents=True, exist_ok=True)

    s3  = _s3_client()
    pag = s3.get_paginator("list_objects_v2")

    kept = []
    checked = 0
    for page in pag.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key  = obj["Key"]
            name = os.path.basename(key)
            if not (name.lower().endswith(".tif") and
                    "VIIRS-Flood-1day-" in name and
                    f"_s{date_str}" in name):
                continue

            dest = tile_dir / name
            if not dest.exists():
                try:
                    s3.download_file(BUCKET, key, str(dest))
                except Exception as exc:
                    log.warning("  download failed %s: %s", name, exc)
                    continue

            checked += 1
            try:
                with rasterio.open(dest) as src:
                    rb = src.bounds
                w, s_b, e, n = bbox
                if rb.right > w and rb.left < e and rb.top > s_b and rb.bottom < n:
                    log.info("  AOI overlap: %s  bounds=[%.2f %.2f %.2f %.2f]",
                             name, rb.left, rb.bottom, rb.right, rb.top)
                    kept.append(dest)
                else:
                    log.debug("  outside AOI: %s  bounds=[%.2f %.2f %.2f %.2f]",
                              name, rb.left, rb.bottom, rb.right, rb.top)
                    dest.unlink(missing_ok=True)
            except Exception as exc:
                log.warning("  could not read %s: %s", name, exc)
                dest.unlink(missing_ok=True)

    log.info("[%s] scanned %d tiles → %d overlap AOI", date_str, checked, len(kept))
    return kept


def mosaic_and_normalise(tile_paths, out_path):
    """
    Merge VIIRS tiles, filter to flood class (100-200),
    normalise to flood fraction [0,1], reproject to EPSG:4326, save GeoTIFF.
    """
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.crs import CRS
    TARGET_CRS = CRS.from_epsg(4326)

    sources = []
    try:
        for p in tile_paths:
            if p and Path(p).exists():
                sources.append(rasterio.open(p))
        if not sources:
            return False

        mosaic, transform = merge(sources)
        src_crs = sources[0].crs
        mosaic  = mosaic.astype(np.float32)

        # Keep only flood-class pixels; everything else → NaN
        mosaic[mosaic < 100] = np.nan
        mosaic[mosaic > 200] = np.nan

        n_flood = int(np.sum(~np.isnan(mosaic)))
        if n_flood > 0:
            log.info("  flood pixels: %d  range [%.0f, %.0f]",
                     n_flood, float(np.nanmin(mosaic)), float(np.nanmax(mosaic)))
        else:
            log.warning("  no flood pixels in mosaic")

        # Normalise: 100 → 0.0, 200 → 1.0
        with np.errstate(invalid="ignore"):
            mosaic = (mosaic - 100.0) / 100.0

        meta = sources[0].meta.copy()
        meta.update(driver="GTiff",
                    height=mosaic.shape[1], width=mosaic.shape[2],
                    transform=transform, dtype=np.float32,
                    nodata=np.nan, compress="lzw", predictor=3,
                    tiled=True, blockxsize=512, blockysize=512)

        # Reproject to EPSG:4326 if the native CRS differs
        if src_crs and src_crs != TARGET_CRS:
            log.info("  reprojecting %s → EPSG:4326", src_crs)
            dst_tr, dst_w, dst_h = calculate_default_transform(
                src_crs, TARGET_CRS,
                mosaic.shape[2], mosaic.shape[1], transform=transform,
            )
            reprojected = np.full((1, dst_h, dst_w), np.nan, dtype=np.float32)
            reproject(
                source=mosaic, destination=reprojected,
                src_transform=transform, src_crs=src_crs,
                dst_transform=dst_tr, dst_crs=TARGET_CRS,
                resampling=Resampling.nearest,
                src_nodata=np.nan, dst_nodata=np.nan,
            )
            mosaic    = reprojected
            transform = dst_tr
            meta.update(crs=TARGET_CRS, height=dst_h, width=dst_w,
                        transform=dst_tr)
        else:
            meta.update(crs=src_crs or TARGET_CRS)

        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(mosaic)

        with rasterio.open(out_path) as chk:
            rb = chk.bounds
        log.info("  mosaic saved: %s  bounds=[%.3f %.3f %.3f %.3f]",
                 Path(out_path).name, rb.left, rb.bottom, rb.right, rb.top)
        return True

    except Exception as exc:
        log.error("  mosaic error: %s", exc)
        return False
    finally:
        for s in sources:
            s.close()


def download_viirs_for_event(dates, out_dir, bbox, tiles=None):
    """
    Download and mosaic VIIRS tiles for a list of dates.

    Tiles are discovered automatically by spatial overlap with bbox=[W, S, E, N].
    The optional `tiles` argument is kept for call-site compatibility but ignored —
    spatial discovery is always used because NOAA GLB tile numbers do not follow
    a simple geographic grid.

    Returns dict {date_str: mosaic_path}
    """
    if tiles is not None:
        log.info("Note: explicit tile list %s ignored — using spatial auto-discovery", tiles)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for obs_date in dates:
        date_str    = obs_date.strftime("%Y%m%d")
        mosaic_path = out_dir / f"{date_str}_VIIRS_mosaic.tif"

        if mosaic_path.exists():
            log.info("[%s] mosaic exists — verifying bounds", date_str)
            if _overlaps(mosaic_path, bbox):
                results[date_str] = mosaic_path
                continue
            log.warning("[%s] cached mosaic outside AOI — deleting and re-downloading",
                        date_str)
            mosaic_path.unlink()

        log.info("=== %s ===", obs_date.isoformat())
        tile_paths = find_tiles_by_bbox(
            obs_date.year, obs_date.month, obs_date.day,
            bbox, out_dir / "tiles",
        )

        if not tile_paths:
            log.warning("[%s] no tiles found overlapping AOI bbox %s", date_str, bbox)
            continue

        ok = mosaic_and_normalise(tile_paths, mosaic_path)
        if ok:
            if _overlaps(mosaic_path, bbox):
                results[date_str] = mosaic_path
                for p in tile_paths:
                    try: Path(p).unlink()
                    except: pass
            else:
                log.error("[%s] mosaic still outside AOI after spatial discovery — "
                          "keeping file for manual inspection", date_str)
        else:
            log.warning("[%s] mosaic failed", date_str)

    return results
