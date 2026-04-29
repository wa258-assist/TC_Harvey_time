"""
scripts/dem_fetch.py
--------------------
Automated DEM download. Zero-auth for Harvey/CONUS (USGS 3DEP).

Usage:
  python scripts/dem_fetch.py --aoi HARVEY --output-dir ./data/dem
  python scripts/dem_fetch.py --aoi CNMI_GUAM --source opentopo --api-key KEY
"""
import argparse, math, os, sys, time
from pathlib import Path
import numpy as np
import requests
import rasterio
from rasterio.merge import merge
from rasterio.windows import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import get_logger, set_gha_output, load_aoi

log = get_logger("dem_fetch")

USGS_BASE    = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1/TIFF/current"
OPENTOPO_URL = "https://portal.opentopography.org/API/globaldem"


def resolve_bbox(aoi_input):
    try:
        return load_aoi(aoi_input)["bbox"]
    except Exception:
        parts = [float(x.strip()) for x in aoi_input.split(",")]
        assert len(parts) == 4
        return parts


# ── USGS 3DEP (zero auth) ────────────────────────────────────────────────────

def _usgs_url(lat_n, lon_w):
    name = f"USGS_1_n{lat_n:02d}w{lon_w:03d}"
    return f"{USGS_BASE}/n{lat_n:02d}w{lon_w:03d}/{name}.tif"


def bbox_to_usgs_tiles(bbox, buf=0.05):
    """
    Return (lat_north, lon_west_abs) tuples for every 1°×1° USGS tile
    that covers the bbox (with buffer).

    USGS tile n30w097 covers: lat 29–30N, lon 97–96W
      lat_north = north edge = 30
      lon_west_abs = west edge absolute = 97
    """
    w, s, e, n = bbox[0]-buf, bbox[1]-buf, bbox[2]+buf, bbox[3]+buf
    tiles = set()
    # lat: from floor(s)+1 up to ceil(n) inclusive
    for lat in range(math.floor(s) + 1, math.ceil(n) + 1):
        # lon: bbox is negative (W), abs gives positive values
        # tile lon_west_abs = absolute west edge of tile
        # For lon=-97.05 to -92.95:
        #   abs(west) = 98.05 → floor = 98
        #   abs(east) = 92.95 → floor = 92
        #   tiles needed: 93,94,95,96,97,98
        for lon in range(math.floor(abs(e)), math.floor(abs(w)) + 1):
            if lon > 0:
                tiles.add((lat, lon))
    return sorted(tiles)


def download_usgs_3dep(bbox, out_dir, retries=3):
    mosaic_path = out_dir / "dem_usgs3dep_30m.tif"
    if mosaic_path.exists():
        log.info("DEM already exists: %s", mosaic_path)
        return mosaic_path

    tiles = bbox_to_usgs_tiles(bbox)
    log.info("USGS 3DEP: %d tiles for Harvey bbox", len(tiles))

    tile_dir = out_dir / "usgs_tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for lat, lon in tiles:
        url  = _usgs_url(lat, lon)
        dest = tile_dir / f"USGS_1_n{lat:02d}w{lon:03d}.tif"
        if dest.exists():
            log.info("  cached n%02dw%03d", lat, lon)
            downloaded.append(dest)
            continue
        log.info("  downloading n%02dw%03d ...", lat, lon)
        for attempt in range(1, retries + 1):
            try:
                r = requests.get(url, timeout=180, stream=True)
                if r.status_code == 404:
                    log.warning("  n%02dw%03d 404 — skipping", lat, lon)
                    break
                r.raise_for_status()
                sz = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(512 * 1024):
                        f.write(chunk)
                        sz += len(chunk)
                log.info("  n%02dw%03d OK (%.1f MB)", lat, lon, sz / 1e6)
                downloaded.append(dest)
                time.sleep(0.3)
                break
            except Exception as exc:
                log.warning("  attempt %d/%d: %s", attempt, retries, exc)
                if attempt < retries:
                    time.sleep(5 * attempt)
                if dest.exists():
                    dest.unlink()

    if not downloaded:
        log.error("No USGS tiles downloaded")
        return None

    log.info("Mosaicking %d tiles...", len(downloaded))
    sources = [rasterio.open(p) for p in downloaded]
    try:
        mosaic, transform = merge(sources)
        meta = sources[0].meta.copy()
        meta.update(height=mosaic.shape[1], width=mosaic.shape[2],
                    transform=transform, compress="lzw", predictor=2,
                    tiled=True, blockxsize=512, blockysize=512)
        with rasterio.open(mosaic_path, "w", **meta) as dst:
            dst.write(mosaic)
        _log_stats(mosaic_path)
        return mosaic_path
    finally:
        for s in sources:
            s.close()


# ── OpenTopography SRTM (global, free key) ───────────────────────────────────

def download_opentopo_srtm(bbox, out_dir, api_key, buf=0.05):
    out = out_dir / "dem_srtm30m.tif"
    if out.exists():
        log.info("SRTM already exists: %s", out)
        return out
    if not api_key:
        log.error("OpenTopography API key required. Get free key: opentopography.org/newUser")
        return None
    w, s, e, n = bbox[0]-buf, bbox[1]-buf, bbox[2]+buf, bbox[3]+buf
    params = dict(demtype="SRTMGL1", south=s, north=n, west=w, east=e,
                  outputFormat="GTiff", API_Key=api_key)
    log.info("Fetching SRTM from OpenTopography...")
    try:
        r = requests.get(OPENTOPO_URL, params=params, timeout=300, stream=True)
        ct = r.headers.get("Content-Type", "")
        if "html" in ct or "json" in ct:
            log.error("OpenTopography error: %s", r.text[:300])
            return None
        r.raise_for_status()
        sz = 0
        with open(out, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)
                sz += len(chunk)
        log.info("SRTM saved: %.1f MB", sz / 1e6)
        _log_stats(out)
        return out
    except Exception as exc:
        log.error("OpenTopography failed: %s", exc)
        if out.exists():
            out.unlink()
        return None


def clip_dem(src, bbox, out, buf=0.01):
    if out.exists():
        return out
    w, s, e, n = bbox[0]-buf, bbox[1]-buf, bbox[2]+buf, bbox[3]+buf
    with rasterio.open(src) as r:
        win  = from_bounds(w, s, e, n, r.transform)
        data = r.read(1, window=win)
        meta = r.meta.copy()
        meta.update(height=data.shape[0], width=data.shape[1],
                    transform=r.window_transform(win), compress="lzw")
    with rasterio.open(out, "w", **meta) as r:
        r.write(data, 1)
    log.info("Clipped DEM -> %s", out)
    return out


def _log_stats(path):
    with rasterio.open(path) as r:
        d  = r.read(1).astype(float)
        nd = r.nodata
        v  = d[d != nd] if nd is not None else d[~np.isnan(d)]
        if len(v):
            log.info("  DEM: res=%.5f° valid=%d elev=[%.1f,%.1f]m CRS=EPSG:%s",
                     abs(r.transform[0]), len(v), v.min(), v.max(), r.crs.to_epsg())


def fetch_dem(aoi_or_bbox, out_dir, source="usgs_3dep", api_key="", existing=""):
    if existing and Path(existing).exists():
        log.info("Using existing DEM: %s", existing)
        return Path(existing)
    bbox = resolve_bbox(aoi_or_bbox) if isinstance(aoi_or_bbox, str) else list(aoi_or_bbox)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if source == "usgs_3dep":
        raw = download_usgs_3dep(bbox, out_dir)
        if raw:
            return clip_dem(raw, bbox, out_dir / "dem_clipped.tif")
    elif source == "opentopo":
        return download_opentopo_srtm(bbox, out_dir, api_key)
    else:
        log.error("Unknown source '%s'", source)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aoi",        default=None)
    p.add_argument("--bbox",       default=None)
    p.add_argument("--output-dir", default="./data/dem")
    p.add_argument("--source",     default="usgs_3dep",
                   choices=["usgs_3dep", "opentopo"])
    p.add_argument("--api-key",    default=os.environ.get("OPENTOPO_API_KEY", ""))
    p.add_argument("--existing",   default="")
    args = p.parse_args()
    if not args.aoi and not args.bbox:
        p.error("Provide --aoi or --bbox")
    dem = fetch_dem(args.aoi or args.bbox, Path(args.output_dir),
                    args.source, args.api_key, args.existing)
    if dem:
        log.info("DEM ready: %s", dem)
        set_gha_output("dem_path", str(dem))
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
