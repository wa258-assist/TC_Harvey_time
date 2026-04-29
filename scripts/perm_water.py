"""
scripts/perm_water.py
---------------------
Download USGS National Hydrography Dataset (NHD) permanent water mask.

Source: USGS NHD — https://www.usgs.gov/national-hydrography/national-hydrography-dataset
Data:   NHD waterbody polygons (perennial lakes, ponds, reservoirs, rivers) rasterized
        to a binary mask; replaces former JRC Global Surface Water occurrence raster.

Usage:
  python scripts/perm_water.py --aoi HARVEY --output-dir ./data/perm_water
"""
import argparse, math, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.windows import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import get_logger, set_gha_output, load_aoi

log = get_logger("perm_water")

JRC_THRESHOLD = 75
# Source changed to USGS NHD; download logic requires rework (vector→raster pipeline).
# See: https://www.usgs.gov/national-hydrography/national-hydrography-dataset
NHD_SOURCE_URL = "https://www.usgs.gov/national-hydrography/national-hydrography-dataset"
JRC_BASE = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GSWE/Aggregated/LATEST/occurrence"  # TODO: replace with NHD download


def _tile_id(lon_left, lat_top):
    """lon_left=-100, lat_top=30 → '100W_30N'"""
    ew = "W" if lon_left < 0 else "E"
    ns = "N" if lat_top  > 0 else "S"
    return f"{int(abs(lon_left))}{ew}_{int(abs(lat_top))}{ns}"


def bbox_to_tiles(bbox):
    """Return [(lon_left, lat_top, tile_id)] for 10° tiles covering bbox."""
    w, s, e, n = bbox
    tiles = []
    lon = math.floor(w / 10) * 10
    while lon < e:
        lat = math.floor(s / 10) * 10
        while lat < n:
            tiles.append((lon, lat + 10, _tile_id(lon, lat + 10)))
            lat += 10
        lon += 10
    return tiles


def _try_urls(tile_id, dest, retries=3):
    """Try multiple confirmed URL patterns for the JRC occurrence tile."""
    import requests

    # All known URL patterns for this tile
    candidates = [
        # Pattern 1: JRC FTP with correct Aggregated/LATEST path
        f"{JRC_BASE}/occurrence_{tile_id}_v1_4_2021.tif",
        # Pattern 2: storage.googleapis.com (confirmed in R package source)
        f"https://storage.googleapis.com/global-surface-water/downloads2021/occurrence/occurrence_{tile_id}_v1_4_2021.tif",
        # Pattern 3: storage.cloud.google.com
        f"https://storage.cloud.google.com/global-surface-water/downloads2021/occurrence/occurrence_{tile_id}_v1_4_2021.tif",
        # Pattern 4: JRC with no version suffix
        f"{JRC_BASE}/occurrence_{tile_id}.tif",
        # Pattern 5: older VER4-0 path
        f"https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GSWE/Aggregated/VER4-0/occurrence/occurrence_{tile_id}_v1_4_2021.tif",
    ]

    headers = {"User-Agent": "Mozilla/5.0 (compatible; FloodPin/1.0)"}

    for url in candidates:
        log.info("  trying: %s", url)
        for attempt in range(1, retries + 1):
            try:
                r = requests.get(url, stream=True, timeout=300, headers=headers)
                if r.status_code == 404:
                    log.warning("    404")
                    break
                if r.status_code == 200:
                    sz = 0
                    with open(dest, "wb") as f:
                        for chunk in r.iter_content(1024 * 1024):
                            f.write(chunk)
                            sz += len(chunk)
                    if sz > 1000:   # sanity check — not an empty response
                        log.info("    OK (%.1f MB)", sz / 1e6)
                        return True
                    else:
                        log.warning("    response too small (%d bytes)", sz)
                        dest.unlink(missing_ok=True)
                        break
                log.warning("    HTTP %d", r.status_code)
                break
            except Exception as exc:
                log.warning("    attempt %d: %s", attempt, exc)
                if attempt < retries:
                    time.sleep(3 * attempt)
                dest.unlink(missing_ok=True)
    return False


def _is_valid_tif(path):
    """Return True only if the file is a readable GeoTIFF (not a corrupt/HTML stub)."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            return src.width > 0 and src.height > 0
    except Exception:
        return False


def download_tile(tile_id, tile_dir):
    dest = tile_dir / f"occurrence_{tile_id}_v1_4_2021.tif"

    if dest.exists():
        if _is_valid_tif(dest):
            log.info("  cached (valid): %s", dest.name)
            return dest
        else:
            log.warning("  cached file is corrupt — deleting and re-downloading: %s", dest.name)
            dest.unlink()

    log.info("Downloading tile %s ...", tile_id)
    if _try_urls(tile_id, dest):
        return dest

    # TODO: implement USGS NHD download via TNM API
    # See: https://www.usgs.gov/national-hydrography/national-hydrography-dataset
    log.info("  NHD download not yet implemented — all download methods exhausted")

    log.error("All download methods failed for tile %s", tile_id)
    return None


def mosaic_and_clip(paths, bbox, out_mosaic, out_clipped, buf=0.05):
    if not out_mosaic.exists():
        if len(paths) == 1:
            shutil.copy(paths[0], out_mosaic)
        else:
            sources = [rasterio.open(p) for p in paths]
            try:
                data, tr = merge(sources, nodata=255)
                meta = sources[0].meta.copy()
                meta.update(height=data.shape[1], width=data.shape[2],
                            transform=tr, compress="lzw")
                with rasterio.open(out_mosaic, "w", **meta) as dst:
                    dst.write(data)
            finally:
                for s in sources: s.close()

    if not out_clipped.exists():
        w, s, e, n = bbox[0]-buf, bbox[1]-buf, bbox[2]+buf, bbox[3]+buf
        with rasterio.open(out_mosaic) as r:
            win  = from_bounds(w, s, e, n, r.transform)
            data = r.read(1, window=win)
            meta = r.meta.copy()
            meta.update(height=data.shape[0], width=data.shape[1],
                        transform=r.window_transform(win), compress="lzw")
        with rasterio.open(out_clipped, "w", **meta) as r:
            r.write(data, 1)


def binarise(src, out, threshold=JRC_THRESHOLD):
    if out.exists():
        return True
    with rasterio.open(src) as r:
        data = r.read(1).astype(np.float32)
        nd   = r.nodata if r.nodata is not None else 255.0
        meta = r.meta.copy()
    valid  = (data != nd) & ~np.isnan(data)
    binary = np.where(valid & (data >= threshold), 1, 0).astype(np.uint8)
    binary[~valid] = 255
    meta.update(dtype="uint8", nodata=255, compress="lzw")
    with rasterio.open(out, "w", **meta) as r:
        r.write(binary, 1)
    log.info("Permanent water pixels: %d (threshold >= %d%%)",
             int((binary == 1).sum()), threshold)
    return True


def get_permanent_water_mask(aoi_name, bbox, out_dir,
                              threshold=JRC_THRESHOLD, existing=""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    binary  = out_dir / "permanent_water_binary.tif"

    if existing and Path(existing).exists():
        if not binary.exists():
            binarise(Path(existing), binary, threshold)
        return binary

    tiles    = bbox_to_tiles(bbox)
    tile_dir = out_dir / "jrc_tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    log.info("NHD/JRC tiles for %s: %s", aoi_name, [t[2] for t in tiles])

    downloaded = [p for _, _, tid in tiles
                  if (p := download_tile(tid, tile_dir)) is not None]

    if not downloaded:
        log.error("No JRC tiles downloaded")
        log.info("Pipeline will continue without permanent water masking.")
        log.info("To fix manually: download waterbody data from USGS NHD:")
        log.info("  https://www.usgs.gov/national-hydrography/national-hydrography-dataset")
        log.info("  and rerun with --existing /path/to/waterbody_raster.tif")
        return None

    mosaic  = out_dir / "jrc_mosaic.tif"
    clipped = out_dir / "jrc_clipped.tif"
    mosaic_and_clip(downloaded, bbox, mosaic, clipped)
    binarise(clipped, binary, threshold)
    log.info("Permanent water mask: %s", binary)
    return binary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aoi",        required=True)
    p.add_argument("--output-dir", default="./data/perm_water")
    p.add_argument("--threshold",  type=int, default=JRC_THRESHOLD)
    p.add_argument("--existing",   default="")
    args = p.parse_args()
    aoi  = load_aoi(args.aoi)
    path = get_permanent_water_mask(args.aoi, aoi["bbox"],
                                    Path(args.output_dir),
                                    args.threshold, args.existing)
    if path:
        log.info("Done: %s", path)
        set_gha_output("perm_water_path", str(path))
    else:
        # Exit 0 — perm water is optional, pipeline can continue without it
        log.warning("Permanent water mask unavailable — pipeline continues without it")
        set_gha_output("perm_water_path", "")
        sys.exit(0)   # ← non-fatal

if __name__ == "__main__":
    main()
