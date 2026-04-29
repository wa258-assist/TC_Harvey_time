import glob
import math
import time
import traceback
from collections import deque
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from scipy import ndimage

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


class Config:
    input_folder: str = ""
    dem_path: str = ""
    permanent_water_path: str = ""
    output_dir: str = "./output"
    scale_ratio: int = 0

    min_water_fraction: float = 0.01
    elevation_step: float = 0.1
    fraction_tolerance: float = 0.02

    polygon_max_box: int = 25
    polygon_min_size: int = 3
    polygon_elev_threshold: float = 10.0

    dryland_inundation_threshold: float = 0.30
    enable_dryland_correction: bool = True

    enable_flood_extension: bool = True

    min_depth: float = 0.0
    max_depth: float = 50.0



def compute_pixel_water_levels(wf, dem, perm_water, scale, cfg):
    Hc, Wc = wf.shape
    pixel_wsl = np.full((Hc, Wc), np.nan, dtype=np.float32)

    print("Step 1: Computing pixel water levels...")
    t0 = time.time()

    for r in range(Hc):
        for c in range(Wc):
            fw = wf[r, c]
            if np.isnan(fw) or fw < cfg.min_water_fraction:
                continue

            r0 = r * scale
            c0 = c * scale
            dem_block = dem[r0:r0 + scale, c0:c0 + scale].ravel()

            valid = ~np.isnan(dem_block)
            dem_valid = dem_block[valid]
            n_valid = len(dem_valid)
            if n_valid == 0:
                continue

            if perm_water is not None:
                pw_block = perm_water[r0:r0 + scale, c0:c0 + scale].ravel()[valid]
                if np.any(pw_block):
                    min_h = np.min(dem_valid[pw_block])
                else:
                    min_h = np.min(dem_valid)
            else:
                min_h = np.min(dem_valid)

            max_h_possible = np.max(dem_valid)

            if fw >= 0.99:
                pixel_wsl[r, c] = max_h_possible
                continue

            h = min_h
            target_fraction = fw
            found = False

            while h <= max_h_possible + cfg.elevation_step:
                inundated_count = np.sum(dem_valid <= h)
                dem_fraction = inundated_count / n_valid

                if dem_fraction >= target_fraction - cfg.fraction_tolerance:
                    pixel_wsl[r, c] = h
                    found = True
                    break

                h += cfg.elevation_step

            if not found:
                pixel_wsl[r, c] = max_h_possible

    elapsed = time.time() - t0
    n_water = np.count_nonzero(~np.isnan(pixel_wsl))
    print(f"  Computed pixel water levels for {n_water} coarse pixels in {elapsed:.1f}s")
    return pixel_wsl



def cluster_water_polygons(wf, pixel_wsl, dem, scale, cfg):
    Hc, Wc = wf.shape
    print("Step 2: Clustering water polygons...")
    t0 = time.time()

    water_mask = (~np.isnan(pixel_wsl)).astype(np.uint8)

    coarse_median_elev = np.full((Hc, Wc), np.nan, dtype=np.float32)
    for r in range(Hc):
        for c in range(Wc):
            if water_mask[r, c]:
                r0, c0 = r * scale, c * scale
                block = dem[r0:r0 + scale, c0:c0 + scale]
                coarse_median_elev[r, c] = np.nanmedian(block)

    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    raw_labels, n_raw = ndimage.label(water_mask, structure=structure)

    polygon_labels = np.zeros_like(raw_labels, dtype=np.int32)
    next_id = 1
    elev_thresh = cfg.polygon_elev_threshold

    for raw_id in range(1, n_raw + 1):
        comp_members = set()
        for r, c in np.argwhere(raw_labels == raw_id):
            if not np.isnan(coarse_median_elev[r, c]):
                comp_members.add((int(r), int(c)))

        while comp_members:
            sr, sc = min(comp_members, key=lambda rc: coarse_median_elev[rc[0], rc[1]])
            seed_elev = coarse_median_elev[sr, sc]

            polygon_labels[sr, sc] = next_id
            comp_members.discard((sr, sc))
            stack = [(sr, sc)]
            while stack:
                cr, cc = stack.pop()
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if (nr, nc) in comp_members:
                        if abs(coarse_median_elev[nr, nc] - seed_elev) < elev_thresh:
                            polygon_labels[nr, nc] = next_id
                            comp_members.discard((nr, nc))
                            stack.append((nr, nc))
            next_id += 1

    total_polygons = next_id - 1

    polygon_wsl = {}
    for pid in range(1, total_polygons + 1):
        members = np.argwhere(polygon_labels == pid)
        if len(members) == 0:
            continue

        wsls = np.array([pixel_wsl[r, c] for r, c in members])
        fws = np.array([wf[r, c] for r, c in members])

        if len(members) <= cfg.polygon_max_box ** 2:
            partial_mask = fws < 0.99
            if np.any(partial_mask):
                polygon_wsl[pid] = float(np.mean(wsls[partial_mask]))
            else:
                polygon_wsl[pid] = float(np.max(wsls))
        else:
            m = cfg.polygon_max_box
            local_wsls = []
            for idx, (r, c) in enumerate(members):
                r_lo = max(0, r - m // 2)
                r_hi = min(Hc, r + m // 2 + 1)
                c_lo = max(0, c - m // 2)
                c_hi = min(Wc, c + m // 2 + 1)
                box_wsls = []
                for br in range(r_lo, r_hi):
                    for bc in range(c_lo, c_hi):
                        if polygon_labels[br, bc] == pid and wf[br, bc] < 0.99:
                            if not np.isnan(pixel_wsl[br, bc]):
                                box_wsls.append(pixel_wsl[br, bc])
                if box_wsls:
                    local_wsls.append(np.mean(box_wsls))
                else:
                    local_wsls.append(wsls[idx])

            polygon_wsl[pid] = float(np.mean(local_wsls))

    for pid in list(polygon_wsl.keys()):
        members = np.argwhere(polygon_labels == pid)
        if len(members) < cfg.polygon_min_size:
            all_wsls = []
            for r, c in members:
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < Hc and 0 <= nc < Wc:
                            if not np.isnan(pixel_wsl[nr, nc]) and wf[nr, nc] < 0.99:
                                all_wsls.append(pixel_wsl[nr, nc])
            if all_wsls:
                polygon_wsl[pid] = float(np.mean(all_wsls))

    elapsed = time.time() - t0
    print(f"  Created {len(polygon_wsl)} polygons ({n_raw} raw components "
          f"split by elevation to {total_polygons}) in {elapsed:.1f}s")
    return polygon_labels, polygon_wsl



def correct_polygon_water_levels(polygon_labels, polygon_wsl, wf, dem, scale, cfg):
    if not cfg.enable_dryland_correction:
        print("Step 3: Dryland correction disabled, skipping.")
        return polygon_wsl

    print("Step 3: Applying non-flooding neighbour correction...")
    t0 = time.time()
    Hc, Wc = wf.shape

    for pid, wsl in list(polygon_wsl.items()):
        members = np.argwhere(polygon_labels == pid)
        if len(members) == 0:
            continue

        dryland_cells = set()
        for r, c in members:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < Hc and 0 <= nc < Wc:
                    if polygon_labels[nr, nc] == 0:
                        dryland_cells.add((nr, nc))

        if not dryland_cells:
            continue

        total_dryland_cells = 0
        total_inundated_cells = 0
        for r, c in dryland_cells:
            r0, c0 = r * scale, c * scale
            block = dem[r0:r0 + scale, c0:c0 + scale].ravel()
            valid = ~np.isnan(block)
            n = np.sum(valid)
            if n == 0:
                continue
            total_dryland_cells += n
            total_inundated_cells += np.sum(block[valid] <= wsl)

        if total_dryland_cells == 0:
            continue

        dryland_fraction = total_inundated_cells / total_dryland_cells

        if dryland_fraction > cfg.dryland_inundation_threshold:
            all_dryland_elevs = []
            for r, c in dryland_cells:
                r0, c0 = r * scale, c * scale
                block = dem[r0:r0 + scale, c0:c0 + scale].ravel()
                valid_elevs = block[~np.isnan(block)]
                all_dryland_elevs.extend(valid_elevs.tolist())

            if all_dryland_elevs:
                all_dryland_elevs = np.sort(all_dryland_elevs)
                idx = int(cfg.dryland_inundation_threshold * len(all_dryland_elevs))
                idx = min(idx, len(all_dryland_elevs) - 1)
                corrected_wsl = all_dryland_elevs[idx]

                if corrected_wsl < wsl:
                    polygon_wsl[pid] = corrected_wsl

    elapsed = time.time() - t0
    print(f"  Dryland correction completed in {elapsed:.1f}s")
    return polygon_wsl



@njit
def _flood_fill_n4(dem_block, wsl, start_r, start_c, flooded):
    H, W = dem_block.shape
    queue_r = np.empty(H * W, dtype=np.int32)
    queue_c = np.empty(H * W, dtype=np.int32)
    head = 0
    tail = 0

    if (0 <= start_r < H and 0 <= start_c < W and
            dem_block[start_r, start_c] <= wsl and
            not flooded[start_r, start_c]):
        flooded[start_r, start_c] = True
        queue_r[tail] = start_r
        queue_c[tail] = start_c
        tail += 1

    while head < tail:
        cr = queue_r[head]
        cc = queue_c[head]
        head += 1

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr = cr + dr
            nc = cc + dc
            if 0 <= nr < H and 0 <= nc < W:
                if not flooded[nr, nc] and dem_block[nr, nc] <= wsl:
                    flooded[nr, nc] = True
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    tail += 1


def _extend_flood_monotonic(dem, flood_extent, wsl_map, polygon_labels, scale):

    Hf, Wf = dem.shape
    Hc, Wc = polygon_labels.shape

    queue = deque()
    for r, c in np.argwhere(flood_extent == 1):
        queue.append((int(r), int(c)))

    n_extended = 0
    neighbours = ((-1, 0), (1, 0), (0, -1), (0, 1))

    while queue:
        cr, cc = queue.popleft()
        cur_dem = dem[cr, cc]
        cur_wsl = wsl_map[cr, cc]
        if np.isnan(cur_wsl) or np.isnan(cur_dem):
            continue

        for dr, dc in neighbours:
            nr = cr + dr
            nc = cc + dc
            if not (0 <= nr < Hf and 0 <= nc < Wf):
                continue
            if flood_extent[nr, nc] != 0:
                continue
            ncr = nr // scale
            ncc = nc // scale
            if not (0 <= ncr < Hc and 0 <= ncc < Wc):
                continue
            if polygon_labels[ncr, ncc] != 0:
                continue  # only propagate into non-flood coarse pixels
            ndem = dem[nr, nc]
            if np.isnan(ndem):
                continue
            if ndem < cur_dem:
                flood_extent[nr, nc] = 1
                wsl_map[nr, nc] = cur_wsl
                queue.append((nr, nc))
                n_extended += 1

    return n_extended


def downscale_to_fine(wf, dem, perm_water, polygon_labels,
                      polygon_wsl, scale, cfg):
    Hc, Wc = wf.shape
    Hf = Hc * scale
    Wf = Wc * scale

    print("Step 4: Downscaling to fine resolution...")
    t0 = time.time()

    flood_extent = np.zeros((Hf, Wf), dtype=np.uint8)
    wsl_map = np.full((Hf, Wf), np.nan, dtype=np.float32)

    for r in range(Hc):
        for c in range(Wc):
            pid = polygon_labels[r, c]
            if pid == 0 or pid not in polygon_wsl:
                continue

            wsl = polygon_wsl[pid]
            r0 = r * scale
            c0 = c * scale

            dem_block = dem[r0:r0 + scale, c0:c0 + scale].copy()
            nan_mask = np.isnan(dem_block)
            dem_block[nan_mask] = 99999.0

            if perm_water is not None:
                pw_block = perm_water[r0:r0 + scale, c0:c0 + scale]
                if np.any(pw_block):
                    masked = dem_block.copy()
                    masked[~pw_block] = 99999.0
                    start_idx = np.argmin(masked)
                else:
                    start_idx = np.argmin(dem_block)
            else:
                start_idx = np.argmin(dem_block)

            start_r_local = start_idx // scale
            start_c_local = start_idx % scale

            flooded = np.zeros((scale, scale), dtype=np.bool_)
            _flood_fill_n4(dem_block, wsl, start_r_local, start_c_local, flooded)

            for lr in range(scale):
                for lc in range(scale):
                    if flooded[lr, lc] and not nan_mask[lr, lc]:
                        flood_extent[r0 + lr, c0 + lc] = 1
                        wsl_map[r0 + lr, c0 + lc] = wsl

    if cfg.enable_flood_extension:
        print("  Extending flood into neighbouring non-flood pixels...")
        n_extended = _extend_flood_monotonic(
            dem, flood_extent, wsl_map, polygon_labels, scale)
        print(f"  Extended {n_extended} fine cells into non-flood pixels.")

    elapsed = time.time() - t0
    n_flooded = np.count_nonzero(flood_extent)
    print(f"  Downscaled {n_flooded} fine cells as flooded in {elapsed:.1f}s")

    return flood_extent, wsl_map



def compute_water_depth(flood_extent, wsl_map, dem, cfg):
    print("Step 5: Computing water depth (D = W_l - E_s)...")

    water_depth = np.full(dem.shape, np.nan, dtype=np.float32)
    flooded = flood_extent == 1

    depth = wsl_map[flooded] - dem[flooded]
    depth = np.clip(depth, cfg.min_depth, cfg.max_depth)
    water_depth[flooded] = depth

    valid_depths = depth[~np.isnan(depth)]
    if len(valid_depths) > 0:
        print(f"  Depth stats: mean={np.mean(valid_depths):.2f}m, "
              f"median={np.median(valid_depths):.2f}m, "
              f"max={np.max(valid_depths):.2f}m, "
              f"std={np.std(valid_depths):.2f}m")
    else:
        print("  WARNING: No valid depth values computed.")

    return water_depth



def write_date_outputs(date_str, flood_extent, water_depth, dem_profile, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extent_profile = dem_profile.copy()
    extent_profile.update(dtype="uint8", count=1, nodata=255, compress="lzw")
    path = output_dir / f"{date_str}_flood_extent.tif"
    with rasterio.open(path, "w", **extent_profile) as dst:
        dst.write(flood_extent, 1)
    print(f"  Wrote: {path.name}")

    depth_profile = dem_profile.copy()
    depth_profile.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
    path = output_dir / f"{date_str}_water_depth.tif"
    with rasterio.open(path, "w", **depth_profile) as dst:
        dst.write(water_depth, 1)
    print(f"  Wrote: {path.name}")


def load_static_data(cfg: Config, reference_wf_path: str):
    print("Loading static data (DEM + permanent water)...")

    dem_src = rasterio.open(cfg.dem_path)
    wf_src = rasterio.open(reference_wf_path)
    pw_src = rasterio.open(cfg.permanent_water_path) if cfg.permanent_water_path else None

    dem_res = abs(dem_src.transform[0])
    wf_res = abs(wf_src.transform[0])
    wf_full_transform = wf_src.transform

    if cfg.scale_ratio == 0:
        scale = int(round(wf_res / dem_res))
    else:
        scale = cfg.scale_ratio
    fine_res = wf_res / scale
    print(f"Scale ratio: {scale} (coarse res {wf_res:.6f} / fine res {fine_res:.8f})")

    bounds_list = [dem_src.bounds, wf_src.bounds]
    if pw_src is not None:
        bounds_list.append(pw_src.bounds)

    common_left = max(b.left for b in bounds_list)
    common_bottom = max(b.bottom for b in bounds_list)
    common_right = min(b.right for b in bounds_list)
    common_top = min(b.top for b in bounds_list)

    if common_left >= common_right or common_bottom >= common_top:
        raise ValueError(
            f"No geographic overlap!\n"
            f"  DEM: {dem_src.bounds}\n  WF: {wf_src.bounds}\n"
            f"  PW: {pw_src.bounds if pw_src else 'N/A'}"
        )

    inv_wf = ~wf_full_transform
    col_left, row_top = inv_wf * (common_left, common_top)
    col_right, row_bottom = inv_wf * (common_right, common_bottom)
    col_start = int(math.ceil(col_left))
    row_start = int(math.ceil(row_top))
    col_end = int(math.floor(col_right))
    row_end = int(math.floor(row_bottom))

    Hc = row_end - row_start
    Wc = col_end - col_start
    if Hc <= 0 or Wc <= 0:
        raise ValueError("Common extent too small after snapping to coarse grid.")

    snap_left, snap_top = wf_full_transform * (col_start, row_start)
    snap_right, snap_bottom = wf_full_transform * (col_end, row_end)

    print(f"Snapped extent: W={snap_left:.6f}, S={snap_bottom:.6f}, "
          f"E={snap_right:.6f}, N={snap_top:.6f}")
    print(f"Coarse grid: {Hc} rows × {Wc} cols")

    wf_window = Window(col_off=col_start, row_off=row_start, width=Wc, height=Hc)
    wf_snap_transform = wf_src.window_transform(wf_window)

    Hf = Hc * scale
    Wf = Wc * scale
    fine_transform = Affine(fine_res, 0.0, snap_left, 0.0, -fine_res, snap_top)
    print(f"Fine grid: {Hf} rows × {Wf} cols")

    dem = np.full((Hf, Wf), np.nan, dtype=np.float32)
    src_nodata = dem_src.nodata if dem_src.nodata is not None else -9999.0
    reproject(
        source=rasterio.band(dem_src, 1),
        destination=dem,
        src_transform=dem_src.transform,
        src_crs=dem_src.crs,
        dst_transform=fine_transform,
        dst_crs=dem_src.crs,
        src_nodata=src_nodata,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    print(f"DEM: {np.count_nonzero(~np.isnan(dem))} valid cells")

    dem_profile = dem_src.profile.copy()
    dem_profile.update(
        height=Hf, width=Wf,
        transform=fine_transform,
        dtype="float32",
        nodata=np.nan,
    )

    perm_water = None
    if pw_src is not None:
        pw_temp = np.zeros((Hf, Wf), dtype=np.uint8)
        pw_nodata = pw_src.nodata if pw_src.nodata is not None else 255
        reproject(
            source=rasterio.band(pw_src, 1),
            destination=pw_temp,
            src_transform=pw_src.transform,
            src_crs=pw_src.crs,
            dst_transform=fine_transform,
            dst_crs=pw_src.crs,
            src_nodata=pw_nodata,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )
        perm_water = pw_temp > 0

    dem_src.close()
    wf_src.close()
    if pw_src is not None:
        pw_src.close()

    return {
        "dem": dem,
        "perm_water": perm_water,
        "scale": scale,
        "dem_profile": dem_profile,
        "Hc": Hc, "Wc": Wc,
        "wf_col_start": col_start,
        "wf_row_start": row_start,
        "wf_snap_transform": wf_snap_transform,
    }


def load_water_fraction(wf_path, static_data):
    Hc = static_data["Hc"]
    Wc = static_data["Wc"]
    wf_window = Window(col_off=static_data["wf_col_start"],
                       row_off=static_data["wf_row_start"],
                       width=Wc, height=Hc)

    with rasterio.open(wf_path) as src:
        wf = src.read(1, window=wf_window).astype(np.float32)
        if src.nodata is not None:
            wf[wf == src.nodata] = np.nan

    return wf


def process_single_date(wf, static_data, cfg):
    dem = static_data["dem"]
    perm_water = static_data["perm_water"]
    scale = static_data["scale"]

    pixel_wsl = compute_pixel_water_levels(wf, dem, perm_water, scale, cfg)
    polygon_labels, polygon_wsl = cluster_water_polygons(
        wf, pixel_wsl, dem, scale, cfg)
    polygon_wsl = correct_polygon_water_levels(
        polygon_labels, polygon_wsl, wf, dem, scale, cfg)
    flood_extent, wsl_map = downscale_to_fine(
        wf, dem, perm_water, polygon_labels, polygon_wsl, scale, cfg)
    water_depth = compute_water_depth(flood_extent, wsl_map, dem, cfg)

    return flood_extent, water_depth


def run_batch(cfg: Config):
    input_folder = Path(cfg.input_folder)
    wf_files = sorted(glob.glob(str(input_folder / "*_VIIRS_mosaic.tif")))

    if not wf_files:
        wf_files = sorted(glob.glob(str(input_folder / "*.tif")))

    if not wf_files:
        print(f"ERROR: No .tif files found in {input_folder}")
        return

    print("=" * 60)
    print("Li et al. Downscaling Model — BATCH MODE")
    print(f"Found {len(wf_files)} water fraction files")
    print("=" * 60)

    static_data = load_static_data(cfg, wf_files[0])

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_batch_start = time.time()

    for i, wf_path in enumerate(wf_files, 1):
        fname = Path(wf_path).stem
        date_str = fname.split("_")[0]

        print(f"\n{'─' * 60}")
        print(f"[{i}/{len(wf_files)}] Processing {fname}  (date: {date_str})")
        print(f"{'─' * 60}")

        t0 = time.time()

        try:
            wf = load_water_fraction(wf_path, static_data)

            n_water = np.count_nonzero(
                (~np.isnan(wf)) & (wf >= cfg.min_water_fraction))
            print(f"  Water pixels in coarse grid: {n_water}")

            if n_water == 0:
                print(f"  WARNING: No water pixels — skipping {date_str}")
                continue

            flood_extent, water_depth = process_single_date(
                wf, static_data, cfg)

            write_date_outputs(
                date_str, flood_extent, water_depth,
                static_data["dem_profile"], cfg.output_dir)

            elapsed = time.time() - t0
            n_flooded = np.count_nonzero(flood_extent)
            print(f"  Done: {n_flooded} flooded cells in {elapsed:.1f}s")

        except Exception as e:
            print(f"  ERROR: FAILED on {fname}: {e}")
            traceback.print_exc()
            continue

    t_total = time.time() - t_batch_start
    print(f"\n{'=' * 60}")
    print(f"Batch complete: {len(wf_files)} files in {t_total:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    cfg = Config()

    cfg.input_folder = r"C:\Users\Chloe\OneDrive - RMIT University\Data\Australia\NSW\test"
    cfg.dem_path = r"C:\Users\Chloe\OneDrive - RMIT University\Data\Australia\NSW\Topography\FABDEM_QL.tif"
    cfg.permanent_water_path = r""
    cfg.output_dir = r"C:\Users\Chloe\OneDrive - RMIT University\Data\Australia\NSW\test"

    cfg.scale_ratio = 0
    cfg.min_water_fraction = 0.01
    cfg.elevation_step = 0.1
    cfg.polygon_max_box = 25
    cfg.enable_dryland_correction = True
    cfg.enable_flood_extension = True
    cfg.max_depth = 50.0

    run_batch(cfg)
