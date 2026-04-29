"""
scripts/run_harvey.py
----------------------
Single entry point for the Harvey FloodPin pipeline.
Run this script directly or call it from GitHub Actions.

Usage:
  python scripts/run_harvey.py --output-dir ./harvey_output
  python scripts/run_harvey.py --output-dir ./harvey_output --dem-path ./data/dem/dem_clipped.tif
  python scripts/run_harvey.py --output-dir ./harvey_output --skip-download --skip-downscale
"""
import argparse, datetime, json, logging, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import ACTIVE_EVENT_ID, FLOOD_THRESHOLD_PCT, H3_RESOLUTION, KNOWN_EVENTS, get_event_dates
from scripts.utils    import get_logger, set_gha_output, utcnow_iso, load_aoi
from scripts.dem_fetch    import fetch_dem
from scripts.perm_water   import get_permanent_water_mask
from scripts.viirs_download import download_viirs_for_event
from scripts.downscale    import run_downscaling
from scripts.h3_grid      import generate_h3_cells, cells_over_raster
from scripts.zonal_stats  import run_zonal_stats
from scripts.pin_schema   import (
    pins_from_zonal_stats, validate_pins,
    save_csv, save_jsonl, save_geojson,
    pin_summary, to_dataframe,
)
from scripts.notify import (
    notify_validation_errors, notify_validation_passed,
    notify_pipeline_complete, notify_pipeline_failed,
)

log = get_logger("run_harvey")

EVENT_ID = ACTIVE_EVENT_ID
EVENT    = KNOWN_EVENTS[EVENT_ID]
DATES    = get_event_dates(EVENT_ID)


def run(args):
    t_start = time.time()
    out = Path(args.output_dir)
    dirs = {k: out/k for k in ["viirs","downscaled","zonal","pins","dem","perm_water","results"]}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    notify_to = args.notify_email or os.environ.get("NOTIFY_EMAIL", "")

    # Load AOI bbox
    bbox = load_aoi("HARVEY")["bbox"]

    log.info("=" * 60)
    log.info("FloodPin Harvey Pipeline  event=%s", EVENT_ID)
    log.info("Dates: %s to %s", DATES[0], DATES[-1])
    log.info("AOI:   %s", bbox)
    log.info("H3 resolution: %d  |  flood threshold: %.2f",
             H3_RESOLUTION, args.flood_threshold)
    log.info("=" * 60)

    # ── Step 1: DEM ──────────────────────────────────────────────────────────
    log.info("\n[1/5] DEM")
    dem_path = fetch_dem(
        aoi_or_bbox = bbox,
        out_dir     = dirs["dem"],
        source      = args.dem_source,
        api_key     = args.api_key,
        existing    = args.dem_path,
    )
    log.info("  DEM: %s", dem_path or "unavailable (depth will be null)")

    # ── Step 2: Permanent water mask ─────────────────────────────────────────
    log.info("\n[2/5] Permanent water mask")
    pw_path = get_permanent_water_mask(
        aoi_name  = "HARVEY",
        bbox      = bbox,
        out_dir   = dirs["perm_water"],
        existing  = args.perm_water,
    )
    log.info("  Perm water: %s", pw_path or "unavailable")

    # ── Step 3: VIIRS download ────────────────────────────────────────────────
    log.info("\n[3/5] VIIRS download")
    if args.skip_download:
        daily_mosaics = {p.stem.split("_")[0]: p
                         for p in sorted(dirs["viirs"].glob("*_VIIRS_mosaic.tif"))}
        log.info("  --skip-download: found %d existing mosaics", len(daily_mosaics))
    else:
        daily_mosaics = download_viirs_for_event(
            dates   = DATES,
            out_dir = dirs["viirs"],
            bbox    = bbox,
        )
    if not daily_mosaics:
        log.error("No VIIRS mosaics — aborting")
        notify_pipeline_failed(notify_to, "No VIIRS mosaics available", EVENT_ID)
        return 1
    log.info("  Mosaics: %s", sorted(daily_mosaics.keys()))

    # ── Step 4: Per-day processing ────────────────────────────────────────────
    log.info("\n[4/5] H3 grid + per-day processing")
    h3_cells = generate_h3_cells(bbox, resolution=H3_RESOLUTION)
    log.info("  H3 cells total: %d", len(h3_cells))

    all_pins, daily_summaries = [], []

    for date_str, mosaic_path in sorted(daily_mosaics.items()):
        log.info("\n  ── %s ──────────────────────────────────────", date_str)

        # Downscale
        if args.skip_downscale or not dem_path:
            extent_path = dirs["downscaled"] / f"{date_str}_flood_extent.tif"
            depth_path  = dirs["downscaled"] / f"{date_str}_water_depth.tif"
            extent_path = extent_path if extent_path.exists() else None
            depth_path  = depth_path  if depth_path.exists()  else None
        else:
            extent_path, depth_path = run_downscaling(
                mosaic_path, str(dem_path),
                str(pw_path) if pw_path else None,
                dirs["downscaled"], date_str,
            )

        # Filter cells to raster extent
        active_cells = cells_over_raster(mosaic_path, h3_cells)
        log.info("  Active cells: %d", len(active_cells))

        # Zonal stats
        stats_df = run_zonal_stats(
            viirs_path = mosaic_path,
            depth_path = depth_path,
            pw_path    = pw_path,
            cells      = active_cells,
        )
        if stats_df.empty:
            log.warning("  No stats for %s", date_str)
            continue

        # Save zonal stats
        z_csv = dirs["zonal"] / f"{date_str}_zonal_stats.csv"
        stats_df.to_csv(z_csv, index=False)

        # Build pins
        daily_pins = pins_from_zonal_stats(
            stats_df        = stats_df,
            event_id        = EVENT_ID,
            obs_date_str    = date_str,
            valid_start_utc = EVENT["valid_start"],
            valid_end_utc   = EVENT["valid_end"],
            hazard_type     = "flood",
            sensor          = "VIIRS",
            flood_threshold = args.flood_threshold,
        )
        valid_pins, errors = validate_pins(daily_pins)
        all_pins.extend(valid_pins)

        if errors:
            notify_validation_errors(notify_to, errors, EVENT_ID, date_str)
        if valid_pins:
            day_csv = dirs["results"] / f"{EVENT_ID}_{date_str}_pins.csv"
            save_csv(valid_pins,  day_csv)
            save_csv(valid_pins,  dirs["pins"] / f"{EVENT_ID}_{date_str}_pins.csv")
            save_jsonl(valid_pins, dirs["pins"] / f"{EVENT_ID}_{date_str}_pins.jsonl")

        s = pin_summary(valid_pins, date_str)
        daily_summaries.append(s)
        log.info("  pins=%d  mean_prob=%.3f  mean_conf=%.3f",
                 s["n_pins"], s["mean_hazard_prob"], s["mean_confidence"])

    # ── Step 5: Event-level outputs ───────────────────────────────────────────
    log.info("\n[5/5] Event-level outputs")
    if not all_pins:
        log.error("No pins generated")
        notify_pipeline_failed(notify_to, "No pins generated after all processing", EVENT_ID)
        return 1

    all_pins_csv = dirs["results"] / f"{EVENT_ID}_Harvey_all_pins.csv"
    save_csv(all_pins,     all_pins_csv)
    save_csv(all_pins,     out / f"{EVENT_ID}_Harvey_all_pins.csv")
    save_jsonl(all_pins,   out / f"{EVENT_ID}_Harvey_all_pins.jsonl")
    save_geojson(all_pins, out / f"{EVENT_ID}_Harvey_all_pins.geojson")

    df       = to_dataframe(all_pins)
    mean_conf = float(df["confidence_overall"].mean())

    summary = {
        "event_id":       EVENT_ID,
        "event_name":     EVENT["name"],
        "hazard_type":    "flood",
        "valid_start":    EVENT["valid_start"],
        "valid_end":      EVENT["valid_end"],
        "aoi_bbox":       bbox,
        "h3_resolution":  H3_RESOLUTION,
        "flood_threshold": args.flood_threshold,
        "total_pins":     len(all_pins),
        "mean_confidence": round(mean_conf, 4),
        "days":           sorted(daily_mosaics.keys()),
        "daily_summaries": daily_summaries,
        "generated_utc":  utcnow_iso(),
        "runtime_s":      round(time.time()-t_start, 1),
    }
    summary_path = out / f"{EVENT_ID}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    (dirs["results"] / f"{EVENT_ID}_summary.json").write_text(json.dumps(summary, indent=2))

    # Console summary
    log.info("\n%s", "="*60)
    log.info("Harvey FloodPin COMPLETE")
    log.info("  Total pins:      %d", len(all_pins))
    log.info("  QA pass:         %d", int(df["qa_flag"].astype(bool).sum()))
    log.info("  Mean confidence: %.3f", mean_conf)
    log.info("  Mean flood prob: %.3f", df["hazard_probability_pct"].mean())
    if "depth_bin_m" in df.columns:
        log.info("  Depth bins:\n%s", df["depth_bin_m"].value_counts().to_string())
    log.info("  Runtime:         %.0fs", time.time()-t_start)
    log.info("  Output:          %s", out.resolve())
    log.info("%s", "="*60)

    # Email notification
    result_csvs = sorted(dirs["results"].glob("*.csv"))
    notify_pipeline_complete(notify_to, summary, attachments=result_csvs)

    # GitHub Actions outputs
    set_gha_output("total_pins",      str(len(all_pins)))
    set_gha_output("mean_confidence", f"{mean_conf:.4f}")
    set_gha_output("event_csv",       str(out / f"{EVENT_ID}_Harvey_all_pins.csv"))
    set_gha_output("summary_json",    str(out / f"{EVENT_ID}_summary.json"))

    # Write GitHub step summary
    gha_sum = os.environ.get("GITHUB_STEP_SUMMARY")
    if gha_sum:
        with open(gha_sum, "a") as f:
            f.write(f"## Harvey FloodPin Results\n")
            f.write(f"**Total pins:** {len(all_pins)} | **Mean confidence:** {mean_conf:.3f}\n\n")
            f.write("| Date | Pins | Mean prob | Mean conf |\n")
            f.write("|------|------|-----------|----------|\n")
            for s in daily_summaries:
                f.write(f"| {s['label']} | {s['n_pins']} | {s['mean_hazard_prob']:.3f} | {s['mean_confidence']:.3f} |\n")

    return 0


def main():
    p = argparse.ArgumentParser(description="FloodPin Harvey pipeline entry point")
    p.add_argument("--output-dir",      default="./harvey_output")
    p.add_argument("--dem-path",        default="",
                   help="Path to existing DEM GeoTIFF (skips download)")
    p.add_argument("--perm-water",      default="",
                   help="Path to existing permanent water binary mask")
    p.add_argument("--dem-source",      default="usgs_3dep",
                   choices=["usgs_3dep","opentopo"],
                   help="DEM source: usgs_3dep (zero-auth CONUS) or opentopo (SRTM global)")
    p.add_argument("--api-key",         default=os.environ.get("OPENTOPO_API_KEY",""),
                   help="OpenTopography API key (only needed with --dem-source opentopo)")
    p.add_argument("--flood-threshold", type=float, default=FLOOD_THRESHOLD_PCT,
                   help=f"Min flood fraction to emit a pin (default {FLOOD_THRESHOLD_PCT})")
    p.add_argument("--skip-download",   action="store_true",
                   help="Skip VIIRS download; use existing mosaics in output-dir/viirs/")
    p.add_argument("--skip-downscale",  action="store_true",
                   help="Skip downscaling; use existing depth rasters if present")
    p.add_argument("--notify-email",    default="",
                   help="Engineer email for validation/completion notifications (overrides NOTIFY_EMAIL env var)")
    args = p.parse_args()
    sys.exit(run(args))

if __name__ == "__main__":
    main()
