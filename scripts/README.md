# FloodPin — Hurricane Harvey Pipeline

End-to-end flood pin generation for Tropical Cyclone Harvey (Aug 25–29, 2017).
Produces FloodPin v1–v36 schema records at H3 resolution-8 (~460m).

---

## Repository layout

```
├── config/
│   ├── settings.py          # all constants, thresholds, event registry
│   └── aoi_registry.json    # AOI bounding boxes
├── scripts/
│   ├── run_harvey.py        # ← single local entry point
│   ├── viirs_download.py    # VIIRS S3 download + mosaic
│   ├── dem_fetch.py         # USGS 3DEP (zero-auth) or OpenTopo SRTM
│   ├── perm_water.py        # Copernicus water bodies mask
│   ├── downscale.py         # wrapper for downscaling_classic.py
│   ├── h3_grid.py           # H3 cell generation + geometry
│   ├── zonal_stats.py       # raster → H3 pixel extraction + p10/p90
│   ├── pin_schema.py        # v1–v36 builder, validator, CSV/JSONL/GeoJSON
│   ├── send_email.py        # SendGrid email with attachments
│   └── utils.py             # shared logging, AOI loader, GHA helpers
├── downscaling_classic.py   # ← PUT YOUR FILE HERE (Li et al. algorithm)
├── requirements.txt
├── main.yml                 # → place at .github/workflows/main.yml
└── README.md
```

---

## Quick start (local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place downscaling_classic.py in the repo root (required for depth fields)
cp /path/to/downscaling_classic.py .

# 3. Run — DEM is downloaded automatically, no credentials needed
python scripts/run_harvey.py --output-dir ./harvey_output

# With a pre-downloaded DEM (skip the 875 MB download):
python scripts/run_harvey.py \
  --output-dir ./harvey_output \
  --dem-path /path/to/your/dem.tif

# Skip re-downloading VIIRS if mosaics already exist:
python scripts/run_harvey.py \
  --output-dir ./harvey_output \
  --skip-download

# Skip downscaling (depth fields will be null but pins still generated):
python scripts/run_harvey.py \
  --output-dir ./harvey_output \
  --skip-downscale
```

---

## GitHub Actions

```bash
# Place workflow file
cp main.yml .github/workflows/main.yml
git add . && git commit -m "Add FloodPin Harvey pipeline"
git push
```

Required GitHub secrets (Settings → Secrets → Actions):

| Secret | Required | Description |
|--------|----------|-------------|
| `SENDGRID_API_KEY` | Optional | Email delivery |
| `CLIENT_EMAIL` | Optional | Recipient email |
| `OPENTOPO_API_KEY` | Not needed for Harvey | Only if using opentopo DEM source |

No secrets are required to run the pipeline — VIIRS and USGS 3DEP are both fully public.

---

## Key fixes vs. previous version

| Issue | Fixed |
|-------|-------|
| VIIRS tiles 097–104 (southern hemisphere) | → **GLB038 + GLB039** (Texas Gulf Coast) |
| CNMI/Guam tiles 119–126 (south Pacific) | → **GLB067** (western Pacific, 0–22.5°N) |
| OpenTopography now requires API key | → Default changed to **USGS 3DEP (zero-auth)** |
| Circular imports between scripts/ and config/ | → Fixed import order |
| `harvey_floodpin_pipeline.py` used as monolith | → Split into focused modules |
| `h3.geo_to_cells` vs old API compatibility | → Pinned `h3>=4.0` |
| `support_radius_m` import missing in pin_schema | → Imported from h3_grid |

---

## Output files

```
harvey_output/
├── viirs/
│   ├── 20170825_VIIRS_mosaic.tif   # flood fraction [0,1]
│   └── ...                          # one per day Aug 25–29
├── dem/
│   └── dem_clipped.tif             # USGS 3DEP 30m, Harvey AOI
├── perm_water/
│   └── permanent_water_binary.tif  # Copernicus 2017, 50% threshold
├── downscaled/
│   ├── 20170825_flood_extent.tif   # binary flood mask (fine res)
│   └── 20170825_water_depth.tif    # water depth in metres
├── zonal/
│   └── 20170825_zonal_stats.csv    # per-H3 stats (p10/mean/p90)
├── pins/
│   └── H001_20170825_pins.csv      # daily pins
├── H001_Harvey_all_pins.csv        # ← main output, all days, v1–v36
├── H001_Harvey_all_pins.jsonl      # streaming-friendly format
├── H001_Harvey_all_pins.geojson    # for QGIS / Mapbox
└── H001_summary.json               # event-level stats
```

---

## FloodPin schema fields

| Field | Variable | Description |
|-------|----------|-------------|
| event_id | v1 | H001 |
| pin_id | v2 | H001_YYYYMMDD_NNNN |
| revision | v3 | 1 |
| issued_utc | v4 | Timestamp of generation |
| obs_time_utc | v5 | VIIRS observation time |
| hazard_type | v6 | flood |
| product | v9 | FloodSENS |
| pin_mode | v11 | Observed |
| H3_res8 | v18 | H3 index at resolution 8 |
| depth_bin_m | v22 | 0.0-0.1m / 0.1-0.3m / ... / >2.0m |
| depth_p10_m | v23 | 10th percentile depth within cell |
| depth_p90_m | v24 | 90th percentile depth within cell |
| hazard_probability_pct | v26 | Mean VIIRS flood fraction in cell |
| hazard_probability_p10 | v27 | P10 flood fraction |
| hazard_probability_p90 | v28 | P90 flood fraction |
| confidence_overall | v29 | Valid pixel coverage fraction |
| coverage_flag | v36 | good / cloud / partial / unknown |
