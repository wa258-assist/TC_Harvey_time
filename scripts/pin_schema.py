"""
scripts/pin_schema.py
----------------------
Build, validate, and serialise FloodPin v1-v36 records.
"""
import datetime, json, logging, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    DATA_VERSION, FLOOD_THRESHOLD_PCT, H3_RESOLUTION,
    PRODUCT_FLOOD, PRODUCT_FIRE, SOURCE_ORG, SOURCE_SYSTEM,
    VIIRS_GSD_M,
)
from scripts.h3_grid import support_radius_m, h3_to_polygon
from scripts.utils import get_logger, utcnow_iso

log = get_logger("pin_schema")

SCHEMA = [
    "event_id","pin_id","revision","issued_utc","obs_time_utc",
    "hazard_type","data_version","evidence_type","product","source_system",
    "pin_mode","valid_start_utc","valid_end_utc","latitude","longitude",
    "spatial_res_m","support_radius_m","H3_res8","source_org",
    "qa_flag","qa_reason","depth_bin_m","depth_p10_m","depth_p90_m",
    "depth_source","hazard_probability_pct","hazard_probability_p10",
    "hazard_probability_p90","confidence_overall","confidence_interval_basis",
    "validation_metric","independent_source","validation_score",
    "sensor","gsd_m","coverage_flag",
]

REQUIRED = {
    "event_id","pin_id","revision","issued_utc","obs_time_utc","hazard_type",
    "product","source_system","pin_mode","valid_start_utc","valid_end_utc",
    "latitude","longitude","spatial_res_m","support_radius_m","H3_res8",
    "source_org","qa_flag","depth_source","hazard_probability_pct",
    "hazard_probability_p10","hazard_probability_p90","confidence_overall",
    "confidence_interval_basis","sensor","gsd_m","coverage_flag",
}

SENSOR_META = {
    "VIIRS":      {"gsd_m":375,   "evidence_type":"Optical",         "depth_source":"VIIRS_extent+DEM_30m"},
    "Sentinel-1": {"gsd_m":10,    "evidence_type":"SAR_backscatter", "depth_source":"S1_extent+DEM_30m"},
    "Landsat":    {"gsd_m":30,    "evidence_type":"Optical",         "depth_source":"Landsat_extent+DEM_30m"},
    "AMSR2":      {"gsd_m":10000, "evidence_type":"Passive_MW",      "depth_source":"AMSR2_extent+DEM_30m"},
}


def build_pin(event_id, obs_date_str, valid_start_utc, valid_end_utc,
              hazard_type="flood", sensor="VIIRS", pin_counter=1, revision=1,
              h3_index="", lat=0.0, lon=0.0,
              hazard_probability_pct=0.0, hazard_probability_p10=0.0, hazard_probability_p90=0.0,
              confidence_overall=0.0, depth_bin_m=None, depth_p10_m=None, depth_p90_m=None,
              qa_flag=True, qa_reason=None, coverage_flag="good",
              validation_metric=None, independent_source=None, validation_score=None,
              confidence_interval_basis="heuristic"):

    obs_dt  = datetime.datetime.strptime(obs_date_str, "%Y%m%d")
    obs_utc = obs_dt.strftime("%Y-%m-%dT00:00:00Z")
    sm      = SENSOR_META.get(sensor, SENSOR_META["VIIRS"])
    product = PRODUCT_FLOOD if hazard_type == "flood" else PRODUCT_FIRE
    sup_r   = support_radius_m(H3_RESOLUTION)
    pin_id  = f"{event_id}_{obs_date_str}_{pin_counter:04d}"

    return {
        "event_id":                  event_id,
        "pin_id":                    pin_id,
        "revision":                  revision,
        "issued_utc":                utcnow_iso(),
        "obs_time_utc":              obs_utc,
        "hazard_type":               hazard_type,
        "data_version":              DATA_VERSION,
        "evidence_type":             sm["evidence_type"],
        "product":                   product,
        "source_system":             SOURCE_SYSTEM,
        "pin_mode":                  "Observed",
        "valid_start_utc":           valid_start_utc,
        "valid_end_utc":             valid_end_utc,
        "latitude":                  round(lat, 6),
        "longitude":                 round(lon, 6),
        "spatial_res_m":             int(sm["gsd_m"]),
        "support_radius_m":          sup_r,
        "H3_res8":                   h3_index,
        "source_org":                SOURCE_ORG,
        "qa_flag":                   qa_flag,
        "qa_reason":                 qa_reason,
        "depth_bin_m":               depth_bin_m,
        "depth_p10_m":               depth_p10_m,
        "depth_p90_m":               depth_p90_m,
        "depth_source":              sm["depth_source"],
        "hazard_probability_pct":    round(float(hazard_probability_pct), 4),
        "hazard_probability_p10":    round(float(hazard_probability_p10), 4),
        "hazard_probability_p90":    round(float(hazard_probability_p90), 4),
        "confidence_overall":        round(float(confidence_overall), 4),
        "confidence_interval_basis": confidence_interval_basis,
        "validation_metric":         validation_metric,
        "independent_source":        independent_source,
        "validation_score":          validation_score,
        "sensor":                    sensor,
        "gsd_m":                     float(sm["gsd_m"]),
        "coverage_flag":             coverage_flag,
    }


def pins_from_zonal_stats(stats_df, event_id, obs_date_str, valid_start_utc,
                           valid_end_utc, hazard_type="flood", sensor="VIIRS",
                           flood_threshold=FLOOD_THRESHOLD_PCT):
    pins, counter = [], 1
    for _, row in stats_df.iterrows():
        if row.get("is_perm_water", False):
            continue
        prob = float(row.get("hazard_probability_pct") or 0)
        if prob < flood_threshold:
            continue
        pin = build_pin(
            event_id=event_id, obs_date_str=obs_date_str,
            valid_start_utc=valid_start_utc, valid_end_utc=valid_end_utc,
            hazard_type=hazard_type, sensor=sensor, pin_counter=counter,
            h3_index=row["h3_index"], lat=row["lat"], lon=row["lon"],
            hazard_probability_pct=prob,
            hazard_probability_p10=float(row.get("hazard_probability_p10") or prob),
            hazard_probability_p90=float(row.get("hazard_probability_p90") or prob),
            confidence_overall=float(row.get("confidence_overall") or 0),
            depth_bin_m=row.get("depth_bin_m"),
            depth_p10_m=row.get("depth_p10_m"),
            depth_p90_m=row.get("depth_p90_m"),
            qa_flag=bool(row.get("qa_flag", True)),
            qa_reason=row.get("qa_reason"),
            coverage_flag=row.get("coverage_flag", "good"),
        )
        pins.append(pin)
        counter += 1
    log.info("[%s] %d pins (threshold=%.2f, excl perm water)", obs_date_str, len(pins), flood_threshold)
    return pins


def validate_pins(pins):
    valid, errors = [], []
    for i, p in enumerate(pins):
        missing = [f for f in REQUIRED if p.get(f) is None]
        if missing:
            errors.append({"index": i, "pin_id": p.get("pin_id"), "missing": missing})
        else:
            valid.append(p)
    if errors:
        log.warning("%d pins failed validation", len(errors))
    return valid, errors


def to_dataframe(pins):
    df = pd.DataFrame(pins)
    cols = [c for c in SCHEMA if c in df.columns]
    extra = [c for c in df.columns if c not in SCHEMA]
    return df[cols + extra]


def save_csv(pins, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    to_dataframe(pins).to_csv(path, index=False)
    log.info("CSV saved: %s (%d pins)", path, len(pins))


def save_jsonl(pins, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for p in pins:
            f.write(json.dumps(p, default=str) + "\n")
    log.info("JSONL saved: %s", path)


def save_geojson(pins, path):
    from shapely.geometry import mapping as shp_mapping
    feats = []
    for p in pins:
        cell = p.get("H3_res8","")
        try:
            geom = shp_mapping(h3_to_polygon(cell)) if cell else None
        except Exception:
            geom = None
        feats.append({"type":"Feature","geometry":geom,
                       "properties":{k:v for k,v in p.items() if k!="H3_res8"}})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"type":"FeatureCollection","features":feats}, default=str))
    log.info("GeoJSON saved: %s", path)


def pin_summary(pins, label=""):
    if not pins: return {"label":label,"n_pins":0,"mean_hazard_prob":0.0,"p10_hazard_prob":0.0,"p90_hazard_prob":0.0,"mean_confidence":0.0,"depth_bins":{},"coverage_flags":{}}
    df = to_dataframe(pins)
    return {
        "label":            label,
        "n_pins":           len(pins),
        "qa_pass":          int(df["qa_flag"].astype(bool).sum()),
        "mean_hazard_prob": round(float(df["hazard_probability_pct"].mean()),4),
        "p10_hazard_prob":  round(float(df["hazard_probability_p10"].mean()),4),
        "p90_hazard_prob":  round(float(df["hazard_probability_p90"].mean()),4),
        "mean_confidence":  round(float(df["confidence_overall"].mean()),4),
        "depth_bins":       df["depth_bin_m"].value_counts().to_dict() if "depth_bin_m" in df.columns else {},
        "coverage_flags":   df["coverage_flag"].value_counts().to_dict(),
    }
