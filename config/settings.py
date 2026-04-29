"""
config/settings.py  —  FloodPin central configuration
"""
import datetime
import os
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent
CONFIG_DIR  = REPO_ROOT / "config"
SCRIPTS_DIR = REPO_ROOT / "scripts"

PRODUCT_FLOOD  = "FloodSENS"
PRODUCT_FIRE   = "FireSENS_GEO"
SOURCE_SYSTEM  = "FloodPin"
SOURCE_ORG     = "FloodPin"
DATA_VERSION   = "1.0"

H3_RESOLUTION       = 8
H3_SUPPORT_RADIUS_M = 300

VIIRS_BUCKET      = "noaa-jpss"
VIIRS_BASE_PREFIX = "JPSS_Blended_Products/VFM_1day_GLB/TIF/"
VIIRS_GSD_M       = 375
VIIRS_FLOOD_MIN   = 100
VIIRS_FLOOD_MAX   = 200

FLOOD_THRESHOLD_PCT        = float(os.environ.get("FLOOD_THRESHOLD", "0.01"))
MIN_COVERAGE_FOR_PIN       = 0.20
CLOUD_FRACTION_THRESHOLD   = 0.80
PERM_WATER_CELL_THRESHOLD  = 0.50
PERM_WATER_PIXEL_THRESHOLD = 50.0

DEPTH_BINS = [
    (0.0,   0.1,  "0.0-0.1m"),
    (0.1,   0.3,  "0.1-0.3m"),
    (0.3,   0.6,  "0.3-0.6m"),
    (0.6,   1.0,  "0.6-1.0m"),
    (1.0,   2.0,  "1.0-2.0m"),
    (2.0, 999.0,  ">2.0m"),
]

def depth_to_bin(depth_m):
    if depth_m is None:
        return None
    for lo, hi, label in DEPTH_BINS:
        if lo <= depth_m < hi:
            return label
    return ">2.0m"

DS_MIN_WATER_FRACTION        = 0.01
DS_ELEVATION_STEP            = 0.1
DS_FRACTION_TOLERANCE        = 0.02
DS_POLYGON_MAX_BOX           = 25
DS_POLYGON_MIN_SIZE          = 3
DS_POLYGON_ELEV_THRESHOLD    = 10.0
DS_DRYLAND_THRESHOLD         = 0.30
DS_ENABLE_DRYLAND_CORRECTION = True
DS_ENABLE_FLOOD_EXTENSION    = True
DS_MIN_DEPTH                 = 0.0
DS_MAX_DEPTH                 = 50.0

CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.75"))

COPERNICUS_LC100_TEMPLATE = (
    "https://s3-eu-west-1.amazonaws.com/vito.landcover.global/v3.0.1/"
    "{year}/{tile}/{tile}_PROBAV_LC100_global_v3.0.1_{year}-conso"
    "_Water-Seasonal-Coverfraction-layer_EPSG-4326.tif"
)

# VIIRS tiles are discovered spatially at runtime (find_tiles_by_bbox in viirs_download.py).
# Hardcoded GLB numbers are not used — the NOAA JPSS VFM tile numbering does not
# follow the expected SSEC geographic grid.
def get_event_dates(event_id: str) -> list:
    """Return a list of datetime.date objects for every day in the event's date range."""
    ev = KNOWN_EVENTS[event_id]
    start = datetime.date.fromisoformat(ev["start_date"])
    end   = datetime.date.fromisoformat(ev["end_date"])
    n     = (end - start).days + 1
    return [start + datetime.timedelta(days=i) for i in range(n)]


KNOWN_EVENTS = {
    "H001": {
        "name":        "Harvey",
        "hazard":      "flood",
        "aoi":         "HARVEY",
        "start_date":  "2017-08-25",
        "end_date":    "2017-09-01",
        "valid_start": "2017-08-25T00:00:00Z",
        "valid_end":   "2017-09-02T00:00:00Z",
        "fema_region": "VI",
        "dem_source":  "usgs_3dep",
        "copernicus_tiles": [{"year": "2017", "tile": "W100N40"}],
    },
    "CNMI001": {
        "name":        "CNMI_Guam_Storm",
        "hazard":      "flood",
        "aoi":         "CNMI_GUAM",
        "start_date":  None,
        "end_date":    None,
        "valid_start": None,
        "valid_end":   None,
        "fema_region": "IX",
        "dem_source":  "opentopo",
        "copernicus_tiles": [{"year": "2019", "tile": "E140N20"}],
    },
}
