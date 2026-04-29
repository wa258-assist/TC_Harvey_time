"""scripts/utils.py — shared utilities"""
import datetime, json, logging, os, sys
from pathlib import Path

def get_logger(name, level="INFO"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "[FloodPin][%(name)s] %(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ"))
        logger.addHandler(h)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger

_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "config" / "aoi_registry.json"

def load_aoi(name):
    if _REGISTRY_PATH.exists():
        reg = json.loads(_REGISTRY_PATH.read_text())
        if name in reg:
            return reg[name]
    # fallback: raw bbox string "west,south,east,north"
    parts = [float(x) for x in name.split(",")]
    assert len(parts) == 4
    return {"description": name, "bbox": parts, "islands": [], "fema_region": "", "us_territory": False}

def utcnow_iso():
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def set_gha_output(key, value):
    gha = os.environ.get("GITHUB_OUTPUT")
    if gha:
        with open(gha, "a") as f:
            f.write(f"{key}={value}\n")
    else:
        print(f"[GHA_OUTPUT] {key}={value}")

def safe_read_csv(path, required_cols=None):
    import pandas as pd
    df = pd.read_csv(path)
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
    return df
