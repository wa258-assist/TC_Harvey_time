"""
Send a human-review notification for the permanent water mask source change.

Usage:
  export SMTP_USER=angk0063@gmail.com
  export SMTP_PASS=<gmail-app-password>
  python send_review_email.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.notify import _send

TO      = "wang@rss-hydro.lu"
SUBJECT = "[FloodPin Automation] Permanent water mask — source change for review"
BODY    = """\
Hi,

This is an automated review request for a change to the FloodPin Harvey pipeline.

Change: Permanent water mask source updated
  Old: EC JRC Global Surface Water (GSWE occurrence raster, 30 m)
       https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GSWE/
  New: USGS National Hydrography Dataset (NHD waterbody polygons)
       https://www.usgs.gov/national-hydrography/national-hydrography-dataset

Action needed:
  The NHD download logic has not yet been implemented (vector → raster pipeline
  required). The current JRC fallback URLs are exhausted / unreliable.
  Please review scripts/perm_water.py and advise on the NHD integration approach
  (TNM API vs bulk geodatabase download, rasterisation method).

Branch: claude/flamboyant-heyrovsky-c123f6
File:   scripts/perm_water.py

—
FloodPin automation (TC Harvey pilot2)
"""

if __name__ == "__main__":
    ok = _send(TO, SUBJECT, BODY)
    if ok:
        print(f"Email sent to {TO}")
    else:
        print("Failed — check SMTP_USER / SMTP_PASS env vars and try again.")
        sys.exit(1)
