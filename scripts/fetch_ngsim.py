"""Fetch the real NGSIM I-80 trajectory slice used for ground-truth validation,
from the USDOT ITS DataHub (Socrata resource 8ect-6jqj), converting native US feet
to SI metres. Cached to data/raw/ngsim_i80.csv.gz.
"""
from __future__ import annotations

import csv
import gzip
import json
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://data.transportation.gov/resource/8ect-6jqj.json"
COLS = ["vehicle_id", "frame_id", "local_y", "v_vel", "v_acc"]
FT_TO_M = 0.3048
PAGE = 50000
MAX_ROWS = 1_000_000
LOCATIONS = {"i-80": "ngsim_i80.csv.gz", "us-101": "ngsim_us101.csv.gz"}


def page(location: str, offset: int):
    params = {"$select": ",".join(COLS), "$where": f"location='{location}'",
              "$order": "vehicle_id,frame_id", "$limit": PAGE, "$offset": offset}
    url = BASE + "?" + urllib.parse.urlencode(params, safe="(),'=")
    req = urllib.request.Request(url, headers={"User-Agent": "phystmc-usa/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def fetch(location: str):
    out = ROOT / "data/raw" / LOCATIONS[location]
    if out.exists() and out.stat().st_size > 1_000_000:
        print("already cached:", out)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    n, offset = 0, 0
    with gzip.open(out, "wt", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        while n < MAX_ROWS:
            rows = page(location, offset)
            if not rows:
                break
            for row in rows:
                rec = {c: row.get(c, "") for c in COLS}
                for c in ("local_y", "v_vel", "v_acc"):   # feet -> SI metres
                    try:
                        rec[c] = float(rec[c]) * FT_TO_M
                    except (TypeError, ValueError):
                        rec[c] = ""
                w.writerow(rec)
            n += len(rows)
            print(f"  {location}: {n} rows", flush=True)
            if len(rows) < PAGE:
                break
            offset += PAGE
    print(f"wrote {out} ({n} rows)")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", default="all", choices=["i-80", "us-101", "all"])
    args = ap.parse_args()
    locs = list(LOCATIONS) if args.location == "all" else [args.location]
    for loc in locs:
        fetch(loc)


if __name__ == "__main__":
    main()
