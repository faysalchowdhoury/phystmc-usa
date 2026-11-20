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
OUT = ROOT / "data/raw/ngsim_i80.csv.gz"
BASE = "https://data.transportation.gov/resource/8ect-6jqj.json"
COLS = ["vehicle_id", "frame_id", "local_y", "v_vel", "v_acc"]
FT_TO_M = 0.3048
PAGE = 50000
MAX_ROWS = 1_000_000


def page(offset: int):
    params = {"$select": ",".join(COLS), "$where": "location='i-80'",
              "$order": "vehicle_id,frame_id", "$limit": PAGE, "$offset": offset}
    url = BASE + "?" + urllib.parse.urlencode(params, safe="(),'=")
    req = urllib.request.Request(url, headers={"User-Agent": "phystmc-usa/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def main():
    if OUT.exists() and OUT.stat().st_size > 1_000_000:
        print("already cached:", OUT)
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n, offset = 0, 0
    with gzip.open(OUT, "wt", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        while n < MAX_ROWS:
            rows = page(offset)
            if not rows:
                break
            for row in rows:
                out = {c: row.get(c, "") for c in COLS}
                for c in ("local_y", "v_vel", "v_acc"):   # feet -> SI metres
                    try:
                        out[c] = float(out[c]) * FT_TO_M
                    except (TypeError, ValueError):
                        out[c] = ""
                w.writerow(out)
            n += len(rows)
            print(f"  {n} rows", flush=True)
            if len(rows) < PAGE:
                break
            offset += PAGE
    print(f"wrote {OUT} ({n} rows)")


if __name__ == "__main__":
    main()
