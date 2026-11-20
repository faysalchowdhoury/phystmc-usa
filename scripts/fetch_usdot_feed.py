"""Data acquisition for PhysTMC-USA (honest, reproducible).

Attempts the public MDOT CHART ArcGIS camera API first and records the outcome.
Live DOT *video* is not reliably capturable here (CHART video endpoints 404, no
system ffmpeg, DOT snapshots refresh ~0.1 FPS), so -- per an explicit decision to
use real recorded footage rather than fabricate frames -- it falls back to a real,
publicly distributed overhead highway-traffic recording, caches it, and extracts a
reference frame. Provenance is written to data/raw/camera_metadata.json.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
CHART_API = ("https://mdgeodata.md.gov/imap/rest/services/Transportation/"
             "MD_TrafficCameras/MapServer/0/query?where=1%3D1&outFields=*&"
             "resultRecordCount=5&f=json")
RECORDED_VIDEO = "https://media.roboflow.com/supervision/video-examples/vehicles.mp4"


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "phystmc-usa/1.0"})
    return urllib.request.urlopen(req, timeout=timeout).read()


def probe_mdot_chart() -> dict:
    out = {"api": CHART_API, "metadata_ok": False, "video_ok": False, "cameras": 0}
    try:
        data = json.loads(_get(CHART_API).decode())
        feats = data.get("features", [])
        out["metadata_ok"] = True
        out["cameras"] = len(feats)
        if feats:
            vurl = feats[0]["attributes"].get("url", "")
            try:
                _get(vurl, timeout=20)
                out["video_ok"] = True
            except Exception as e:
                out["video_error"] = repr(e)[:120]
    except Exception as e:
        out["metadata_error"] = repr(e)[:120]
    return out


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    print("[1] Probing MDOT CHART camera API ...")
    chart = probe_mdot_chart()
    print("    metadata_ok=%s cameras=%s video_ok=%s"
          % (chart["metadata_ok"], chart["cameras"], chart["video_ok"]))

    dst = RAW / "source_traffic.mp4"
    if not dst.exists() or dst.stat().st_size < 1_000_000:
        print("[2] Live DOT video unavailable -> caching real recorded clip ...")
        with open(dst, "wb") as f:
            f.write(_get(RECORDED_VIDEO, timeout=180))
    print(f"    video cached: {dst} ({dst.stat().st_size/1e6:.1f} MB)")

    import cv2
    cap = cv2.VideoCapture(str(dst))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, frame = cap.read()
    if ok:
        cv2.imwrite(str(RAW / "reference_frame.jpg"), frame)
    cap.release()
    print(f"    reference frame extracted ({'ok' if ok else 'FAILED'})")

    # append the live-probe outcome into the provenance file
    meta_path = RAW / "camera_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["mdot_chart_probe"] = chart
    meta_path.write_text(json.dumps(meta, indent=2))
    print("[3] Provenance written to", meta_path)


if __name__ == "__main__":
    main()
