"""Multi-FPS ablation: Baseline raw tracker vs PhysTMC-USA (kinematic filter) across
native / 5 / 2 FPS on the real recorded traffic clip. Writes the results table,
virtual BSM telemetry, and the split-screen dashboard video deliverables.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.run_pipeline import (cache_detections, load_detections, build_calibrator,
                              run_config, geh)
from src.dashboard import render_dashboard


def main():
    import cv2
    source = str(ROOT / "data/raw/source_traffic.mp4")
    cap = cv2.VideoCapture(source)
    native_fps = round(cap.get(cv2.CAP_PROP_FPS) or 25.0, 2)
    cap.release()

    cache = cache_detections(source)
    boxes, cls, conf, n = load_detections(cache)
    calib = build_calibrator()
    print(f"native_fps={native_fps} frames={n}")

    rates = [native_fps, 5.0, 2.0]
    configs = [(fps, kf) for fps in rates for kf in (False, True)]
    results, bsm_by_label = [], {}
    for fps, kf in configs:
        res, bsm, _ = run_config(boxes, cls, conf, native_fps, fps, kf, calib)
        results.append(res)
        bsm_by_label[res.label] = bsm
        print(f"  {res.label:18s} tracks={res.unique_tracks:3d} count={res.total_count:3d} "
              f"meanV={res.mean_speed_mph:5.1f} unphys%={res.unphysical_accel_rate_pct:5.2f} "
              f"CO2={res.mean_co2_gps:.3f} LOS={res.hcm_los}")

    # reference = PhysTMC at native rate (best estimate of ground truth counts)
    ref = next(r for r in results if r.kinematic_filter and abs(r.fps - native_fps) < 1e-6)
    rows = []
    for r in results:
        mape = 100 * abs(r.total_count - ref.total_count) / max(1, ref.total_count)
        rows.append({
            "configuration": r.label,
            "tracker": "PhysTMC-USA" if r.kinematic_filter else "Baseline raw",
            "fps": r.fps, "unique_tracks": r.unique_tracks,
            "total_count": r.total_count,
            "count_MAPE_pct_vs_native": round(mape, 2),
            "GEH_vs_native": round(geh(r.total_count, ref.total_count), 2),
            "mean_speed_mph": r.mean_speed_mph,
            "unphysical_accel_spike_pct": r.unphysical_accel_rate_pct,
            "mean_vtmicro_co2_gps": r.mean_co2_gps,
            "co2_drift_pct_vs_native": round(100 * (r.mean_co2_gps - ref.mean_co2_gps) /
                                             max(1e-6, ref.mean_co2_gps), 2),
            "pet_conflicts": r.pet_conflicts,
            "hcm7_control_delay_s": r.hcm_delay_s, "hcm7_los": r.hcm_los,
            "hcm7_vc": r.hcm_vc, "phf": r.phf, "bsm_records": r.bsm_records,
        })
    df = pd.DataFrame(rows)
    (ROOT / "outputs/tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "outputs/tables/ablation_results.csv", index=False)
    with open(ROOT / "outputs/tables/ablation_results.md", "w", encoding="utf-8") as f:
        f.write("# PhysTMC-USA Multi-FPS Ablation Results\n\n")
        f.write(f"Real recorded traffic clip, native {native_fps} FPS, {n} frames. "
                f"Reference = PhysTMC-USA @ native FPS.\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")

    # export BSM telemetry from the native PhysTMC config
    (ROOT / "outputs/data").mkdir(parents=True, exist_ok=True)
    bsm_by_label[ref.label].to_csv(str(ROOT / "outputs/data/virtual_bsm_telemetry.csv"))
    bsm_by_label[ref.label].to_jsonl(str(ROOT / "outputs/data/virtual_bsm_telemetry.jsonl"))

    # dashboard videos (showcase: PhysTMC filter ON, at a watchable rate)
    print("rendering dashboard video ...", flush=True)
    render_dashboard(source, boxes, cls, conf, native_fps, target_fps=12.0,
                     kinematic_filter=True,
                     out_path=str(ROOT / "outputs/videos/phystmc_usa_demo.mp4"))
    render_dashboard(source, boxes, cls, conf, native_fps, target_fps=12.0,
                     kinematic_filter=True, max_seconds=15.0,
                     out_path=str(ROOT / "outputs/videos/phystmc_usa_linkedin_clip.mp4"))

    print("\n=== ABLATION RESULTS ===")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
