"""PhysTMC-USA end-to-end pipeline: detect -> track -> physics-informed trajectory
reconstruction -> HCM/emissions/BSM auditing, on real recorded traffic video.

Detections are cached once at native frame rate (expensive YOLO pass); each
frame-rate / kinematic-filter configuration reuses them, so the ablation is fast
and apples-to-apples (only the sampling rate and the physics layer change).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

from .kinematics import (MUTCDHomographyCalibrator, KinematicTrajectoryFilter,
                         vt_micro_co2_gps, detect_pet_conflicts, MPH_PER_MPS)
from .hcm7_audit import hcm7_signalized_delay, peak_hour_factor
from .v2x_bsm import BSMEncoder

ROOT = Path(__file__).resolve().parents[1]
COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Approximate ground-plane control points for the testbed clip (3840x2160),
# spanning the visible road plane. Scale is approximate (see camera_metadata.json).
DEFAULT_SRC_PTS = [[600, 2050], [3400, 2050], [2100, 720], [1700, 720]]
ROAD_WIDTH_M = 18.0
ROAD_LENGTH_M = 80.0
GATE_Y_M = ROAD_LENGTH_M / 2.0     # counting gate at mid-segment


def build_calibrator() -> MUTCDHomographyCalibrator:
    c = MUTCDHomographyCalibrator(DEFAULT_SRC_PTS, lane_width_m=3.6,
                                  lane_count=5, longitudinal_m=ROAD_LENGTH_M)
    return c


# --------------------------------------------------------------------------- #
# Detection caching (single native-rate YOLO pass)
# --------------------------------------------------------------------------- #
def cache_detections(video: str, model_name: str = "yolo11s.pt", imgsz: int = 1920,
                     conf: float = 0.3, cache: str | None = None) -> str:
    import cv2
    from ultralytics import YOLO
    cache = cache or str(ROOT / "outputs/data/detections_cache.npz")
    if os.path.exists(cache):
        return cache
    model = YOLO(model_name)
    cap = cv2.VideoCapture(video)
    boxes_all, cls_all, conf_all, idx = [], [], [], []
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, imgsz=imgsz, conf=conf, classes=list(COCO_VEHICLES),
                          verbose=False)[0]
        b = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        c = r.boxes.cls.cpu().numpy().astype(np.int16)
        s = r.boxes.conf.cpu().numpy().astype(np.float32)
        boxes_all.append(b); cls_all.append(c); conf_all.append(s); idx.append(fi)
        fi += 1
    cap.release()
    np.savez_compressed(cache,
                        boxes=np.array(boxes_all, dtype=object),
                        cls=np.array(cls_all, dtype=object),
                        conf=np.array(conf_all, dtype=object),
                        n_frames=fi)
    return cache


def load_detections(cache: str):
    d = np.load(cache, allow_pickle=True)
    return list(d["boxes"]), list(d["cls"]), list(d["conf"]), int(d["n_frames"])


# --------------------------------------------------------------------------- #
# One ablation configuration
# --------------------------------------------------------------------------- #
@dataclass
class ConfigResult:
    label: str
    fps: float
    kinematic_filter: bool
    unique_tracks: int
    directional_counts: dict
    total_count: int
    mean_speed_mph: float
    unphysical_accel_rate_pct: float
    mean_co2_gps: float
    pet_conflicts: int
    hcm_delay_s: float
    hcm_los: str
    hcm_vc: float
    phf: float
    bsm_records: int


def run_config(boxes, cls, conf, native_fps: float, target_fps: float,
               kinematic_filter: bool, calib, anchor=(39.0, -76.9)):
    from .tracker import ByteStyleTracker
    step = max(1, int(round(native_fps / target_fps)))
    tracker = ByteStyleTracker(max_age=int(round(target_fps)) * 2)
    kf = KinematicTrajectoryFilter(enabled=kinematic_filter, alpha=0.5)
    bsm = BSMEncoder(anchor_lat=anchor[0], anchor_lon=anchor[1])

    track_samples: dict[int, list] = defaultdict(list)
    speeds, co2s = [], []
    gate_dir_counts: dict[str, set] = defaultdict(set)
    counted = set()
    quarter_counts = [0, 0, 0, 0]
    n_frames = len(boxes)
    dt = step / native_fps

    for fi in range(0, n_frames, step):
        t = fi / native_fps
        b, c, s = boxes[fi], cls[fi], conf[fi]
        tracked = tracker.update(b if len(b) else np.zeros((0, 4), np.float32),
                                 s if len(b) else np.zeros(0, np.float32),
                                 c if len(b) else np.zeros(0, np.int64))
        for tid, box, cid in tracked:
            x1, y1, x2, y2 = box
            foot = np.array([[(x1 + x2) / 2.0, y2]], np.float32)
            gx, gy = calib.to_ground(foot)[0]
            st = kf.update(tid, float(gx), float(gy), t)
            vclass = COCO_VEHICLES.get(int(cid), "car")
            speed_mph = st.speed * MPH_PER_MPS
            track_samples[tid].append((t, float(gx), float(gy)))
            if st.n > 1:
                speeds.append(speed_mph)
                co2s.append(vt_micro_co2_gps(st.speed, st.accel, vclass))
            # gate crossing for directional counts
            prev = track_samples[tid][-2][2] if len(track_samples[tid]) >= 2 else None
            if prev is not None and tid not in counted:
                if (prev - GATE_Y_M) * (gy - GATE_Y_M) <= 0:
                    leg = "downstream" if (gy - prev) < 0 else "upstream"
                    gate_dir_counts[leg].add(tid)
                    counted.add(tid)
                    q = min(3, int((t / (n_frames / native_fps)) * 4))
                    quarter_counts[q] += 1
            entry = "upstream" if st.vy < 0 else "downstream"
            bsm.add(t, tid, vclass, gx, gy, speed_mph, st.accel, st.heading,
                    entry_leg=entry, exit_leg=entry)

    unique = len({tid for tid, s in track_samples.items() if len(s) >= 2})
    dir_counts = {k: len(v) for k, v in gate_dir_counts.items()}
    total_count = sum(dir_counts.values())
    pet = detect_pet_conflicts(track_samples)
    mean_speed = float(np.mean(speeds)) if speeds else 0.0
    mean_co2 = float(np.mean(co2s)) if co2s else 0.0

    # HCM: flow extrapolated from observed gate count over the clip duration
    clip_h = (n_frames / native_fps) / 3600.0
    flow_vph = total_count / clip_h if clip_h > 0 else 0.0
    phf = peak_hour_factor(quarter_counts)
    hcm = hcm7_signalized_delay(flow_veh_h=flow_vph, capacity_veh_h=3600.0,
                                phf=phf if phf > 0 else 0.9)

    res = ConfigResult(
        label=f"{'PhysTMC' if kinematic_filter else 'Baseline'}@{int(round(target_fps))}fps",
        fps=target_fps, kinematic_filter=kinematic_filter, unique_tracks=unique,
        directional_counts=dir_counts, total_count=total_count,
        mean_speed_mph=round(mean_speed, 2),
        unphysical_accel_rate_pct=round(100 * kf.unphysical_rate, 2),
        mean_co2_gps=round(mean_co2, 4), pet_conflicts=pet,
        hcm_delay_s=round(hcm.control_delay, 2), hcm_los=hcm.los,
        hcm_vc=round(hcm.v_over_c, 3), phf=round(phf, 3), bsm_records=len(bsm))
    return res, bsm, track_samples


def geh(m: float, c: float) -> float:
    if m + c == 0:
        return 0.0
    return float(np.sqrt(2 * (m - c) ** 2 / (m + c)))


def main():
    ap = argparse.ArgumentParser(description="PhysTMC-USA single-config runner")
    ap.add_argument("--source", default=str(ROOT / "data/raw/source_traffic.mp4"))
    ap.add_argument("--metadata", default=str(ROOT / "data/raw/camera_metadata.json"))
    ap.add_argument("--target-fps", type=float, default=5.0)
    ap.add_argument("--kinematic-filter", choices=["on", "off"], default="on")
    ap.add_argument("--out-dir", default=str(ROOT / "outputs"))
    args = ap.parse_args()

    import cv2
    cap = cv2.VideoCapture(args.source)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    cache = cache_detections(args.source)
    boxes, cls, conf, n = load_detections(cache)
    calib = build_calibrator()
    res, bsm, _ = run_config(boxes, cls, conf, native_fps, args.target_fps,
                             args.kinematic_filter == "on", calib)
    Path(args.out_dir, "data").mkdir(parents=True, exist_ok=True)
    bsm.to_csv(str(Path(args.out_dir, "data/virtual_bsm_telemetry.csv")))
    print(json.dumps(res.__dict__, indent=2))


if __name__ == "__main__":
    main()
