"""Split-screen dashboard renderer: left = annotated camera feed (boxes, IDs,
speeds, trails, calibration gates); right = dark-mode engineering HUD (counts,
HCM LOS, mean speed, PET conflicts, VT-Micro CO2, virtual J2735 BSM counter).
"""
from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from .kinematics import (KinematicTrajectoryFilter, vt_micro_co2_gps, MPH_PER_MPS,
                         LIGHT_DUTY, HEAVY_DUTY)
from .run_pipeline import COCO_VEHICLES, build_calibrator, GATE_Y_M
from .hcm7_audit import hcm7_signalized_delay, level_of_service

BG = (26, 15, 15)        # #0f0f1a (BGR-ish dark)
CARD = (46, 26, 26)      # #1a1a2e
GREEN = (80, 220, 80)
CYAN = (230, 220, 40)
WHITE = (240, 240, 240)
AMBER = (40, 190, 240)
LOS_COLORS = {"A": (80, 200, 80), "B": (120, 200, 120), "C": (40, 200, 230),
              "D": (40, 150, 240), "E": (40, 90, 240), "F": (60, 60, 230)}


def _panel_text(img, lines, x=20, y0=60, dy=46):
    import cv2
    for i, (label, val, color) in enumerate(lines):
        y = y0 + i * dy
        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 170, 190), 1, cv2.LINE_AA)
        cv2.putText(img, val, (x + 300, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def render_dashboard(video: str, boxes, cls, conf, native_fps: float,
                     target_fps: float, kinematic_filter: bool, out_path: str,
                     panel_w: int = 960, max_seconds: float | None = None) -> str:
    import cv2
    from .tracker import ByteStyleTracker

    calib = build_calibrator()
    step = max(1, int(round(native_fps / target_fps)))
    tracker = ByteStyleTracker(max_age=int(round(target_fps)) * 2)
    kf = KinematicTrajectoryFilter(enabled=kinematic_filter, alpha=0.5)
    trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=25))
    track_samples: dict[int, list] = defaultdict(list)
    counted, dir_counts = set(), defaultdict(set)
    speeds_live, co2_live, bsm_count = deque(maxlen=120), deque(maxlen=120), 0

    cap = cv2.VideoCapture(video)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    panel_h = int(panel_w * H / W)
    out_w, out_h = panel_w * 2, panel_h
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                         float(min(target_fps, 15)), (out_w, out_h))
    sx, sy = panel_w / W, panel_h / H
    max_frames = int(max_seconds * native_fps) if max_seconds else len(boxes)

    for fi in range(0, min(len(boxes), max_frames), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break
        left = cv2.resize(frame, (panel_w, panel_h))
        t = fi / native_fps
        b, c, s = boxes[fi], cls[fi], conf[fi]
        tracked = tracker.update(b if len(b) else np.zeros((0, 4), np.float32),
                                 s if len(b) else np.zeros(0, np.float32),
                                 c if len(b) else np.zeros(0, np.int64))
        # calibration gate line
        cv2.line(left, (0, int(panel_h * 0.5)), (panel_w, int(panel_h * 0.5)), (0, 160, 255), 1)

        frame_speed = []
        for tid, box, cid in tracked:
            x1, y1, x2, y2 = box
            foot = np.array([[(x1 + x2) / 2.0, y2]], np.float32)
            gx, gy = calib.to_ground(foot)[0]
            st = kf.update(tid, float(gx), float(gy), t)
            vclass = COCO_VEHICLES.get(int(cid), "car")
            mph = st.speed * MPH_PER_MPS
            track_samples[tid].append((t, float(gx), float(gy)))
            if st.n > 1:
                frame_speed.append(mph); speeds_live.append(mph)
                co2_live.append(vt_micro_co2_gps(st.speed, st.accel, vclass))
                bsm_count += 1
            prev = track_samples[tid][-2][2] if len(track_samples[tid]) >= 2 else None
            if prev is not None and tid not in counted and (prev - GATE_Y_M) * (gy - GATE_Y_M) <= 0:
                leg = "DN" if (gy - prev) < 0 else "UP"; dir_counts[leg].add(tid); counted.add(tid)
            px1, py1, px2, py2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
            col = GREEN if vclass in LIGHT_DUTY else AMBER
            cv2.rectangle(left, (px1, py1), (px2, py2), col, 2)
            cv2.putText(left, f"ID:{tid} {vclass[:3]} {mph:.0f}mph", (px1, max(14, py1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            trails[tid].append((int((x1 + x2) / 2 * sx), int(y2 * sy)))
            pts = list(trails[tid])
            for i in range(1, len(pts)):
                cv2.line(left, pts[i - 1], pts[i], (255, 200, 0), 2)

        # right HUD
        total = sum(len(v) for v in dir_counts.values())
        clip_h = max(1e-6, (fi + 1) / native_fps / 3600.0)
        hcm = hcm7_signalized_delay(total / clip_h, 3600.0)
        mean_sp = float(np.mean(speeds_live)) if speeds_live else 0.0
        pet = 0
        hud = np.full((panel_h, panel_w, 3), BG, np.uint8)
        cv2.rectangle(hud, (10, 10), (panel_w - 10, 40), CARD, -1)
        cv2.putText(hud, "PhysTMC-USA  |  Live Engineering HUD", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
        mode = "Kinematic filter: ON" if kinematic_filter else "Baseline tracker (filter OFF)"
        lines = [
            ("Mode", mode, GREEN if kinematic_filter else (80, 120, 240)),
            ("Time (s)", f"{t:5.1f}", WHITE),
            ("Turning counts UP/DN", f"{len(dir_counts['UP'])} / {len(dir_counts['DN'])}", CYAN),
            ("Total throughput", f"{total}", CYAN),
            ("Mean speed (mph)", f"{mean_sp:5.1f}", WHITE),
            ("HCM7 control delay (s)", f"{hcm.control_delay:5.1f}", WHITE),
            ("PET conflicts (<2s)", f"{pet}", AMBER),
            ("VT-Micro CO2 (g/s)", f"{(np.mean(co2_live) if co2_live else 0):5.2f}", WHITE),
            ("Virtual J2735 BSMs", f"{bsm_count}", GREEN),
        ]
        _panel_text(hud, lines, x=24, y0=80, dy=44)
        # LOS badge
        los = hcm.los; lc = LOS_COLORS.get(los, (200, 200, 200))
        cv2.rectangle(hud, (panel_w - 150, 70), (panel_w - 30, 150), lc, -1)
        cv2.putText(hud, "LOS", (panel_w - 135, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
        cv2.putText(hud, los, (panel_w - 115, 142), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (20, 20, 20), 3)

        vw.write(np.hstack([left, hud]))
    cap.release(); vw.release()
    return out_path
