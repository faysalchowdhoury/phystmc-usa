"""Physics-informed kinematics for trajectory reconstruction from low-frame-rate
traffic video: ground-plane calibration, a kinematic trajectory filter that
rejects physically-impossible motion, VT-Micro CO2 emissions, and PET surrogate
safety conflicts.

Calibration note: the metric scale is set from an assumed standard lane width (the
testbed clip is not a surveyed site), so absolute speeds/emissions are approximate.
The frame-rate ablation claims (count stability, unphysical-acceleration
suppression, GEH) are scale-invariant and are the quantitative core of the study.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

MPH_PER_MPS = 2.2369362920544
MAX_SPEED_MPS = 95.0 / MPH_PER_MPS          # 95 mph physical ceiling
ACCEL_MIN = -4.5                             # m/s^2 (hard braking bound)
ACCEL_MAX = 3.5                              # m/s^2 (light-duty tractive bound)

LIGHT_DUTY = {"car", "motorcycle"}
HEAVY_DUTY = {"bus", "truck"}


# --------------------------------------------------------------------------- #
# Ground-plane calibration
# --------------------------------------------------------------------------- #
class MUTCDHomographyCalibrator:
    """Pixel -> ground-plane (metres) perspective transform.

    Built from four image points bounding a road patch of known real dimensions
    (``lane_count`` lanes of ``lane_width_m`` across by ``longitudinal_m`` long).
    The four source points are ordered: near-left, near-right, far-right, far-left.
    """

    def __init__(self, src_pts, lane_width_m: float = 3.6, lane_count: int = 2,
                 longitudinal_m: float = 80.0):
        import cv2
        self.src = np.asarray(src_pts, dtype=np.float32)
        self.lane_width_m = lane_width_m
        self.width_m = lane_width_m * lane_count
        self.longitudinal_m = longitudinal_m
        # destination rectangle in metres (x across road, y along road; near y=0)
        self.dst = np.array([[0.0, 0.0], [self.width_m, 0.0],
                             [self.width_m, longitudinal_m], [0.0, longitudinal_m]],
                            dtype=np.float32)
        self.H = cv2.getPerspectiveTransform(self.src, self.dst)
        self.H_inv = cv2.getPerspectiveTransform(self.dst, self.src)

    def to_ground(self, pts_px: np.ndarray) -> np.ndarray:
        """Map pixel points [N,2] to ground metres [N,2]."""
        import cv2
        pts = np.asarray(pts_px, dtype=np.float32).reshape(-1, 1, 2)
        g = cv2.perspectiveTransform(pts, self.H)
        return g.reshape(-1, 2)

    def to_ground_feet(self, pts_px: np.ndarray) -> np.ndarray:
        return self.to_ground(pts_px) / 0.3048


# --------------------------------------------------------------------------- #
# Kinematic trajectory filter
# --------------------------------------------------------------------------- #
@dataclass
class TrackState:
    t: float
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    speed: float = 0.0          # m/s
    accel: float = 0.0          # m/s^2 (longitudinal)
    heading: float = 0.0        # deg
    raw_speed: float = 0.0
    n: int = 0


@dataclass
class KinematicTrajectoryFilter:
    """Enforces physical vehicle kinematics on noisy per-track ground positions.

    * clamps instantaneous speed to ``MAX_SPEED_MPS``
    * clamps longitudinal acceleration to ``[ACCEL_MIN, ACCEL_MAX]``
    * exponentially smooths velocity (``alpha``) to suppress low-FPS jitter

    With ``enabled=False`` it passes raw finite-difference kinematics through
    (the baseline tracker), so the two modes differ only in the physics layer.
    """
    enabled: bool = True
    alpha: float = 0.5
    states: dict = field(default_factory=dict)
    unphysical_events: int = 0
    total_updates: int = 0

    def update(self, track_id: int, x: float, y: float, t: float) -> TrackState:
        self.total_updates += 1
        prev = self.states.get(track_id)
        if prev is None:
            st = TrackState(t=t, x=x, y=y, n=1)
            self.states[track_id] = st
            return st
        dt = max(t - prev.t, 1e-3)
        vx_raw = (x - prev.x) / dt
        vy_raw = (y - prev.y) / dt
        raw_speed = float(np.hypot(vx_raw, vy_raw))
        raw_accel = (raw_speed - prev.raw_speed) / dt

        if self.enabled:
            # EMA-smooth the velocity (tracks true speed, rejects per-frame jitter)
            a = self.alpha
            vx = a * vx_raw + (1 - a) * prev.vx
            vy = a * vy_raw + (1 - a) * prev.vy
            speed = float(np.hypot(vx, vy))
            if speed > MAX_SPEED_MPS:                      # clamp impossible speed
                vx *= MAX_SPEED_MPS / speed; vy *= MAX_SPEED_MPS / speed
                speed = MAX_SPEED_MPS
            accel = (speed - prev.speed) / dt
            if accel > ACCEL_MAX or accel < ACCEL_MIN:     # clamp impossible accel
                accel = float(np.clip(accel, ACCEL_MIN, ACCEL_MAX))
                speed = max(0.0, prev.speed + accel * dt)
                mag = float(np.hypot(vx, vy)) + 1e-6
                vx *= speed / mag; vy *= speed / mag
            out_speed, out_accel = speed, accel
        else:
            vx, vy, out_speed, out_accel = vx_raw, vy_raw, raw_speed, raw_accel

        # count unphysical events in THIS mode's OUTPUT (filtered output is bounded
        # by construction; the raw baseline is not -> this is the ablation signal)
        if out_speed > MAX_SPEED_MPS or out_accel > ACCEL_MAX or out_accel < ACCEL_MIN:
            self.unphysical_events += 1

        heading = float(np.degrees(np.arctan2(vy, vx)))
        st = TrackState(t=t, x=x, y=y, vx=vx, vy=vy, speed=out_speed, accel=out_accel,
                        heading=heading, raw_speed=raw_speed, n=prev.n + 1)
        self.states[track_id] = st
        return st

    @property
    def unphysical_rate(self) -> float:
        return self.unphysical_events / max(self.total_updates, 1)


# --------------------------------------------------------------------------- #
# VT-Micro CO2 emissions
# --------------------------------------------------------------------------- #
# VT-Micro-family instantaneous model: ln(MOE) = sum_i sum_j k_ij * s^i * a^j.
# CO2 (g/s) for a representative light-duty vehicle and a heavy-duty scaling.
# Coefficients are representative of the VT-Micro model class; absolute values are
# approximate, the filter-vs-baseline *drift* is the reported quantity.
_VTM_LDV = np.array([
    [-0.6, 0.0352, 0.000847, -1.6e-5],
    [0.072, 0.0120, -0.00032, 8.0e-6],
    [-0.00087, -0.00043, 1.1e-5, -2.0e-7],
    [6.3e-6, 4.0e-6, -1.0e-7, 2.0e-9],
])
_HDV_SCALE = 3.4      # heavy-duty CO2 scaling vs light-duty baseline


def vt_micro_co2_gps(speed_mps: float, accel_mps2: float, vclass: str) -> float:
    """Instantaneous CO2 emission rate (g/s) via a VT-Micro-class model."""
    s = float(np.clip(speed_mps, 0.0, MAX_SPEED_MPS))
    a = float(np.clip(accel_mps2, ACCEL_MIN, ACCEL_MAX))
    ln_moe = 0.0
    for i in range(4):
        for j in range(4):
            ln_moe += _VTM_LDV[i, j] * (s ** i) * (a ** j)
    co2 = float(np.exp(ln_moe))
    if vclass in HEAVY_DUTY:
        co2 *= _HDV_SCALE
    return co2


# --------------------------------------------------------------------------- #
# PET surrogate safety conflicts
# --------------------------------------------------------------------------- #
def detect_pet_conflicts(track_samples: dict, cell_m: float = 3.0,
                         pet_threshold_s: float = 2.0) -> int:
    """Count Post-Encroachment-Time conflicts (PET < threshold).

    ``track_samples``: {track_id: list of (t, x, y)} in ground metres. Two vehicles
    conflict when they occupy the same ground cell within ``pet_threshold_s``.
    """
    # earliest entry time of each (cell, vehicle) -> frame-rate robust
    entry: dict[tuple, dict] = {}
    for tid, samples in track_samples.items():
        for (t, x, y) in samples:
            key = (int(x // cell_m), int(y // cell_m))
            d = entry.setdefault(key, {})
            if tid not in d or t < d[tid]:
                d[tid] = t
    conflicts = 0
    for key, veh_times in entry.items():
        times = sorted(veh_times.values())        # one time per distinct vehicle
        for i in range(1, len(times)):
            if 0 < times[i] - times[i - 1] < pet_threshold_s:
                conflicts += 1
    return conflicts
