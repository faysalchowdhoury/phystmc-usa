"""Ground-truth validation of low-frame-rate kinematic reconstruction.

Uses real NGSIM I-80 vehicle trajectories (10 Hz, measured position local_y and
speed v_vel) as ground truth. Each trajectory is decimated to a lower effective
frame rate and corrupted with realistic detection-localization noise, emulating a
low-FPS camera tracker. Three reconstructions of speed/acceleration are compared
against the ground truth by RMSE:

  1. raw        -- naive finite differencing (baseline tracker)
  2. ema_clamp  -- the PhysTMC-USA kinematic filter (EMA + physical clamp)
  3. smoother   -- PhysTMC+: constrained least-squares trajectory smoother with a
                   second-difference (acceleration) penalty, then physical clamp

Because the error is measured against real ground truth, the comparison is NOT
tautological: it quantifies which reconstruction recovers true motion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinematics import ACCEL_MIN, ACCEL_MAX, MAX_SPEED_MPS


# --------------------------------------------------------------------------- #
# NGSIM ground-truth trajectories
# --------------------------------------------------------------------------- #
def load_trajectories(path: str, min_len: int = 150, max_vehicles: int = 300):
    """Return list of dicts with t[s], y[m] (position), and ground-truth v[m/s],
    a[m/s^2] from Savitzky-Golay differentiation of the full-rate positions."""
    import pandas as pd
    from scipy.signal import savgol_filter
    df = pd.read_csv(path, usecols=["vehicle_id", "frame_id", "local_y"])
    df = df.dropna(subset=["local_y"])
    trajs = []
    for vid, g in df.groupby("vehicle_id"):
        g = g.sort_values("frame_id").drop_duplicates("frame_id")
        # keep only contiguous 10 Hz runs (no missing frames)
        fr = g["frame_id"].to_numpy(int)
        if len(g) < min_len or (np.diff(fr) != 1).mean() > 0.02:
            continue
        y = g["local_y"].to_numpy(float)
        dt = 0.1
        # reject known NGSIM position artifacts (non-physical jumps/kinematics)
        if np.ptp(y) < 20 or np.abs(np.diff(y)).max() > 5.0:   # >50 m/s step
            continue
        w = 11
        v = savgol_filter(y, w, 3, deriv=1, delta=dt)   # ground-truth speed
        a = savgol_filter(y, w, 3, deriv=2, delta=dt)   # ground-truth acceleration
        if np.abs(v).max() > MAX_SPEED_MPS or np.abs(a).max() > 20.0:
            continue
        t = (fr - fr[0]) * dt
        trajs.append({"t": t, "y": y, "v": v, "a": a})
        if len(trajs) >= max_vehicles:
            break
    return trajs


# --------------------------------------------------------------------------- #
# Reconstructors (operate on noisy, decimated positions)
# --------------------------------------------------------------------------- #
def recon_raw(t, y):
    v = np.gradient(y, t)
    a = np.gradient(v, t)
    return v, a


def recon_ema_clamp(t, y, alpha=0.5):
    n = len(y)
    v = np.zeros(n); a = np.zeros(n)
    if n < 2:
        return v, a
    dt0 = max(t[1] - t[0], 1e-3)
    v[0] = v[1] = (y[1] - y[0]) / dt0          # initialise from real first velocity
    for k in range(2, n):
        dt = max(t[k] - t[k - 1], 1e-3)
        vr = (y[k] - y[k - 1]) / dt
        vs = alpha * vr + (1 - alpha) * v[k - 1]    # EMA tracks true speed
        ac = float(np.clip((vs - v[k - 1]) / dt, ACCEL_MIN, ACCEL_MAX))  # de-spike
        vs = float(np.clip(v[k - 1] + ac * dt, 0.0, MAX_SPEED_MPS))
        v[k] = vs; a[k] = ac
    return v, a


def recon_smoother(t, y, lam=12.0):
    """Constrained least-squares smoother: minimise ||p-y||^2 + lam*||D2 p||^2,
    then derive speed/accel and clamp to the physical envelope."""
    n = len(y)
    if n < 4:
        return recon_raw(t, y)
    # second-difference operator (n-2 x n)
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i] = 1.0; D[i, i + 1] = -2.0; D[i, i + 2] = 1.0
    A = np.eye(n) + lam * (D.T @ D)
    p = np.linalg.solve(A, y)
    v = np.gradient(p, t)
    a = np.gradient(v, t)
    a = np.clip(a, ACCEL_MIN, ACCEL_MAX)
    v = np.clip(v, -MAX_SPEED_MPS, MAX_SPEED_MPS)
    return v, a


RECONSTRUCTORS = {"raw": recon_raw, "ema_clamp": recon_ema_clamp, "smoother": recon_smoother}


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
@dataclass
class ValResult:
    method: str
    hz: float
    speed_mae: float
    accel_mae: float
    unphysical_pct: float
    n_samples: int


def run_validation(trajs, native_hz: float = 10.0, rates=(10.0, 5.0, 2.0),
                   noise_m: float = 0.5, seed: int = 7) -> list[ValResult]:
    rng = np.random.default_rng(seed)
    acc = {(m, hz): {"ae_v": [], "ae_a": [], "unphys": 0, "n": 0}
           for m in RECONSTRUCTORS for hz in rates}

    for tr in trajs:
        t, y, v_true, a_true = tr["t"], tr["y"], tr["v"], tr["a"]
        for hz in rates:
            step = max(1, int(round(native_hz / hz)))
            idx = np.arange(0, len(t), step)
            if len(idx) < 5:
                continue
            td = t[idx]
            yd = y[idx] + rng.normal(0, noise_m, len(idx))   # detection noise
            v_ref = v_true[idx]; a_ref = a_true[idx]
            sl = slice(1, len(idx) - 1)                      # drop boundary samples
            for m, fn in RECONSTRUCTORS.items():
                vr, ar = fn(td, yd)
                acc[(m, hz)]["ae_v"].append(np.abs(vr[sl] - v_ref[sl]))
                acc[(m, hz)]["ae_a"].append(np.abs(ar[sl] - a_ref[sl]))
                acc[(m, hz)]["unphys"] += int(np.sum((np.abs(ar[sl]) > ACCEL_MAX) |
                                                     (np.abs(vr[sl]) > MAX_SPEED_MPS)))
                acc[(m, hz)]["n"] += (sl.stop - sl.start)

    out = []
    for (m, hz), d in acc.items():
        if d["n"] == 0:
            continue
        av = np.concatenate(d["ae_v"]); aa = np.concatenate(d["ae_a"])
        out.append(ValResult(method=m, hz=hz, speed_mae=float(av.mean()),
                             accel_mae=float(aa.mean()),
                             unphysical_pct=round(100 * d["unphys"] / d["n"], 2),
                             n_samples=d["n"]))
    return out
