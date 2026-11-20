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
    """Return list of dicts with t[s], y[m] (position), and INDEPENDENT ground-truth
    v[m/s], a[m/s^2] obtained from a quartic smoothing spline fit to the full-rate
    clean positions. The spline truth uses a different estimator and basis than any
    tested reconstruction (which see only decimated, noised positions), avoiding
    truth/method circularity."""
    import pandas as pd
    from scipy.interpolate import UnivariateSpline
    df = pd.read_csv(path, usecols=["vehicle_id", "frame_id", "local_y"])
    df = df.dropna(subset=["local_y"])
    trajs = []
    dt = 0.1
    for vid, g in df.groupby("vehicle_id"):
        g = g.sort_values("frame_id").drop_duplicates("frame_id")
        fr_all = g["frame_id"].to_numpy(int)
        y_all = g["local_y"].to_numpy(float)
        # NGSIM reuses vehicle_id across time periods/segments; split into contiguous
        # runs wherever the frame sequence or longitudinal position breaks.
        if len(fr_all) < 2:
            continue
        brk = np.where((np.diff(fr_all) > 5) | (np.abs(np.diff(y_all)) > 10))[0] + 1
        for seg in np.split(np.arange(len(fr_all)), brk):
            if len(seg) < min_len:
                continue
            fr = fr_all[seg]; y0 = y_all[seg]
            fr_u = np.arange(fr[0], fr[-1] + 1)
            if len(fr_u) < min_len or (np.diff(fr).max() > 5):
                continue
            y = np.interp(fr_u, fr, y0)
            t = (fr_u - fr_u[0]) * dt
            _maybe_add(trajs, t, y)
            if len(trajs) >= max_vehicles:
                return trajs
    return trajs


def _maybe_add(trajs, t, y):
    """De-spike, fit independent spline ground truth, and append if physical."""
    from scipy.interpolate import UnivariateSpline
    from scipy.ndimage import median_filter
    med = median_filter(y, size=5, mode="nearest")
    dev = np.abs(y - med)
    mad = np.median(dev) + 1e-6
    outliers = dev > 8.0 * mad
    if outliers.mean() > 0.02:                       # too corrupted to trust
        return
    y = np.where(outliers, med, y)
    if np.ptp(y) < 20 or np.abs(np.diff(y)).max() > 5.0:
        return
    try:
        sp = UnivariateSpline(t, y, k=4, s=len(t) * 0.03)   # independent truth
    except Exception:
        return
    v = sp.derivative(1)(t)
    a = sp.derivative(2)(t)
    if np.abs(v).max() > MAX_SPEED_MPS or np.abs(a).max() > 25.0:
        return
    trajs.append({"t": t, "y": y, "v": v, "a": a})


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


def recon_savgol(t, y):
    """Standard baseline: Savitzky-Golay differentiation of the sparse noisy
    positions (window adapted to length)."""
    from scipy.signal import savgol_filter
    n = len(y)
    if n < 5:
        return recon_raw(t, y)
    w = min(n if n % 2 else n - 1, 9)
    w = max(5, w if w % 2 else w - 1)
    dt = t[1] - t[0]
    v = savgol_filter(y, w, 2, deriv=1, delta=dt)
    a = savgol_filter(y, w, 2, deriv=2, delta=dt)
    return v, a


def recon_kalman(t, y, sigma_pos=0.5, sigma_jerk=3.0):
    """Standard baseline: constant-acceleration Kalman filter + RTS smoother.
    State [pos, vel, acc], position measurements, jerk process noise."""
    n = len(y)
    if n < 3:
        return recon_raw(t, y)
    dt = float(t[1] - t[0])
    F = np.array([[1, dt, 0.5 * dt * dt], [0, 1, dt], [0, 0, 1]], float)
    H = np.array([[1.0, 0.0, 0.0]])
    q = sigma_jerk ** 2
    Q = q * np.array([[dt**5 / 20, dt**4 / 8, dt**3 / 6],
                      [dt**4 / 8, dt**3 / 3, dt**2 / 2],
                      [dt**3 / 6, dt**2 / 2, dt]], float)
    R = np.array([[sigma_pos ** 2]])
    x = np.array([y[0], (y[1] - y[0]) / dt, 0.0])
    P = np.eye(3) * 10.0
    xs_f, Ps_f, xs_p, Ps_p = [], [], [], []
    for k in range(n):
        if k > 0:
            x = F @ x; P = F @ P @ F.T + Q
        xs_p.append(x.copy()); Ps_p.append(P.copy())
        S = H @ P @ H.T + R
        K = (P @ H.T) @ np.linalg.inv(S)
        x = x + (K @ (np.array([y[k]]) - H @ x)).ravel()
        P = (np.eye(3) - K @ H) @ P
        xs_f.append(x.copy()); Ps_f.append(P.copy())
    # RTS backward smoother
    xs = [a.copy() for a in xs_f]
    Ps = [a.copy() for a in Ps_f]
    for k in range(n - 2, -1, -1):
        C = Ps_f[k] @ F.T @ np.linalg.inv(Ps_p[k + 1])
        xs[k] = xs_f[k] + C @ (xs[k + 1] - xs_p[k + 1])
    v = np.array([s[1] for s in xs]); a = np.array([s[2] for s in xs])
    return v, a


RECONSTRUCTORS = {
    "raw": recon_raw,                 # naive baseline
    "savgol": recon_savgol,           # standard baseline
    "kalman": recon_kalman,           # standard baseline (RTS smoother)
    "ema_clamp": recon_ema_clamp,     # streaming physics filter
    "smoother": recon_smoother,       # proposed constrained smoother
}


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
                vr, ar = fn(td, yd, sigma_pos=max(noise_m, 1e-3)) if m == "kalman" else fn(td, yd)
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
