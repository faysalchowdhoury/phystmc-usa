"""Ground-truth benchmark on real NGSIM trajectories (multi-site, multi-noise).

Decimates real 10 Hz trajectories to low effective frame rates, adds detection
noise, and scores five reconstructions of speed/acceleration against an INDEPENDENT
spline ground truth (MAE). Compares the proposed constrained smoother to two standard
baselines (Savitzky-Golay, Kalman RTS) plus raw differencing and an EMA+clamp filter,
across two real freeway sites (I-80, US-101) and a noise sweep.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.validation import load_trajectories, run_validation

SITES = {"I-80": ROOT / "data/raw/ngsim_i80.csv.gz",
         "US-101": ROOT / "data/raw/ngsim_us101.csv.gz"}
METHOD_ORDER = ["raw", "savgol", "kalman", "ema_clamp", "smoother"]
LABELS = {"raw": "Raw finite-diff", "savgol": "Savitzky-Golay",
          "kalman": "Kalman RTS", "ema_clamp": "EMA + clamp",
          "smoother": "Constrained smoother (ours)"}
COLORS = {"raw": "#d1495b", "savgol": "#8d96a3", "kalman": "#66a182",
          "ema_clamp": "#edae49", "smoother": "#2e86ab"}
MARK = {"raw": "o", "savgol": "v", "kalman": "D", "ema_clamp": "^", "smoother": "s"}


def _setup():
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 11, "figure.dpi": 300, "savefig.dpi": 300,
                         "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3})


def fig_methods(df, out):
    _setup()
    sub = df[df["site"] == "I-80"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, yl, title in [
            (a1, "speed_mae_mps", "Speed MAE (m/s)", "(a) Speed recovery"),
            (a2, "accel_mae_mps2", "Accel MAE (m/s²)", "(b) Acceleration recovery")]:
        for m in METHOD_ORDER:
            s = sub[sub["method"] == m].sort_values("hz")
            ax.plot(s["hz"], s[col], MARK[m] + "-", color=COLORS[m], label=LABELS[m])
        ax.set_xlabel("Effective frame rate (Hz)"); ax.set_ylabel(yl); ax.set_title(title)
        ax.set_xticks([2, 5, 10])
    a2.set_yscale("log"); a1.legend(fontsize=8)
    fig.suptitle("NGSIM I-80 ground-truth benchmark (σ=0.5 m detection noise)", y=1.02)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig4_ngsim_ground_truth_validation.{ext}")
    plt.close(fig)


def fig_noise(sweep, out):
    _setup()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for m in METHOD_ORDER:
        s = sweep[(sweep["method"] == m) & (sweep["hz"] == 10.0)].sort_values("noise_m")
        ax.plot(s["noise_m"], s["accel_mae_mps2"], MARK[m] + "-", color=COLORS[m], label=LABELS[m])
    ax.set_xlabel("Detection localization noise σ (m)")
    ax.set_ylabel("Accel MAE (m/s²)"); ax.set_yscale("log")
    ax.set_title("Acceleration recovery vs detection noise (I-80, 10 Hz)")
    ax.legend(fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig5_noise_sensitivity.{ext}")
    plt.close(fig)


def main():
    rows = []
    counts = {}
    for site, path in SITES.items():
        if not path.exists():
            print(f"skip {site}: {path} not found"); continue
        trajs = load_trajectories(str(path), min_len=150, max_vehicles=800)
        counts[site] = len(trajs)
        print(f"{site}: {len(trajs)} clean trajectories")
        for r in run_validation(trajs, rates=(10.0, 5.0, 2.0), noise_m=0.5, seed=7):
            rows.append({"site": site, "method": r.method, "hz": r.hz,
                         "speed_mae_mps": round(r.speed_mae, 3),
                         "accel_mae_mps2": round(r.accel_mae, 3),
                         "unphysical_pct": r.unphysical_pct, "n_samples": r.n_samples})
    df = pd.DataFrame(rows)
    (ROOT / "outputs/tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "outputs/tables/ngsim_validation.csv", index=False)

    # noise sweep on I-80
    sweep_rows = []
    i80 = SITES["I-80"]
    if i80.exists():
        trajs = load_trajectories(str(i80), min_len=150, max_vehicles=800)
        for noise in (0.25, 0.5, 1.0, 2.0):
            for r in run_validation(trajs, rates=(10.0, 5.0, 2.0), noise_m=noise, seed=7):
                sweep_rows.append({"method": r.method, "hz": r.hz, "noise_m": noise,
                                   "speed_mae_mps": round(r.speed_mae, 3),
                                   "accel_mae_mps2": round(r.accel_mae, 3)})
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(ROOT / "outputs/tables/ngsim_noise_sweep.csv", index=False)

    with open(ROOT / "outputs/tables/ngsim_validation.md", "w", encoding="utf-8") as f:
        f.write("# NGSIM Ground-Truth Benchmark (multi-site, multi-noise)\n\n")
        f.write("Clean trajectories: " + ", ".join(f"{k} = {v}" for k, v in counts.items())
                + ". Ground truth = quartic smoothing spline on full-rate clean positions "
                "(independent of all tested reconstructions). Detection noise σ=0.5 m. "
                "MAE, lower is better.\n\n")
        f.write(df.to_markdown(index=False)); f.write("\n\n## Noise sensitivity (I-80)\n\n")
        f.write(sweep[sweep["hz"] == 10.0].to_markdown(index=False)); f.write("\n")

    fig_methods(df, ROOT / "outputs/figures")
    if not sweep.empty:
        fig_noise(sweep, ROOT / "outputs/figures")
    print("\n=== I-80 (sigma=0.5) ===")
    print(df[df.site == "I-80"].to_string(index=False))
    if "US-101" in counts:
        print("\n=== US-101 (sigma=0.5) ===")
        print(df[df.site == "US-101"].to_string(index=False))


if __name__ == "__main__":
    main()
