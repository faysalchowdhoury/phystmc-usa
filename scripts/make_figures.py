"""Publication figures (300 DPI, Times New Roman): calibration+trajectories,
multi-FPS ablation metrics, and speed/emissions profiles. Uses real ablation
outputs and reconstructed trajectories."""
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
from src.run_pipeline import (cache_detections, load_detections, build_calibrator,
                              run_config, DEFAULT_SRC_PTS, COCO_VEHICLES)
from src.kinematics import vt_micro_co2_gps, MPH_PER_MPS

FIG = ROOT / "outputs/figures"


def _setup():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11, "axes.titlesize": 12, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3})


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)


def fig1(frame_path, track_samples, calib):
    import cv2
    _setup()
    img = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.3))
    axL.imshow(img)
    src = np.array(DEFAULT_SRC_PTS + [DEFAULT_SRC_PTS[0]])
    axL.plot(src[:, 0], src[:, 1], "-o", color="#e4572e", lw=2, ms=4, label="homography control polygon")
    axL.set_title("(a) Camera reference frame + ground-plane calibration")
    axL.axis("off"); axL.legend(loc="lower right", fontsize=8)
    # reconstructed ground trajectories
    for tid, s in track_samples.items():
        if len(s) < 3:
            continue
        a = np.array([(x, y) for (_, x, y) in s])
        axR.plot(a[:, 0], a[:, 1], lw=1.2, alpha=0.8)
    axR.set_xlabel("Ground X across road (m)"); axR.set_ylabel("Ground Y along road (m)")
    axR.set_title("(b) Physics-filtered reconstructed trajectories")
    axR.invert_yaxis()
    _save(fig, "fig1_architecture_and_calibration")


def fig2(df):
    _setup()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    rates = sorted(df["fps"].unique())
    for ax, (col, title, ylab) in zip(axes, [
            ("unphysical_accel_spike_pct", "Unphysical acceleration spikes", "|a|>4.5 m/s^2 rate (%)"),
            ("count_MAPE_pct_vs_native", "Count error vs native FPS", "Directional count MAPE (%)"),
            ("co2_drift_pct_vs_native", "VT-Micro CO2 drift", "CO2 drift vs native (%)")]):
        for tracker, color, mk in [("Baseline raw", "#d1495b", "o"), ("PhysTMC-USA", "#2e86ab", "s")]:
            sub = df[df["tracker"] == tracker].sort_values("fps")
            ax.plot(sub["fps"], sub[col], mk + "-", color=color, label=tracker)
        ax.set_xlabel("Frame rate (FPS)"); ax.set_ylabel(ylab); ax.set_title(title)
        ax.set_xticks(rates)
    axes[0].legend(fontsize=8)
    fig.suptitle("Multi-FPS ablation: physics-informed filtering vs raw tracking", y=1.03)
    _save(fig, "fig2_fps_ablation_metrics")


def fig3(bsm_csv):
    _setup()
    df = pd.read_csv(bsm_csv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    # speed time-series for a few representative tracks
    for tid, g in df.groupby("temp_id"):
        if len(g) >= 6:
            ax1.plot(g["timestamp_s"], g["speed_mph"], lw=1, alpha=0.7)
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Speed (mph)")
    ax1.set_title("(a) Reconstructed vehicle speed profiles (virtual J2735 BSMs)")
    # speed-acceleration phase portrait colored by CO2
    co2 = [vt_micro_co2_gps(v / MPH_PER_MPS, a, "car") for v, a in
           zip(df["speed_mph"], df["lon_accel_mps2"])]
    sc = ax2.scatter(df["speed_mph"], df["lon_accel_mps2"], c=co2, s=8, cmap="viridis")
    ax2.set_xlabel("Speed (mph)"); ax2.set_ylabel("Longitudinal accel (m/s^2)")
    ax2.set_title("(b) Speed-acceleration phase portrait (color = VT-Micro CO2)")
    fig.colorbar(sc, ax=ax2, label="CO2 (g/s)")
    _save(fig, "fig3_speed_emissions_profiles")


def main():
    source = str(ROOT / "data/raw/source_traffic.mp4")
    import cv2
    cap = cv2.VideoCapture(source); native = round(cap.get(cv2.CAP_PROP_FPS) or 25.0, 2); cap.release()
    cache = cache_detections(source)
    boxes, cls, conf, n = load_detections(cache)
    calib = build_calibrator()
    _, _, tracks = run_config(boxes, cls, conf, native, native, True, calib)
    fig1(ROOT / "data/raw/reference_frame.jpg", tracks, calib)
    fig2(pd.read_csv(ROOT / "outputs/tables/ablation_results.csv"))
    fig3(ROOT / "outputs/data/virtual_bsm_telemetry.csv")
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
