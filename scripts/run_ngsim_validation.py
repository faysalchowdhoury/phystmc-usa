"""Ground-truth validation on real NGSIM I-80 trajectories: decimate 10 Hz
trajectories to low effective frame rates, add detection-localization noise, and
measure how well each reconstruction recovers TRUE speed/acceleration (MAE vs
Savitzky-Golay ground truth). Writes a table and a figure.
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

# real NGSIM I-80 slice cached by the companion project (10 Hz, SI metres)
NGSIM_CANDIDATES = [
    ROOT / "data/raw/ngsim_i80.csv.gz",
    ROOT.parent / "convoy-cyber-physical-control/data/raw/ngsim/i-80_raw.csv.gz",
]
LABELS = {"raw": "Raw finite-diff (baseline)",
          "ema_clamp": "EMA + physical clamp",
          "smoother": "Constrained smoother (PhysTMC+)"}
COLORS = {"raw": "#d1495b", "ema_clamp": "#edae49", "smoother": "#2e86ab"}
MARK = {"raw": "o", "ema_clamp": "^", "smoother": "s"}


def _ngsim_path():
    for p in NGSIM_CANDIDATES:
        if p.exists():
            return str(p)
    raise FileNotFoundError("NGSIM I-80 cache not found; expected one of "
                            + ", ".join(str(p) for p in NGSIM_CANDIDATES))


def make_figure(df, out):
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 11, "figure.dpi": 300, "savefig.dpi": 300,
                         "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, title, yl in [(a1, "speed_mae_mps", "Speed recovery vs ground truth", "Speed MAE (m/s)"),
                               (a2, "accel_mae_mps2", "Acceleration recovery vs ground truth", "Accel MAE (m/s²)")]:
        for m in ["raw", "ema_clamp", "smoother"]:
            sub = df[df["method"] == m].sort_values("hz")
            ax.plot(sub["hz"], sub[col], MARK[m] + "-", color=COLORS[m], label=LABELS[m])
        ax.set_xlabel("Effective frame rate (Hz)"); ax.set_ylabel(yl); ax.set_title(title)
        ax.set_xticks([2, 5, 10])
    a2.set_yscale("log")
    a1.legend(fontsize=8)
    fig.suptitle("NGSIM I-80 ground-truth validation (real trajectories, decimated + noised)", y=1.02)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig4_ngsim_ground_truth_validation.{ext}")
    plt.close(fig)


def main():
    path = _ngsim_path()
    trajs = load_trajectories(path, min_len=150, max_vehicles=800)
    print(f"clean NGSIM trajectories: {len(trajs)}")
    res = run_validation(trajs, rates=(10.0, 5.0, 2.0), noise_m=0.5, seed=7)
    rows = [{"method": r.method, "hz": r.hz, "speed_mae_mps": round(r.speed_mae, 3),
             "accel_mae_mps2": round(r.accel_mae, 3),
             "unphysical_pct": r.unphysical_pct, "n_samples": r.n_samples} for r in res]
    df = pd.DataFrame(rows).sort_values(["hz", "method"], ascending=[False, True])
    (ROOT / "outputs/tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "outputs/tables/ngsim_validation.csv", index=False)
    with open(ROOT / "outputs/tables/ngsim_validation.md", "w", encoding="utf-8") as f:
        f.write("# NGSIM I-80 Ground-Truth Validation\n\n")
        f.write(f"{len(trajs)} real clean trajectories; ground truth = Savitzky-Golay "
                "differentiation of full-rate positions; detection noise σ=0.5 m. "
                "Lower MAE is better.\n\n")
        f.write(df.to_markdown(index=False)); f.write("\n")
    make_figure(df, ROOT / "outputs/figures")
    print(df.to_string(index=False))
    print("\nwrote outputs/tables/ngsim_validation.{csv,md} + "
          "outputs/figures/fig4_ngsim_ground_truth_validation.{png,pdf}")


if __name__ == "__main__":
    main()
