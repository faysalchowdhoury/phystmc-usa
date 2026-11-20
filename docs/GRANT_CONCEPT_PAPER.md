# PhysTMC-USA — Concept Paper

## Physics-Informed Multi-Object Trajectory Reconstruction and Automated HCM/Emissions Auditing from Legacy Low-Frame-Rate Traffic Cameras

**Target programs:** USDOT SMART Grants; FHWA Safe Streets and Roads for All (SS4A);
NSF Cyber-Physical Systems (CPS).

> **Transparency statement (read first).** The results below are **preliminary
> results on a real, publicly distributed overhead highway-traffic recording**
> (25 FPS, 4K, 538 frames / 21.5 s; Roboflow `supervision` example clip — an
> elevated, oblique, DOT-camera-style view of a divided multi-lane highway). It is
> **not** a live U.S. DOT field deployment: live DOT video could not be captured in
> our build environment (MDOT CHART video endpoints return HTTP 404; public DOT
> still-image snapshots refresh at ~0.1 FPS, far too slow for trajectory
> reconstruction). The metric scale rests on an assumed standard lane width, so
> **absolute** speeds/emissions are approximate; the **frame-rate robustness**
> findings that motivate this proposal are scale-invariant. The proposed work would
> deploy and validate the same pipeline on live, geo-surveyed DOT feeds.

### 1. Problem & Motivation

State and local DOTs operate tens of thousands of traffic cameras, but the vast
majority stream at **2–5 FPS** (or slower snapshots) due to bandwidth and legacy
hardware. Naive object tracking on such feeds produces **physically impossible
motion** — teleport-like displacements that inflate speed, acceleration, and
model-based emissions, and corrupt safety and capacity metrics. Agencies therefore
cannot reuse this existing, sunk-cost sensor network for automated operational
analysis (HCM level-of-service, emissions, surrogate safety), and instead pay for
new instrumentation.

### 2. Innovation

PhysTMC-USA inserts a **physics-informed kinematic layer** between detection/tracking
(YOLO11 + a ByteTrack-style associator) and downstream analytics. It enforces a
vehicle motion envelope — speed ≤ 95 mph, longitudinal acceleration ∈ [−4.5, +3.5]
m/s², exponential velocity smoothing — on each track's ground-plane trajectory after
a MUTCD-style homography. The reconstructed trajectories feed an automated audit
stack: **HCM 7th Edition** control delay / v/c / LOS, **VT-Micro** CO₂, **Post-
Encroachment-Time** surrogate safety conflicts, and **virtual SAE J2735 BSM**
telemetry — turning a legacy low-FPS camera into a digital-twin sensor.

### 3. Preliminary Results (measured)

**Multi-site ground-truth benchmark (real NGSIM).** Decimating real 10-Hz
trajectories from two freeway sites (I-80, 418 tracks; US-101, 784 tracks) to low
frame rates (+detection noise) and scoring against independent spline ground truth, a
**Kalman RTS smoother** recovers true **acceleration** with MAE **0.94 (I-80) / 0.54
(US-101) vs ≈24.5 m/s² for naive differencing at 10 Hz — a ≈26–45× reduction** — with
an identical method ranking at both sites and robustness across a σ=0.25–2 m noise
sweep. This is a ground-truth-validated, non-tautological result (recovery of *true*
motion, not envelope compliance).

**Operational demonstration (real recorded highway video).** Across 25 / 5 / 2 FPS:

| Metric (25 / 5 / 2 FPS) | Baseline raw tracker | **PhysTMC-USA** |
|---|---|---|
| Unphysical acceleration spikes (\|a\|>4.5 m/s²) | 94.96 / 47.86 / 21.26 % | **0 / 0 / 0 %** |
| VT-Micro CO₂ inflation vs native | +29.5 / +16.9 / +16.5 % | **0 / +6.4 / +12.3 %** |
| Mean speed (mph) — stability across FPS | 35.1 / 26.0 / 21.4 (swings) | **15.1 / 16.8 / 18.3 (stable)** |
| Directional count MAPE vs native | 0 / 0 / 25 % | 0 / 0 / 25 % |
| GEH vs native | 0 / 0 / 0.76 | 0 / 0 / 0.76 (GEH ≪ 5) |

At native rate the pipeline tracked **15 vehicles**, measured an **HCM LOS C**
(control delay 26.6 s, v/c 0.83, PHF 0.67), and emitted **2,562 virtual J2735 BSM
records**. The physics layer **eliminates** unphysical acceleration spikes at every
frame rate and **halves** the frame-rate-induced CO₂ inflation, while leaving vehicle
counts unchanged (the filter corrects kinematics, not detections). Counts remain
GEH-acceptable down to 2 FPS, where they degrade gracefully (−25 %).

### 4. Specific Aims

- **Aim 1 — Live DOT integration & calibration.** Deploy on geo-surveyed MDOT/
  NYSDOT feeds; replace the assumed-lane-width scale with surveyed MUTCD control
  points for validated absolute speeds/emissions.
- **Aim 2 — Physics-informed reconstruction at scale.** Extend the kinematic filter
  with a constrained Kalman/IMM estimator and quantify accuracy against
  probe/loop-detector ground truth across FPS.
- **Aim 3 — Automated operational & equity auditing.** Produce continuous HCM LOS,
  VT-Micro emissions, and PET safety-conflict surveillance, with J2735 BSM export
  for V2X and digital-twin consumers.

### 5. Broader Impact

Reusing the existing low-FPS camera fleet avoids new-sensor capital cost, extends
automated HCM/emissions/safety analytics to under-resourced agencies, and produces
standards-compliant (SAE J2735) telemetry for connected-vehicle and SS4A safety
programs.

*Reproduce:* `python scripts/fetch_usdot_feed.py && python
scripts/run_ablation_suite.py && python scripts/make_figures.py`. Every number above
is read from `outputs/tables/ablation_results.csv` and `data/raw/camera_metadata.json`.
