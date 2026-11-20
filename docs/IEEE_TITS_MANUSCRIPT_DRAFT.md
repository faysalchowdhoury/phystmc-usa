# Physics-Informed Multi-Object Trajectory Reconstruction and Automated HCM/Emissions Auditing from Legacy Low-Frame-Rate Traffic Cameras

*Draft structured for IEEE Transactions on Intelligent Transportation Systems.*

---

### Abstract

Most deployed traffic cameras stream at 2–5 FPS, at which naive multi-object
tracking yields physically impossible motion that corrupts speed, acceleration, and
every downstream operational metric. We present **PhysTMC-USA**, a pipeline that
pairs a kinematic reconstruction layer with an automated audit stack (HCM 7th Edition
level-of-service, VT-Micro CO₂, Post-Encroachment-Time safety conflicts, and virtual
SAE J2735 Basic Safety Messages), and we contribute a **rigorous ground-truth
benchmark** of reconstruction methods for low-FPS cameras. Real NGSIM trajectories
from **two freeway sites (I-80, 418 tracks; US-101, 784 tracks)** are decimated to
10/5/2 Hz, corrupted with detection-localization noise, and scored against an
independent spline ground truth. Across both sites, every frame rate, and a noise
sweep (σ = 0.25–2 m), a **constant-acceleration Kalman RTS smoother** is the most
accurate and most noise-robust reconstruction, cutting acceleration error from **24.7
m/s² (naive differencing) to 0.94 m/s² at 10 Hz (≈26×)** and remaining near-flat as
noise grows; a constrained least-squares smoother is a competitive second and
Savitzky-Golay third. We integrate the validated reconstruction into the audit stack
and demonstrate it operationally on a real overhead highway recording (HCM LOS,
VT-Micro CO₂, 2,562 virtual J2735 BSMs). The study turns legacy low-FPS
infrastructure into a standards-compliant digital-twin sensor and gives a reproducible
method recommendation.

**Index terms —** traffic video analytics, trajectory reconstruction, Highway
Capacity Manual, vehicle emissions, SAE J2735, low-frame-rate cameras.

---

### I. Introduction

Departments of transportation operate large camera networks that, for bandwidth and
legacy reasons, deliver low-frame-rate imagery. Tracking-by-detection on such feeds
suffers from large inter-frame displacement: finite-difference velocity and
acceleration are dominated by localization noise, producing teleport-like motion.
Because microscopic emissions models (e.g., VT-Micro) are exponential in
acceleration, and surrogate safety and HCM metrics depend on speed/trajectory
fidelity, these artifacts propagate into every analytic. We contribute: (i) a
**multi-site, ground-truth benchmark** of five reconstruction methods for low-FPS
traffic cameras on real NGSIM trajectories, with an independent ground truth and a
noise sweep; (ii) a reproducible recommendation — a constant-acceleration **Kalman RTS
smoother** recovers true acceleration ≈26× more accurately than naive differencing at
10 Hz and is the most noise-robust at both sites; (iii) an integrated
HCM/emissions/safety/J2735 audit built on the validated reconstruction; and (iv) an
operational multi-frame-rate demonstration on real recorded video.

### II. Methodology & Kinematic Formulation

**A. Detection & association.** Vehicles are detected per frame by YOLO11 (COCO
classes car/motorcycle/bus/truck) and associated by a ByteTrack-style two-stage IoU
tracker with constant-velocity box prediction (`src/tracker.py`), which preserves
identities under the large inter-frame motion of 2–5 FPS feeds.

**B. Ground-plane calibration.** A four-point homography `H` maps image pixels to a
metric ground plane from visible lane geometry (`MUTCDHomographyCalibrator`). Each
vehicle's ground position is the perspective transform of its bounding-box
ground-contact point.

**C. Kinematic reconstruction (five methods compared).** From ground positions we
reconstruct per-track speed/acceleration by: (i) *Raw* finite differencing
(baseline); (ii) *Savitzky-Golay* differentiation (standard baseline); (iii)
*Kalman RTS* — a constant-acceleration Kalman filter with Rauch-Tung-Striebel
smoothing, position measurements, jerk process noise; (iv) *EMA+clamp*
(`KinematicTrajectoryFilter`): exponential velocity smoothing with the 95 mph /
[−4.5, +3.5] m/s² envelope, for streaming use; and (v) *Constrained smoother*: solve
`min_p ‖p − y‖² + λ‖D²p‖²` (`D²` the second-difference operator), then differentiate
and clamp. All are in `src/validation.py`. §IV-A benchmarks them against ground truth
and selects the Kalman RTS smoother for the audit stack.

**D. Operational audit.** VT-Micro gives instantaneous CO₂ `g/s` from speed and
acceleration for light- and heavy-duty classes (`vt_micro_co2_gps`); HCM 7th Edition
control delay `d = d₁ + d₂`, v/c, PHF, and LOS A–F are computed from measured
gate-crossing flow (`hcm7_audit.py`); PET < 2 s surrogate conflicts are counted from
ground-cell co-occupancy (`detect_pet_conflicts`); and every sample is encoded as a
virtual SAE J2735 BSM (MsgID 20) with core mobility fields (`v2x_bsm.py`).

### III. Real-World Experimental Setup

The testbed is a real, publicly distributed overhead highway-traffic recording
(Roboflow `supervision` example clip): a fixed, elevated, oblique view of a divided
multi-lane highway, 3840×2160 at 25 FPS, 538 frames (21.5 s). **Honest scope:** this
is not a live U.S. DOT feed — live DOT video was not capturable in our environment
(CHART video endpoints 404; DOT snapshots refresh ~0.1 FPS). The metric scale uses an
assumed standard lane width, so absolute speeds/emissions are approximate; the
frame-rate ablation claims are scale-invariant. Detections are computed once at
native rate (YOLO11s, imgsz 1920) and cached; each configuration (Baseline vs
PhysTMC × 25/5/2 FPS) reuses them, so comparisons differ only in sampling rate and
the physics layer. See Fig. 1 for calibration and reconstructed trajectories.

### IV. Results

#### A. Multi-site ground-truth benchmark (primary result)

We decimate real NGSIM trajectories from two freeway sites (I-80, 418 tracks, 318 k
samples; US-101, 784 tracks, 241 k samples) to 10/5/2 Hz, add σ = 0.5 m detection
noise, and score each reconstruction against an independent quartic-spline ground
truth (MAE; Table II, Fig. 4; `outputs/tables/ngsim_validation.csv`).

**Table II — Reconstruction error at 10 Hz (MAE; lower is better).**

| Site | Method | Speed MAE (m/s) | Accel MAE (m/s²) | Unphysical % |
|---|---|--:|--:|--:|
| I-80 | Raw finite-diff | 2.88 | 24.66 | 91.1 |
| I-80 | Savitzky-Golay | 0.63 | 4.93 | 59.6 |
| I-80 | EMA + clamp | 2.69 | 3.97 | 40.0 |
| I-80 | Constrained smoother (ours) | 0.58 | 2.26 | 18.0 |
| I-80 | **Kalman RTS** | **0.33** | **0.94** | 15.0 |
| US-101 | Raw finite-diff | 2.84 | 24.49 | 91.0 |
| US-101 | Savitzky-Golay | 0.59 | 4.61 | 54.7 |
| US-101 | Constrained smoother (ours) | 0.54 | 2.06 | 11.2 |
| US-101 | **Kalman RTS** | **0.26** | **0.54** | 0.2 |

The ranking is **identical at both independent sites**: Kalman RTS < constrained
smoother < Savitzky-Golay < EMA+clamp < raw on accuracy. *Acceleration* — which drives
microscopic emissions and surrogate safety — is where naive differencing fails worst:
its 10 Hz accel MAE (≈24.5 m/s²) is **≈26× the Kalman smoother's (0.94)**, because
small localization error divided by a small Δt explodes. **The standard Kalman RTS
smoother is the best method**; our constrained smoother is a competitive second
(beating Savitzky-Golay and raw at every rate) but does **not** surpass Kalman — a
result we report as measured rather than claim a novel winner. A noise sweep
(σ = 0.25–2 m, Fig. 5) confirms robustness: Kalman's 10 Hz accel MAE rises only
0.86 → 1.48, while raw explodes 12.7 → 97.8. **Honest nuance:** at 2 Hz, raw speed is
competitive again (sparse sampling self-averages noise), so the smoothers' advantage
concentrates in acceleration and at higher frame rates. This is a ground-truth,
non-tautological result — it measures recovery of *true* motion, not envelope
compliance.

#### B. Operational demonstration (multi-FPS ablation on recorded video)

Table I (`outputs/tables/ablation_results.csv`) reports the full operational ablation
on the recorded highway clip; Fig. 2 summarizes it; Fig. 3 shows reconstructed speed
profiles and the speed–acceleration–CO₂ phase portrait.

**Table I — Multi-FPS ablation (real clip, reference = PhysTMC @ 25 FPS).**

| Config | Tracker | FPS | Unique tracks | Count | MAPE% | GEH | Mean mph | Unphys.% | CO₂ g/s | CO₂ drift% | HCM delay (LOS) |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Baseline@25 | raw | 25 | 15 | 8 | 0 | 0 | 35.1 | 94.96 | 1.702 | 29.5 | 26.6 (C) |
| **PhysTMC@25** | phys | 25 | 15 | 8 | 0 | 0 | 15.1 | **0** | 1.315 | 0 | 26.6 (C) |
| Baseline@5 | raw | 5 | 13 | 8 | 0 | 0 | 26.0 | 47.86 | 1.538 | 16.9 | 26.6 (C) |
| **PhysTMC@5** | phys | 5 | 13 | 8 | 0 | 0 | 16.8 | **0** | 1.399 | 6.4 | 26.6 (C) |
| Baseline@2 | raw | 2 | 8 | 6 | 25 | 0.76 | 21.4 | 21.26 | 1.532 | 16.5 | 20.7 (C) |
| **PhysTMC@2** | phys | 2 | 8 | 6 | 25 | 0.76 | 18.3 | **0** | 1.477 | 12.3 | 20.7 (C) |

**Unphysical motion.** The raw baseline flags 94.96 % of updates as unphysical at 25
FPS, falling to 21.26 % at 2 FPS (longer Δt averages localization noise). The physics
layer removes all such events at every rate — the central reliability claim.

**Emissions fidelity.** Because VT-Micro is exponential in acceleration, baseline
noise inflates mean CO₂ by up to 29.5 % relative to the physics-filtered native
reference; the filter limits inflation to ≤ 12.3 % across rates.

**Count integrity.** Vehicle counts are unchanged by the physics layer (it corrects
kinematics, not detection) and remain GEH-acceptable (GEH 0.76) at 2 FPS, where two
of eight through-vehicles are missed (MAPE 25 %).

**Speed stability.** Baseline mean speed swings with frame rate (35→21 mph) as noise
inflates apparent displacement; the filtered estimate is stable (15–18 mph).

**Limitations.** Reconstruction is validated on two real NGSIM freeway sites with an
independent ground truth and a noise sweep (§IV-A), but both sites are U.S. freeways
(no arterial/urban), detection noise is injected rather than produced by an end-to-end
tracker, and ground truth is a spline on dense clean positions (not an instrumented
probe vehicle). The *operational demonstration* (§IV-B) runs on a different,
uncalibrated recorded clip, so its absolute speeds/emissions are approximate; PET
counts are frame-rate sensitive. A single-platform, geo-surveyed deployment unifying
validation and live auditing is the next step.

### V. Conclusion

A multi-site, ground-truth benchmark establishes that smoothing-based reconstruction —
best realized by a constant-acceleration **Kalman RTS smoother** — makes legacy
low-FPS traffic cameras usable for automated HCM/emissions/safety analytics,
recovering true acceleration ≈26× more accurately than naive differencing at 10 Hz
and remaining robust across a noise sweep and two independent freeway sites. The
proposed constrained smoother is competitive but does not surpass Kalman, which we
report honestly. Built on the validated reconstruction, an operational demonstration
produces HCM LOS, VT-Micro CO₂, and virtual J2735 BSMs on real recorded video. Future
work: a single-platform, geo-surveyed DOT deployment unifying validation and live
auditing; richer non-freeway sites; and real (not injected) detection noise from an
end-to-end tracker.
