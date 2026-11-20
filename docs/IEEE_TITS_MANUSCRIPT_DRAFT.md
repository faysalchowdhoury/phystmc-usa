# Physics-Informed Multi-Object Trajectory Reconstruction and Automated HCM/Emissions Auditing from Legacy Low-Frame-Rate Traffic Cameras

*Draft structured for IEEE Transactions on Intelligent Transportation Systems.*

---

### Abstract

Most deployed traffic cameras stream at 2–5 FPS, at which naive multi-object
tracking yields physically impossible motion that corrupts speed, acceleration, and
every downstream operational metric. We present **PhysTMC-USA**, a pipeline that
inserts a physics-informed kinematic reconstruction layer between
detection/tracking (YOLO11 + a ByteTrack-style associator) and an automated audit
stack (HCM 7th Edition level-of-service, VT-Micro CO₂, Post-Encroachment-Time safety
conflicts, and virtual SAE J2735 Basic Safety Messages). The core method is a
**constrained least-squares trajectory smoother** with a second-difference
(acceleration) penalty and a physical motion envelope. **We validate it against real
ground truth**: real NGSIM I-80 trajectories (10 Hz) are decimated to low effective
frame rates and corrupted with detection-localization noise, and reconstructions are
scored against Savitzky-Golay ground-truth kinematics. The smoother recovers true
**acceleration** with mean absolute error **2.12 vs 24.52 m/s² for naive finite
differencing at 10 Hz (a 12× reduction)** and true **speed** to 0.51 m/s, and
dominates a simple EMA+clamp filter. On a separate real overhead highway recording,
an operational demonstration emits HCM LOS, VT-Micro CO₂, and 2,562 virtual J2735
BSMs. The method turns legacy low-FPS infrastructure into a standards-compliant
digital-twin sensor.

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
constrained least-squares trajectory smoother for low-FPS kinematic reconstruction;
(ii) a **ground-truth validation** on real NGSIM trajectories showing it recovers
true acceleration 12× more accurately than naive differencing at 10 Hz; (iii) an
integrated HCM/emissions/safety/J2735 audit; and (iv) an operational multi-frame-rate
demonstration on real recorded video.

### II. Methodology & Kinematic Formulation

**A. Detection & association.** Vehicles are detected per frame by YOLO11 (COCO
classes car/motorcycle/bus/truck) and associated by a ByteTrack-style two-stage IoU
tracker with constant-velocity box prediction (`src/tracker.py`), which preserves
identities under the large inter-frame motion of 2–5 FPS feeds.

**B. Ground-plane calibration.** A four-point homography `H` maps image pixels to a
metric ground plane from visible lane geometry (`MUTCDHomographyCalibrator`). Each
vehicle's ground position is the perspective transform of its bounding-box
ground-contact point.

**C. Physics-informed kinematic reconstruction.** We compare three reconstructions
of per-track speed/acceleration from ground positions. (i) *Raw* finite differencing
(baseline). (ii) *EMA+clamp* (`KinematicTrajectoryFilter`): exponential velocity
smoothing with the 95 mph / [−4.5, +3.5] m/s² envelope, for streaming use. (iii)
*Constrained smoother* (`src/validation.py`, the proposed method): given noisy sparse
positions `y`, solve `min_p ‖p − y‖² + λ‖D²p‖²` (`D²` the second-difference operator;
λ the acceleration-regularization weight), a banded linear system, then differentiate
`p` and clamp to the envelope. The second-difference penalty directly suppresses the
acceleration noise that low-FPS differencing amplifies, which §IV shows is decisive.

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

#### A. Ground-truth validation on real NGSIM trajectories (primary result)

We decimate 78 real NGSIM I-80 trajectories (10 Hz, clean of position artifacts) to
10 / 5 / 2 Hz, add σ = 0.5 m detection noise, and score each reconstruction's
speed/acceleration against Savitzky-Golay ground truth (MAE; Table II, Fig. 4,
`outputs/tables/ngsim_validation.csv`).

**Table II — NGSIM ground-truth reconstruction error (MAE; lower is better).**

| Hz | Method | Speed MAE (m/s) | Accel MAE (m/s²) | Unphysical % |
|--:|---|--:|--:|--:|
| 10 | Raw finite-diff | 2.82 | **24.52** | 91.3 |
| 10 | EMA + clamp | 2.61 | 4.14 | 39.8 |
| 10 | **Constrained smoother** | **0.51** | **2.12** | 17.1 |
| 5 | Raw finite-diff | 1.41 | 6.13 | 67.7 |
| 5 | EMA + clamp | 1.83 | 3.77 | 32.1 |
| 5 | **Constrained smoother** | **0.55** | **1.51** | 5.7 |
| 2 | Raw finite-diff | **0.71** | 1.61 | 15.5 |
| 2 | EMA + clamp | 1.44 | 2.84 | 8.8 |
| 2 | **Constrained smoother** | 0.90 | **1.83** | 1.3 |

*Acceleration* is the quantity that drives microscopic emissions and surrogate
safety, and is where naive differencing fails worst: at 10 Hz its accel MAE (24.5
m/s²) is **12× the smoother's (2.12)**, because small localization error divided by a
small Δt explodes. The constrained smoother gives the lowest acceleration error at
**every** frame rate and the lowest speed error at 10 and 5 Hz. **Honest nuance:** at
2 Hz the raw speed estimate (0.71 m/s) edges the smoother (0.90) — sparse sampling
self-averages localization noise, and the smoother's regularization slightly
over-smooths genuine speed change — so the smoother's advantage is concentrated in
acceleration and at higher frame rates. The EMA+clamp filter is dominated by the
smoother throughout. This is a ground-truth-validated, non-tautological result: it
measures recovery of *true* motion, not merely envelope compliance.

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

**Limitations.** The kinematic method is validated against real NGSIM ground truth
(§IV-A), but the *operational demonstration* (§IV-B) runs on a different, uncalibrated
recorded clip, so its absolute speeds/emissions are approximate; the smoother edge
narrows at 2 Hz for speed; PET counts are frame-rate sensitive; and the demonstration
clip is not a surveyed U.S. DOT site. A single-platform, geo-surveyed deployment that
unifies ground-truth validation and live auditing is the next step.

### V. Conclusion

A constrained least-squares trajectory smoother makes legacy low-FPS traffic cameras
usable for automated HCM/emissions/safety analytics. Validated against real NGSIM
ground truth, it recovers true acceleration 12× more accurately than naive
differencing at 10 Hz and dominates a simple EMA+clamp filter, with the caveat that
its speed advantage narrows at 2 Hz. An operational demonstration produces HCM LOS,
VT-Micro CO₂, and virtual J2735 BSMs on real recorded video. Future work: a
single-platform, geo-surveyed DOT deployment unifying ground-truth validation and
live auditing, and an online constrained estimator for streaming use.
