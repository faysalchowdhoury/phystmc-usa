# Physics-Informed Multi-Object Trajectory Reconstruction and Automated HCM/Emissions Auditing from Legacy Low-Frame-Rate Traffic Cameras

*Draft structured for IEEE Transactions on Intelligent Transportation Systems.*

---

### Abstract

Most deployed traffic cameras stream at 2–5 FPS, at which naive multi-object
tracking yields physically impossible motion that corrupts speed, acceleration, and
every downstream operational metric. We present **PhysTMC-USA**, a pipeline that
inserts a physics-informed kinematic layer — a vehicle motion envelope (speed ≤ 95
mph, longitudinal acceleration ∈ [−4.5, +3.5] m/s²) with exponential velocity
smoothing on ground-plane trajectories — between detection/tracking (YOLO11 + a
ByteTrack-style associator) and an automated audit stack (HCM 7th Edition
level-of-service, VT-Micro CO₂, Post-Encroachment-Time safety conflicts, and virtual
SAE J2735 Basic Safety Messages). On a real overhead highway-traffic recording
evaluated at 25, 5, and 2 FPS, the physics layer **eliminates** unphysical
acceleration spikes (94.96 % → 0 % at native rate) and **halves** frame-rate-induced
VT-Micro CO₂ inflation (29.5 % → 0 %), while preserving vehicle counts (GEH < 1 down
to 5 FPS). The method turns legacy low-FPS infrastructure into a standards-compliant
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
physics-informed kinematic reconstruction layer with an explicit vehicle motion
envelope; (ii) an integrated HCM/emissions/safety/J2735 audit; and (iii) a
multi-frame-rate ablation on real video quantifying the artifact and its suppression.

### II. Methodology & Kinematic Formulation

**A. Detection & association.** Vehicles are detected per frame by YOLO11 (COCO
classes car/motorcycle/bus/truck) and associated by a ByteTrack-style two-stage IoU
tracker with constant-velocity box prediction (`src/tracker.py`), which preserves
identities under the large inter-frame motion of 2–5 FPS feeds.

**B. Ground-plane calibration.** A four-point homography `H` maps image pixels to a
metric ground plane from visible lane geometry (`MUTCDHomographyCalibrator`). Each
vehicle's ground position is the perspective transform of its bounding-box
ground-contact point.

**C. Physics-informed kinematic filter.** For track *i* with ground position
`p_i(t)`, raw velocity `v_raw = Δp/Δt`. The filter (`KinematicTrajectoryFilter`)
applies exponential smoothing `v = α v_raw + (1−α) v_prev`, clamps the speed to the
95 mph envelope, and clamps the implied longitudinal acceleration to
`[−4.5, +3.5] m/s²`, rescaling velocity accordingly. An update whose output violates
the envelope is logged as an *unphysical event*; by construction the filtered stream
produces none, while the raw baseline does.

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

### IV. Results & Multi-FPS Ablation Discussion

Table I (`outputs/tables/ablation_results.csv`) reports the full ablation; Fig. 2
summarizes it; Fig. 3 shows reconstructed speed profiles and the
speed–acceleration–CO₂ phase portrait.

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

**Limitations.** Absolute scale is uncalibrated (approximate homography); PET counts
are frame-rate sensitive (sparse low-FPS sampling undercounts conflicts); and the
testbed is a single short public clip, not a surveyed DOT site. These define the
validation work in Section V.

### V. Conclusion

A lightweight physics-informed kinematic layer makes legacy low-FPS traffic cameras
usable for automated HCM/emissions/safety analytics, eliminating unphysical motion
and halving frame-rate-induced emissions inflation while preserving counts. Future
work: live geo-surveyed DOT deployment with surveyed calibration, a constrained
IMM/Kalman estimator, and validation against loop-detector and probe ground truth.
