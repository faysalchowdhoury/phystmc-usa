"""Highway Capacity Manual (HCM 7th Edition) operational audit.

Implements signalized control delay d = d1 + d2, volume-to-capacity ratio (v/c),
Peak Hour Factor (PHF), and Level of Service (LOS A-F). Flows are MEASURED from the
reconstructed trajectories (gate crossings); signal timing parameters are explicit
inputs (the testbed is an uninterrupted-flow highway, so the signalized-delay audit
is applied with stated assumed timing and reported transparently).
"""
from __future__ import annotations

from dataclasses import dataclass


# HCM signalized LOS thresholds by control delay (s/veh)
LOS_DELAY_BOUNDS = [(10, "A"), (20, "B"), (35, "C"), (55, "D"), (80, "E")]


def level_of_service(control_delay_s: float) -> str:
    for bound, los in LOS_DELAY_BOUNDS:
        if control_delay_s <= bound:
            return los
    return "F"


def peak_hour_factor(counts_per_subinterval: list[int]) -> float:
    """PHF = V / (4 * V15), using the observed sub-interval with the peak count.
    For a short clip the sub-intervals are the clip quarters (reported as a
    short-horizon PHF proxy)."""
    if not counts_per_subinterval:
        return 0.0
    V = sum(counts_per_subinterval)
    peak = max(counts_per_subinterval)
    n = len(counts_per_subinterval)
    if peak == 0:
        return 0.0
    return float(V / (n * peak))


@dataclass
class HCMResult:
    flow_veh_h: float
    capacity_veh_h: float
    v_over_c: float
    phf: float
    uniform_delay_d1: float
    incremental_delay_d2: float
    control_delay: float
    los: str


def hcm7_signalized_delay(flow_veh_h: float, capacity_veh_h: float,
                          cycle_s: float = 90.0, green_ratio: float = 0.45,
                          phf: float = 0.9, analysis_period_h: float = 0.25,
                          k_incr: float = 0.5, I_upstream: float = 1.0) -> HCMResult:
    """HCM 7th Ed. control delay for a signalized movement.

    d1 = 0.5 C (1 - g/C)^2 / (1 - min(1,X) g/C)
    d2 = 900 T [ (X-1) + sqrt((X-1)^2 + 8 k I X / (c T)) ]
    d  = d1 * PF + d2        (PF taken as 1.0 for isolated operation)
    """
    gC = green_ratio
    c_cap = max(capacity_veh_h * gC, 1e-6)          # movement capacity
    X = flow_veh_h / c_cap                          # v/c (degree of saturation)
    d1 = (0.5 * cycle_s * (1 - gC) ** 2) / (1 - min(1.0, X) * gC)
    T = analysis_period_h
    d2 = 900 * T * ((X - 1) + ((X - 1) ** 2 + (8 * k_incr * I_upstream * X) / (c_cap * T)) ** 0.5)
    d2 = max(d2, 0.0)
    d = d1 + d2
    return HCMResult(flow_veh_h=flow_veh_h, capacity_veh_h=c_cap, v_over_c=X,
                     phf=phf, uniform_delay_d1=d1, incremental_delay_d2=d2,
                     control_delay=d, los=level_of_service(d))
