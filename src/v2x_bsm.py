"""Encode reconstructed trajectories into virtual SAE J2735 Basic Safety Messages
(BSM, MsgID 20) and export to CSV + JSONL.

Each sample becomes one BSM core-data record. Ground (x, y) metres are mapped to
synthetic WGS84 lat/lon about a configurable camera anchor so the stream is
format-faithful to J2735 even though the testbed clip is not geo-surveyed.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict

M_PER_DEG_LAT = 111_320.0

BSM_FIELDS = ["timestamp_s", "temp_id", "vehicle_class", "lat", "lon",
              "local_x_ft", "local_y_ft", "speed_mph", "lon_accel_mps2",
              "heading_deg", "entry_leg", "exit_leg"]


@dataclass
class BSM:
    timestamp_s: float
    temp_id: int
    vehicle_class: str
    lat: float
    lon: float
    local_x_ft: float
    local_y_ft: float
    speed_mph: float
    lon_accel_mps2: float
    heading_deg: float
    entry_leg: str
    exit_leg: str
    msg_id: int = 20          # SAE J2735 BSM MsgID


class BSMEncoder:
    def __init__(self, anchor_lat: float = 39.0, anchor_lon: float = -76.9):
        self.anchor_lat = anchor_lat
        self.anchor_lon = anchor_lon
        self.records: list[BSM] = []

    def _latlon(self, x_m: float, y_m: float) -> tuple[float, float]:
        lat = self.anchor_lat + (y_m / M_PER_DEG_LAT)
        lon = self.anchor_lon + (x_m / (M_PER_DEG_LAT *
              max(0.1, abs(__import__("math").cos(__import__("math").radians(self.anchor_lat))))))
        return round(lat, 7), round(lon, 7)

    def add(self, t: float, track_id: int, vclass: str, x_m: float, y_m: float,
            speed_mph: float, accel: float, heading: float,
            entry_leg: str = "", exit_leg: str = "") -> None:
        lat, lon = self._latlon(x_m, y_m)
        self.records.append(BSM(
            timestamp_s=round(t, 3), temp_id=int(track_id), vehicle_class=vclass,
            lat=lat, lon=lon, local_x_ft=round(x_m / 0.3048, 2),
            local_y_ft=round(y_m / 0.3048, 2), speed_mph=round(speed_mph, 2),
            lon_accel_mps2=round(accel, 3), heading_deg=round(heading, 1),
            entry_leg=entry_leg, exit_leg=exit_leg))

    def __len__(self) -> int:
        return len(self.records)

    def to_csv(self, path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=BSM_FIELDS)
            w.writeheader()
            for r in self.records:
                w.writerow({k: getattr(r, k) for k in BSM_FIELDS})

    def to_jsonl(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(asdict(r), default=float) + "\n")
