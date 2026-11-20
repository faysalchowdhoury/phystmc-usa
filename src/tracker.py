"""A compact ByteTrack-style multi-object tracker (two-stage high/low-confidence
IoU association with constant-velocity box prediction).

Implemented in-repo because `supervision>=0.32` removed its bundled `ByteTrack`.
It reproduces ByteTrack's core ideas -- associate confident detections first, then
recover low-confidence ones against surviving tracks, with Kalman-free velocity
prediction so association survives the large inter-frame motion of 2-5 FPS feeds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a[N,4] and b[M,4] (x1,y1,x2,y2)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = area_a[:, None] + area_b[None, :] - inter + 1e-6
    return (inter / union).astype(np.float32)


@dataclass
class Track:
    tid: int
    box: np.ndarray
    vel: np.ndarray             # center velocity (px/frame)
    cls: int
    hits: int = 1
    age: int = 0               # frames since last match
    confirmed: bool = False

    def predict(self) -> np.ndarray:
        cx = (self.box[0] + self.box[2]) / 2 + self.vel[0]
        cy = (self.box[1] + self.box[3]) / 2 + self.vel[1]
        w = self.box[2] - self.box[0]; h = self.box[3] - self.box[1]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], np.float32)


@dataclass
class ByteStyleTracker:
    high_thr: float = 0.5
    low_thr: float = 0.1
    iou_thr: float = 0.2
    max_age: int = 30
    min_hits: int = 2
    _next: int = 1
    tracks: list = field(default_factory=list)

    def _match(self, tracks, boxes, idxs):
        """Greedy IoU match between given tracks and detection indices."""
        if not tracks or len(idxs) == 0:
            return [], list(range(len(tracks))), list(idxs)
        pred = np.array([t.predict() for t in tracks], np.float32)
        M = iou(pred, boxes[idxs])
        matches, ut, ud = [], set(range(len(tracks))), set(range(len(idxs)))
        order = np.dstack(np.unravel_index(np.argsort(-M, axis=None), M.shape))[0]
        for ti, di in order:
            if M[ti, di] < self.iou_thr:
                break
            if ti in ut and di in ud:
                matches.append((ti, idxs[di])); ut.discard(ti); ud.discard(di)
        return matches, list(ut), [idxs[d] for d in ud]

    def update(self, boxes: np.ndarray, confs: np.ndarray, classes: np.ndarray):
        """Advance one frame. Returns list of (tid, box, cls) for confirmed tracks."""
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        confs = np.asarray(confs, np.float32).reshape(-1)
        classes = np.asarray(classes, np.int64).reshape(-1)
        hi = np.where(confs >= self.high_thr)[0]
        lo = np.where((confs >= self.low_thr) & (confs < self.high_thr))[0]

        for t in self.tracks:
            t.age += 1
        # stage 1: confident detections vs all tracks
        m1, ut1, ud_hi = self._match(self.tracks, boxes, list(hi))
        for ti, di in m1:
            self._apply(self.tracks[ti], boxes[di], int(classes[di]))
        # stage 2: low-confidence detections vs still-unmatched tracks
        rem = [self.tracks[i] for i in ut1]
        m2, ut2, _ = self._match(rem, boxes, list(lo))
        for ti, di in m2:
            self._apply(rem[ti], boxes[di], int(classes[di]))
        # new tracks from unmatched confident detections
        for di in ud_hi:
            self.tracks.append(Track(self._next, boxes[di].copy(),
                                     np.zeros(2, np.float32), int(classes[di])))
            self._next += 1
        # cull stale tracks
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]

        out = []
        for t in self.tracks:
            if t.hits >= self.min_hits or t.confirmed:
                t.confirmed = True
                if t.age == 0:
                    out.append((t.tid, t.box.copy(), t.cls))
        return out

    def _apply(self, t: Track, box: np.ndarray, cls: int):
        new_c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], np.float32)
        old_c = np.array([(t.box[0] + t.box[2]) / 2, (t.box[1] + t.box[3]) / 2], np.float32)
        t.vel = 0.5 * t.vel + 0.5 * (new_c - old_c)
        t.box = box.astype(np.float32); t.cls = cls
        t.hits += 1; t.age = 0
