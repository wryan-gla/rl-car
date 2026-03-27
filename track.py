"""
track.py — Track geometry and presets.

Supports both:
- `oval`: analytic ellipse centreline.
- `monaco`: Monaco-inspired closed street-style polyline.

Walls are built by offsetting the centreline along per-point unit normals
estimated from local tangent vectors, which works for non-convex tracks.
"""

from __future__ import annotations

import numpy as np

TRACK_NAMES = ("oval", "monaco", "spa")


class BaseTrack:
    def __init__(self, centreline: np.ndarray, half_width: float = 45.0, name: str = "custom"):
        cl = np.asarray(centreline, dtype=np.float64)
        if cl.ndim != 2 or cl.shape[1] != 2:
            raise ValueError("centreline must have shape (N, 2)")
        if cl.shape[0] < 8:
            raise ValueError("centreline must contain at least 8 points")

        self.name = name
        self.half_width = float(half_width)
        self.centreline = cl
        self.n_points = int(cl.shape[0])

        self.normals = _estimate_normals_closed(self.centreline)
        self.inner_wall = self.centreline - self.half_width * self.normals
        self.outer_wall = self.centreline + self.half_width * self.normals
        self.wall_segs = self._build_seg_array()

        p0 = self.centreline
        p1 = np.roll(self.centreline, -1, axis=0)
        seg = p1 - p0
        self._seg = seg
        self._seg_len = np.linalg.norm(seg, axis=1)
        self._cum_len = np.concatenate([[0.0], np.cumsum(self._seg_len)])
        self.track_length = float(self._cum_len[-1])

        self.start_pos = self.centreline[0].copy()
        d = self.centreline[1] - self.centreline[0]
        self.start_angle = float(np.arctan2(d[1], d[0]))

    def _build_seg_array(self) -> np.ndarray:
        n = self.n_points
        segs = []
        src_idxs = []
        for wall in (self.inner_wall, self.outer_wall):
            for i in range(n):
                segs.append([wall[i], wall[(i + 1) % n]])
                src_idxs.append(i)
        self.seg_cl_idxs: np.ndarray = np.array(src_idxs, dtype=np.int32)
        return np.array(segs, dtype=np.float64)

    def get_progress(self, position: np.ndarray) -> float:
        """
        Continuous lap progress ∈ [0, 1) from projection on nearest centreline segment.
        """
        pos = np.asarray(position, dtype=np.float64)
        p0 = self.centreline
        p1 = np.roll(self.centreline, -1, axis=0)
        seg = p1 - p0
        seg_len_sq = np.maximum((seg * seg).sum(axis=1), 1e-10)

        rel = pos[None, :] - p0
        t = np.clip((rel * seg).sum(axis=1) / seg_len_sq, 0.0, 1.0)
        proj = p0 + t[:, None] * seg
        dist_sq = ((proj - pos[None, :]) ** 2).sum(axis=1)

        idx = int(np.argmin(dist_sq))
        return float((idx + t[idx]) / self.n_points)

    def point_and_heading_at_progress(self, progress: float) -> tuple[np.ndarray, float]:
        """
        Centreline point and tangent heading at lap progress in [0, 1).
        """
        if self.track_length <= 1e-9:
            raise ValueError("track length must be > 0")

        prog = float(progress) % 1.0
        target_len = prog * self.track_length

        idx = int(np.searchsorted(self._cum_len, target_len, side="right") - 1)
        idx = int(np.clip(idx, 0, self.n_points - 1))

        seg_len = max(float(self._seg_len[idx]), 1e-10)
        frac = (target_len - float(self._cum_len[idx])) / seg_len

        point = self.centreline[idx] + frac * self._seg[idx]
        heading = float(np.arctan2(self._seg[idx, 1], self._seg[idx, 0]))
        return point.astype(np.float64), heading


class OvalTrack(BaseTrack):
    def __init__(
        self,
        cx: float = 400,
        cy: float = 300,
        rx: float = 260,
        ry: float = 160,
        half_width: float = 45,
        n_points: int = 120,
    ):
        t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
        centreline = np.column_stack([cx + rx * np.cos(t), cy + ry * np.sin(t)])
        super().__init__(centreline=centreline, half_width=half_width, name="oval")


class MonacoTrack(BaseTrack):
    """Monaco-inspired layout (not official CAD-accurate)."""

    def __init__(self, half_width: float = 12.0, n_points: int = 240):
        anchors = 3*np.array(
            [
                [ 95,  73],
       [107,  63],
       [118,  54],
       [126,  47],
       [134,  40],
       [150,  33],
       [161,  38],
       [171,  45],
       [182,  51],
       [193,  58],
       [206,  68],
       [214,  74],
       [223,  79],
       [233,  86],
       [242,  89],
       [252,  96],
       [268, 104],
       [279, 111],
       [294, 117],
       [307, 114],
       [313, 104],
       [314,  92],
       [316,  79],
       [326,  72],
       [360,  66],
       [386,  62],
       [392,  65],
       [386,  74],
       [377,  80],
       [371,  91],
       [371, 108],
       [386, 101],
       [394,  86],
       [401,  90],
       [410,  97],
       [406, 104],
       [397, 115],
       [374, 124],
       [356, 134],
       [331, 140],
       [286, 137],
       [254, 117],
       [235, 105],
       [213, 114],
       [193, 103],
       [192,  84],
       [170,  65],
       [139,  68],
       [124,  84],
       [123,  98],
       [111, 114],
       [101, 125],
       [ 85, 130],
       [ 79, 142],
       [ 81, 151],
       [ 87, 173],
       [ 70, 181],
       [ 45, 155],
       [ 57, 126],
       [ 69, 110],
       [ 78,  82]
            ],
            dtype=np.float64,
        )

        centreline = _resample_closed_polyline(anchors, n_points)
        super().__init__(centreline=centreline, half_width=half_width, name="monaco")


class BrazilTrack(BaseTrack):
    """Brazil-inspired layout (not official CAD-accurate)."""

    def __init__(self, half_width: float = 12.0, n_points: int = 240):
        anchors = 5*np.array(
            [88, 24,
                71, 30,
                51, 35,
                34, 41,
                24, 46,
                21, 50,
                22, 56,
                30, 61,
                33, 67,
                29, 80,
                24, 92,
                28, 106,
                34, 119,
                46, 125,
                62, 128,
                74, 131,
                89, 135,
                112, 140,
                122, 144,
                143, 150,
                158, 153,
                179, 158,
                190, 159,
                195, 154,
                198, 138,
                199, 121,
                192, 111,
                181, 102,
                165, 90,
                154, 81,
                139, 69,
                129, 60,
                122, 52,
                124, 42,
                129, 32,
                139, 30,
                153, 25,
                167, 23,
                170, 32,
                163, 40,
                163, 53,
                170, 55,
                181, 50,
                191, 41,
                201, 34,
                217, 32,
                221, 37,
                212, 48,
                202, 57,
                195, 72,
                201, 83,
                214, 94,
                229, 104,
                246, 111,
                256, 75,
                249, 55,
                242, 38,
                224, 24,
                206, 20,
                177, 15,
                151, 10,
                125, 16,
                102, 22,
            ],
            dtype=np.float64,
        )
        anchors = anchors.reshape(-1, 2)
        centreline = _resample_closed_polyline(anchors, n_points)
        super().__init__(centreline=centreline, half_width=half_width, name="interlagos")

class SpaTrack(BaseTrack):
    """Spa-inspired layout (not official CAD-accurate)."""

    def __init__(self, half_width: float = 12.0, n_points: int = 240):
        anchors = 3/4*np.array(
            [
                (558, 812),(443, 887),(318, 989),(263, 953),(309, 805),(384, 739),
                (489, 624), (552, 555),(597, 440), (683, 403), (758, 403),(844, 328),
                (967, 278),(1061, 230),(1167, 186), (1284, 130), (1376, 92),(1426, 78),
                (1484, 117),(1574, 86),(1641, 142),(1704, 203),(1766, 276),(1749, 326),
                (1683, 301),(1603, 228),(1464, 288),(1265, 361),(1232, 420),(1257, 513),
                  (1332, 570),(1478, 605),(1562, 632),(1591, 686),(1578, 762),(1656, 820),
                  (1749, 866),(1728, 945),(1676, 1008),(1551, 995),(1457, 939),(1353, 820),
                  (1269, 712),(1132, 657),(1052, 645),(958, 680),(869, 730),(723, 810),
                  (695, 714),(631, 757)
            ],
            dtype=np.float64,
        )

        centreline = _resample_closed_polyline(anchors, n_points)
        super().__init__(centreline=centreline, half_width=half_width, name="spa")


class Track(OvalTrack):
    """Backward-compatible default track class (oval)."""




def make_track(name: str = "oval") -> BaseTrack:
    key = name.strip().lower()
    if key == "oval":
        return OvalTrack()
    if key == "monaco":
        return MonacoTrack()
    if key == "spa":
        return SpaTrack()
    if key == "interlagos" or key == "brazil":
        return BrazilTrack()
    raise ValueError(f"Unknown track '{name}'. Valid options: {', '.join(TRACK_NAMES)}")



def _estimate_normals_closed(points: np.ndarray) -> np.ndarray:
    prev_pts = np.roll(points, 1, axis=0)
    next_pts = np.roll(points, -1, axis=0)
    tangents = next_pts - prev_pts
    tan_norm = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(tan_norm, 1e-10)

    # Screen coords are y-down, so right-hand normal is (ty, -tx).
    normals = np.column_stack([tangents[:, 1], -tangents[:, 0]])
    nor_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(nor_norm, 1e-10)



def _resample_closed_polyline(points: np.ndarray, n_points: int) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")

    closed = np.vstack([p, p[0]])
    seg = closed[1:] - closed[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    if total <= 1e-9:
        raise ValueError("polyline length must be > 0")

    targets = np.linspace(0.0, total, n_points, endpoint=False)
    out = np.empty((n_points, 2), dtype=np.float64)

    seg_idx = 0
    for i, d in enumerate(targets):
        while seg_idx < len(seg_len) - 1 and d > cum[seg_idx + 1]:
            seg_idx += 1
        span = max(seg_len[seg_idx], 1e-10)
        frac = (d - cum[seg_idx]) / span
        out[i] = closed[seg_idx] + frac * seg[seg_idx]

    return out
