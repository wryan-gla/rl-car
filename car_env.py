"""
car_env.py — 2-D kinematic car racing environment.

Observation space ((2·n_rays + 2)-D, all values ∈ [0, 1] except angle):
    [near_ray_0, …, near_ray_(n-1),   ← cast from the car's current position
    far_ray_0,  …, far_ray_(n-1),    ← cast from centreline lookahead probe
     speed]
    Both ray fans share the same angular spread (default ±90°) and are ordered
    left→right.  The far (lookahead) rays let the policy anticipate upcoming
    corners before the car physically arrives at them.

Action space (2-D, values ∈ [−1, 1]):
    [steering, throttle]
    steering > 0 → turn right (clockwise in screen coords, y-down)
    throttle > 0 → accelerate, < 0 → brake

Control smoothing
-----------------
Optional slew-rate limits can constrain how much steering/throttle may change
between consecutive steps. This improves control consistency and reduces jerky
commanding from the policy during training.

Reward:
    Each step: Δprogress × PROGRESS_SCALE
        One full lap ≈ +100 reward.
        Going backwards gives negative reward.
    On collision: −COLLISION_PENALTY (terminal).
"""

from __future__ import annotations
import numpy as np
from track import Track


# ── Physics ─────────────────────────────────────────────────────────────
MAX_SPEED       = 10.0    # pixels per timestep
ACCEL           = 2.    # Δspeed per step at full throttle
BRAKE_MULT      = 2.0    # braking is this many times faster than accelerating
MAX_STEER_RAD      = 0.12   # max heading change (rad/step) at full speed
SLOW_STEER_FRAC    = 1.2   # steering authority at rest, as a fraction of MAX_STEER_RAD
#                             0.50 → car can turn at half-authority even when stationary.
#                             Authority then interpolates linearly up to 1.0 at MAX_SPEED.
#                             Keep this ≥ 0.3 so the car can always navigate hairpins
#                             from rest; keep it ≤ 0.7 so high-speed inputs aren't too
#                             twitchy relative to low-speed ones.

# ── Sensors ──────────────────────────────────────────────────────────────
MAX_RAY_DIST   = 300.0                            # pixels (saturates sensor)
LOOKAHEAD_DIST = 80.0                             # px ahead along centreline for probe fan
RAY_ANGLES     = np.linspace(-np.pi / 2, np.pi / 2, 5)   # 5 rays, ±90°
FAR_RAY_ANGLES = np.linspace(-np.pi / 2, np.pi / 2, 5)   # 5 rays, ±90°

# ── Collision ────────────────────────────────────────────────────────────
CAR_RADIUS = 4.0   # collision sphere radius (px) — car is modelled as a circle

# ── Reward ───────────────────────────────────────────────────────────────
PROGRESS_SCALE    = 100.0   # reward multiplier: one full lap → ~100 reward
COLLISION_PENALTY = 5.0    # subtracted on wall hit


class CarEnv:
    """
    Gym-compatible 2-D car environment.

    Usage
    -----
    env = CarEnv()
    obs = env.reset()
    obs, reward, done, info = env.step(action)
    """

    # Expose constants so external code (rendering etc.) can read them.
    MAX_SPEED      = MAX_SPEED
    MAX_RAY_DIST   = MAX_RAY_DIST
    LOOKAHEAD_DIST = LOOKAHEAD_DIST
    RAY_ANGLES     = RAY_ANGLES
    FAR_RAY_ANGLES = FAR_RAY_ANGLES
    CAR_RADIUS     = CAR_RADIUS

    def __init__(
        self,
        track: Track | None = None,
        n_rays: int = 5,
        ray_span: float = np.pi,
        far_ray_span: float = np.pi,
        local_window: float = 0.30,
        max_steer_delta: float = 0.20,
        max_throttle_delta: float = 0.20,
        lookahead_dist: float = LOOKAHEAD_DIST,
    ):
        self.track = track or Track()
        self.n_rays = max(1, int(n_rays))
        self.ray_span = float(ray_span)
        self.far_ray_span = float(far_ray_span)
        self.ray_angles = np.linspace(
            -self.ray_span / 2.0,
            self.ray_span / 2.0,
            self.n_rays,
            dtype=np.float64,
        )
        self.far_ray_angles = np.linspace(
            -self.far_ray_span / 2.0,
            self.far_ray_span / 2.0,
            self.n_rays,
            dtype=np.float64,
        )
        # Lookahead probe: sample a point lookahead_dist px ahead *along the
        # centreline arc* from current progress. This gives corner context
        # without allowing through-wall straight-line peeking.
        self.lookahead_dist = float(max(0.0, lookahead_dist))
        # Observation = near_rays + far_rays + speed
        self.obs_dim = 2 * self.n_rays + 1
        self.max_steer_delta = float(max(0.0, max_steer_delta))
        self.max_throttle_delta = float(max(0.0, max_throttle_delta))

        # Progress-local segment filtering.
        # On tracks that double back on themselves (e.g. Monaco), the naive
        # wall-segment array contains "phantom" walls where offset polygons from
        # two different sections cross each other.  Restricting collision and
        # ray-cast checks to segments whose source centreline index is within
        # ±local_window of the car's current lap fraction eliminates all
        # cross-track phantoms while keeping every genuinely nearby wall.
        # ±0.30 (30 % of the lap) is safe: Monaco's phantom pairs are 42–45 %
        # apart, and the tightest hairpin spans only 9 %.
        self._local_window = float(np.clip(local_window, 0.05, 0.49))
        self._n_cl = self.track.n_points
        self._seg_cl_idxs: np.ndarray = self.track.seg_cl_idxs  # (2N,) int32

        # Full segment endpoint arrays — filtered views are built per-step.
        self._all_p1: np.ndarray = self.track.wall_segs[:, 0, :]
        self._all_p2: np.ndarray = self.track.wall_segs[:, 1, :]

        # Active views (updated each step via _refresh_local_segs)
        self._p1: np.ndarray = self._all_p1
        self._p2: np.ndarray = self._all_p2

        self.reset()

    # ================================================================== #
    #  Gym interface
    # ================================================================== #

    def reset(self) -> np.ndarray:
        pos, angle = self.track.start_pos.copy(), self.track.start_angle
        self.pos          = pos.astype(float)
        self.angle        = float(angle)
        self.speed        = 0.0
        self._progress    = self.track.get_progress(self.pos)
        self._total_laps  = 0.0
        self.done         = False
        self._prev_steer = 0.0
        self._prev_throttle = 0.0
        self.last_action_applied = np.zeros(2, dtype=np.float32)
        return self._obs()

    def step(self, action: np.ndarray):
        """
        Parameters
        ----------
        action : array-like of shape (2,)
            [steering, throttle], both clipped to [-1, 1].

        Returns
        -------
        obs    : np.ndarray (6,)
        reward : float
        done   : bool
        info   : dict  {"laps": float}
        """
        assert not self.done, "Episode has ended — call reset()."

        steer_cmd    = float(np.clip(action[0], -1.0, 1.0))
        throttle_cmd = float(np.clip(action[1], -1.0, 1.0))

        steer = float(
            np.clip(
                steer_cmd,
                self._prev_steer - self.max_steer_delta,
                self._prev_steer + self.max_steer_delta,
            )
        )
        throttle = float(
            np.clip(
                throttle_cmd,
                self._prev_throttle - self.max_throttle_delta,
                self._prev_throttle + self.max_throttle_delta,
            )
        )
        self._prev_steer = steer
        self._prev_throttle = throttle
        self.last_action_applied = np.array([steer, throttle], dtype=np.float32)

        # ── Kinematics ──────────────────────────────────────────────────
        if throttle >= 0.0:
            self.speed += throttle * ACCEL
        else:
            self.speed += throttle * ACCEL * BRAKE_MULT
        self.speed = float(np.clip(self.speed, 0.0, MAX_SPEED))

        # Steering authority: interpolates from SLOW_STEER_FRAC at rest up to
        # 1.0 at MAX_SPEED.  This lets the car turn through hairpins from a
        # standing start while keeping high-speed inputs proportionally larger
        # (more committed steering at speed = more realistic handling feel).
        speed_frac   = self.speed / MAX_SPEED
        steer_factor = SLOW_STEER_FRAC + (1.0 - SLOW_STEER_FRAC) * speed_frac
        self.angle  += steer * MAX_STEER_RAD * steer_factor

        new_pos = self.pos + np.array(
            [np.cos(self.angle), np.sin(self.angle)]
        ) * self.speed

        # ── Refresh progress-local segment view ──────────────────────────
        self._refresh_local_segs()

        # ── Collision check ──────────────────────────────────────────────
        if self._collides(self.pos, new_pos):
            self.done = True
            return self._obs(), -COLLISION_PENALTY, True, {
                "laps": self._total_laps,
                "applied_action": self.last_action_applied.copy(),
            }

        self.pos = new_pos

        # ── Progress reward ──────────────────────────────────────────────
        new_prog = self.track.get_progress(self.pos)
        delta    = new_prog - self._progress

        # Handle the wraparound at the lap boundary.
        # Assumption: car can't travel > half the track in a single step.
        if delta < -0.5:
            delta += 1.0   # crossed start line going forward
        elif delta > 0.5:
            delta -= 1.0   # crossed start line going backward

        self._progress   = new_prog
        self._total_laps += delta
        reward = delta * PROGRESS_SCALE

        return self._obs(), float(reward), False, {
            "laps": self._total_laps,
            "applied_action": self.last_action_applied.copy(),
        }

    # ================================================================== #
    #  Observation
    # ================================================================== #

    def _obs(self) -> np.ndarray:
        near_rays = self._cast_rays(self.pos)
        probe_pos, probe_angle = self._probe_pos_and_angle()
        far_rays  = self._cast_far_rays(probe_pos, probe_angle)
        return np.concatenate([near_rays, far_rays, [self.speed / MAX_SPEED]]).astype(np.float32)

    def _probe_pos(self) -> np.ndarray:
        """Point lookahead_dist px ahead along centreline arc from current progress."""
        probe_pos, _ = self._probe_pos_and_angle()
        return probe_pos

    def _probe_pos_and_angle(self) -> tuple[np.ndarray, float]:
        """Centreline lookahead probe position and local tangent heading."""
        if self.track.track_length <= 1e-9 or self.lookahead_dist <= 0.0:
            return self.pos.copy(), float(self.angle)

        delta_prog = self.lookahead_dist / self.track.track_length
        probe_progress = (self._progress + delta_prog) % 1.0
        return self.track.point_and_heading_at_progress(probe_progress)

    def _cast_rays_with_angles(self, origin: np.ndarray, base_angle: float, angles: np.ndarray) -> np.ndarray:
        """Cast rays from *origin* for a provided set of relative angles."""
        out = np.empty(len(angles), dtype=np.float32)
        for i, rel in enumerate(angles):
            ang = base_angle + rel
            direction = np.array([np.cos(ang), np.sin(ang)])
            out[i] = self._cast_ray(origin, direction) / MAX_RAY_DIST
        return out

    def _cast_rays(self, origin: np.ndarray) -> np.ndarray:
        """Cast the full ray fan from *origin* and return normalised distances."""
        return self._cast_rays_with_angles(origin, self.angle, self.ray_angles)
    
    def _cast_far_rays(self, origin: np.ndarray, base_angle: float) -> np.ndarray:
        """Cast the full ray fan from *origin* and return normalised distances."""
        return self._cast_rays_with_angles(origin, base_angle, self.far_ray_angles)

    def _cast_ray(self, origin: np.ndarray, direction: np.ndarray) -> float:
        """
        Vectorised ray-vs-all-wall-segments intersection.

        For a ray  P + t·D  and segment  A + s·(B-A),  solving the 2×2
        linear system gives:

            denom = Dx·(Ay−By) + Dy·(Bx−Ax)      [= Dx·sy − Dy·sx  where s=B−A]
            t     = [(Ax−Px)·sy − (Ay−Py)·sx] / denom
            s     = [(Ax−Px)·Dy − (Ay−Py)·Dx] / denom

        Intersection exists when t ≥ 0 and 0 ≤ s ≤ 1.
        We return the minimum valid t, clamped to MAX_RAY_DIST.
        """
        dx, dy  = direction
        ox, oy  = origin
        p1, p2  = self._p1, self._p2          # (2N, 2) each

        sx = p2[:, 0] - p1[:, 0]
        sy = p2[:, 1] - p1[:, 1]

        denom = dx * sy - dy * sx             # (2N,)
        valid = np.abs(denom) > 1e-10

        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(
                valid,
                ((p1[:, 0] - ox) * sy - (p1[:, 1] - oy) * sx) / denom,
                np.inf,
            )
            s = np.where(
                valid,
                ((p1[:, 0] - ox) * dy - (p1[:, 1] - oy) * dx) / denom,
                -1.0,
            )

        # Keep only hits that are in front of the ray and within segment
        hit = valid & (t > 1e-6) & (s >= 0.0) & (s <= 1.0)
        t   = np.where(hit, t, MAX_RAY_DIST)
        return float(t.min())

    # ================================================================== #
    #  Progress-local segment filtering
    # ================================================================== #

    def _refresh_local_segs(self) -> None:
        """
        Rebuild _p1 / _p2 to only include wall segments whose source
        centreline index is within ±local_window of the car's current
        lap progress.  Called once per step (cheap: pure numpy boolean mask).

        This eliminates "phantom" walls that arise when a track doubles back
        on itself and the offset polygons of two sections cross each other.
        """
        car_cl = int(round(self._progress * self._n_cl)) % self._n_cl
        win    = int(round(self._local_window * self._n_cl))

        # Circular distance on the track
        raw_dist  = np.abs(self._seg_cl_idxs - car_cl)
        circ_dist = np.minimum(raw_dist, self._n_cl - raw_dist)
        mask = circ_dist <= win

        self._p1 = self._all_p1[mask]
        self._p2 = self._all_p2[mask]

    # ================================================================== #
    #  Collision detection
    # ================================================================== #

    def _collides(self, old_pos: np.ndarray, new_pos: np.ndarray) -> bool:
        """
        True if the car (circle of radius CAR_RADIUS) moving from old_pos
        to new_pos penetrates any wall segment.

        Two complementary checks:
          1. Proximity — new centre within CAR_RADIUS of any wall segment.
          2. Crossing  — movement ray intersects a wall within the step length.
             (Catches fast cars that could tunnel through thin walls.)
        """
        # (1) Proximity of new centre to walls
        if self._dist_to_walls(new_pos) < CAR_RADIUS:
            return True

        # (2) Ray from old → new
        move     = new_pos - old_pos
        move_len = float(np.linalg.norm(move))
        if move_len < 1e-9:
            return False
        return self._cast_ray(old_pos, move / move_len) < move_len

    def _dist_to_walls(self, pos: np.ndarray) -> float:
        """Minimum distance from pos to any wall segment (vectorised)."""
        p1, p2  = self._p1, self._p2
        seg     = p2 - p1                                   # (2N, 2)
        seg_sq  = (seg * seg).sum(axis=1)                   # (2N,)
        diff    = pos - p1                                   # (2N, 2)
        t       = np.clip(
            (diff * seg).sum(axis=1) / np.maximum(seg_sq, 1e-10),
            0.0, 1.0,
        )
        closest = p1 + t[:, None] * seg                     # (2N, 2)
        return float(np.linalg.norm(pos - closest, axis=1).min())