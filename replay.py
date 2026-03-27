"""
replay.py — Offline lap replay viewer for saved training laps.

Usage
-----
python replay.py
python replay.py --dir replays --lap 3

Controls
--------
ESC / Q      quit
SPACE        pause/resume
LEFT / RIGHT step frame (when paused)
N / P        next / previous lap
+ / -        playback speed up / down
T            toggle trail
H            toggle HUD
M            toggle auto-advance laps
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pygame

from track import TRACK_NAMES, make_track

W, H = 1600, 1200
BG = (12, 12, 12)
TRACK_BG = (40, 40, 40)
WALL_IN = (210, 60, 60)
WALL_OUT = (60, 60, 210)
CAR_COL = (255, 210, 30)
HUD_COL = (220, 220, 220)
PATH_COL = (100, 220, 120)
CAR_VW = 18
CAR_VH = 11
BASE_FPS = 60.0
TARGET_FPS = 120

# ── Runtime defaults (edit these to avoid passing CLI args each run) ─────
DEFAULT_REPLAY_DIR = "replays"
DEFAULT_START_LAP = 2850
DEFAULT_TRACK_MODE = "interlagos"      # auto | one of TRACK_NAMES
DEFAULT_TRACK_FILTER = "all"            # all  | one of TRACK_NAMES


class ReplayViewer:
    def __init__(
        self,
        replay_dir: Path,
        lap_index: int = 0,
        track_override: str | None = None,
        track_filter: str = "all",
    ):
        self.replay_dir = replay_dir
        self.track_filter = track_filter
        self.files = self._discover_replay_files()
        if not self.files:
            filter_note = "" if self.track_filter == "all" else f" for track_filter='{self.track_filter}'"
            raise FileNotFoundError(
                f"No replay files found in '{replay_dir}'{filter_note}. "
                "Expected files like replays/<track>/lap_000001.npz"
            )

        self.track_override = track_override

        self.lap_idx = int(np.clip(lap_index, 0, len(self.files) - 1))
        self.frame_idx = 0
        self.frame_pos = 0.0
        self.playing = True
        self.speed = 1.0
        self.show_trail = True
        self.show_hud = True
        self.auto_advance_laps = True
        self.render_fps = 0.0
        self.active_track_name = "interlagos"

        self.track = make_track(self.active_track_name)
        self.static_surface: pygame.Surface | None = None
        self.path_surface: pygame.Surface | None = None
        self.path_drawn_until = 0

        self._load_lap(self.lap_idx)

    def _discover_replay_files(self) -> list[Path]:
        files = sorted(
            self.replay_dir.rglob("lap_*.npz"),
            key=lambda p: str(p.relative_to(self.replay_dir)),
        )
        if self.track_filter == "all":
            return files

        # Primary filter: directory layout replays/<track>/lap_*.npz
        filtered = [p for p in files if p.parent.name.lower() == self.track_filter]
        if filtered:
            return filtered

        # Fallback for legacy layouts: infer from saved metadata.
        filtered = []
        for path in files:
            try:
                with np.load(path, allow_pickle=False) as data:
                    if self._lap_track_name(data).lower() == self.track_filter:
                        filtered.append(path)
            except Exception:
                continue
        return filtered

    def _build_static_surface(self) -> None:
        surf = pygame.Surface((W, H)).convert()
        surf.fill(BG)

        outer_px = [(int(p[0]), int(p[1])) for p in self.track.outer_wall]
        inner_px = [(int(p[0]), int(p[1])) for p in self.track.inner_wall]
        pygame.draw.polygon(surf, TRACK_BG, outer_px)
        pygame.draw.polygon(surf, BG, inner_px)
        pygame.draw.polygon(surf, WALL_OUT, outer_px, 2)
        pygame.draw.polygon(surf, WALL_IN, inner_px, 2)

        cl_pts = [(int(p[0]), int(p[1])) for p in self.track.centreline]
        pygame.draw.lines(surf, (70, 70, 70), True, cl_pts, 1)

        self.static_surface = surf
        self.path_surface = pygame.Surface((W, H), pygame.SRCALPHA).convert_alpha()
        self.path_surface.fill((0, 0, 0, 0))

    def _load_lap(self, idx: int) -> None:
        self.lap_idx = int(np.clip(idx, 0, len(self.files) - 1))
        self.data = np.load(self.files[self.lap_idx], allow_pickle=False)
        self.n_frames = int(self.data["n_steps"])
        self.lap_num_rays = int(self.data["num_rays"]) if "num_rays" in self.data.files else None

        lap_track = self._lap_track_name(self.data)
        selected_track = self.track_override or lap_track
        if selected_track != self.active_track_name:
            self.active_track_name = selected_track
            self.track = make_track(self.active_track_name)
            if self.static_surface is not None:
                self._build_static_surface()

        self.x = np.asarray(self.data["x"], dtype=np.float32)
        self.y = np.asarray(self.data["y"], dtype=np.float32)
        self.angle = np.asarray(self.data["angle"], dtype=np.float32)
        self.speed_arr = np.asarray(self.data["speed"], dtype=np.float32)
        self.steering = np.asarray(self.data["steering"], dtype=np.float32)
        self.throttle = np.asarray(self.data["throttle"], dtype=np.float32)

        self.xi = np.rint(self.x).astype(np.int32)
        self.yi = np.rint(self.y).astype(np.int32)
        self.path_points = list(zip(self.xi.tolist(), self.yi.tolist()))

        self.cos_angle = np.cos(self.angle)
        self.sin_angle = np.sin(self.angle)

        self.frame_idx = 0
        self.frame_pos = 0.0
        self.path_drawn_until = 0
        if self.path_surface is not None:
            self.path_surface.fill((0, 0, 0, 0))

    def _lap_track_name(self, data: np.lib.npyio.NpzFile) -> str:
        if "track_name" not in data.files:
            return "oval"
        value = data["track_name"]
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return str(value.item())
            if len(value) > 0:
                return str(value[0])
        return str(value)

    def _redraw_path_to(self, frame_idx: int) -> None:
        if self.path_surface is None:
            return

        self.path_surface.fill((0, 0, 0, 0))
        target = int(np.clip(frame_idx, 0, self.n_frames - 1))
        if target >= 1:
            pygame.draw.lines(
                self.path_surface,
                PATH_COL,
                False,
                self.path_points[: target + 1],
                2,
            )
        self.path_drawn_until = target

    def _extend_path_to(self, frame_idx: int) -> None:
        if self.path_surface is None:
            return

        target = int(np.clip(frame_idx, 0, self.n_frames - 1))
        if target <= self.path_drawn_until:
            return

        start = max(1, self.path_drawn_until)
        for idx in range(start, target + 1):
            pygame.draw.line(
                self.path_surface,
                PATH_COL,
                self.path_points[idx - 1],
                self.path_points[idx],
                2,
            )
        self.path_drawn_until = target

    def _interp_pose(self, frame_pos: float):
        i0 = int(np.clip(frame_pos, 0, self.n_frames - 1))
        i1 = min(self.n_frames - 1, i0 + 1)
        alpha = float(frame_pos - i0)

        x = float((1.0 - alpha) * self.x[i0] + alpha * self.x[i1])
        y = float((1.0 - alpha) * self.y[i0] + alpha * self.y[i1])
        speed = float((1.0 - alpha) * self.speed_arr[i0] + alpha * self.speed_arr[i1])
        steer = float((1.0 - alpha) * self.steering[i0] + alpha * self.steering[i1])
        throttle = float((1.0 - alpha) * self.throttle[i0] + alpha * self.throttle[i1])

        ca = float((1.0 - alpha) * self.cos_angle[i0] + alpha * self.cos_angle[i1])
        sa = float((1.0 - alpha) * self.sin_angle[i0] + alpha * self.sin_angle[i1])
        norm = float(np.hypot(ca, sa))
        if norm > 1e-8:
            ca /= norm
            sa /= norm
        else:
            ca, sa = 1.0, 0.0

        return i0, x, y, ca, sa, speed, steer, throttle

    def _draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        if self.static_surface is None:
            return

        screen.blit(self.static_surface, (0, 0))

        i, x, y, ca, sa, speed, steer, throttle = self._interp_pose(self.frame_pos)

        if self.show_trail and self.path_surface is not None:
            self._extend_path_to(i)
            screen.blit(self.path_surface, (0, 0))

        hw, hh = CAR_VW / 2.0, CAR_VH / 2.0
        c1 = (int(x + ca * hw - sa * hh), int(y + sa * hw + ca * hh))
        c2 = (int(x - ca * hw - sa * hh), int(y - sa * hw + ca * hh))
        c3 = (int(x - ca * hw + sa * hh), int(y - sa * hw - ca * hh))
        c4 = (int(x + ca * hw + sa * hh), int(y + sa * hw - ca * hh))
        pygame.draw.polygon(screen, CAR_COL, [c1, c2, c3, c4])
        pygame.draw.line(
            screen,
            (255, 255, 255),
            (int(x), int(y)),
            (int(x + ca * (hw + 3.0)), int(y + sa * (hw + 3.0))),
            2,
        )

        if self.show_hud:
            lap_rel = str(self.files[self.lap_idx].relative_to(self.replay_dir))
            lines = [
                f"Replay: {lap_rel}   ({self.lap_idx + 1}/{len(self.files)})",
                f"Track: {self.active_track_name}",
                f"Rays: {self.lap_num_rays if self.lap_num_rays is not None else 'n/a'}",
                f"List filter: {self.track_filter}",
                f"Frame: {i + 1:>5}/{self.n_frames:<5}   {'PLAY' if self.playing else 'PAUSE'}   speed x{self.speed:.2f}",
                f"Lap flow: {'AUTO' if self.auto_advance_laps else 'STOP'}",
                f"Car: x={x:6.1f}  y={y:6.1f}  v={speed:4.2f}",
                f"Control: steer={steer:+.2f}  throttle={throttle:+.2f}",
                f"Lap return: {float(self.data['lap_return']):.1f}    Lap time: {float(self.data['lap_time']):.2f}s",
                f"Render FPS: {self.render_fps:5.1f}",
                "ESC/Q quit  SPACE play/pause  ←/→ step  N/P lap  +/- speed  T trail  H HUD  M auto",
            ]

            for row, text in enumerate(lines):
                surf = font.render(text, True, HUD_COL)
                screen.blit(surf, (10, 10 + row * 22))

        pygame.display.flip()

    def run(self) -> None:
        pygame.init()
        screen = pygame.display.set_mode((W, H), pygame.DOUBLEBUF)
        pygame.display.set_caption("F1 RL Car — Lap Replay")
        font = pygame.font.SysFont("monospace", 16)
        clock = pygame.time.Clock()

        self._build_static_surface()
        self._redraw_path_to(0)

        running = True
        while running:
            dt = clock.tick(TARGET_FPS) / 1000.0
            self.render_fps = float(clock.get_fps())

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.playing = not self.playing
                    elif event.key == pygame.K_RIGHT and not self.playing:
                        self.frame_idx = min(self.n_frames - 1, self.frame_idx + 1)
                        self.frame_pos = float(self.frame_idx)
                        self._redraw_path_to(self.frame_idx)
                    elif event.key == pygame.K_LEFT and not self.playing:
                        self.frame_idx = max(0, self.frame_idx - 1)
                        self.frame_pos = float(self.frame_idx)
                        self._redraw_path_to(self.frame_idx)
                    elif event.key == pygame.K_n:
                        self._load_lap(min(len(self.files) - 1, self.lap_idx + 1))
                        self._redraw_path_to(0)
                    elif event.key == pygame.K_p:
                        self._load_lap(max(0, self.lap_idx - 1))
                        self._redraw_path_to(0)
                    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                        self.speed = min(8.0, self.speed * 1.25)
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        self.speed = max(0.1, self.speed / 1.25)
                    elif event.key == pygame.K_t:
                        self.show_trail = not self.show_trail
                    elif event.key == pygame.K_h:
                        self.show_hud = not self.show_hud
                    elif event.key == pygame.K_m:
                        self.auto_advance_laps = not self.auto_advance_laps

            if self.playing:
                self.frame_pos = min(self.n_frames - 1, self.frame_pos + self.speed * BASE_FPS * dt)
                self.frame_idx = int(self.frame_pos)
                if self.frame_idx >= self.n_frames - 1:
                    if self.auto_advance_laps and len(self.files) > 1:
                        next_idx = (self.lap_idx + 1) % len(self.files)
                        self._load_lap(next_idx)
                        self._redraw_path_to(0)
                        self.playing = True
                    else:
                        self.playing = False

            self._draw(screen, font)

        pygame.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay saved F1 RL laps")
    parser.add_argument("--dir", type=str, default=DEFAULT_REPLAY_DIR, help="Replay root directory")
    parser.add_argument("--lap", type=int, default=DEFAULT_START_LAP, help="1-based lap index to start from")
    parser.add_argument(
        "--track",
        type=str,
        default=DEFAULT_TRACK_MODE,
        choices=["auto", *TRACK_NAMES],
        help="Replay track: auto uses lap metadata",
    )
    parser.add_argument(
        "--track-filter",
        type=str,
        default=DEFAULT_TRACK_FILTER,
        choices=["all", *TRACK_NAMES],
        help="Filter replay list to one track subfolder when browsing a replay root",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replay_dir = Path(args.dir)
    track_override = None if args.track == "auto" else args.track
    effective_filter = args.track_filter
    # Keep replay list aligned with explicit track overrides by default.
    # Users can still force a different filter by passing --track-filter.
    if track_override is not None and args.track_filter == "all":
        effective_filter = track_override
    viewer = ReplayViewer(
        replay_dir=replay_dir,
        lap_index=max(0, args.lap - 1),
        track_override=track_override,
        track_filter=effective_filter,
    )
    viewer.run()


if __name__ == "__main__":
    main()
