"""
train.py — Training loop with live Pygame visualisation.

Keyboard controls (window must have focus)
------------------------------------------
  ESC / Q  — quit and save
  S        — save checkpoint immediately
  F        — toggle fast mode (skip rendering to train ~10× faster)
  R        — reset the current episode
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pygame

from track import TRACK_NAMES, make_track
from car_env import CarEnv, MAX_RAY_DIST, MAX_SPEED, LOOKAHEAD_DIST
from agent import PPO

# ── Training ─────────────────────────────────────────────────────────────
N_ROLLOUT    = 2048          # env steps between PPO updates
MAX_EP_LEN   = 10000          # max steps per episode (prevents infinite loops)
TOTAL_STEPS  = 2_000_000     # stop after this many env steps
SAVE_EVERY   = 100_000       # auto-save checkpoint interval (steps)
CHECKPOINT   = "checkpoint.pt"
REPLAY_DIR   = "replays"

# ── Runtime defaults (edit these to avoid passing CLI args each run) ─────
DEFAULT_TRACK      = "monaco"
DEFAULT_REPLAY_DIR = REPLAY_DIR
DEFAULT_NUM_RAYS   = 13
DEFAULT_MAX_STEER_DELTA = 1.
DEFAULT_MAX_THROTTLE_DELTA = 1.
DEFAULT_LOOKAHEAD  = LOOKAHEAD_DIST   # px — probe point ahead of car

# ── Window / colours ─────────────────────────────────────────────────────
W, H = 1600, 1200
BG        = (12,  12,  12)
TRACK_BG  = (40,  40,  40)
WALL_IN   = (210, 60,  60)
WALL_OUT  = (60,  60, 210)
CAR_COL   = (255, 210, 30)
HUD_COL   = (220, 220, 220)
CAR_VW    = 18//2   # visual car width  (px) — purely cosmetic
CAR_VH    = 11//2   # visual car height (px)


class LapRecorder:
    """Collect per-step telemetry and save completed laps as compressed npz files."""

    def __init__(
        self,
        replay_dir: str = REPLAY_DIR,
        track_name: str = "monaco",
        num_rays: int = DEFAULT_NUM_RAYS,
    ):
        self.track_name = track_name
        self.num_rays = int(num_rays)
        self.replay_root = Path(replay_dir)
        self.replay_dir = self.replay_root / self.track_name
        self.replay_dir.mkdir(parents=True, exist_ok=True)

        existing = sorted(self.replay_dir.glob("lap_*.npz"))
        if existing:
            self.next_lap_id = max(int(p.stem.split("_")[-1]) for p in existing) + 1
        else:
            self.next_lap_id = 1

        self.current_steps: list[dict[str, float]] = []
        self.prev_total_laps = 0.0

    def on_reset(self, env: CarEnv) -> None:
        self.current_steps = []
        self.prev_total_laps = float(env._total_laps)

    def record_step(
        self,
        env: CarEnv,
        action: np.ndarray,
        reward: float,
        done: bool,
        global_step: int,
        updates: int,
        ep_len: int,
    ) -> None:
        total_laps = float(env._total_laps)
        self.current_steps.append(
            {
                "x": float(env.pos[0]),
                "y": float(env.pos[1]),
                "angle": float(env.angle),
                "speed": float(env.speed),
                "steering": float(action[0]),
                "throttle": float(action[1]),
                "reward": float(reward),
                "done": float(done),
                "total_laps": total_laps,
                "global_step": float(global_step),
                "update": float(updates),
                "ep_step": float(ep_len),
            }
        )

        prev_floor = int(np.floor(self.prev_total_laps + 1e-9))
        curr_floor = int(np.floor(total_laps + 1e-9))

        if curr_floor > prev_floor:
            self._save_current_lap()
            self.current_steps = []

        self.prev_total_laps = total_laps

    def discard_partial(self, env: CarEnv) -> None:
        self.current_steps = []
        self.prev_total_laps = float(env._total_laps)

    def _save_current_lap(self) -> None:
        if len(self.current_steps) < 2:
            return

        arr = {k: np.array([s[k] for s in self.current_steps], dtype=np.float32)
               for k in self.current_steps[0]}

        lap_id = self.next_lap_id
        self.next_lap_id += 1

        out_path = self.replay_dir / f"lap_{lap_id:06d}.npz"
        np.savez_compressed(
            out_path,
            **arr,
            n_steps=np.int32(len(self.current_steps)),
            lap_return=np.float32(np.sum(arr["reward"])),
            lap_time=np.float32(len(self.current_steps) / 60.0),
            track_name=np.array(self.track_name),
            num_rays=np.int32(self.num_rays),
        )
        print(
            f"Replay saved → {out_path}  "
            f"(steps={len(self.current_steps)}, ret={float(np.sum(arr['reward'])):.1f})"
        )


# ================================================================== #
#  Rendering
# ================================================================== #

def build_static_track_surface(track, size: tuple[int, int]) -> pygame.Surface:
    surf = pygame.Surface(size).convert()
    surf.fill(BG)

    outer_px = [(int(p[0]), int(p[1])) for p in track.outer_wall]
    inner_px = [(int(p[0]), int(p[1])) for p in track.inner_wall]
    pygame.draw.polygon(surf, TRACK_BG, outer_px)
    pygame.draw.polygon(surf, BG, inner_px)

    pygame.draw.polygon(surf, WALL_OUT, outer_px, 2)
    pygame.draw.polygon(surf, WALL_IN, inner_px, 2)

    cl_pts = [(int(p[0]), int(p[1])) for p in track.centreline]
    pygame.draw.lines(surf, (70, 70, 70), True, cl_pts, 1)

    return surf

def draw(
    screen: pygame.Surface,
    static_track_surface: pygame.Surface,
    font,
    env: CarEnv,
    action: np.ndarray,
    total_steps: int,
    updates: int,
    fast: bool,
) -> None:
    screen.blit(static_track_surface, (0, 0))

    # ── Vision rays — near fan (green, from car) ──────────────────────
    cx, cy = int(env.pos[0]), int(env.pos[1])
    for rel in env.ray_angles:
        ang = env.angle + rel
        d   = env._cast_ray(env.pos, np.array([np.cos(ang), np.sin(ang)]))
        brightness = int(80 + 175 * (1.0 - d / MAX_RAY_DIST))
        color = (0, brightness, int(brightness * 0.45))
        ex = int(env.pos[0] + np.cos(ang) * d)
        ey = int(env.pos[1] + np.sin(ang) * d)
        pygame.draw.line(screen, color, (cx, cy), (ex, ey), 1)
        pygame.draw.circle(screen, color, (ex, ey), 2)

    # ── Vision rays — far fan (cyan, from lookahead probe) ────────────
    probe, probe_angle = env._probe_pos_and_angle()
    px, py = int(probe[0]), int(probe[1])
    # Small diamond marker at the probe origin
    pygame.draw.circle(screen, (80, 200, 220), (px, py), 3)
    # Thin dashed connector: draw as a short line from car to probe
    pygame.draw.line(screen, (60, 100, 110), (cx, cy), (px, py), 1)
    for rel in env.far_ray_angles:
        ang = probe_angle + rel
        d   = env._cast_ray(probe, np.array([np.cos(ang), np.sin(ang)]))
        brightness = int(60 + 150 * (1.0 - d / MAX_RAY_DIST))
        color = (int(brightness * 0.3), int(brightness * 0.9), brightness)
        ex = int(probe[0] + np.cos(ang) * d)
        ey = int(probe[1] + np.sin(ang) * d)
        pygame.draw.line(screen, color, (px, py), (ex, ey), 1)
        pygame.draw.circle(screen, color, (ex, ey), 2)

    # ── Car rectangle ─────────────────────────────────────────────────
    ca, sa = np.cos(env.angle), np.sin(env.angle)
    rot    = np.array([[ca, -sa], [sa, ca]])
    hw, hh = CAR_VW / 2, CAR_VH / 2
    local_corners = np.array([
        [ hw,  hh], [-hw,  hh], [-hw, -hh], [ hw, -hh],
    ])
    world_corners = [
        (int(env.pos[0] + (rot @ c)[0]), int(env.pos[1] + (rot @ c)[1]))
        for c in local_corners
    ]
    pygame.draw.polygon(screen, CAR_COL, world_corners)
    # Tiny heading arrow
    tip = (int(env.pos[0] + ca * (hw + 3)), int(env.pos[1] + sa * (hw + 3)))
    pygame.draw.line(screen, (255, 255, 255), (cx, cy), tip, 2)

    # ── HUD ───────────────────────────────────────────────────────────
    mode_str = " [FAST]" if fast else ""
    lines = [
        f"Steps:    {total_steps:>9,}{mode_str}",
        f"Updates:  {updates:>6}",
        f"Speed:    {env.speed:5.2f} / {MAX_SPEED:.0f}  px/tick",
        f"Steering: {action[0]:+.2f}   Throttle: {action[1]:+.2f}",
        f"Progress: {env._total_laps:.3f} laps",
        "",
        "ESC/Q quit  S save  F fast  R reset",
    ]
    for i, text in enumerate(lines):
        surf = font.render(text, True, HUD_COL)
        screen.blit(surf, (10, 10 + i * 22))

    pygame.display.flip()


# ================================================================== #
#  Rollout collection
# ================================================================== #

def collect_rollout(
    env:         CarEnv,
    agent:       PPO,
    n_steps:     int,
    screen,
    static_track_surface,
    font,
    clock,
    fast:        bool,
    updates:     int,
    total_steps: int,
    recorder:    LapRecorder,
):
    """
    Collect n_steps of experience.

    Returns
    -------
    (rollout, last_value, status)
    status ∈ {"ok", "quit", "toggle_fast", "reset_ep"}
    """
    buf = {k: [] for k in ("obs", "actions", "log_probs", "rewards", "values", "dones")}
    obs     = env.reset()
    recorder.on_reset(env)
    ep_len  = 0
    last_action = np.zeros(2)

    for step in range(n_steps):

        # ── Event handling ───────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, 0.0, "quit"
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return None, 0.0, "quit"
                if event.key == pygame.K_s:
                    agent.save(CHECKPOINT)
                if event.key == pygame.K_f:
                    return None, 0.0, "toggle_fast"
                if event.key == pygame.K_r:
                    obs = env.reset(); ep_len = 0
                    recorder.on_reset(env)

        # ── Act ──────────────────────────────────────────────────────
        action, log_p, value = agent.net.act(obs)
        next_obs, reward, done, info = env.step(action)
        last_action = np.asarray(info.get("applied_action", action), dtype=np.float32)
        ep_len += 1

        recorder.record_step(
            env=env,
            action=last_action,
            reward=reward,
            done=done,
            global_step=total_steps + step,
            updates=updates,
            ep_len=ep_len,
        )

        buf["obs"].append(obs)
        buf["actions"].append(action)
        buf["log_probs"].append(log_p)
        buf["rewards"].append(reward)
        buf["values"].append(value)
        buf["dones"].append(float(done))

        obs     = next_obs

        # ── Render ───────────────────────────────────────────────────
        if not fast:
            draw(
                screen,
                static_track_surface,
                font,
                env,
                last_action,
                total_steps + step,
                updates,
                fast,
            )
            clock.tick(60)

        # ── Episode reset ─────────────────────────────────────────────
        if done or ep_len >= MAX_EP_LEN:
            obs    = env.reset()
            ep_len = 0
            recorder.discard_partial(env)

    _, _, last_val = agent.net.act(obs)
    return buf, last_val, "ok"


# ================================================================== #
#  Main
# ================================================================== #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F1 RL PPO trainer")
    parser.add_argument(
        "--track",
        type=str,
        default=DEFAULT_TRACK,
        choices=TRACK_NAMES,
        help="Track preset to train on",
    )
    parser.add_argument(
        "--replay-dir",
        type=str,
        default=DEFAULT_REPLAY_DIR,
        help="Directory for saved lap replay files",
    )
    parser.add_argument(
        "--num-rays",
        type=int,
        default=DEFAULT_NUM_RAYS,
        help="Number of vision rays in the observation (>=1)",
    )
    parser.add_argument(
        "--max-steer-delta",
        type=float,
        default=DEFAULT_MAX_STEER_DELTA,
        help="Max steering change per step (0 disables steering changes)",
    )
    parser.add_argument(
        "--max-throttle-delta",
        type=float,
        default=DEFAULT_MAX_THROTTLE_DELTA,
        help="Max throttle change per step (0 disables throttle changes)",
    )
    parser.add_argument(
        "--lookahead-dist",
        type=float,
        default=DEFAULT_LOOKAHEAD,
        help="Distance (px) of centreline lookahead probe ahead of current progress (0 to disable)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_rays < 1:
        raise ValueError("--num-rays must be >= 1")
    if args.max_steer_delta < 0.0 or args.max_throttle_delta < 0.0:
        raise ValueError("--max-steer-delta and --max-throttle-delta must be >= 0")

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(f"F1 RL Car — PPO Training ({args.track})")
    font  = pygame.font.SysFont("monospace", 16)
    clock = pygame.time.Clock()

    track = make_track(args.track)
    env   = CarEnv(
        track,
        n_rays=args.num_rays,
        max_steer_delta=args.max_steer_delta,
        max_throttle_delta=args.max_throttle_delta,
        lookahead_dist=args.lookahead_dist,
    )
    agent = PPO(obs_dim=env.obs_dim)
    agent.load(CHECKPOINT)   # no-op if file doesn't exist
    recorder = LapRecorder(args.replay_dir, track_name=track.name, num_rays=args.num_rays)
    static_track_surface = build_static_track_surface(track, (W, H))

    total_steps  = 0
    updates      = 0
    fast         = False
    last_save    = 0

    print("=" * 60)
    print(f"  F1 RL Car — PPO Training ({track.name})")
    print(f"  Rays: {args.num_rays}")
    print(
        "  Action slew limits: "
        f"steer Δ≤{args.max_steer_delta:.3f}, throttle Δ≤{args.max_throttle_delta:.3f}"
    )
    print(f"  Centreline lookahead: {args.lookahead_dist:.1f}px")
    print("  Controls: ESC/Q quit  S save  F fast-mode  R reset ep")
    print("=" * 60)

    while total_steps < TOTAL_STEPS:
        rollout, last_val, status = collect_rollout(
            env,
            agent,
            N_ROLLOUT,
            screen,
            static_track_surface,
            font,
            clock,
            fast,
            updates,
            total_steps,
            recorder,
        )

        if status == "quit":
            print("Quitting — saving checkpoint...")
            agent.save(CHECKPOINT)
            break

        if status == "toggle_fast":
            fast = not fast
            print(f"Fast mode: {'ON  (rendering paused)' if fast else 'OFF (rendering live)'}")
            continue

        # ── PPO update ───────────────────────────────────────────────
        metrics = agent.update(rollout, last_val)
        total_steps += N_ROLLOUT
        updates     += 1

        mean_r   = float(np.mean(rollout["rewards"]))
        mean_ret = float(
            np.sum(rollout["rewards"]) /
            max(1, int(np.sum(rollout["dones"])) + 1)
        )

        print(
            f"[{total_steps:>9,}]  updates={updates:>4}  "
            f"p={metrics['policy']:+.4f}  v={metrics['value']:.4f}  "
            f"ent={metrics['entropy']:.4f}  "
            f"mean_r/step={mean_r:.4f}  mean_ep_ret≈{mean_ret:.1f}"
        )

        # ── Auto-save ────────────────────────────────────────────────
        if total_steps - last_save >= SAVE_EVERY:
            agent.save(CHECKPOINT)
            last_save = total_steps

    pygame.quit()
    print("Done.")


if __name__ == "__main__":
    main()
