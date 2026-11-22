"""Run the scripted FSM expert on seeded Lift episodes and report success rate.

Example:
    python scripts/run_expert.py --episodes 50 --seed-start 0 --videos 3
Outputs go to results/expert/<run_name>/:
    episodes.csv   one row per episode (seed, success, length, failure phase, ...)
    summary.json   config + aggregate success rate
    steps_seed*.csv  per-step control log for episodes that were recorded
    seed*_{success,fail}.mp4  agentview | wrist camera side by side
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import imageio
import numpy as np
import yaml

from newbody.env.lift_env import cube_pos, get_image, is_grasping, make_lift_env, reset_with_seed, table_height
from newbody.evaluation.success import HoldSuccess
from newbody.expert.fsm_expert import ExpertConfig, LiftFSMExpert

ROOT = Path(__file__).resolve().parents[1]


def run_episode(env, expert, checker, seed, record, env_cfg):
    obs = reset_with_seed(env, seed)
    expert.reset()
    checker.reset()
    start_cube = cube_pos(env).copy()
    w, x, y, z = env.sim.data.body_xquat[env.cube_body_id]
    yaw = float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))
    yaw_mod90 = yaw % 90  # cube is 4-fold symmetric: 0 and 90 deg look identical to a parallel gripper
    frames, steps = [], []
    success, max_cube_lift = False, 0.0
    for t in range(env_cfg["horizon"]):
        action = expert.act(obs)
        phase = expert.phase.value
        obs, _, _, _ = env.step(action)
        success = checker.update(env)
        lift = cube_pos(env)[2] - table_height(env)
        max_cube_lift = max(max_cube_lift, lift)
        if record:
            frames.append(np.concatenate([get_image(obs, c) for c in env_cfg["camera_names"]], axis=1))
            steps.append(dict(t=t, phase=phase, ax=action[0], ay=action[1], az=action[2], grip=action[3],
                              eef_x=obs["robot0_eef_pos"][0], eef_y=obs["robot0_eef_pos"][1], eef_z=obs["robot0_eef_pos"][2],
                              cube_z_above_table=lift, grasping=int(is_grasping(env))))
        if success or expert.timed_out_phase is not None:
            break
    if success:
        fail_reason = ""
    elif expert.timed_out_phase is not None:
        fail_reason = f"timeout_{expert.timed_out_phase.value}"
    else:
        fail_reason = f"horizon_in_{expert.phase.value}"
    row = dict(seed=seed, success=int(success), length=t + 1, fail_reason=fail_reason,
               final_phase=expert.phase.value, cube_x0=start_cube[0], cube_y0=start_cube[1], cube_yaw_mod90=round(yaw_mod90, 1),
               max_cube_lift=round(max_cube_lift, 4))
    return row, frames, steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs/lift_kinova3.yaml"))
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--videos", type=int, default=2, help="record first N successes (all failures are recorded too)")
    p.add_argument("--cube-xy-range", type=float, default=None, help="override env.cube_xy_range")
    p.add_argument("--set", nargs="*", default=[], help="override expert params, e.g. approach_height=0.05")
    p.add_argument("--name", default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.cube_xy_range is not None:
        cfg["env"]["cube_xy_range"] = args.cube_xy_range
    for kv in args.set:
        k, v = kv.split("=")
        cfg["expert"][k] = type(cfg["expert"][k])(v)

    out = ROOT / "results/expert" / (args.name or time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    env = make_lift_env(cfg["env"], render_cameras=True)
    expert = LiftFSMExpert(ExpertConfig(**cfg["expert"]))
    checker = HoldSuccess(**cfg["success"])

    rows, n_succ_videos, t0 = [], 0, time.time()
    for i in range(args.episodes):
        seed = args.seed_start + i
        # Record every episode cheaply (frames are only kept if we save them).
        row, frames, steps = run_episode(env, expert, checker, seed, record=True, env_cfg=cfg["env"])
        rows.append(row)
        save = (not row["success"]) or n_succ_videos < args.videos
        if save:
            tag = "success" if row["success"] else "fail"
            n_succ_videos += row["success"]
            imageio.mimsave(out / f"seed{seed}_{tag}.mp4", frames, fps=cfg["env"]["control_freq"])
            with open(out / f"steps_seed{seed}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(steps[0].keys()))
                w.writeheader()
                w.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()} for s in steps)
        print(f"seed {seed:4d}  {'OK  ' if row['success'] else 'FAIL'}  len={row['length']:3d}  {row['fail_reason']}")

    with open(out / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    k = sum(r["success"] for r in rows)
    summary = dict(success=k, episodes=n, success_rate=k / n, seeds=[args.seed_start, args.seed_start + n - 1],
                   wall_time_s=round(time.time() - t0, 1), config=cfg)
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(f"\nSuccess {k}/{n} = {k / n:.1%}   -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
