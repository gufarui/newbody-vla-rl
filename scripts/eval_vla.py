"""Closed-loop evaluation of a SmolVLA checkpoint in Kinova3 PickPlaceCan.

    python scripts/eval_vla.py --checkpoint outputs/sft/checkpoints/last/pretrained_model \
        --split test --episodes 100 --n-action-steps 10 --videos 5 --name sft_last_k10

The VLA acts alone (no expert). Privileged simulator state is used ONLY to score episodes and
label failure modes, never as policy input.

Outputs results/eval/<name>/: episodes.csv, summary.json (success rate + 95% Wilson CI,
failure-mode counts, chunk inference latency), and MP4s (all failures up to --videos, plus successes).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import imageio
import numpy as np
import yaml

from newbody.env.pick_place_env import can_in_target, can_pos, get_image, is_grasping, make_env, reset_with_seed
from newbody.evaluation.stats import wilson_interval
from newbody.evaluation.success import PlaceSuccess
from newbody.policies.vla import VLAController

ROOT = Path(__file__).resolve().parents[1]


def in_any_bin2_compartment(env) -> int | None:
    p = can_pos(env)
    for i in range(4):
        if not env.not_in_bin(p, i):
            return i
    return None


def run_episode(env, ctrl, checker, seed, horizon, cameras, record):
    obs = reset_with_seed(env, seed)
    ctrl.reset()
    checker.reset()
    start = can_pos(env).copy()
    frames, grip_raw = [], []
    was_grasped, max_lift, grasp_lost_outside = False, 0.0, False
    success = False
    for t in range(horizon):
        action, raw = ctrl.act(obs)
        grip_raw.append(float(raw[3]))
        obs, _, _, _ = env.step(action)
        g = is_grasping(env)
        if was_grasped and not g and in_any_bin2_compartment(env) is None and can_pos(env)[2] > start[2] + 0.03:
            grasp_lost_outside = True  # can let go while lifted and not above bin2 -> dropped
        was_grasped |= g
        max_lift = max(max_lift, can_pos(env)[2] - start[2])
        if record:
            frames.append(np.concatenate([get_image(obs, c) for c in cameras], axis=1))
        success = checker.update(env)
        if success:
            break
    comp = in_any_bin2_compartment(env)
    if success:
        mode = "success"
    elif not was_grasped:
        mode = "never_grasped"
    elif can_in_target(env) and is_grasping(env):
        mode = "in_target_not_released"
    elif comp is not None:
        mode = "wrong_compartment"
    elif grasp_lost_outside:
        mode = "dropped"
    elif is_grasping(env):
        mode = "still_holding_outside_target"
    else:
        mode = "other"
    row = dict(seed=seed, success=int(success), length=t + 1, failure_mode=mode,
               can_x0=round(start[0], 4), can_y0=round(start[1], 4), max_can_lift=round(max_lift, 3),
               final_compartment=-1 if comp is None else comp,
               gripper_raw_min=round(min(grip_raw), 2), gripper_raw_max=round(max(grip_raw), 2))
    return row, frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=str(ROOT / "configs/pickplace_can.yaml"))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed-offset", type=int, default=0, help="start at split seed_start + offset")
    p.add_argument("--n-action-steps", type=int, default=50, help="actions executed per predicted chunk")
    p.add_argument("--horizon", type=int, default=None, help="default: env.horizon from config")
    p.add_argument("--videos", type=int, default=5, help="max videos saved per outcome (success / failure)")
    p.add_argument("--device", default="auto")
    p.add_argument("--name", required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    horizon = args.horizon or cfg["env"]["horizon"]
    out = ROOT / "results/eval" / args.name
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(cfg["env"], render_cameras=True)
    ctrl = VLAController(args.checkpoint, cfg["data"]["task"], args.n_action_steps, args.device)
    checker = PlaceSuccess(**cfg["success"])
    seed0 = cfg["data"]["splits"][args.split]["seed_start"] + args.seed_offset

    rows, saved, t0 = [], Counter(), time.time()
    for i in range(args.episodes):
        seed = seed0 + i
        row, frames = run_episode(env, ctrl, checker, seed, horizon, cfg["env"]["camera_names"], record=args.videos > 0)
        rows.append(row)
        tag = "success" if row["success"] else "fail"
        if frames and saved[tag] < args.videos:
            saved[tag] += 1
            imageio.mimsave(out / f"seed{seed}_{row['failure_mode']}.mp4", frames, fps=cfg["env"]["control_freq"])
        print(f"seed {seed}: {row['failure_mode']:28s} len={row['length']:3d}", flush=True)

    with open(out / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n, k = len(rows), sum(r["success"] for r in rows)
    lo, hi = wilson_interval(k, n)
    lat = np.array(ctrl.chunk_times[1:] or ctrl.chunk_times)  # drop first call (warm-up)
    summary = dict(
        checkpoint=str(args.checkpoint), split=args.split, seeds=[seed0, seed0 + n - 1], n_action_steps=args.n_action_steps,
        horizon=horizon, success=k, episodes=n, success_rate=k / n, ci95_wilson=[round(lo, 3), round(hi, 3)],
        failure_modes=dict(Counter(r["failure_mode"] for r in rows)),
        chunk_latency_ms=dict(mean=round(1000 * lat.mean(), 1), p95=round(1000 * np.percentile(lat, 95), 1)),
        device=ctrl.device, wall_time_s=round(time.time() - t0, 1),
    )
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
