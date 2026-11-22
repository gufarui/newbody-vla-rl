"""Run the scripted FSM expert on seeded PickPlaceCan episodes and report success rate.

    python scripts/run_expert.py --episodes 50 --videos 3
    python scripts/run_expert.py --episodes 20 --set carry_z=1.0 --name lower_carry
Outputs in results/expert/<name>/:
    episodes.csv   one row per episode (seed, success, length, failure reason, ...)
    summary.json   config + aggregate success rate
    steps_seed*.csv + seed*_{success,fail}.mp4 for recorded episodes (all failures + first N successes)
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

from newbody.env.pick_place_env import (can_in_target, can_pos, can_target, get_image, is_grasping, make_env,
                                        reset_with_seed)
from newbody.evaluation.success import PlaceSuccess
from newbody.expert.fsm_expert import ExpertConfig, PickPlaceFSMExpert

ROOT = Path(__file__).resolve().parents[1]


def run_episode(env, expert, checker, seed, env_cfg, record=True):
    obs = reset_with_seed(env, seed)
    expert.reset(goal=can_target(env))
    checker.reset()
    start = can_pos(env).copy()
    frames, steps = [], []
    success, was_grasped, max_can_z = False, False, start[2]
    for t in range(env_cfg["horizon"]):
        action = expert.act(obs)
        phase = expert.phase.value
        obs, _, _, _ = env.step(action)
        success = checker.update(env)
        grasping = is_grasping(env)
        was_grasped |= grasping
        cp = can_pos(env)
        max_can_z = max(max_can_z, cp[2])
        if record:
            frames.append(np.concatenate([get_image(obs, c) for c in env_cfg["camera_names"]], axis=1))
            e = obs["robot0_eef_pos"]
            steps.append(dict(t=t, phase=phase, ax=action[0], ay=action[1], az=action[2], grip=action[3],
                              eef_x=e[0], eef_y=e[1], eef_z=e[2], can_x=cp[0], can_y=cp[1], can_z=cp[2],
                              grasping=int(grasping), can_in_target=int(can_in_target(env))))
        if success or expert.timed_out_phase is not None:
            break
    if success:
        reason = ""
    elif not was_grasped:
        reason = "never_grasped"
    elif expert.timed_out_phase is not None:
        reason = f"timeout_{expert.timed_out_phase.value}"
    else:
        reason = f"horizon_in_{expert.phase.value}"
    row = dict(seed=seed, success=int(success), length=t + 1, fail_reason=reason, final_phase=expert.phase.value,
               can_x0=round(start[0], 4), can_y0=round(start[1], 4), max_can_lift=round(max_can_z - start[2], 3))
    return row, frames, steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs/pickplace_can.yaml"))
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--videos", type=int, default=2, help="record first N successes (all failures are recorded too)")
    p.add_argument("--set", nargs="*", default=[], help="override expert params, e.g. carry_z=1.0")
    p.add_argument("--name", default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    for kv in args.set:
        k, v = kv.split("=")
        cfg["expert"][k] = type(cfg["expert"][k])(v)

    out = ROOT / "results/expert" / (args.name or time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(cfg["env"], render_cameras=True)
    expert = PickPlaceFSMExpert(ExpertConfig(**cfg["expert"]))
    checker = PlaceSuccess(**cfg["success"])

    rows, n_succ_videos, t0 = [], 0, time.time()
    for i in range(args.episodes):
        seed = args.seed_start + i
        row, frames, steps = run_episode(env, expert, checker, seed, cfg["env"])
        rows.append(row)
        if (not row["success"]) or n_succ_videos < args.videos:
            tag = "success" if row["success"] else "fail"
            n_succ_videos += row["success"]
            imageio.mimsave(out / f"seed{seed}_{tag}.mp4", frames, fps=cfg["env"]["control_freq"])
            with open(out / f"steps_seed{seed}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(steps[0].keys()))
                w.writeheader()
                w.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()} for s in steps)
        print(f"seed {seed:4d}  {'OK  ' if row['success'] else 'FAIL'}  len={row['length']:3d}  {row['fail_reason']}", flush=True)

    with open(out / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n, k = len(rows), sum(r["success"] for r in rows)
    lengths = [r["length"] for r in rows if r["success"]]
    summary = dict(success=k, episodes=n, success_rate=k / n, seeds=[args.seed_start, args.seed_start + n - 1],
                   mean_success_length=float(np.mean(lengths)) if lengths else None,
                   wall_time_s=round(time.time() - t0, 1), config=cfg)
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(f"\nSuccess {k}/{n} = {k / n:.1%}   -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
