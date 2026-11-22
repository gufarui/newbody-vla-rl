"""Record scripted-expert demonstrations into a LeRobot dataset + privileged sidecar.

Output layout (root = data/<name>/<split>):
    <root>/                  LeRobot dataset (videos, parquet, meta/) -- what the VLA trains on
    <root>/../<split>_sim.jsonl   one line per kept episode: seed, cube pose, length, ...
    <root>/../<split>_sim/ep{i}.npz   per-step privileged state (cube pos) for replay checks
Only successful episodes (strict PlaceSuccess) are written to the dataset; failures are
counted and logged in the sidecar's `rejected` list.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from newbody.data import schema
from newbody.env.pick_place_env import can_pos, can_target, make_env, reset_with_seed
from newbody.evaluation.success import PlaceSuccess
from newbody.expert.fsm_expert import ExpertConfig, PickPlaceFSMExpert


def rollout_expert(env, expert, checker, seed: int, horizon: int, task: str):
    """Run one episode. Returns (frames, privileged per-step dict, success, info)."""
    obs = reset_with_seed(env, seed)
    goal = can_target(env)
    expert.reset(goal=goal)
    checker.reset()
    info = dict(seed=seed, can_pos0=can_pos(env).round(5).tolist(), target=goal.round(5).tolist())
    frames, can_traj, eef_traj = [], [], []
    success = False
    for _ in range(horizon):
        action = expert.act(obs)
        frames.append(schema.make_frame(obs, action, task))  # (o_t, a_t)
        can_traj.append(can_pos(env).copy())
        eef_traj.append(obs["robot0_eef_pos"].copy())
        obs, _, _, _ = env.step(action)
        success = checker.update(env)
        if success or expert.timed_out_phase is not None:
            break
    info["length"] = len(frames)
    priv = dict(can_pos=np.array(can_traj), eef_pos=np.array(eef_traj), final_can_pos=can_pos(env).copy())
    return frames, priv, success, info


def record(cfg: dict, out_dir: Path, split: str, n_episodes: int, seed_start: int, max_attempts: int | None = None):
    env = make_env(cfg["env"], render_cameras=True)
    expert = PickPlaceFSMExpert(ExpertConfig(**cfg["expert"]))
    checker = PlaceSuccess(**cfg["success"])
    root = out_dir / split
    ds = LeRobotDataset.create(
        repo_id=f"local/{out_dir.name}_{split}", fps=cfg["env"]["control_freq"], root=root,
        features=schema.features(cfg["env"]["camera_size"]), robot_type="kinova3_robotiq85_osc_position",
        use_videos=True, image_writer_threads=4,
    )
    sim_dir = out_dir / f"{split}_sim"
    sim_dir.mkdir(parents=True, exist_ok=True)
    kept, rejected, seed = [], [], seed_start
    max_attempts = max_attempts or 2 * n_episodes
    while len(kept) < n_episodes and seed < seed_start + max_attempts:
        frames, priv, success, info = rollout_expert(env, expert, checker, seed, cfg["env"]["horizon"], cfg["data"]["task"])
        if success:
            for f in frames:
                ds.add_frame(f)
            ds.save_episode()
            info["episode_index"] = len(kept)
            np.savez_compressed(sim_dir / f"ep{len(kept)}.npz", **priv)
            kept.append(info)
        else:
            rejected.append(info)
        print(f"[{split}] seed {seed}: {'kept' if success else 'REJECTED'} len={info['length']}  ({len(kept)}/{n_episodes})", flush=True)
        seed += 1
    ds.finalize()
    with open(out_dir / f"{split}_sim.jsonl", "w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")
    summary = dict(split=split, kept=len(kept), rejected=len(rejected), attempts=len(kept) + len(rejected),
                   seed_range=[seed_start, seed - 1], rejected_seeds=[r["seed"] for r in rejected])
    json.dump(summary, open(out_dir / f"{split}_summary.json", "w"), indent=2)
    env.close()
    return summary
