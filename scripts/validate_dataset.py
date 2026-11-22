"""Validate a recorded demo dataset. Every check prints PASS/FAIL with numbers.

    python scripts/validate_dataset.py --name kinova3_lift_v1 --split train --replay 5

Checks
 1. Loads through the official LeRobotDataset API; schema, fps, episode/frame counts.
 2. Sample tensors: shapes, dtypes, image value range, channel-first layout.
 3. Image orientation + color order: decoded video frame 0 vs. a fresh render of the same seed,
    compared against the vertically flipped and BGR-swapped alternatives.
 4. Replay (observation/action alignment): reset to the recorded seed, execute the recorded
    actions open-loop, and compare every resulting state to the recorded observation.state.
    If (o_t, a_t) were misaligned by even one step, this diverges immediately.
 5. Success labels: replayed episodes must end in strict hold-success.
 6. No privileged keys in the dataset; stats.json consistent with the data.
Also writes plots to results/dataset/<name>/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from newbody.data import schema
from newbody.env.lift_env import get_image, make_lift_env, reset_with_seed
from newbody.evaluation.success import HoldSuccess

ROOT = Path(__file__).resolve().parents[1]
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--replay", type=int, default=5, help="number of episodes to replay")
    args = p.parse_args()

    data_dir = ROOT / "data" / args.name
    cfg = yaml.safe_load(open(data_dir / "collection_config.yaml"))
    sim_rows = [json.loads(line) for line in open(data_dir / f"{args.split}_sim.jsonl")]
    ds = LeRobotDataset(repo_id=f"local/{args.name}_{args.split}", root=data_dir / args.split)
    out = ROOT / "results/dataset" / args.name
    out.mkdir(parents=True, exist_ok=True)

    # 1. schema
    meta = ds.meta
    check("loads via LeRobotDataset", True, f"{meta.total_episodes} episodes, {meta.total_frames} frames, fps={meta.fps}")
    check("fps == control_freq", meta.fps == cfg["env"]["control_freq"])
    check("episode count matches sidecar", meta.total_episodes == len(sim_rows))
    check("frame count matches sidecar", meta.total_frames == sum(r["length"] for r in sim_rows))
    expected = set(schema.features(cfg["env"]["camera_size"]))
    policy_keys = {k for k in meta.features if k.startswith("observation") or k.startswith("action")}
    check("no extra/privileged policy keys", policy_keys == expected, f"{sorted(policy_keys)}")

    # 2. sample tensors
    item = ds[0]
    img = item["observation.images.agentview"]
    check("image tensor CHW float in [0,1]", img.shape == (3, cfg["env"]["camera_size"], cfg["env"]["camera_size"])
          and 0.0 <= float(img.min()) and float(img.max()) <= 1.0, f"shape={tuple(img.shape)} range=[{img.min():.2f},{img.max():.2f}]")
    check("state/action shapes", tuple(item["observation.state"].shape) == (9,) and tuple(item["action"].shape) == (4,))
    check("task string", item["task"] == schema.TASK, repr(item["task"]))

    # 3 + 4 + 5. orientation, replay, success
    env = make_lift_env(cfg["env"], render_cameras=True)
    checker = HoldSuccess(**cfg["success"])
    ep_from = ds.meta.episodes["dataset_from_index"]
    ep_to = ds.meta.episodes["dataset_to_index"]
    worst_state_err, replay_success = 0.0, 0
    n_replay = min(args.replay, len(sim_rows))
    for ep in range(n_replay):
        seed = sim_rows[ep]["seed"]
        idx = range(int(ep_from[ep]), int(ep_to[ep]))
        rec_state = np.stack([ds.hf_dataset[i]["observation.state"].numpy() for i in idx])
        rec_action = np.stack([ds.hf_dataset[i]["action"].numpy() for i in idx])
        obs = reset_with_seed(env, seed)
        checker.reset()
        if ep == 0:
            for cam, key in schema.CAMERAS.items():
                decoded = (ds[int(ep_from[0])][key].permute(1, 2, 0).numpy() * 255).astype(np.float32)
                fresh = get_image(obs, cam).astype(np.float32)
                d_ok = np.abs(decoded - fresh).mean()
                d_flip = np.abs(decoded - fresh[::-1]).mean()
                check(f"{cam}: orientation", d_ok < 0.3 * d_flip,
                      f"mean |diff| as-recorded={d_ok:.1f}  if flipped={d_flip:.1f} (0-255)")
                # Color order: the scene is mostly white/gray, so test only saturated pixels (red cube)
                sat = np.abs(fresh[..., 0] - fresh[..., 2]) > 60
                c_ok = np.abs(decoded - fresh)[sat].mean()
                c_bgr = np.abs(decoded - fresh[..., ::-1])[sat].mean()
                check(f"{cam}: RGB channel order", sat.sum() > 20 and c_ok < 0.3 * c_bgr,
                      f"on {sat.sum()} saturated px: |diff| RGB={c_ok:.1f}  if BGR={c_bgr:.1f}")
                plt.imsave(out / f"frame0_{cam}.png", decoded.astype(np.uint8))
        errs = []
        for t, a in enumerate(rec_action):
            errs.append(np.abs(schema.state_vector(obs) - rec_state[t]).max())
            obs, _, _, _ = env.step(a)
            ok = checker.update(env)
        worst_state_err = max(worst_state_err, max(errs))
        replay_success += int(ok)
    check("replay reproduces recorded states (alignment)", worst_state_err < 1e-4,
          f"max |state diff| over {n_replay} episodes = {worst_state_err:.2e}")
    check("replayed episodes end in hold-success", replay_success == n_replay, f"{replay_success}/{n_replay}")

    # 6. stats + plots
    actions = np.stack(ds.hf_dataset["action"]).astype(np.float32)
    states = np.stack(ds.hf_dataset["observation.state"]).astype(np.float32)
    st = meta.stats
    check("stats.json action mean matches data", np.allclose(np.asarray(st["action"]["mean"]), actions.mean(0), atol=1e-3))
    check("actions within [-1, 1]", np.abs(actions).max() <= 1.0 + 1e-6)

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for d, n in enumerate(schema.ACTION_NAMES):
        ax.flat[d].hist(actions[:, d], bins=50)
        ax.flat[d].set_title(f"action.{n}  mean={actions[:, d].mean():.2f} std={actions[:, d].std():.2f}")
    lengths = [r["length"] for r in sim_rows]
    ax.flat[4].hist(lengths, bins=20)
    ax.flat[4].set_title(f"episode length (steps)  mean={np.mean(lengths):.1f}")
    xy = np.array([r["cube_pos0"][:2] for r in sim_rows])
    ax.flat[5].scatter(xy[:, 0], xy[:, 1], s=8)
    ax.flat[5].set_title("initial cube xy (m)")
    ax.flat[5].set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out / f"{args.split}_distributions.png", dpi=80)

    ep0 = range(int(ep_from[0]), int(ep_to[0]))
    a0 = actions[list(ep0)]
    s0 = states[list(ep0)]
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for d, n in enumerate(schema.ACTION_NAMES):
        ax[0].plot(a0[:, d], label=n)
    ax[0].set_ylabel("action")
    ax[0].legend()
    for d in range(3):
        ax[1].plot(s0[:, d], label=schema.STATE_NAMES[d])
    ax[1].set_ylabel("eef position (m)")
    ax[1].set_xlabel("step (20 Hz)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / f"{args.split}_episode0_timeseries.png", dpi=80)

    n_fail = sum(not ok for _, ok in results)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed. Plots -> {out.relative_to(ROOT)}")
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
