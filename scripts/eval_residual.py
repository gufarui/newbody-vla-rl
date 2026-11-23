"""Final evaluation of SFT VLA + residual SAC (deterministic actor) on the same seeds as the SFT baseline.

    python scripts/eval_residual.py --run rl_v1 --checkpoint ep01499.pt --split test --episodes 200 \
        --baseline results/eval/sft_v1_010000_testwide_k10/episodes.csv --name rl_v1_test

Output results/eval/<name>/: episodes.csv, summary.json (success + Wilson CI, inside/outside demo region,
failure modes, residual magnitude, paired comparison with the baseline on identical seeds), MP4s.
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
import torch
import yaml
from scipy.stats import binomtest

from newbody.env.pick_place_env import can_in_target, get_image, is_grasping, make_env
from newbody.evaluation.stats import wilson_interval
from newbody.policies.vla import VLAController
from newbody.rl.residual_env import ResidualChunkEnv, ResidualConfig
from newbody.rl.sac import OBS_KEYS, SAC, SACConfig

ROOT = Path(__file__).resolve().parents[1]


def classify(renv, info) -> str:
    env = renv.env
    if info["success"]:
        return "success"
    if info["tipped"]:
        return "tipped"
    if not renv.was_grasped:
        return "never_grasped"
    if can_in_target(env) and is_grasping(env):
        return "in_target_not_released"
    if any(not env.not_in_bin(np.array(env.sim.data.body_xpos[env.obj_body_id[env.objects[env.object_id].name]]), i)
           for i in range(4)):
        return "wrong_compartment"
    if renv.dropped:
        return "dropped"
    return "still_holding_outside_target" if is_grasping(env) else "other"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoint", default=None, help="file in outputs/<run>/checkpoints (default: latest)")
    p.add_argument("--split", default="test", choices=["val", "test", "rlval"])
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--baseline", default=None, help="baseline episodes.csv on the same seeds for a paired comparison")
    p.add_argument("--zero-residual", action="store_true", help="sanity: run the same loop with δ = 0")
    p.add_argument("--videos", type=int, default=8)
    p.add_argument("--name", required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs/pickplace_can.yaml"))
    rl = cfg["rl"]
    cfg["env"]["can_x_range"], cfg["env"]["can_y_range"] = rl["can_x_range"], rl["can_y_range"]
    seed0 = rl["val_seed_start"] if args.split == "rlval" else cfg["data"]["splits"][args.split]["seed_start"]
    out = ROOT / "results/eval" / args.name
    out.mkdir(parents=True, exist_ok=True)

    vla = VLAController(ROOT / rl["vla_checkpoint"], cfg["data"]["task"], rl["residual"]["chunk_steps"])
    vla.enable_feature_capture()
    env = make_env(cfg["env"], render_cameras=True)
    renv = ResidualChunkEnv(env, vla, cfg["success"], ResidualConfig(**rl["residual"]), cfg["env"]["bin1_pos"])
    o, pr = renv.reset(seed0)
    dims = {k: int(np.asarray(o[k]).size) for k in OBS_KEYS}
    sac = SAC(dims, ResidualChunkEnv.ACT_DIM, pr.size, SACConfig(**rl["sac"]), vla.device)
    ck_dir = ROOT / "outputs" / args.run / "checkpoints"
    ck_name = args.checkpoint or (ck_dir / "latest.txt").read_text().strip()
    sac.load_state_dict(torch.load(ck_dir / ck_name, weights_only=False, map_location="cpu")["sac"])

    rows, saved, t0 = [], Counter(), time.time()
    for i in range(args.episodes):
        seed = seed0 + i
        o, _ = renv.reset(seed)
        frames, deltas = [], []
        while True:
            d = np.zeros(3) if args.zero_residual else sac.act(o, deterministic=True)
            deltas.append(np.abs(d))
            o, _, _, term, trunc, info = renv.step(d)
            if args.videos:
                frames.append(np.concatenate([get_image(renv._obs, c) for c in cfg["env"]["camera_names"]], axis=1))
            if term or trunc:
                break
        mode = classify(renv, info)
        rows.append(dict(seed=seed, success=int(info["success"]), failure_mode=mode, env_steps=info["env_steps"],
                         inside_demo_region=int(renv.inside_demo_region),
                         mean_abs_delta=round(float(np.mean(deltas)), 4), max_abs_delta=round(float(np.max(deltas)), 4)))
        tag = "success" if info["success"] else "fail"
        if frames and saved[tag] < args.videos:  # note: frames are chunk ends only (1 per decision)
            saved[tag] += 1
            imageio.mimsave(out / f"seed{seed}_{mode}.mp4", frames, fps=4)
        print(f"seed {seed}: {mode:28s} steps={info['env_steps']:3d} |δ|={rows[-1]['mean_abs_delta']}", flush=True)

    with open(out / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n, k = len(rows), sum(r["success"] for r in rows)
    ins = [r for r in rows if r["inside_demo_region"]]
    outs = [r for r in rows if not r["inside_demo_region"]]
    summary = dict(run=args.run, checkpoint=ck_name, split=args.split, seeds=[seed0, seed0 + n - 1],
                   zero_residual=args.zero_residual, success=k, episodes=n, success_rate=k / n,
                   ci95_wilson=[round(x, 3) for x in wilson_interval(k, n)],
                   inside=[sum(r["success"] for r in ins), len(ins)], outside=[sum(r["success"] for r in outs), len(outs)],
                   failure_modes=dict(Counter(r["failure_mode"] for r in rows)),
                   mean_abs_delta=round(float(np.mean([r["mean_abs_delta"] for r in rows])), 4),
                   wall_time_s=round(time.time() - t0, 1))
    if args.baseline:
        base = {int(r["seed"]): int(r["success"]) for r in csv.DictReader(open(ROOT / args.baseline))}
        pairs = [(base[r["seed"]], r["success"]) for r in rows if r["seed"] in base]
        only_rl = sum(1 for b, x in pairs if x and not b)
        only_base = sum(1 for b, x in pairs if b and not x)
        p_mcnemar = binomtest(only_rl, only_rl + only_base, 0.5).pvalue if only_rl + only_base else 1.0
        summary["paired_vs_baseline"] = dict(pairs=len(pairs), baseline_success=sum(b for b, _ in pairs),
                                             rl_success=sum(x for _, x in pairs), only_rl=only_rl,
                                             only_baseline=only_base, mcnemar_exact_p=round(p_mcnemar, 4))
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
