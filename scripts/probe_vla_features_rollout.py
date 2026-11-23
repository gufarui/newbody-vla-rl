"""Probe: at each RL decision point (every 10 steps while the VLA acts), can frozen VLA features predict the
can's xy offset RELATIVE TO THE GRIPPER (what a residual must correct)? Compares token pooling choices.

    python scripts/probe_vla_features_rollout.py --checkpoint outputs/sft_v1/checkpoints/010000/pretrained_model
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from check_vla_features import ridge_cv_r2
from newbody.env.pick_place_env import can_pos, is_grasping, make_env, reset_with_seed
from newbody.policies.vla import VLAController

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max-chunks", type=int, default=6, help="decision points per episode (approach/descent phase)")
    args = p.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs/pickplace_can.yaml"))
    cfg["env"]["can_x_range"], cfg["env"]["can_y_range"] = [-0.145, 0.145], [-0.195, 0.195]
    env = make_env(cfg["env"], render_cameras=True)
    ctrl = VLAController(args.checkpoint, cfg["data"]["task"], 10)
    ctrl.enable_feature_capture()
    X = {"all_tokens": [], "front_cam_tokens": [], "wrist_cam_tokens": [], "robot_state": []}
    Y = []
    for ep in range(args.episodes):
        obs = reset_with_seed(env, 510000 + ep)
        for _ in range(args.max_chunks):
            if is_grasping(env):
                break  # only pre-grasp decisions matter for the xy-offset failure
            ref, f = ctrl.predict_chunk(obs, 10)
            h = ctrl.last_hidden[0]
            X["all_tokens"].append(f)
            X["front_cam_tokens"].append(h[:64].mean(0).cpu().numpy())
            X["wrist_cam_tokens"].append(h[64:128].mean(0).cpu().numpy())
            X["robot_state"].append(obs["robot0_eef_pos"].copy())
            Y.append((can_pos(env) - obs["robot0_eef_pos"])[:2])
            for a in ref:
                obs, *_ = env.step(a)
    Y = np.array(Y)
    rep = {"samples": len(Y), "target": "can xy minus gripper xy (m)", "target_std_cm": (Y.std(0) * 100).round(2).tolist()}
    for k, v in X.items():
        r2, rmse, a = ridge_cv_r2(np.array(v), Y)
        rep[k] = dict(R2_x=round(float(r2[0]), 3), R2_y=round(float(r2[1]), 3), rmse_cm=(rmse * 100).round(2).tolist())
    json.dump(rep, open(ROOT / "results/probe_vla_features_rollout.json", "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
