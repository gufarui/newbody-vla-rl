"""Verify the frozen-VLA feature interface for residual RL (Milestone 3 pre-check).

    python scripts/check_vla_features.py --checkpoint outputs/sft_v1/checkpoints/010000/pretrained_model

Checks: feature shape; exactly one VLM prefix pass per chunk; features depend on the current observation
(identical for identical obs, different across scenes); VLA weights unchanged; and a linear probe:
can the can's (privileged) xy position be read out of the features? Compared with probing the robot state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from newbody.env.pick_place_env import can_pos, make_env, reset_with_seed
from newbody.policies.vla import VLAController

ROOT = Path(__file__).resolve().parents[1]


def weights_hash(policy) -> str:
    h = hashlib.sha256()
    for name, p in sorted(policy.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def ridge_cv_r2(X, Y, alphas=(1e-2, 1e-1, 1, 10, 100), k=5, seed=0):
    """k-fold CV of ridge regression; returns best (R2 per output, RMSE per output) over alphas."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    folds = np.array_split(idx, k)
    best = None
    for a in alphas:
        pred = np.zeros_like(Y)
        for f in folds:
            tr = np.setdiff1d(idx, f)
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            Xt, Xv = (X[tr] - mu) / sd, (X[f] - mu) / sd
            ym = Y[tr].mean(0)
            W = np.linalg.solve(Xt.T @ Xt + a * np.eye(X.shape[1]), Xt.T @ (Y[tr] - ym))
            pred[f] = Xv @ W + ym
        r2 = 1 - ((Y - pred) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
        rmse = np.sqrt(((Y - pred) ** 2).mean(0))
        if best is None or r2.mean() > best[0].mean():
            best = (r2, rmse, a)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n", type=int, default=120, help="number of scenes for the linear probe")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs/pickplace_can.yaml"))
    cfg["env"]["can_x_range"], cfg["env"]["can_y_range"] = [-0.145, 0.145], [-0.195, 0.195]
    env = make_env(cfg["env"], render_cameras=True)
    ctrl = VLAController(args.checkpoint, cfg["data"]["task"], 10, args.device)
    ctrl.enable_feature_capture()
    h0 = weights_hash(ctrl.policy)
    report = {}

    obs = reset_with_seed(env, 500000)
    calls0 = ctrl.feature_calls
    ref, f1 = ctrl.predict_chunk(obs, 10)
    report["feature_shape"] = list(f1.shape)
    report["ref_chunk_shape"] = list(ref.shape)
    report["vlm_prefix_passes_per_chunk"] = ctrl.feature_calls - calls0
    _, f2 = ctrl.predict_chunk(obs, 10)
    report["same_obs_same_features_maxdiff"] = float(np.abs(f1 - f2).max())

    feats_mean, feats_img, states, cans = [], [], [], []
    for i in range(args.n):
        obs = reset_with_seed(env, 500000 + i)
        ctrl.predict_chunk(obs, 10)
        h, m = ctrl.last_hidden[0], ctrl.last_mask[0, :, 0]
        feats_mean.append((ctrl.last_hidden[0] * ctrl.last_mask[0]).sum(0).cpu().numpy() / float(ctrl.last_mask[0].sum()))
        feats_img.append(h[:128].mean(0).cpu().numpy())  # 2 cameras x 64 image tokens
        states.append(obs["robot0_eef_pos"].copy())
        cans.append(can_pos(env)[:2].copy())
    F, Fi, S, C = map(np.array, (feats_mean, feats_img, states, cans))
    report["different_scenes_feature_std_mean"] = float(F.std(0).mean())
    for name, X in (("pooled_all_tokens_960", F), ("pooled_image_tokens_960", Fi), ("robot_state_only_3", S)):
        r2, rmse, a = ridge_cv_r2(X, C)
        report[f"probe_can_xy_from_{name}"] = dict(R2_x=round(float(r2[0]), 3), R2_y=round(float(r2[1]), 3),
                                                   rmse_cm=[round(float(v) * 100, 2) for v in rmse], alpha=a)
    report["can_xy_std_cm"] = [round(float(v) * 100, 2) for v in C.std(0)]
    report["weights_unchanged"] = weights_hash(ctrl.policy) == h0
    report["chunk_latency_ms_mean"] = round(1000 * float(np.mean(ctrl.chunk_times[2:])), 1)
    out = ROOT / "results/check_vla_features.json"
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
