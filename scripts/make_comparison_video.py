"""Side-by-side video: SFT-only VLA vs SFT + residual SAC on the same scene (illustrative, not a statistic).

    python scripts/make_comparison_video.py --seeds 200003 200006 ...
Runs both policies locally on each seed with full frame recording; keeps the seeds where the local rerun
reproduces "SFT fails, SFT+RL succeeds" (the VLA's flow-matching noise makes reruns stochastic).
Writes results/videos/compare_seed<seed>.mp4 and a static frame strip of the first kept seed to docs/figures.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from newbody.env.pick_place_env import get_image, make_env, reset_with_seed
from newbody.evaluation.success import PlaceSuccess
from newbody.policies.vla import VLAController
from newbody.rl.residual_env import ResidualChunkEnv, ResidualConfig
from newbody.rl.sac import OBS_KEYS, SAC, SACConfig

ROOT = Path(__file__).resolve().parents[1]
CAM = "agentview"


def label(frame, text, ok):
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, im.width, 18), fill=(252, 252, 251))
    d.text((4, 3), text, fill=(26, 26, 25))
    if ok is not None:
        d.rectangle((0, im.height - 18, im.width, im.height), fill=(42, 120, 214) if ok else (95, 94, 88))
        d.text((4, im.height - 15), "SUCCESS" if ok else "FAILED", fill=(252, 252, 251))
    return np.array(im)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--keep", type=int, default=2)
    p.add_argument("--rl-checkpoint", default="ep00499.pt")
    args = p.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs/pickplace_can.yaml"))
    rl = cfg["rl"]
    cfg["env"]["can_x_range"], cfg["env"]["can_y_range"] = rl["can_x_range"], rl["can_y_range"]
    cfg["env"]["camera_size"] = 256
    env = make_env(cfg["env"], render_cameras=True)
    vla = VLAController(ROOT / rl["vla_checkpoint"], cfg["data"]["task"], 10)
    vla.enable_feature_capture()
    renv = ResidualChunkEnv(env, vla, cfg["success"], ResidualConfig(**rl["residual"]), cfg["env"]["bin1_pos"])
    o, pr = renv.reset(args.seeds[0])
    sac = SAC({k: int(np.asarray(o[k]).size) for k in OBS_KEYS}, 3, pr.size, SACConfig(**rl["sac"]), vla.device)
    sac.load_state_dict(torch.load(ROOT / "outputs/rl_v1/checkpoints" / args.rl_checkpoint, weights_only=False, map_location="cpu")["sac"])
    out = ROOT / "results/videos"
    out.mkdir(parents=True, exist_ok=True)
    kept = 0
    for seed in args.seeds:
        # SFT only (same loop as scripts/eval_vla.py)
        obs = reset_with_seed(env, seed)
        vla.reset()
        chk = PlaceSuccess(**cfg["success"])
        a_frames, a_ok = [], False
        for _ in range(cfg["env"]["horizon"]):
            act, _ = vla.act(obs)
            obs, *_ = env.step(act)
            a_frames.append(get_image(obs, CAM))
            if chk.update(env):
                a_ok = True
                break
        # SFT + residual SAC (deterministic actor)
        renv.record_camera = CAM
        o, _ = renv.reset(seed)
        while True:
            o, _, _, term, trunc, info = renv.step(sac.act(o, deterministic=True))
            if term or trunc:
                break
        b_frames, b_ok = list(renv.frames), info["success"]
        print(f"seed {seed}: SFT only {'OK' if a_ok else 'FAIL'} ({len(a_frames)} steps) | "
              f"SFT+RL {'OK' if b_ok else 'FAIL'} ({len(b_frames)} steps)", flush=True)
        if a_ok or not b_ok:
            continue
        n = max(len(a_frames), len(b_frames))
        a_frames += [a_frames[-1]] * (n - len(a_frames))
        b_frames += [b_frames[-1]] * (n - len(b_frames))
        video = [np.concatenate([label(a, "SFT only (frozen VLA)", a_ok if i == n - 1 else None),
                                 label(b, "SFT + residual SAC", b_ok if i == n - 1 else None)], axis=1)
                 for i, (a, b) in enumerate(zip(a_frames, b_frames))]
        video += [video[-1]] * 20  # hold the final frame for 1 s
        imageio.mimsave(out / f"compare_seed{seed}.mp4", video, fps=20)
        if kept == 0:  # small static strip for the report: 4 moments, both policies side by side
            idx = [0, n // 4, n // 2, n - 1]
            strip = np.concatenate([video[i] for i in idx], axis=0)
            Image.fromarray(strip).resize((strip.shape[1] // 2, strip.shape[0] // 2)).save(ROOT / "docs/figures/m4_compare_strip.png")
        kept += 1
        if kept >= args.keep:
            break


if __name__ == "__main__":
    main()
