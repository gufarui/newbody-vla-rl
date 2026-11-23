"""Check: residual env with δ = 0 executes exactly what the SFT baseline evaluation executes.

Baseline path  = VLAController.act (select_action queue, 10 actions per chunk), as in scripts/eval_vla.py
Residual path  = ResidualChunkEnv.step(zeros) (predict_chunk every 10 steps)
Same scene seed and same torch seed (flow-matching noise) -> executed actions must be identical.
"""
from pathlib import Path

import numpy as np
import torch
import yaml

from newbody.env.pick_place_env import make_env, reset_with_seed
from newbody.evaluation.success import PlaceSuccess
from newbody.policies.vla import VLAController
from newbody.rl.residual_env import ResidualChunkEnv, ResidualConfig

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "configs/pickplace_can.yaml"))
rl = cfg["rl"]
cfg["env"]["can_x_range"], cfg["env"]["can_y_range"] = rl["can_x_range"], rl["can_y_range"]
env = make_env(cfg["env"], render_cameras=True)
vla = VLAController(ROOT / rl["vla_checkpoint"], cfg["data"]["task"], 10)
vla.enable_feature_capture()
renv = ResidualChunkEnv(env, vla, cfg["success"], ResidualConfig(**rl["residual"]), cfg["env"]["bin1_pos"])

for seed in (450000, 450001):
    torch.manual_seed(123)
    obs = reset_with_seed(env, seed)
    vla.reset()
    chk = PlaceSuccess(**cfg["success"])
    base, base_ok = [], False
    for t in range(300):
        a, _ = vla.act(obs)
        base.append(a)
        obs, *_ = env.step(a)
        if chk.update(env):
            base_ok = True
            break
    torch.manual_seed(123)
    renv.reset(seed)
    res = []
    while True:
        *_, term, trunc, info = renv.step(np.zeros(3))
        res.extend(info["executed"])
        if term or trunc:
            break
    n = min(len(base), len(res))
    diff = np.abs(np.array(base[:n]) - np.array(res[:n])).max()
    print(f"seed {seed}: baseline steps={len(base)} success={base_ok} | residual(δ=0) steps={len(res)} "
          f"success={info['success']} | max |action diff| over {n} steps = {diff:.2e}", flush=True)
