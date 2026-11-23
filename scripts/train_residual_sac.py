"""Train the residual SAC policy on top of the frozen SFT SmolVLA (Milestone 3).

    python scripts/train_residual_sac.py --name rl_v1                 # full run (config: rl section)
    python scripts/train_residual_sac.py --name rl_v1 --resume        # continue after an interruption
    python scripts/train_residual_sac.py --name smoke --episodes 6 --learning-starts 20 --val-episodes 2

Outputs in outputs/<name>/: episodes.csv (one row per training episode), updates.csv (SAC stats),
evals.csv (deterministic actor on RL-val seeds), checkpoints/ (actor/critics/optimizers + replay buffer),
run_info.json (config, VLA weight hashes before/after).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from newbody.env.pick_place_env import make_env
from newbody.policies.vla import VLAController
from newbody.rl.residual_env import ResidualChunkEnv, ResidualConfig
from newbody.rl.sac import OBS_KEYS, SAC, ReplayBuffer, SACConfig

ROOT = Path(__file__).resolve().parents[1]


def weights_hash(module) -> str:
    h = hashlib.sha256()
    for name, p in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


class CsvLog:
    def __init__(self, path: Path, append: bool):
        self.path, self.append, self.writer, self.f = path, append, None, None

    def write(self, row: dict):
        if self.writer is None:
            exists = self.path.exists() and self.append
            self.f = open(self.path, "a" if exists else "w", newline="")
            self.writer = csv.DictWriter(self.f, fieldnames=list(row.keys()))
            if not exists:
                self.writer.writeheader()
        self.writer.writerow(row)
        self.f.flush()


def run_episode(renv, sac, seed, deterministic, buf=None, learn=None):
    o, p = renv.reset(seed)
    ret, deltas, info = 0.0, [], {}
    while True:
        a = sac.act(o, deterministic=deterministic)
        o2, p2, r, term, trunc, info = renv.step(a)
        if buf is not None:
            buf.add(o, p, a, r, o2, p2, term)  # bootstrap through truncation, not through termination
            if learn is not None:
                learn()
        ret += r
        deltas.append(np.abs(a))
        o, p = o2, p2
        if term or trunc:
            break
    d = np.array(deltas)
    return dict(success=int(info["success"]), tipped=int(info["tipped"]), ret=round(ret, 4), decisions=len(d),
                env_steps=info["env_steps"], mean_abs_delta=round(float(d.mean()), 4),
                max_abs_delta=round(float(d.max()), 4), inside_demo_region=int(info["inside_demo_region"]))


def save_ckpt(out: Path, sac: SAC, buf: ReplayBuffer, episode: int):
    ck = out / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    torch.save(dict(sac=sac.state_dict(), episode=episode, np_rng=np.random.get_state(),
                    torch_rng=torch.get_rng_state(), py_rng=random.getstate()), ck / f"ep{episode:05d}.pt")
    n = buf.n
    np.savez(ck / "replay.npz", n=n, i=buf.i, p=buf.p[:n], p2=buf.p2[:n], a=buf.a[:n], r=buf.r[:n], d=buf.d[:n],
             **{f"o_{k}": buf.o[k][:n] for k in OBS_KEYS}, **{f"o2_{k}": buf.o2[k][:n] for k in OBS_KEYS})
    (ck / "latest.txt").write_text(f"ep{episode:05d}.pt")


def load_ckpt(out: Path, sac: SAC, buf: ReplayBuffer) -> int:
    ck = out / "checkpoints"
    s = torch.load(ck / (ck / "latest.txt").read_text().strip(), weights_only=False)
    sac.load_state_dict(s["sac"])
    np.random.set_state(s["np_rng"])
    torch.set_rng_state(s["torch_rng"])
    random.setstate(s["py_rng"])
    z = np.load(ck / "replay.npz")
    n = int(z["n"])
    for k in ("p", "p2", "a", "r", "d"):
        getattr(buf, k)[:n] = z[k]
    for k in OBS_KEYS:
        buf.o[k][:n], buf.o2[k][:n] = z[f"o_{k}"], z[f"o2_{k}"]
    buf.n, buf.i = n, int(z["i"])
    return int(s["episode"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs/pickplace_can.yaml"))
    p.add_argument("--name", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--learning-starts", type=int, default=None)
    p.add_argument("--val-episodes", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    rl = cfg["rl"]
    for k, v in (("total_episodes", args.episodes), ("learning_starts", args.learning_starts),
                 ("val_episodes", args.val_episodes), ("eval_every_episodes", args.eval_every)):
        if v is not None:
            rl[k] = v
    cfg["env"]["can_x_range"], cfg["env"]["can_y_range"] = rl["can_x_range"], rl["can_y_range"]
    out = ROOT / "outputs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    vla = VLAController(ROOT / rl["vla_checkpoint"], cfg["data"]["task"], rl["residual"]["chunk_steps"], args.device)
    vla.enable_feature_capture()
    for prm in vla.policy.parameters():
        prm.requires_grad_(False)
    vla_hash_before = weights_hash(vla.policy)
    env = make_env(cfg["env"], render_cameras=True)
    renv = ResidualChunkEnv(env, vla, cfg["success"], ResidualConfig(**rl["residual"]), cfg["env"]["bin1_pos"])

    o, pr = renv.reset(rl["train_seed_start"] - 1)  # probe shapes (seed outside every range used below)
    dims = {k: int(np.asarray(o[k]).size) for k in OBS_KEYS}
    sac = SAC(dims, ResidualChunkEnv.ACT_DIM, pr.size, SACConfig(**rl["sac"]), vla.device)
    buf = ReplayBuffer(rl["buffer_size"], dims, ResidualChunkEnv.ACT_DIM, pr.size)
    start_ep = load_ckpt(out, sac, buf) + 1 if args.resume else 0
    json.dump(dict(config=cfg, dims=dims, priv_dim=int(pr.size), vla_hash_before=vla_hash_before,
                   resumed_from=start_ep, device=vla.device), open(out / f"run_info_start{start_ep}.json", "w"), indent=2)

    ep_log = CsvLog(out / "episodes.csv", append=args.resume)
    up_log = CsvLog(out / "updates.csv", append=args.resume)
    ev_log = CsvLog(out / "evals.csv", append=args.resume)
    acc: list[dict] = []

    def learn():
        if buf.n < rl["learning_starts"]:
            return
        for _ in range(rl["updates_per_transition"]):
            logs = sac.update(buf)
            if "pi_loss" in logs:
                acc.append(logs)

    def evaluate(episode):
        rows = [run_episode(renv, sac, rl["val_seed_start"] + i, deterministic=True) for i in range(rl["val_episodes"])]
        row = dict(episode=episode, success=sum(r["success"] for r in rows), n=len(rows),
                   success_inside=sum(r["success"] for r in rows if r["inside_demo_region"]),
                   n_inside=sum(r["inside_demo_region"] for r in rows),
                   tipped=sum(r["tipped"] for r in rows),
                   mean_abs_delta=round(float(np.mean([r["mean_abs_delta"] for r in rows])), 4))
        ev_log.write(row)
        print(f"[eval ep {episode}] val success {row['success']}/{row['n']}  inside {row['success_inside']}/{row['n_inside']}", flush=True)

    if start_ep == 0 and rl["val_episodes"] > 0:
        evaluate(0)  # deterministic actor at init ≈ SFT baseline on the RL-val seeds
    t0 = time.time()
    for ep in range(start_ep, rl["total_episodes"]):
        row = run_episode(renv, sac, rl["train_seed_start"] + ep, deterministic=False, buf=buf, learn=learn)
        row = dict(episode=ep, seed=rl["train_seed_start"] + ep, **row, transitions=buf.n, updates=sac.updates,
                   alpha=round(sac.alpha, 5), wall_s=round(time.time() - t0, 1))
        ep_log.write(row)
        if acc:
            keys = acc[0].keys()
            up_log.write(dict(episode=ep, **{k: round(float(np.mean([a[k] for a in acc])), 5) for k in keys}))
            acc.clear()
        if ep % 10 == 0:
            print(f"ep {ep} success={row['success']} tipped={row['tipped']} |δ|={row['mean_abs_delta']} "
                  f"transitions={buf.n} alpha={row['alpha']} wall={row['wall_s']}s", flush=True)
        if (ep + 1) % rl["save_every_episodes"] == 0:
            save_ckpt(out, sac, buf, ep)
        if rl["val_episodes"] > 0 and (ep + 1) % rl["eval_every_episodes"] == 0:
            evaluate(ep + 1)
    save_ckpt(out, sac, buf, rl["total_episodes"] - 1)
    vla_hash_after = weights_hash(vla.policy)
    json.dump(dict(vla_hash_before=vla_hash_before, vla_hash_after=vla_hash_after,
                   vla_unchanged=vla_hash_before == vla_hash_after, episodes=rl["total_episodes"]),
              open(out / "run_end.json", "w"), indent=2)
    print("VLA weights unchanged:", vla_hash_before == vla_hash_after)


if __name__ == "__main__":
    main()
