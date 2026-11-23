"""Soft Actor-Critic with an asymmetric critic, adapted from CleanRL's `sac_continuous_action.py`.

Kept from CleanRL: tanh-squashed Gaussian actor with log-std clamp, twin Q networks + Polyak target networks,
automatic entropy tuning (target entropy = -|A|), delayed policy updates.
Changed for this project:
- Dict observations. Actor sees only deployable inputs: frozen-VLA features (1920), reference chunk (40),
  robot state (5), time (1). Critic additionally sees privileged simulator state (training only).
- Actor output layer initialised near zero -> mean residual ≈ 0 -> the initial policy is the SFT VLA.
- Replay buffer stores features in float16 (they dominate memory).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
OBS_KEYS = ("features", "ref", "state", "time")


@dataclass
class SACConfig:
    feature_proj: int = 128
    hidden: int = 256
    gamma: float = 0.98            # per decision (= per 10 env steps)
    tau: float = 0.005
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    batch_size: int = 256
    policy_frequency: int = 2
    init_log_std: float = -1.2     # initial exploration std ≈ 0.3 in δ units (≈ 3 mm/step target offset)
    autotune: bool = True
    alpha: float = 0.05            # used only if autotune is False


class ObsEncoder(nn.Module):
    """features -> Linear -> LayerNorm -> ReLU, concatenated with the small vectors."""

    def __init__(self, dims: dict, proj: int, extra_dim: int = 0):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dims["features"], proj), nn.LayerNorm(proj), nn.ReLU())
        self.out_dim = proj + dims["ref"] + dims["state"] + dims["time"] + extra_dim

    def forward(self, o: dict, extra: torch.Tensor | None = None):
        parts = [self.proj(o["features"].float()), o["ref"], o["state"], o["time"]]
        if extra is not None:
            parts.append(extra)
        return torch.cat(parts, dim=-1)


class Actor(nn.Module):
    def __init__(self, dims: dict, act_dim: int, cfg: SACConfig):
        super().__init__()
        self.enc = ObsEncoder(dims, cfg.feature_proj)
        self.body = nn.Sequential(nn.Linear(self.enc.out_dim, cfg.hidden), nn.ReLU(),
                                  nn.Linear(cfg.hidden, cfg.hidden), nn.ReLU())
        self.mean = nn.Linear(cfg.hidden, act_dim)
        self.log_std = nn.Linear(cfg.hidden, act_dim)
        nn.init.uniform_(self.mean.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.mean.bias)
        nn.init.uniform_(self.log_std.weight, -1e-3, 1e-3)
        # log_std head is tanh-rescaled to [MIN, MAX]; choose bias so the initial std = exp(init_log_std)
        target = (cfg.init_log_std - LOG_STD_MIN) / (LOG_STD_MAX - LOG_STD_MIN) * 2 - 1
        nn.init.constant_(self.log_std.bias, float(np.arctanh(target)))

    def forward(self, o):
        h = self.body(self.enc(o))
        log_std = torch.tanh(self.log_std(h))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return self.mean(h), log_std

    def sample(self, o):
        mean, log_std = self(o)
        normal = torch.distributions.Normal(mean, log_std.exp())
        x = normal.rsample()
        y = torch.tanh(x)
        log_prob = (normal.log_prob(x) - torch.log(1 - y.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return y, log_prob, torch.tanh(mean)


class QNet(nn.Module):
    def __init__(self, dims: dict, act_dim: int, priv_dim: int, cfg: SACConfig):
        super().__init__()
        self.enc = ObsEncoder(dims, cfg.feature_proj, extra_dim=priv_dim)
        self.body = nn.Sequential(nn.Linear(self.enc.out_dim + act_dim, cfg.hidden), nn.ReLU(),
                                  nn.Linear(cfg.hidden, cfg.hidden), nn.ReLU(), nn.Linear(cfg.hidden, 1))

    def forward(self, o, priv, a):
        return self.body(torch.cat([self.enc(o, priv), a], dim=-1))


class ReplayBuffer:
    def __init__(self, capacity: int, dims: dict, act_dim: int, priv_dim: int):
        self.cap, self.n, self.i = capacity, 0, 0

        def store(d, dtype):
            return np.zeros((capacity, d), dtype=dtype)

        self.o = {k: store(dims[k], np.float16 if k == "features" else np.float32) for k in OBS_KEYS}
        self.o2 = {k: store(dims[k], np.float16 if k == "features" else np.float32) for k in OBS_KEYS}
        self.p, self.p2 = store(priv_dim, np.float32), store(priv_dim, np.float32)
        self.a, self.r, self.d = store(act_dim, np.float32), store(1, np.float32), store(1, np.float32)

    def add(self, o, p, a, r, o2, p2, terminated):
        i = self.i
        for k in OBS_KEYS:
            self.o[k][i], self.o2[k][i] = o[k], o2[k]
        self.p[i], self.p2[i], self.a[i], self.r[i], self.d[i] = p, p2, a, r, float(terminated)
        self.i = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch: int, device):
        idx = np.random.randint(0, self.n, size=batch)

        def t(x):
            return torch.as_tensor(x[idx], device=device).float()

        return ({k: t(self.o[k]) for k in OBS_KEYS}, t(self.p), t(self.a), t(self.r),
                {k: t(self.o2[k]) for k in OBS_KEYS}, t(self.p2), t(self.d))


class SAC:
    def __init__(self, dims: dict, act_dim: int, priv_dim: int, cfg: SACConfig, device: str):
        self.cfg, self.device, self.act_dim = cfg, device, act_dim
        self.actor = Actor(dims, act_dim, cfg).to(device)
        self.q1, self.q2 = QNet(dims, act_dim, priv_dim, cfg).to(device), QNet(dims, act_dim, priv_dim, cfg).to(device)
        self.q1_t, self.q2_t = QNet(dims, act_dim, priv_dim, cfg).to(device), QNet(dims, act_dim, priv_dim, cfg).to(device)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())
        self.q_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.q_lr)
        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.policy_lr)
        self.target_entropy = -float(act_dim)
        self.log_alpha = torch.tensor(float(np.log(cfg.alpha)), dtype=torch.float32, device=device, requires_grad=True)
        self.a_opt = torch.optim.Adam([self.log_alpha], lr=cfg.q_lr)
        self.updates = 0

    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().item()

    def _to_t(self, o):
        return {k: torch.as_tensor(np.asarray(o[k]), device=self.device).float().unsqueeze(0) for k in OBS_KEYS}

    @torch.no_grad()
    def act(self, o, deterministic: bool = False) -> np.ndarray:
        y, _, mean_action = self.actor.sample(self._to_t(o))
        return (mean_action if deterministic else y).squeeze(0).cpu().numpy()

    def update(self, buf: ReplayBuffer) -> dict:
        c = self.cfg
        o, p, a, r, o2, p2, d = buf.sample(c.batch_size, self.device)
        with torch.no_grad():
            a2, logp2, _ = self.actor.sample(o2)
            q_next = torch.min(self.q1_t(o2, p2, a2), self.q2_t(o2, p2, a2)) - self.alpha * logp2
            target = r + (1 - d) * c.gamma * q_next
        q1, q2 = self.q1(o, p, a), self.q2(o, p, a)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()
        logs = dict(q_loss=q_loss.item(), q1_mean=q1.mean().item(), target_mean=target.mean().item())

        if self.updates % c.policy_frequency == 0:
            for _ in range(c.policy_frequency):  # CleanRL: compensate for the delay
                pi, logp, _ = self.actor.sample(o)
                q_pi = torch.min(self.q1(o, p, pi), self.q2(o, p, pi))
                pi_loss = (self.alpha * logp - q_pi).mean()
                self.pi_opt.zero_grad()
                pi_loss.backward()
                self.pi_opt.step()
                if c.autotune:
                    with torch.no_grad():
                        _, logp, _ = self.actor.sample(o)
                    a_loss = (-self.log_alpha.exp() * (logp + self.target_entropy)).mean()
                    self.a_opt.zero_grad()
                    a_loss.backward()
                    self.a_opt.step()
            logs.update(pi_loss=pi_loss.item(), alpha=self.alpha, entropy=-logp.mean().item())

        with torch.no_grad():
            for net, tgt in ((self.q1, self.q1_t), (self.q2, self.q2_t)):
                for w, wt in zip(net.parameters(), tgt.parameters()):
                    wt.mul_(1 - c.tau).add_(c.tau * w)
        self.updates += 1
        return logs

    def state_dict(self):
        return dict(actor=self.actor.state_dict(), q1=self.q1.state_dict(), q2=self.q2.state_dict(),
                    q1_t=self.q1_t.state_dict(), q2_t=self.q2_t.state_dict(), log_alpha=self.log_alpha.detach().cpu(),
                    q_opt=self.q_opt.state_dict(), pi_opt=self.pi_opt.state_dict(), a_opt=self.a_opt.state_dict(),
                    updates=self.updates)

    def load_state_dict(self, s):
        for k in ("actor", "q1", "q2", "q1_t", "q2_t"):
            getattr(self, k).load_state_dict(s[k])
        with torch.no_grad():
            self.log_alpha.copy_(s["log_alpha"].to(self.device))
        for k in ("q_opt", "pi_opt", "a_opt"):
            getattr(self, k).load_state_dict(s[k])
        self.updates = s["updates"]
