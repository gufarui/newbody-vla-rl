"""Chunk-level residual RL environment around a FROZEN VLA (Milestone 3).

One RL decision = one VLA action chunk of K env steps (K = 10, same as SFT evaluation):

    o_k  = VLA features + reference chunk + robot state  (+ privileged state for the critic only)
    δ_k  ∈ [-1, 1]^3  (RL actor output)
    a_t  = clip(a_ref_t + λ·[δ_k, 0], -1, 1)  for the K steps   (gripper never modified)
    r_k  = Σ_t r_t − β‖δ_k‖²,   r_t = +1 on strict success, episode ends on success or tipped can

The VLA is only called through `vla.predict_chunk(obs, K) -> (ref (K,4), features)`, so tests can pass a stub.
Privileged state is returned separately (`priv`) and must only reach the critic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from newbody.env.pick_place_env import can_obj, can_pos, can_target, is_grasping, reset_with_seed
from newbody.evaluation.success import PlaceSuccess

DEMO_REGION_HALF = (0.055, 0.075)  # effective can-center region of the demonstrations (rel. bin1 center)


@dataclass
class ResidualConfig:
    chunk_steps: int = 10
    residual_scale: float = 0.2       # λ: δ=1 -> +0.2 action units = 1 cm target offset per step
    residual_penalty: float = 0.005   # β
    success_reward: float = 1.0
    max_decisions: int = 30           # 30 x 10 = 300 env steps (same horizon as SFT evaluation)
    tip_cos_threshold: float = 0.5    # can axis more than 60 deg from vertical -> tipped


def can_up_cos(env) -> float:
    """cos(angle between the can's axis and world z). 1 = upright, 0 = lying on its side."""
    q = env.sim.data.body_xquat[env.obj_body_id[can_obj(env).name]]  # (w, x, y, z)
    w, x, y, z = q
    return float(1 - 2 * (x * x + y * y))  # R[2,2]


class ResidualChunkEnv:
    ACT_DIM = 3

    def __init__(self, env, vla, success_cfg: dict, cfg: ResidualConfig, bin1_xy):
        self.env, self.vla, self.cfg = env, vla, cfg
        self.checker = PlaceSuccess(**success_cfg)
        self.bin1_xy = np.asarray(bin1_xy[:2], dtype=float)

    # ------------------------------------------------------------------ observations
    def _actor_obs(self):
        st = self._obs
        state5 = np.concatenate([st["robot0_eef_pos"], st["robot0_gripper_qpos"][[0, 3]]]).astype(np.float32)
        return {
            "features": self._feat.astype(np.float32),
            "ref": self._ref.reshape(-1).astype(np.float32),
            "state": state5,
            "time": np.array([self._k / self.cfg.max_decisions], dtype=np.float32),
        }

    def _priv(self):
        e = self.env
        cp, eef = can_pos(e), self._obs["robot0_eef_pos"]
        return np.concatenate([cp, can_target(e), cp - eef, [float(is_grasping(e)), can_up_cos(e)]]).astype(np.float32)

    def _plan(self):
        self._ref, self._feat = self.vla.predict_chunk(self._obs, self.cfg.chunk_steps)

    # ------------------------------------------------------------------ API
    def reset(self, seed: int):
        self._obs = reset_with_seed(self.env, seed)
        self.checker.reset()
        self._k, self._t = 0, 0
        rel = can_pos(self.env)[:2] - self.bin1_xy
        self.inside_demo_region = bool(abs(rel[0]) <= DEMO_REGION_HALF[0] and abs(rel[1]) <= DEMO_REGION_HALF[1])
        self._plan()
        return self._actor_obs(), self._priv()

    def step(self, delta: np.ndarray):
        c = self.cfg
        delta = np.clip(np.asarray(delta, dtype=np.float32), -1.0, 1.0)
        offset = np.zeros(4, dtype=np.float32)
        offset[:3] = c.residual_scale * delta
        reward, success, tipped = -c.residual_penalty * float(delta @ delta), False, False
        executed = []
        for a_ref in self._ref:
            a = np.clip(a_ref + offset, -1.0, 1.0).astype(np.float32)
            a[3] = a_ref[3]  # gripper exactly as the VLA decided
            self._obs, _, _, _ = self.env.step(a)
            executed.append(a)
            self._t += 1
            if self.checker.update(self.env):
                success = True
                reward += c.success_reward
                break
            if can_up_cos(self.env) < c.tip_cos_threshold and not is_grasping(self.env):
                tipped = True
                break
        self._k += 1
        terminated = success or tipped
        truncated = (not terminated) and self._k >= c.max_decisions
        info = dict(success=success, tipped=tipped, env_steps=self._t, executed=np.array(executed),
                    inside_demo_region=self.inside_demo_region)
        if not terminated:
            # Also on truncation: SAC bootstraps Q(s_{k+1}) there, so s_{k+1} needs real VLA features.
            self._plan()
        return self._actor_obs(), self._priv(), reward, terminated, truncated, info
