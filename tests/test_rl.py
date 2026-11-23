"""Residual RL env + SAC tests with a stub VLA (fast, no model download)."""
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from newbody.env.pick_place_env import make_env
from newbody.rl.residual_env import ResidualChunkEnv, ResidualConfig, can_up_cos
from newbody.rl.sac import OBS_KEYS, SAC, ReplayBuffer, SACConfig

CFG = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs/pickplace_can.yaml"))


class StubVLA:
    """Returns a fixed reference chunk and observation-independent features."""

    def __init__(self, k=10):
        self.ref = np.tile(np.array([0.3, -0.2, 0.95, -1.0], dtype=np.float32), (k, 1))
        self.calls = 0

    def predict_chunk(self, obs, n):
        self.calls += 1
        return self.ref[:n].copy(), np.ones(1920, dtype=np.float32)


@pytest.fixture(scope="module")
def renv():
    env = make_env({**CFG["env"], "camera_size": 32}, render_cameras=True)
    rcfg = ResidualConfig(**CFG["rl"]["residual"])
    yield ResidualChunkEnv(env, StubVLA(rcfg.chunk_steps), CFG["success"], rcfg, CFG["env"]["bin1_pos"])
    env.close()


def test_zero_residual_executes_reference_exactly(renv):
    renv.reset(400000)
    *_, info = renv.step(np.zeros(3))
    assert np.array_equal(info["executed"], renv.vla.ref)


def test_residual_scale_bounds_and_gripper_untouched(renv):
    renv.reset(400000)
    *_, info = renv.step(np.array([1.0, 1.0, 1.0]))
    ex, ref = info["executed"], renv.vla.ref
    assert np.allclose(ex[:, :3], np.clip(ref[:, :3] + 0.2, -1, 1))
    assert np.array_equal(ex[:, 3], ref[:, 3])
    *_, info = renv.step(np.array([5.0, -5.0, 0.0]))  # out-of-range δ is clipped to [-1, 1]
    assert np.allclose(info["executed"][:, :2], np.clip(ref[:, :2] + [0.2, -0.2], -1, 1))


def test_reward_penalty_and_truncation(renv):
    renv.reset(400000)
    _, _, r, term, trunc, _ = renv.step(np.array([1.0, 0.0, 0.0]))
    assert np.isclose(r, -renv.cfg.residual_penalty * 1.0) and not term and not trunc
    for _ in range(renv.cfg.max_decisions - 1):
        _, _, _, term, trunc, _ = renv.step(np.zeros(3))
        if term:
            break
    assert trunc and not term  # stub never succeeds; time limit -> truncated, not terminated


def test_actor_obs_is_deployable_only(renv):
    o, priv = renv.reset(400001)
    assert set(o) == set(OBS_KEYS)
    assert o["features"].shape == (1920,) and o["ref"].shape == (40,) and o["state"].shape == (5,)
    assert priv.shape == (11,)  # can pos, target, can - eef, grasping, can upright cos -> critic only


def test_tipped_can_detected(renv):
    renv.reset(400002)
    env = renv.env
    assert can_up_cos(env) > 0.99
    jid = env.sim.model.joint_name2id(env.objects[env.object_id].joints[0])
    adr = env.sim.model.jnt_qposadr[jid]
    env.sim.data.qpos[adr + 3:adr + 7] = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]  # 90 deg about x
    env.sim.forward()
    assert abs(can_up_cos(env)) < 1e-6


def test_sac_initial_policy_is_near_zero_and_updates_are_finite():
    torch.manual_seed(0)
    dims = {"features": 1920, "ref": 40, "state": 5, "time": 1}
    sac = SAC(dims, 3, 11, SACConfig(batch_size=16), "cpu")
    o = {k: np.random.randn(d).astype(np.float32) for k, d in dims.items()}
    assert np.abs(sac.act(o, deterministic=True)).max() < 0.05
    stds = [sac.actor(sac._to_t(o))[1].exp().mean().item()]
    assert 0.2 < stds[0] < 0.4  # initial exploration std ≈ 0.3
    buf = ReplayBuffer(64, dims, 3, 11)
    for _ in range(32):
        buf.add(o, np.zeros(11), np.random.uniform(-1, 1, 3), 1.0, o, np.zeros(11), False)
    for _ in range(4):
        logs = sac.update(buf)
    assert all(np.isfinite(v) for v in logs.values())
    assert buf.o["features"].dtype == np.float16


def test_actor_frozen_during_critic_warmup():
    dims = {"features": 1920, "ref": 40, "state": 5, "time": 1}
    sac = SAC(dims, 3, 11, SACConfig(batch_size=16, actor_warmup_updates=5), "cpu")
    before = [p.clone() for p in sac.actor.parameters()]
    buf = ReplayBuffer(64, dims, 3, 11)
    o = {k: np.random.randn(d).astype(np.float32) for k, d in dims.items()}
    for _ in range(32):
        buf.add(o, np.zeros(11), np.random.uniform(-1, 1, 3), 1.0, o, np.zeros(11), False)
    for _ in range(5):
        sac.update(buf)
    assert all(torch.equal(a, b) for a, b in zip(before, sac.actor.parameters()))
    for _ in range(2):
        sac.update(buf)
    assert not all(torch.equal(a, b) for a, b in zip(before, sac.actor.parameters()))
