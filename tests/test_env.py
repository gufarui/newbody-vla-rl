"""Smoke + regression tests for the Kinova3 PickPlaceCan environment."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from newbody.env.pick_place_env import can_pos, can_target, get_image, make_env, reset_with_seed

CFG = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs/pickplace_can.yaml"))


@pytest.fixture(scope="module")
def env():
    e = make_env({**CFG["env"], "camera_size": 64}, render_cameras=True)
    yield e
    e.close()


def test_action_space(env):
    assert env.action_dim == 4  # dx, dy, dz, gripper
    lo, hi = env.action_spec
    assert np.allclose(lo, -1) and np.allclose(hi, 1)
    assert env.control_freq == 20


def test_seeded_reset_is_deterministic(env):
    reset_with_seed(env, 123)
    a, q_a = can_pos(env).copy(), env.sim.data.qpos.copy()
    reset_with_seed(env, 7)
    b = can_pos(env).copy()
    reset_with_seed(env, 123)
    assert np.allclose(a, can_pos(env)) and np.allclose(q_a, env.sim.data.qpos)
    assert not np.allclose(a, b)


def test_can_starts_in_configured_region(env):
    bin1 = np.array(CFG["env"]["bin1_pos"][:2])
    for s in range(20):
        reset_with_seed(env, s)
        rel = can_pos(env)[:2] - bin1
        assert CFG["env"]["can_x_range"][0] - 1e-6 <= rel[0] <= CFG["env"]["can_x_range"][1] + 1e-6
        assert CFG["env"]["can_y_range"][0] - 1e-6 <= rel[1] <= CFG["env"]["can_y_range"][1] + 1e-6


def test_target_compartment_reachable_with_vertical_gripper(env):
    """Regression: with the stock bin layout the target was out of reach (9.7 cm error)."""
    obs = reset_with_seed(env, 0)
    goal = can_target(env) + [0, 0, 0.15]
    for _ in range(150):
        obs, *_ = env.step(np.append(np.clip((goal - obs["robot0_eef_pos"]) / 0.05, -1, 1), -1))
    assert np.linalg.norm(goal - obs["robot0_eef_pos"]) < 0.02
    tilt = np.degrees(2 * np.arccos(min(1.0, abs(obs["robot0_eef_quat"][0]))))
    assert tilt < 10, f"gripper tilted {tilt:.1f} deg; orientation hold not active?"


def test_gripper_closes_fully(env):
    """Regression: with mujoco 3.9.0 the Robotiq85 stalls with a ~6.3 cm gap."""
    reset_with_seed(env, 0)
    m, d = env.sim.model, env.sim.data
    gid = [m.geom_name2id(f"gripper0_right_{s}_fingerpad_collision") for s in ("left", "right")]
    for _ in range(40):
        env.step(np.array([0, 0, 0, 1.0]))
    gap = np.linalg.norm(d.geom_xpos[gid[0]] - d.geom_xpos[gid[1]])
    assert gap < 0.03, f"fully closed fingerpad gap {gap:.3f} m"


def test_images_upright(env):
    obs = reset_with_seed(env, 0)
    img = get_image(obs, "agentview")
    assert img.shape == (64, 64, 3) and img.dtype == np.uint8
    # Floor/table fills the bottom of a correctly oriented agentview; the top is the back wall.
    assert np.abs(img[-8:].astype(int) - img[:8].astype(int)).mean() > 5
