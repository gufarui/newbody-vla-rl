"""Smoke tests: env creation, action space, seeded reset determinism, image orientation."""
import numpy as np
import pytest
import yaml

from newbody.env.lift_env import cube_pos, get_image, make_lift_env, reset_with_seed
from pathlib import Path

CFG = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs/lift_kinova3.yaml"))


@pytest.fixture(scope="module")
def env():
    e = make_lift_env({**CFG["env"], "camera_size": 64}, render_cameras=True)
    yield e
    e.close()


def test_action_space(env):
    assert env.action_dim == 4  # dx, dy, dz, gripper
    lo, hi = env.action_spec
    assert np.allclose(lo, -1) and np.allclose(hi, 1)
    assert env.control_freq == 20


def test_seeded_reset_is_deterministic(env):
    reset_with_seed(env, 123)
    a = cube_pos(env).copy()
    q_a = env.sim.data.qpos.copy()
    reset_with_seed(env, 7)
    b = cube_pos(env).copy()
    reset_with_seed(env, 123)
    assert np.allclose(a, cube_pos(env))
    assert np.allclose(q_a, env.sim.data.qpos)
    assert not np.allclose(a, b)


def test_cube_within_configured_range(env):
    r = CFG["env"]["cube_xy_range"]
    for s in range(10):
        reset_with_seed(env, s)
        assert np.all(np.abs(cube_pos(env)[:2]) <= r + 1e-6)


def test_action_moves_eef_up(env):
    obs = reset_with_seed(env, 0)
    z0 = obs["robot0_eef_pos"][2]
    for _ in range(10):
        obs, *_ = env.step(np.array([0, 0, 1.0, -1.0]))
    assert obs["robot0_eef_pos"][2] > z0 + 0.05


def test_images_rgb_uint8(env):
    obs = reset_with_seed(env, 0)
    img = get_image(obs, "agentview")
    assert img.shape == (64, 64, 3) and img.dtype == np.uint8
    # The table (bright) fills the bottom of a correctly oriented agentview image.
    assert img[-8:].mean() > img[:8].mean()


def test_gripper_closes_fully(env):
    """Regression test: with mujoco 3.9.0 the Robotiq85 stalls with a ~6.3 cm gap and
    silently cannot grasp a 4 cm cube face-on. Guard against physics-version regressions."""
    reset_with_seed(env, 0)
    m, d = env.sim.model, env.sim.data
    gid = [m.geom_name2id(f"gripper0_right_{s}_fingerpad_collision") for s in ("left", "right")]
    for _ in range(40):
        env.step(np.array([0, 0, 0, 1.0]))
    gap = np.linalg.norm(d.geom_xpos[gid[0]] - d.geom_xpos[gid[1]])
    assert gap < 0.03, f"fully closed fingerpad gap {gap:.3f} m is too wide to grasp a 4 cm cube"
