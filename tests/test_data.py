"""Schema tests: the VLA must never receive privileged simulator state."""
import numpy as np
import yaml
from pathlib import Path

from newbody.data import schema
from newbody.env.pick_place_env import make_env, reset_with_seed

CFG = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs/pickplace_can.yaml"))


def test_policy_keys_are_only_allowed_ones():
    keys = set(schema.features(64))
    assert keys == {"observation.images.agentview", "observation.images.wrist", "observation.state", "action"}


def test_state_has_no_can_information():
    env = make_env({**CFG["env"], "camera_size": 64}, render_cameras=True)
    obs = reset_with_seed(env, 0)
    s0 = schema.state_vector(obs)
    # Move only the can: the state vector must not change.
    jid = env.sim.model.joint_name2id(env.objects[env.object_id].joints[0])
    adr = env.sim.model.jnt_qposadr[jid]
    env.sim.data.qpos[adr:adr + 2] += 0.05
    env.sim.forward()
    obs2 = env._get_observations(force_update=True)
    assert not np.allclose(obs["Can_pos"], obs2["Can_pos"])
    assert np.allclose(s0, schema.state_vector(obs2))
    frame = schema.make_frame(obs, np.zeros(4))
    assert frame["observation.state"].shape == (9,) and frame["action"].dtype == np.float32
    env.close()


def test_seed_splits_do_not_overlap():
    s = CFG["data"]["splits"]
    train = range(s["train"]["seed_start"], s["train"]["seed_start"] + 2 * s["train"]["episodes"])
    val = range(s["val"]["seed_start"], s["val"]["seed_start"] + 2 * s["val"]["episodes"])
    assert max(train) < s["val"]["seed_start"] and max(val) < s["test"]["seed_start"]
