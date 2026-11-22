"""Dataset schema: what the VLA is allowed to see, and how one frame is built.

Rules
- Only keys starting with "observation." or "action" become policy inputs/outputs in LeRobot
  (lerobot.utils.feature_utils.dataset_to_policy_features). Privileged simulator state
  (can pose, target compartment, seed, contacts) therefore never goes into the LeRobot dataset; it is written to
  a separate sidecar file (see record.py).
- Frame t pairs the observation BEFORE acting with the action taken at that observation:
      (o_t, a_t)  ->  env.step(a_t)  ->  o_{t+1}
"""
from __future__ import annotations

import numpy as np

from newbody.env.pick_place_env import get_image

TASK = "Pick up the can and place it in the target bin."

# Robotiq85 has 6 coupled finger joints; indices 0 and 3 are the left/right driver joints.
GRIPPER_QPOS_IDX = [0, 3]

# Gripper orientation (quaternion) is deliberately NOT in the state: we hold it fixed, so it is nearly
# constant (std down to 4e-4) and MEAN_STD normalization would blow a 1-degree tilt up into a huge input.
STATE_NAMES = ["eef_x", "eef_y", "eef_z", "grip_left", "grip_right"]
ACTION_NAMES = ["dx", "dy", "dz", "gripper"]
CAMERAS = {"agentview": "observation.images.agentview", "robot0_eye_in_hand": "observation.images.wrist"}


def features(image_size: int) -> dict:
    img = {"dtype": "video", "shape": (image_size, image_size, 3), "names": ["height", "width", "channels"]}
    return {
        **{key: dict(img) for key in CAMERAS.values()},
        "observation.state": {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (len(ACTION_NAMES),), "names": ACTION_NAMES},
    }


def state_vector(obs: dict) -> np.ndarray:
    return np.concatenate(
        [obs["robot0_eef_pos"], obs["robot0_gripper_qpos"][GRIPPER_QPOS_IDX]]
    ).astype(np.float32)


def make_frame(obs: dict, action: np.ndarray, task: str = TASK) -> dict:
    frame = {key: get_image(obs, cam) for cam, key in CAMERAS.items()}
    frame["observation.state"] = state_vector(obs)
    frame["action"] = np.asarray(action, dtype=np.float32)
    frame["task"] = task
    return frame
