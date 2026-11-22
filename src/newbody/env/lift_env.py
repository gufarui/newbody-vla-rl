"""Kinova3 + Lift environment factory with reproducible, seeded resets.

Design notes
- We use robosuite's official OSC_POSITION part config (3-D delta position) and keep
  the robot's default gripper sub-config. Only changing the "type" field of the default
  OSC_POSE config would keep 6-D output limits and crash.
- Each episode is reset from an explicit integer seed. The same seed always gives the
  same cube position and robot initial-joint noise, so we can build fixed train/val/test
  splits of initial states and compare methods on paired initial states.
- robosuite renders images upside down; `get_image` flips them to normal orientation.
"""
from __future__ import annotations

import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config, load_part_controller_config


def make_controller_config(robot: str, arm_controller: str = "OSC_POSITION") -> dict:
    cfg = load_composite_controller_config(controller="BASIC", robot=robot)
    part = load_part_controller_config(default_controller=arm_controller)
    part["gripper"] = cfg["body_parts"]["right"]["gripper"]
    cfg["body_parts"]["right"] = part
    return cfg


def make_lift_env(env_cfg: dict, render_cameras: bool = True):
    cams = list(env_cfg["camera_names"]) if render_cameras else []
    env = suite.make(
        env_cfg["name"],
        robots=env_cfg["robot"],
        controller_configs=make_controller_config(env_cfg["robot"], env_cfg["arm_controller"]),
        has_renderer=False,
        has_offscreen_renderer=render_cameras,
        use_camera_obs=render_cameras,
        camera_names=cams or "agentview",
        camera_heights=env_cfg["camera_size"],
        camera_widths=env_cfg["camera_size"],
        control_freq=env_cfg["control_freq"],
        horizon=env_cfg["horizon"],
        reward_shaping=False,
        ignore_done=True,  # we decide episode ends ourselves (success / timeout)
    )
    r = float(env_cfg["cube_xy_range"])
    env.placement_initializer.x_range = [-r, r]
    env.placement_initializer.y_range = [-r, r]
    # Cube yaw: robosuite's Lift default (rotation=None) is uniform over 0..360 deg.
    # Config value: null -> keep default; number -> fixed angle (rad); [lo, hi] -> uniform range (rad).
    if env_cfg.get("cube_yaw") is not None:
        env.placement_initializer.rotation = env_cfg["cube_yaw"]
    return env


def reset_with_seed(env, seed: int) -> dict:
    """Reset so that the initial state is a deterministic function of `seed`."""
    rng = np.random.default_rng(seed)
    env.rng = rng  # used by robot joint init noise
    env.placement_initializer.rng = rng  # used by cube placement
    return env.reset()


def get_image(obs: dict, camera: str) -> np.ndarray:
    """HxWx3 uint8 RGB in normal (not upside-down) orientation."""
    return np.ascontiguousarray(obs[f"{camera}_image"][::-1])


def table_height(env) -> float:
    return float(env.model.mujoco_arena.table_offset[2])


def cube_pos(env) -> np.ndarray:
    return np.array(env.sim.data.body_xpos[env.cube_body_id])


def is_grasping(env) -> bool:
    return bool(env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.cube))
