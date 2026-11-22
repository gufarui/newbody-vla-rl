"""Kinova3 + PickPlaceCan environment with reproducible, seeded resets.

Two deliberate changes vs. robosuite's stock PickPlaceCan (both verified 2026-11-22):

1. Bins moved 15 cm closer to the robot (bin1/bin2 x: 0.10 -> -0.05).
   The stock layout puts the can's target compartment at the edge of Kinova3's workspace:
   with the gripper held vertical, the target hover point was 9.7 cm out of reach.
   After the shift, all grasp/release points are reachable within ~1-2 cm.

2. OSC_POSITION holds the gripper orientation fixed (pointing down).
   robosuite's default goal_update_mode="achieved" re-targets orientation to the *current*
   orientation every step, so it is never actively held and drifted up to 40 deg near the
   workspace edge. We capture the orientation goal once after reset and keep it.

Other notes
- robosuite renders images upside down; `get_image` flips them to normal orientation.
- The can's start position is sampled in a central region of bin1 (configurable),
  not the whole bin, so the open gripper does not collide with bin walls.
"""
from __future__ import annotations

import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config, load_part_controller_config
from robosuite.environments.manipulation.pick_place import PickPlaceCan
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler


class KinovaPickPlaceCan(PickPlaceCan):
    """PickPlaceCan with a configurable can start region (relative to bin1 center)."""

    def __init__(self, can_x_range=(-0.08, 0.08), can_y_range=(-0.10, 0.10), **kwargs):
        self.can_x_range = list(can_x_range)
        self.can_y_range = list(can_y_range)
        super().__init__(**kwargs)

    def _get_placement_initializer(self):
        # Stock sampler places ALL 4 objects (milk, bread, cereal, can) anywhere in bin1, then hides
        # the 3 unused ones. Shrinking that shared range makes placement fail ("Cannot place all
        # objects"). So: sample the can first in its own region, then the hidden objects in the
        # full bin around it, then the stock target-marker samplers.
        super()._get_placement_initializer()
        stock = self.placement_initializer.samplers
        full = stock["CollisionObjectSampler"]
        can = self.objects[self.object_id]
        others = [o for o in self.objects if o is not can]

        def uniform(name, objs, xr, yr):
            return UniformRandomSampler(
                name=name, mujoco_objects=objs, x_range=xr, y_range=yr, rotation=self.z_rotation,
                rotation_axis="z", ensure_object_boundary_in_range=True, ensure_valid_placement=True,
                reference_pos=self.bin1_pos, z_offset=self.z_offset, rng=self.rng,
            )

        composite = SequentialCompositeSampler(name="ObjectSampler")
        composite.append_sampler(uniform("CanSampler", can, self.can_x_range, self.can_y_range))
        composite.append_sampler(uniform("HiddenObjectSampler", others, full.x_range, full.y_range))
        for name, sampler in stock.items():
            if name != "CollisionObjectSampler":
                composite.append_sampler(sampler)
        self.placement_initializer = composite


def make_controller_config(robot: str, arm_controller: str = "OSC_POSITION") -> dict:
    cfg = load_composite_controller_config(controller="BASIC", robot=robot)
    part = load_part_controller_config(default_controller=arm_controller)
    part["gripper"] = cfg["body_parts"]["right"]["gripper"]
    cfg["body_parts"]["right"] = part
    return cfg


def make_env(env_cfg: dict, render_cameras: bool = True):
    cams = list(env_cfg["camera_names"]) if render_cameras else []
    env = suite.make(
        "KinovaPickPlaceCan",
        robots=env_cfg["robot"],
        controller_configs=make_controller_config(env_cfg["robot"], env_cfg["arm_controller"]),
        bin1_pos=tuple(env_cfg["bin1_pos"]),
        bin2_pos=tuple(env_cfg["bin2_pos"]),
        can_x_range=env_cfg["can_x_range"],
        can_y_range=env_cfg["can_y_range"],
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
    env._hold_orientation = env_cfg.get("hold_orientation", True)
    return env


def _hold_orientation(env):
    """Make the OSC orientation goal constant for the episode (captured at the first step)."""
    ctrl = env.robots[0].composite_controller.part_controllers["right"]
    original = type(ctrl).compute_goal_ori
    ctrl._held_goal_ori = None

    def compute_goal_ori(delta, goal_update_mode=None):
        if ctrl._held_goal_ori is None:
            ctrl._held_goal_ori = original(ctrl, delta)
        return ctrl._held_goal_ori

    ctrl.compute_goal_ori = compute_goal_ori


def reset_with_seed(env, seed: int) -> dict:
    """Reset so that the initial state is a deterministic function of `seed`."""
    env.rng = np.random.default_rng(seed)  # robot joint init noise + can placement sampler
    obs = env.reset()
    if env._hold_orientation:
        _hold_orientation(env)  # controllers are rebuilt on hard reset, so patch every time
    return obs


def get_image(obs: dict, camera: str) -> np.ndarray:
    """HxWx3 uint8 RGB in normal (not upside-down) orientation."""
    return np.ascontiguousarray(obs[f"{camera}_image"][::-1])


def can_obj(env):
    return env.objects[env.object_id]


def can_pos(env) -> np.ndarray:
    return np.array(env.sim.data.body_xpos[env.obj_body_id[can_obj(env).name]])


def can_target(env) -> np.ndarray:
    """Center of the can's goal compartment in bin2 (z = bin floor). Privileged."""
    return np.array(env.target_bin_placements[env.object_id])


def can_in_target(env) -> bool:
    return not env.not_in_bin(can_pos(env), env.object_id)


def is_grasping(env) -> bool:
    return bool(env._check_grasp(gripper=env.robots[0].gripper, object_geoms=can_obj(env)))


def eef_pos(env) -> np.ndarray:
    return np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]])
