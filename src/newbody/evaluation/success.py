"""Stricter success condition than robosuite's Lift._check_success.

robosuite counts success the instant the cube is >4 cm above the table, so a cube
that is knocked or flicked upward for one frame counts as a success. We require the cube
to stay lifted for `hold_steps` consecutive policy steps while it is grasped.
"""
from __future__ import annotations

from newbody.env.lift_env import cube_pos, is_grasping, table_height


class HoldSuccess:
    def __init__(self, lift_height: float = 0.04, hold_steps: int = 10, require_grasp: bool = True):
        self.lift_height = lift_height
        self.hold_steps = hold_steps
        self.require_grasp = require_grasp
        self.reset()

    def reset(self):
        self.count = 0

    def update(self, env) -> bool:
        lifted = cube_pos(env)[2] > table_height(env) + self.lift_height
        ok = lifted and (is_grasping(env) or not self.require_grasp)
        self.count = self.count + 1 if ok else 0
        return self.count >= self.hold_steps
