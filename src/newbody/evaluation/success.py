"""Stricter success condition for PickPlaceCan.

robosuite's check: can inside the target compartment AND the gripper more than ~7 cm away
(r_reach < 0.6), evaluated at a single instant. A can that is still falling, bouncing, or
being carried past the compartment could be counted for one frame.

Ours: can inside the target compartment AND not grasped by the gripper, for `hold_steps`
consecutive policy steps (so it has been released and stays there).
"""
from __future__ import annotations

from newbody.env.pick_place_env import can_in_target, is_grasping


class PlaceSuccess:
    def __init__(self, hold_steps: int = 10):
        self.hold_steps = hold_steps
        self.reset()

    def reset(self):
        self.count = 0

    def update(self, env) -> bool:
        ok = can_in_target(env) and not is_grasping(env)
        self.count = self.count + 1 if ok else 0
        return self.count >= self.hold_steps
