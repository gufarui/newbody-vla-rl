"""Scripted Finite-State-Machine expert for PickPlaceCan with OSC_POSITION control.

The expert uses PRIVILEGED state: the true can position (obs["Can_pos"]) and the goal
compartment center (given at reset). The learned VLA only gets camera images, robot
proprioception and the language instruction.

Control law in every phase (P-controller in Cartesian space):
    action_xyz = clip(gain * (target - eef_pos) / 0.05, -1, 1)     (0.05 m = OSC max offset)
Gripper convention in robosuite: -1 = open, +1 = close.

Phases:
    APPROACH -> DESCEND -> GRASP -> LIFT -> MOVE -> LOWER -> RELEASE -> RETREAT
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

OSC_MAX_DELTA = 0.05  # must match output_max in osc_position.json
OPEN, CLOSE = -1.0, 1.0


class Phase(str, Enum):
    APPROACH = "APPROACH"
    DESCEND = "DESCEND"
    GRASP = "GRASP"
    LIFT = "LIFT"
    MOVE = "MOVE"
    LOWER = "LOWER"
    RELEASE = "RELEASE"
    RETREAT = "RETREAT"


@dataclass
class ExpertConfig:
    approach_height: float = 0.10
    grasp_z_offset: float = 0.0
    carry_z: float = 1.00
    move_xy_tolerance: float = 0.03
    release_z_above_floor: float = 0.09
    action_gain: float = 1.0
    xy_tolerance: float = 0.01
    z_tolerance: float = 0.01
    gripper_steps: int = 10
    phase_timeout: int = 80


class PickPlaceFSMExpert:
    def __init__(self, cfg: ExpertConfig):
        self.cfg = cfg
        self.goal = None
        self.reset()

    def reset(self, goal: np.ndarray | None = None):
        """goal: center of the target compartment (privileged), z = bin floor."""
        self.goal = None if goal is None else np.asarray(goal, dtype=float)
        self.phase = Phase.APPROACH
        self.phase_steps = 0
        self.anchor = None  # fixed point used by GRASP / RELEASE / RETREAT
        self.timed_out_phase: Phase | None = None

    def _move(self, eef, target):
        return np.clip(self.cfg.action_gain * (target - eef) / OSC_MAX_DELTA, -1.0, 1.0)

    def _goto(self, phase: Phase):
        self.phase, self.phase_steps = phase, 0

    def act(self, obs: dict) -> np.ndarray:
        """Return a 4-D action [dx, dy, dz, gripper] in [-1, 1]."""
        c = self.cfg
        eef = obs["robot0_eef_pos"]
        can = obs["Can_pos"]  # privileged
        self.phase_steps += 1
        if self.phase_steps > c.phase_timeout and self.timed_out_phase is None:
            self.timed_out_phase = self.phase  # caller ends the episode as a failure

        def reached(target, xy_tol=c.xy_tolerance, z_tol=c.z_tolerance):
            e = target - eef
            return np.linalg.norm(e[:2]) < xy_tol and abs(e[2]) < z_tol

        p = self.phase
        if p == Phase.APPROACH:
            target, grip = can + [0, 0, c.approach_height], OPEN
            if reached(target, z_tol=2 * c.z_tolerance):
                self._goto(Phase.DESCEND)
        elif p == Phase.DESCEND:
            target, grip = can + [0, 0, c.grasp_z_offset], OPEN
            if reached(target):
                self.anchor = target.copy()
                self._goto(Phase.GRASP)
        elif p == Phase.GRASP:
            target, grip = self.anchor, CLOSE
            if self.phase_steps >= c.gripper_steps:
                self._goto(Phase.LIFT)
        elif p == Phase.LIFT:
            target, grip = np.array([self.anchor[0], self.anchor[1], c.carry_z]), CLOSE
            if reached(target, xy_tol=0.03, z_tol=2 * c.z_tolerance):
                self._goto(Phase.MOVE)
        elif p == Phase.MOVE:
            target, grip = np.array([self.goal[0], self.goal[1], c.carry_z]), CLOSE
            # Start lowering once roughly above the compartment: the exact point above the target at
            # carry height is the most stretched arm pose (reach limit); LOWER finishes alignment lower.
            if reached(target, xy_tol=c.move_xy_tolerance, z_tol=3 * c.z_tolerance):
                self._goto(Phase.LOWER)
        elif p == Phase.LOWER:
            target, grip = self.goal + [0, 0, c.release_z_above_floor], CLOSE
            if reached(target, xy_tol=1.5 * c.xy_tolerance, z_tol=1.5 * c.z_tolerance):
                self.anchor = target.copy()
                self._goto(Phase.RELEASE)
        elif p == Phase.RELEASE:
            target, grip = self.anchor, OPEN
            if self.phase_steps >= c.gripper_steps:
                self._goto(Phase.RETREAT)
        else:  # RETREAT: move straight up with the gripper open, then hold
            target, grip = self.anchor + [0, 0, 0.12], OPEN
        return np.append(self._move(eef, target), grip)
