"""Scripted Finite-State-Machine expert for Lift with OSC_POSITION control.

The expert uses PRIVILEGED state (true cube position read from the simulator).
That is fine for generating demonstrations, but the learned VLA policy will only
get camera images + robot proprioception + the language instruction.

Control law in each phase (a simple P-controller in Cartesian space):
    error  = target_pos - eef_pos                      (meters)
    action = clip(gain * error / 0.05, -1, 1)          (0.05 m = OSC max delta per step)
The OSC controller then turns this Cartesian delta into joint torques (via the
Jacobian / operational space dynamics), so the expert never touches joint space.

Gripper convention in robosuite: -1 = open, +1 = close.
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
    VERIFY = "VERIFY"


@dataclass
class ExpertConfig:
    approach_height: float = 0.10
    grasp_z_offset: float = 0.0
    lift_height: float = 0.15
    action_gain: float = 1.0
    xy_tolerance: float = 0.01
    z_tolerance: float = 0.01
    grasp_close_steps: int = 10
    phase_timeout: int = 60


class LiftFSMExpert:
    def __init__(self, cfg: ExpertConfig):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.phase = Phase.APPROACH
        self.phase_steps = 0
        self.grasp_pos = None
        self.timed_out_phase: Phase | None = None

    def _move(self, eef: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.clip(self.cfg.action_gain * (target - eef) / OSC_MAX_DELTA, -1.0, 1.0)

    def _goto(self, phase: Phase):
        self.phase, self.phase_steps = phase, 0

    def act(self, obs: dict) -> np.ndarray:
        """Return a 4-D action [dx, dy, dz, gripper] in [-1, 1]."""
        c = self.cfg
        eef = obs["robot0_eef_pos"]
        cube = obs["cube_pos"]  # privileged
        self.phase_steps += 1
        if self.phase_steps > c.phase_timeout and self.timed_out_phase is None:
            self.timed_out_phase = self.phase  # caller ends the episode as a failure

        if self.phase == Phase.APPROACH:
            target = cube + np.array([0, 0, c.approach_height])
            err = target - eef
            if np.linalg.norm(err[:2]) < c.xy_tolerance and abs(err[2]) < 2 * c.z_tolerance:
                self._goto(Phase.DESCEND)
            return np.append(self._move(eef, target), OPEN)

        if self.phase == Phase.DESCEND:
            target = cube + np.array([0, 0, c.grasp_z_offset])
            err = target - eef
            if np.linalg.norm(err) < c.z_tolerance:
                self.grasp_pos = target.copy()
                self._goto(Phase.GRASP)
            return np.append(self._move(eef, target), OPEN)

        if self.phase == Phase.GRASP:
            # hold still while the fingers close
            if self.phase_steps >= c.grasp_close_steps:
                self._goto(Phase.LIFT)
            return np.append(self._move(eef, self.grasp_pos), CLOSE)

        # LIFT and VERIFY: keep closed, move/hold above grasp point
        target = self.grasp_pos + np.array([0, 0, c.lift_height])
        if self.phase == Phase.LIFT and abs(target[2] - eef[2]) < 2 * c.z_tolerance:
            self._goto(Phase.VERIFY)
        return np.append(self._move(eef, target), CLOSE)
