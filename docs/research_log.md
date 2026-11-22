# Research Log

## 2026-11-22 — Milestone 0 + 1 (local, CPU only)

**Setup.** robosuite 1.5.2, Python 3.12 (lerobot 0.7.0 requires >=3.12), macOS M4 CPU.
Kinova3 (7-DoF) + Robotiq85, Lift task, `OSC_POSITION` (action = [dx, dy, dz, gripper] in [-1,1],
dx..dz scaled to +-5 cm per step), control 20 Hz, sim dt 2 ms (25 physics steps per action).

**Finding 1 — mujoco 4.0.1 crashes robosuite 1.5.2** (`AssertionError` in `get_joint_qpos_addr`).

**Finding 2 — mujoco 3.9.0 silently breaks the Robotiq85 gripper.**
First expert run: 5/20 success. Diagnosis chain:
1. Failures reached VERIFY with the gripper closed but `grasping=0`, cube never lifted.
2. Success vs cube yaw (40 eps): only yaw%90 in ~[30, 60] deg succeeded.
3. Fixed-yaw test: 0 deg -> 0/10, 20 deg -> 0/10, 45 deg -> 9/10.
4. Contacts at grasp time: at 0 deg **no gripper geom touches the cube** though eef is centered (<1 mm).
5. Fully closed fingerpad gap = 6.3 cm on mujoco 3.9.0 vs 1.5 cm on 3.3.7 (same for Panda+Robotiq85).
   A 4 cm cube face never gets touched; only its ~6 cm diagonal (yaw 45 deg) does.

Fix: pin `mujoco==3.3.7`; added `test_gripper_closes_fully` regression test.
Lesson: a physics-version change can produce no error, only a lower success rate.

**Finding 3 — robosuite Lift randomizes cube yaw uniformly over 360 deg by default**
(`rotation=None`), not fixed. Kept as default (expert handles it, via a parallel-jaw grasp that self-aligns).

**Stricter success.** robosuite counts success the instant the cube is >4 cm above the table.
We require: lifted >4 cm **and** `_check_grasp` true for 10 consecutive steps.

**Expert results** (mujoco 3.3.7, random yaw, seeds 0-99, hold-success):

| cube xy range | success | episode length (mean / min / max) |
|---|---|---|
| +-3 cm (robosuite default) | 100/100 | 49.9 / 45 / 55 |
| +-10 cm | 100/100 | 50.7 / 40 / 63 |

Caveat: 100% means the scripted expert is not yet a source of failure-recovery behavior;
demos will contain only clean trajectories (relevant to covariate shift in Milestone 4).

---

## 2026-11-22 — Spec v2: task switched from Lift to PickPlaceCan

Scope narrowed to one task (PickPlaceCan), one VLA (SmolVLA), one RL method (RLT-inspired residual SAC).
See `docs/project_spec.md`. The Lift expert/env code was removed (kept in git history).

**Dependency fix for LeRobot.** lerobot 0.7.0 needs numpy>=2; robosuite 1.5.2 -> mink 0.0.5 pins numpy<2.
mink is only used for GR1 humanoid IK. Override `numpy>=2.0,<2.3`; re-verified: tests pass, Lift expert
50/50 with identical per-seed cube positions and episode lengths as on numpy 1.26.

**Stock PickPlaceCan does not fit Kinova3 with a vertical gripper.** Reach tests (P-controller to fixed points, 150 steps):

| setting | target hover error | gripper tilt |
|---|---|---|
| stock bins, default OSC_POSITION | 0.0 cm | 40 deg |
| stock bins, orientation held | 9.7 cm | 8 deg |
| bins shifted -0.05 m | 4.2 cm | 11 deg |
| bins shifted -0.10 m | 3.1 cm | 6 deg |
| bins shifted -0.15 m (chosen) | 0.8 cm | 4 deg |

Root cause of the tilt: OSC `goal_update_mode="achieved"` resets the orientation goal to the current
orientation every step, so OSC_POSITION never actively holds orientation. Fix: capture the goal once per episode.

**Placement crash.** Narrowing the stock object sampler made `RandomizationError: Cannot place all objects`,
because single-object mode still places all 4 objects in bin1 before hiding 3. Fix: separate samplers
(can first in its region, hidden objects in the full bin).

**Expert carry height** (30 episodes each):

| carry_z | success | note |
|---|---|---|
| 1.05 | 24/30 | all failures: MOVE timeout (reach limit above target) |
| 1.00 | 29/30 | 1 MOVE timeout |
| 0.97 | 30/30 | only 1 cm clearance over bin walls (top 0.90 m) -> rejected as fragile |
| 1.00 + start lowering within 3 cm | **100/100** (seeds 0-99) | 4 cm clearance, mean length 128.8 steps (110-148) |

Strict success: can inside the target compartment AND not grasped, for 10 consecutive steps.

## 2026-11-22 — Demonstration dataset v1 (`data/kinova3_pickplace_can_v1`, not in git)

- train: seeds 0–49, 50/50 kept, 6,413 frames; val: seeds 100000–100009, 10/10 kept, 1,308 frames.
- Validator 17/17 on both splits; open-loop replay of all 60 episodes bit-exact.
- **float32 replay drift.** First version executed float64 expert actions but stored float32. Replaying
  stored actions: alignment exact for the first 20 steps in all episodes, but contact-rich steps amplified
  the rounding (max state diff 2.7e-2, 1/50 replays failed). Verified cause: replaying the float64 actions
  gives error 0; float32-rounded gives exactly the observed drift. Fix: cast to float32 before env.step.
- **Collection speed.** 13.2 s/episode -> 3.2 s/episode. Breakdown per 130-step episode: physics 0.6 s,
  2-camera render 1.9 s (128 px vs 256 px same cost), LeRobot `save_episode(parallel_encoding=True)` 15.5 s
  vs `False` 2.3 s (macOS spawns a new process pool per episode that re-imports torch/robosuite).
- Action stats: saturated at |a|=1 for 6% dx, 46% dy, 16% dz; gripper binary. State quaternion std 0.0004–0.018
  (orientation held) -> risk for MEAN_STD normalization; decision deferred to Milestone 2.
- **Dataset v1 final:** state reduced to 5-D (eef xyz + 2 finger joints) after discussion; recollected
  (train 50 / val 10, 160 s + 33 s), 17/17 checks on both splits, replay bit-exact.
