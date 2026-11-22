# Project Spec v2 (2026-11-22) — condensed

**Title:** Lightweight RL Post-training for Cross-Embodiment VLA Adaptation

**Research question:** Can a lightweight feature-conditioned RL policy improve a frozen VLA after
adaptation to a new simulated robot embodiment?

## Fixed scope

| Component | Choice |
|---|---|
| Simulator / framework | MuJoCo 3.3.7 / robosuite 1.5.2 |
| Robot | Kinova3 + Robotiq85 |
| Task | PickPlaceCan (one fixed instruction) |
| Controller | OSC_POSITION (+ held gripper orientation) |
| Expert | Scripted FSM (privileged state) |
| Dataset | LeRobot (lerobot 0.7.0), ~50 demos |
| Foundation policy | `lerobot/smolvla_base` |
| Adaptation | SFT |
| RL | RLT-inspired residual SAC on a frozen VLA (VLA features + reference action, chunk-level, position-only bounded correction) |
| Main comparison | SFT-only vs SFT + residual SAC on the same held-out initial states |

Out of scope: second robot, multi-task, hierarchical planning, world models, memory, ROS2 / real robot,
multiple RL algorithms, reproducing π0.5 / π*0.6 / π0.7 / full RLT (read for concepts only).

## Milestones

1. **Simulation + demonstrations** (local CPU): env, expert, LeRobot dataset, validation.
2. **SmolVLA adaptation** (remote 4090): SFT, closed-loop success rate. If ~0%: failure analysis first.
   If ~100%: widen initial-state range *before* RL so there are failures to improve.
3. **Residual RL** (local tests, remote training): frozen VLA, feature hook, residual SAC, reward design.
4. **Evaluation**: one main comparison, 50–100 fixed eval episodes, confidence intervals, failure analysis.

## Working rules

- Agent does engineering; human decides VLA I/O, RL observation/action/reward, and interprets results.
- No paid GPU job without an explicit command, resource estimate and cost risk approved by the human.
- VLA-in-the-loop simulation runs on the GPU server, never over the network per step.
- Make it work → understand why → evaluate → document → stop. No new research directions without approval.

## Adjustments made while implementing Milestone 1 (with evidence)

The stock robosuite `PickPlaceCan` does not work as-is for Kinova3 with a vertical gripper:

1. **Bins moved 15 cm toward the robot** (x 0.10 → −0.05). Stock layout: target compartment
   9.7 cm out of reach with the gripper held vertical.
2. **Gripper orientation held fixed** in OSC_POSITION. robosuite's default re-targets orientation to the
   current one each step, so it drifted up to 40° near the workspace edge.
3. **Can start region narrowed** to the central part of bin1 (~±5.5 × ±7.5 cm) so the open gripper
   does not hit bin walls; the 3 hidden objects are sampled separately to avoid placement failures.
4. **Carry height 1.00 m** (4 cm above bin walls); start lowering within 3 cm of the target
   (exact point at carry height is the reach limit).

## Decisions agreed with the human (2026-11-22)

- `observation.state` = 5-D (eef xyz + 2 finger joints); gripper quaternion dropped (near-constant -> MEAN_STD blow-up).
- Milestone 2: compare executing 10 vs 50 actions per predicted chunk (task is sensitive to small action errors).
- Milestone 3: residual corrects position only; gripper-timing failures are out of its reach -> report in failure analysis.
- Milestone 4: consider ~200 eval episodes per method (100 episodes cannot resolve <10-point differences).
- Be explicit in the report: SmolVLA was pretrained on 6-D joint actions + 3 cameras; our 4-D Cartesian actions + 2
  cameras mean the action head is largely re-learned from our demos; transfer is mainly visual/language.
- **Hard setting for RL and final evaluation (2026-11-23):** SFT data stays in the central can region; residual RL
  training and the final SFT-only vs SFT+RL comparison use the **full source bin** (can_x ±0.145, can_y ±0.195).
  Evidence: 5k SFT checkpoint 20/20 in-region but 11/32 outside it; expert 59/60 on the full bin.
  Execution: 10 actions per chunk (50 per chunk: 16/20 even in-region). SFT checkpoint chosen on val seeds (full bin).
