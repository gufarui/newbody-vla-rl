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

## 2026-11-22 — Milestone 2 local smoke test (Mac M4, MPS, free)

- Official `lerobot-train --policy.path=lerobot/smolvla_base` on our train split, batch 2, 3 steps: runs, finite loss
  (2.53, 3.30, 1.16), ~0.6 s/step on MPS. Cameras mapped with `--rename_map` agentview->camera1, wrist->camera2;
  camera3 absent (empty_cameras=0 -> skipped, no fake image).
- One-step tensor inspection (`scripts/inspect_smolvla_batch.py`): prefix 177 tokens (2x64 image + 48 language + 1 state),
  images resized 256->512 in [-1,1]; state/action padded to 32; action chunk 50; flow matching verified
  x_t = t*eps + (1-t)*A, u = eps - A (NOISE_AT_ONE); loss masks padded dims and action_is_pad.
- Params: 450M total, 350M VLM frozen, 100M trainable (action expert 98M + state/action projections + time MLPs).

## 2026-11-23 — Milestone 2 GPU run sft_v1 (RTX 4090 D, ¥1.88/h)

- Setup (see scripts/server/setup_autodl.md): ~15 min of installs/downloads via Tsinghua PyPI + hf-mirror.
  Server: torch 2.15.0+cu132, mujoco 3.3.7, 9/9 tests pass with MUJOCO_GL=egl.
- GPU smoke, batch 64: 0.56 s/step, 14 GB VRAM, util ~99%; loss 1.31 -> 0.72 in 40 steps.
- Launched sft_v1: official recipe, batch 64, 20k steps, checkpoints every 5k, seed 1000. ETA ~3.1 h.
- **Checkpoint 5k (epoch ~47, train loss 0.04), val seeds 100000-100019, 10 actions/chunk: 20/20 success**
  (Wilson 95% CI 0.84-1.00), mean chunk latency 409 ms on GPU while training runs concurrently.
  -> SFT already saturates the training distribution; per spec, harden evaluation before RL.
- Expert on the FULL source bin (can_x +-0.145, can_y +-0.195, ~4x training area), seeds 300000-300059: 59/60
  (1 never_grasped near a wall). The wider region is feasible for the task.
- 5k checkpoint, val seeds, **50 actions/chunk** (central region): 16/20 (CI 0.58-0.92; 2 never_grasped, 2 dropped)
  vs 20/20 with 10 actions/chunk -> long open-loop execution hurts, as expected from replay sensitivity.
- 5k checkpoint, **full-bin start region**, val seeds 100020-100059, 10 actions/chunk: **18/40** (CI 0.31-0.60);
  failures: 17 never_grasped, 3 dropped, 2 wrong_compartment. Expert on the same region: 59/60.
- **Checkpoint selection (full-bin val seeds 100020-100059, 10 actions/chunk), pre-registered rule
  "stop when consecutive checkpoints differ by <=3/40":** 5k 18/40, 10k 22/40 (paired: 6 only-10k vs 2 only-5k,
  McNemar p~0.29), 15k 21/40 (paired vs 10k: 4 vs 3). Train loss kept falling (5k 0.041, 10k 0.015, 12k 0.012)
  while closed-loop success plateaued -> training stopped at ~16.4k by user agreement; chosen checkpoint **10k**.
- Final test eval first ran at ~80 s/episode: torch defaulted to 128 CPU threads on an 18-core container quota
  (heavy cgroup throttling). Restarted with OMP_NUM_THREADS=8 -> ~8 s/episode. Partial results of the slow run discarded.

### Milestone 2 result — SFT-only baseline (checkpoint 10k), TEST seeds 200000-200199, full source bin

| actions executed per chunk | success | 95% CI (Wilson) | inside training region | outside | failure modes |
|---|---|---|---|---|---|
| **10** | **120/200 (60%)** | 0.53-0.67 | 43/44 (98%) | 77/156 (49%) | never_grasped 69, dropped 9, wrong_compartment 2 |
| 50 | 60/200 (30%) | 0.24-0.37 | 40/44 (91%) | 20/156 (13%) | never_grasped 132, dropped 5, wrong_compartment 3 |

- Chunk inference latency on the 4090: ~205 ms mean (10 flow steps, 2 cameras), so one chunk per 10 steps (0.5 s) fits.
- Typical never_grasped (video seed 200003): gripper descends with a few-cm horizontal offset, hits the can and knocks it
  over; a lying can never appears in the demos, so no recovery. => distribution shift outside the demo region;
  a residual that corrects xy during approach/descent has a concrete job in Milestone 3.
- Long open-loop execution (50/chunk) halves success, mostly outside the training region (13%).
- SFT-only baseline for Milestone 4 = 10k checkpoint, 10 actions/chunk, full bin: 60% [53, 67].

## 2026-11-23 — Milestone 3 pre-checks: frozen-VLA feature interface (local, Mac MPS)

- Hook on SmolVLA's prefix-only VLM pass (normally discarded hidden states): 1 VLM pass per chunk, identical obs ->
  identical features (max diff 0.0), VLA weights hash unchanged, chunk latency ~320 ms on MPS.
- Linear probe (ridge, 5-fold CV), initial scenes, full bin, n=120: can xy from mean of all tokens R2 0.60/0.70
  (RMSE 4.1/5.3 cm); from robot state R2 ~0.
- Linear probe at pre-grasp decision points during VLA rollouts (n=277), target = can xy minus gripper xy:
  all-token mean R2 0.67/0.87 (2.8/4.8 cm), **front-cam tokens 0.81/0.96 (2.1/2.7 cm)**, wrist-cam 0.61/0.73,
  robot state 0.07/0.42. -> actor features = per-camera means concatenated (1920-D).

## 2026-11-23 — Milestone 3 implementation + local checks (Mac MPS, free)

- `src/newbody/rl/residual_env.py` (chunk-level residual env), `src/newbody/rl/sac.py` (CleanRL-based SAC,
  asymmetric critic, near-zero initial actor), `scripts/train_residual_sac.py` (logs, evals, checkpoint + replay resume).
- Unit tests with a stub VLA (tests/test_rl.py, 6 tests): δ=0 executes the reference exactly; ±λ scaling and clipping;
  gripper never modified; penalty; truncation vs termination; actor obs has no privileged keys; tipped-can detection;
  initial deterministic |δ| < 0.05, exploration std ≈ 0.3; finite SAC updates.
- Real VLA, `scripts/check_residual_zero.py`: residual env with δ=0 vs the SFT eval path, same seeds ->
  **identical executed actions (max diff 0.0)** on 2 full episodes. The RL starting point is exactly the baseline.
- 6-episode smoke run: everything runs, VLA weight hash unchanged, resume works. Observed |δ| rising 0.26 -> 0.55 in
  6 episodes with initial α = 0.05 (entropy bonus dominates while Q is still flat) -> initial α set to 0.01.

## 2026-11-24 — Milestone 3 training run rl_v1 (RTX 4090 D)

- GPU smoke: 0.36 s per decision (VLA chunk + 10 sim steps + 4 SAC updates). Without critic warm-up the actor
  drifted to mean |δ| ~0.6 within 8 episodes -> added `actor_warmup_updates: 2000` (|δ| then stays ~0.2 = noise).
- RL-val (seeds 450000-450039, deterministic actor) during training: ep0 (=SFT) 29/40, ep250 32/40, **ep500 34/40**,
  ep750 23/40 (inside demo region 9/9 throughout).
- Training-time success per 100 episodes: 55, 66, 78, 79, 76, 68, 65, 69% (ep 0-763); mean |δ| 0.31 -> 0.37;
  critic loss 0.0005 (ep 145-445) -> 0.005 (ep 763); Q mean peaked 0.65 (ep 645). Interpretation: off-policy
  degradation (critic error grows, actor exploits it with larger corrections; UTD 4 makes this easier).
- Decision (user, 2026-11-24): stop at ep 763 (~1 h 45 min); select a checkpoint by RL-val among ep 249-599
  (rule fixed beforehand: highest val success, tie -> earlier); then one final test on test seeds 200000-200199.
- RL-val checkpoint selection (40 eps each): ep249 32, ep349 31, ep399 33, ep449 29, **ep499 34**, ep549 29, ep599 27
  -> selected ep00499.pt by the pre-registered rule (note: max of 7 noisy evals -> val number optimistic).

### Milestone 3 result — TEST seeds 200000-200199, full bin, 10 actions/chunk, paired with the SFT baseline

| policy | success | 95% CI | inside demo region | outside | failure modes |
|---|---|---|---|---|---|
| SFT only (10k) | 120/200 (60.0%) | 0.53-0.67 | 43/44 | 77/156 (49%, CI 0.42-0.57) | never_grasped* 69, dropped 9, wrong_compartment 2 |
| **SFT + residual SAC (ep499)** | **147/200 (73.5%)** | 0.67-0.79 | 41/44 | **106/156 (68%, CI 0.60-0.75)** | never_grasped 35, tipped 14, dropped 2, wrong_compartment 2 |

*the baseline classifier had no separate "tipped" label; tipped cans were counted as never_grasped.
- Paired on identical seeds: only-RL successes 31, only-baseline 4 -> **McNemar exact p = 3.5e-6**.
- Mean |δ| = 0.30 (≈ 3 mm target offset per step on average). Gains are almost entirely outside the demo region;
  inside it is 41/44 vs 43/44 (difference of 2 episodes, not significant).
- SFT checkpoint file sha256 unchanged on the server (7aa33326...); the VLA is never written. rl_v1 was interrupted,
  so its run_end.json (in-process weight hash) was not produced; smoke runs verified the in-process hash.
- Cost of the M3 GPU session ≈ 2.5 h ≈ ¥5. Instance shut down.

## 2026-11-24 — Milestone 4 analysis (local)

- Figures: docs/figures/m4_success.png, m4_learning_curve.png, m4_outcome_map.png (scripts/make_figures.py);
  comparison videos results/videos/compare_seed2000{03,08}.mp4 (local reruns reproduced SFT fail / RL success).
- Success by start band (SFT -> RL): y<-11 cm 67->90%, |y|<7.5 76->84%, y>=11 cm 16->29%; x>=8 cm 13->45%, x<-8 cm 32->54%.
- 27/53 RL failures at y>=11 cm (near the wall facing the target bin), mostly never_grasped; 59% of those hit
  max|δ|>=0.95 vs 0% of successes there and 17% elsewhere -> residual bound likely limiting (untested).
- RL broke 4 scenes (all tipped; 3 inside the demo region). Report: docs/technical_report.md.
