# Chapter 1: Simulation, Scripted Teacher, and Demonstrations

## Where this chapter fits

We want SmolVLA to learn to pick up a can and place it in a target bin, using a robot arm it was not
trained on (Kinova3). A VLA learns by **imitation**, so it needs demonstrations first.

In the real world a human records demonstrations with a teleoperation device.
In simulation we write a program, the **scripted expert** (the "teacher"), that does the task automatically.
It is allowed to cheat: it reads the can's exact position and the target location from the simulator.
Its only job is to produce training data. It is a tool, not the research result.

```
Milestone 1  teacher does pick-and-place (privileged state) -> recorded as a LeRobot dataset   <- this chapter
Milestone 2  fine-tune SmolVLA on that data; it only sees camera images + robot state + text
Milestone 3  freeze the VLA; train a small residual SAC policy that corrects its actions
Milestone 4  compare SFT-only vs SFT + RL on the same held-out start states
```

**Results of this chapter**

- Teacher: **100/100** successful pick-and-place episodes (about 6.4 s each).
- Dataset: 50 training + 10 validation demonstrations (8,300 frames), 17/17 checks passed, bit-exact replay (Section 8).

---

## 1. The scene

![Teacher pick-and-place, one frame per FSM phase](../figures/pickplace_phases.png)

*Front camera, one frame at the start of each teacher phase (seed 0).*

- **Left bin (bin1):** the red can starts here, at a random position near the center.
- **Right bin (bin2):** 4 compartments. The can belongs in the back-right one (marked by a gray "ghost" can).
- **Instruction** (the only one in this project): *"Pick up the can and place it in the target bin."*

## 2. What the robot sees: the observation

At every step the simulator returns an **observation**: everything the robot knows at that moment.

| Group | Examples | Available on a real robot? |
|---|---|---|
| **Proprioception** (the robot's sense of its own body) | 7 joint angles, gripper tip position `eef_pos`, finger opening | Yes, from motor encoders |
| **Camera images** | front camera `agentview`, wrist camera `eye_in_hand`, 256×256 RGB | Yes |
| **Exact object state** | `Can_pos`, `Can_quat`, target compartment center | **No** |

Information that exists in simulation but not in reality is **privileged information**.
Project rule: **the teacher may use it; the VLA may not.** The VLA gets only images + proprioception + the instruction.

## 3. What the robot can do: the action

Each step (every 50 ms) the policy outputs **4 numbers** in [-1, 1]:

```
[dx, dy, dz, gripper]
 move the gripper tip in x, y, z  (1.0 = "place a target 5 cm away")
 gripper: -1 = open, +1 = close
```

### Policy vs controller

- The **policy** is the brain: it decides *where to go*.
- The **controller** is the muscles: it turns "go 5 cm up" into torques for the 7 joints.
  We use **OSC (Operational Space Control)**.

An action does **not** teleport the hand. It places a target point, and the controller pulls the hand toward
it like a spring. Measured: commanding "5 cm up" every step moves the hand a steady ~1.2 cm per step;
half the command (0.5) gives ~0.64 cm per step.

> **Why this matters for the project:** the same action produces different motion on a different robot or controller.
> SmolVLA was pretrained on other robots whose 6-D actions were **joint positions**; ours are 4-D **Cartesian
> position offsets**. Learning what the numbers mean for this new body is the core of "cross-embodiment adaptation".

### Why Cartesian position control?

- **Joint space**: command 7 joint angles. Moving 1 cm forward means solving **inverse kinematics** yourself.
- **Cartesian space**: command "1 cm forward"; the controller handles the joints.

A 4-number action is much easier for the VLA and RL to learn. Cost: the gripper always points down; no wrist rotation.
That's fine here: the can is a cylinder, so it looks the same from every rotation around its vertical axis.

### Two clocks

Physics updates every **2 ms** (500 Hz); the policy decides every **50 ms** (20 Hz).
One `env.step(action)` = 1 decision + 25 physics steps. A typical episode is about 130 decisions.

## 4. Making the stock task work for Kinova3

robosuite's `PickPlaceCan` was not designed with Kinova3 in mind. We found three problems by measurement:

**Problem 1: the gripper tilted up to 40°.**
`OSC_POSITION` is supposed to control only position, but robosuite resets the orientation target to
*whatever the current orientation is* at every step. So nothing holds the gripper vertical, and it sags
near the edge of the workspace.
*Fix:* capture the orientation target once at the start of each episode and keep it.

**Problem 2: with the gripper held vertical, the target was out of reach (9.7 cm error).**
The target compartment is at the edge of the arm's **workspace** (the set of positions it can reach).
Without problem 1's fix the arm "reached" it only by tilting 40°, which ruins grasping and releasing.
*Fix:* move both bins 15 cm closer to the robot. All grasp and release points are now reachable within ~1 cm.

**Problem 3: narrowing the can's start region crashed the environment.**
robosuite places all 4 objects (milk, bread, cereal, can) in the left bin and then hides 3 of them.
Squeezing all 4 into a smaller area sometimes has no valid layout.
*Fix:* sample the can first in its small region, then the hidden objects in the full bin.

> **Lesson:** a "standard benchmark task" is still a specific design. When you change the robot, check reach
> and controller behavior with small measurements before writing any policy.

## 5. How the teacher works: a Finite State Machine

The teacher is an **FSM (Finite State Machine)**: a fixed sequence of stages; it moves on when a condition is met.

```
APPROACH  → DESCEND → GRASP     → LIFT      → MOVE           → LOWER       → RELEASE  → RETREAT
above can   to can     close,      up to       over target      into the      open,      straight up
            center     wait 10     carry       compartment      compartment   wait 10
                       steps       height                                     steps
```

Every stage uses the same **P controller (proportional controller)**:

```python
action_xyz = clip(gain * (target - eef_pos) / 0.05, -1, 1)
```

Far away it moves at full speed (the action saturates at ±1); near the target it slows down.
It is **closed-loop**: it re-reads the hand position every step, so lag and small errors get corrected.
The opposite, **open-loop**, sends actions without looking at the result.

> This pair returns in Milestone 2: a VLA that executes a long **action chunk** without looking again is closer to open-loop.

Code: `src/newbody/expert/fsm_expert.py`.

### One design choice we measured: carry height

| carry height | success (30 episodes) | why |
|---|---|---|
| 1.05 m | 24/30 | arm can't stretch to the point above the target at that height |
| 1.00 m | 29/30 | one reach failure |
| 0.97 m | 30/30 | but only **1 cm** above the bin walls: a small slip would hit a wall |
| **1.00 m + start lowering within 3 cm of target** | **100/100** | 4 cm wall clearance; avoids the most stretched pose |

We rejected 0.97 m even though it scored 30/30, because the margin was too small. A good engineering
choice is about robustness, not just the number on one run.

## 6. What counts as "success"?

robosuite's rule: the can is in the target compartment **and** the gripper is a few cm away, **at one instant**.

Our stricter rule: the can is in the target compartment **and** the gripper is no longer holding it,
for **10 consecutive steps**. So the can has really been released and stays there.

> **Why be strict?** In RL (Milestone 3) the agent will exploit any loophole in the scoring rule.
> That is **reward hacking**, one of the most common RL pitfalls.

## 7. Two silent bugs we found (and why they matter)

Both produced **no error message**, only worse behavior:

1. **Physics version.** On MuJoCo 3.9.0 the gripper could not close below 6.3 cm, so it could not hold
   a 4 cm object. On 3.3.7 it closes to 1.5 cm. Found by: success grouped by object rotation → contact check → measuring finger gap.
2. **Gripper tilt** (Section 4). Found by printing the gripper orientation during reach tests.

Both are now covered by regression tests (`tests/test_env.py`).

> **Lesson:** when success is low, first find *which layer* is broken (physics? controller? policy?) with the
> smallest experiment. Otherwise you train a VLA on bad data and blame the model.

## 8. From teacher to dataset

### What one training sample looks like

Every step of every teacher episode becomes one **frame**:

| Key | Shape | Content |
|---|---|---|
| `observation.images.agentview` | 3×256×256 | front camera |
| `observation.images.wrist` | 3×256×256 | wrist camera |
| `observation.state` | 5 | gripper tip xyz, 2 finger joints |
| `action` | 4 | dx, dy, dz, gripper (what the teacher did at this step) |
| `task` | text | "Pick up the can and place it in the target bin." |

| front camera (frame 0) | wrist camera (frame 0) |
|---|---|
| ![](../figures/dataset_v1_frame0_agentview.png) | ![](../figures/dataset_v1_frame0_wrist.png) |

Training the VLA to output the teacher's `action` from the observation is **Behavior Cloning (BC)**:
plain supervised learning on (observation, action) pairs.

**Time alignment matters.** Frame *t* pairs the observation **before** acting with the action taken at it:
`(o_t, a_t) → env.step(a_t) → o_{t+1}`. If they were shifted by one step, the model would learn to
predict the action that already happened.

### No privileged information can leak

In LeRobot, **any** key starting with `observation.` automatically becomes a model input.
So the can position and target location are **not** stored in the dataset at all. They go into a separate
"sidecar" file used only for checks. A unit test moves the can and verifies that `observation.state` does not change.

### Splits: no shared start states

Each episode is defined by its seed, and the seed ranges never overlap:

| split | seeds | episodes | use |
|---|---|---|---|
| train | 0 – 49 | 50 | SFT training |
| val | 100000 – 100009 | 10 | validation loss during SFT |
| test | 200000 – … | never recorded | closed-loop evaluation of SFT-only vs SFT+RL |

If test start states also appeared in training, the success rate would measure memorization, not generalization.
That is **data leakage**.

### How we know the data is correct (17 automatic checks, all passed)

- loads through the official LeRobot API; counts match
- images upright and in RGB order (compared against a fresh render; flipped/BGR versions differ ~20× more)
- **replay test**: reset to each episode's seed, execute the stored actions open-loop, compare every state.
  Result: **bit-exact (error 0.0) for all 60 episodes**, and all 60 still succeed.
- no extra/privileged keys; stats consistent; actions in [-1, 1]

The replay test caught a real subtlety. At first we stored actions as float32 but executed them as float64.
That tiny rounding was enough for 1 of 50 replays to drift 2.7 cm and **fail**. Fix: execute exactly the float32
action we store. Lesson: pick-and-place with contacts is **sensitive to tiny action differences**.
That sensitivity is exactly what can hurt a VLA that executes long action chunks open-loop (Milestone 2).

### What the data looks like

![](../figures/dataset_v1_distributions.png)

![](../figures/dataset_v1_episode0.png)

Things to notice:

- **Many actions are saturated at ±1** (46% of `dy`, 16% of `dz`): during long moves the teacher goes at full speed.
- **The gripper is binary** (only −1 or +1). The VLA will output continuous values, so at execution we must
  decide how to turn them back into open/close (e.g. the sign).
- **Actions jump at phase changes** (e.g. `dy` from ~0 to +1 when MOVE starts). The VLA has no phase variable;
  it must infer "which phase am I in" from the images and the state.
- **Episode length** 110 – 148 steps; **start positions** cover the central region of the bin.
- All demos are **clean**: the teacher never makes and recovers from mistakes. A VLA that drifts into an unseen
  state has no example of how to recover. This is **covariate shift / compounding error**, and it is one
  of the failure modes residual RL could fix in Milestone 3.

### Decision: no gripper orientation in the state

Our first dataset had a 9-D state that included the gripper orientation (a quaternion, 4 numbers).
Because we hold the gripper vertical, those 4 numbers barely change (std as small as 0.0004).
SmolVLA normalizes each state dimension as `(x − mean) / std`. A dimension with a tiny std turns a harmless
1° tilt at test time into a huge input value, a classic **normalization pitfall**.
**Decision:** drop them; the state is 5-D (gripper tip xyz + 2 finger joints). Re-collected; all checks pass.

### Practical note: collection speed

Collection first took ~13 s per episode. Measured breakdown: physics 0.6 s, rendering 2 cameras 1.9 s,
**video encoding ~10 s**. LeRobot's default parallel encoding starts new Python processes for every episode,
and each one re-imports torch and robosuite. Serial encoding: 2.3 s. Now **~3 s per episode**
(50 episodes in under 3 minutes).

---

## Interview check

Answer in English (3–5 sentences each), then send your answers for feedback.

1. *What is the difference between the policy and the controller in your setup, and why does it matter for adapting a pretrained VLA to a new robot?*
2. *Your scripted expert reads the exact can position. Why is that allowed for the expert but not for the VLA, and how did you make sure it never leaks into the training data?*
3. *How do you know your demonstration dataset is correct?*

## Concept recap

- **Observation / Action**, **Proprioception**, **Privileged information**
- **Policy vs Controller**, **OSC**, **Joint space vs Cartesian space**
- **Workspace / reachability**
- **Control frequency (20 Hz) vs simulation frequency (500 Hz)**
- **P controller, closed-loop vs open-loop**
- **FSM**
- **Success condition, reward hacking**
- **Behavior Cloning**, **observation/action time alignment**
- **Data leakage**, **train/val/test splits by seed**
- **Normalization (mean/std) pitfalls**
- **Covariate shift / compounding error**
