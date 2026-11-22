# Chapter 1: The Simulated Robot and the Scripted Teacher

## Where this chapter fits

We want SmolVLA to learn to lift a cube with a robot arm it has never seen (Kinova3).
A VLA learns by **imitation**, so it needs demonstrations first.

In the real world a human records demonstrations with a teleoperation device.
In simulation we can write a program, the **scripted expert** (the "teacher"), that grasps the cube automatically.
It is allowed to cheat: it reads the cube's exact position from the simulator.
Its only job is to produce training data. It is a tool, not the research result.

```
Ch.1  teacher grasps automatically (reads exact cube position)      <- this chapter
Ch.2  record "camera images + robot state + action" as a dataset
Ch.3  fine-tune SmolVLA on that data (it only sees images + state + text)
Ch.4  let the VLA grasp on its own, measure success
Ch.5  freeze the VLA, train a small RL network to correct its mistakes
```

**Result of this chapter:** the teacher lifts the cube in **100/100** episodes, with the cube placed randomly within ±10 cm and rotated randomly.

---

## 1. What the robot sees: the observation

At every step the simulator returns an **observation**, a dictionary of everything the robot knows. It falls into three groups:

| Group | Examples | Available on a real robot? |
|---|---|---|
| **Proprioception** (the robot's sense of its own body) | 7 joint angles, gripper tip position `eef_pos`, finger opening | Yes, from motor encoders |
| **Camera images** | front camera `agentview`, wrist camera `eye_in_hand`, 256×256 RGB | Yes, mount a camera |
| **Exact object state** | `cube_pos`, `cube_quat` | **No**, nobody tells you where the cube is |

Information that exists in simulation but not in reality is **privileged information**.
Project rule: **the teacher may use it; the VLA may not.** The VLA only gets images + proprioception + a text instruction like *"Lift the cube"*.

## 2. What the robot can do: the action

Each step, the policy outputs **4 numbers** in the range [-1, 1]:

```
[dx, dy, dz, gripper]
 move the gripper tip in x, y, z  (1.0 = "place a target 5 cm away")
 gripper: -1 = open, +1 = close
```

### The policy and the controller are different things

- The **policy** is the brain: it decides *where to go* (the 4 numbers).
- The **controller** is the muscles: it turns "go up 5 cm" into torques for the 7 joints.
  We use **OSC (Operational Space Control)**.

An action does **not** teleport the hand. It places a target point, and the controller pulls the hand toward it like a spring.
The next action arrives 50 ms later and places a new target.

Measured: commanding `dz = 1.0` ("5 cm up") for 10 steps raised the hand only **11.7 cm**, a steady ~1.2 cm per step.
`dz = 0.5` gave ~0.64 cm per step, roughly half.

> **Why this matters:** the same action on a different robot or controller produces different motion.
> That is one reason cross-embodiment transfer is hard: SmolVLA was pretrained on other robots whose actions meant something else.

### Why control the hand position instead of the joints?

- **Joint space**: command the 7 joint angles directly. To move 1 cm forward you must solve **inverse kinematics** yourself.
- **Cartesian space**: command "1 cm forward"; the controller handles the joints.

We chose Cartesian position control (`OSC_POSITION`) because a 4-number action is much easier for the VLA and RL to learn than 7 joint commands.
Cost: the gripper always points down and cannot rotate its wrist.

## 3. Two clocks: control frequency vs simulation frequency

- **Physics** updates every **2 ms** (500 Hz). It needs small steps, otherwise fingers pass through objects.
- **The policy** decides every **50 ms** (20 Hz).

So one `env.step(action)` = 1 decision + **25 physics steps**, during which the controller keeps recomputing joint torques.

> Looking ahead (Ch.4): one SmolVLA inference can take tens of milliseconds.
> That's why VLAs use **action chunking**: predict many future actions at once, instead of thinking at every 50 ms step.

## 4. Camera images come out upside down

robosuite renders images upside down (OpenGL's origin is the bottom-left corner). We flip them in `get_image()`.

> SmolVLA was pretrained on normally oriented images. Feeding flipped images would not raise any error; it would just quietly lower the success rate.
> Ch.2 will explicitly check orientation and color order.

## 5. How the teacher works: a Finite State Machine

The teacher is an **FSM (Finite State Machine)**: a fixed sequence of stages, moving to the next one when a condition is met.

```
APPROACH → DESCEND → GRASP → LIFT → VERIFY
fly 10 cm   go down to   close the     lift 15 cm   hold and
above cube  cube center  gripper,      straight up  check success
                         wait 10 steps
```

In every stage the motion uses a **P controller (proportional controller)**:

```python
action_xyz = clip(gain * (target - eef_pos) / 0.05, -1, 1)
```

Far from the target it moves at full speed (the action saturates at ±1); close to the target it slows down.

It is **closed-loop**: it re-reads the hand position every step, so errors and lag are corrected automatically.
The opposite, **open-loop**, sends actions without looking at the result.

> This pair comes back in Ch.4: a VLA that executes a long chunk of predicted actions "with its eyes closed" is closer to open-loop.

Code: `src/newbody/expert/fsm_expert.py` (about 100 lines).

Measured gain effect (moving to the point above the cube, seed 0):

| gain | error first below 1 cm | overshoot |
|---|---|---|
| 0.3 | step ~32–36 | none, slow approach |
| 1.0 (default) | step ~16–20 | small, −0.8 cm |
| 5.0 | step ~16 | almost none |

## 6. What counts as "success"?

robosuite's official rule: the cube's center is more than 4 cm above the table **at any single instant**.
Loophole: a cube that gets **knocked or bounced** into the air for one frame counts as a success.

Our stricter rule (`src/newbody/evaluation/success.py`): the cube is lifted **and** actually held by the gripper, for **10 consecutive steps**.

> **Why be strict?** In RL (Ch.5) the agent will exploit any loophole in the scoring rule,
> for example by learning to flick the cube up instead of grasping it.
> This is **reward hacking**, one of the most common RL pitfalls.

## 7. The real bug: a silent physics regression

**Symptom:** the teacher's first run succeeded in only **25%** of episodes, with **no error**.

**Investigation**, one small experiment at a time:

1. Failed videos: the gripper closed but didn't hold the cube.
2. Success grouped by cube rotation: only cubes rotated about **45°** succeeded.
3. Fixed-rotation test: 0° → 0/10, 45° → 9/10.
4. Physical contacts at grasp time: at 0° the fingers **never touched the cube**, even though the hand was centered within 1 mm.
5. Finger gap when fully closed: **6.3 cm**. The cube is 4 cm wide, so it can't be held; only its ~6 cm diagonal (at 45°) reaches the fingers.
6. Compared MuJoCo versions: 3.9.0 → 6.3 cm gap, 3.3.7 → 1.5 cm gap. With 3.3.7: **100% success**.

**Fix:** pin `mujoco==3.3.7` and add a regression test (`test_gripper_closes_fully`) so a future upgrade can't silently break it again.

> **Lesson:** when success is low, first use the smallest experiment that tells you *which layer* is broken (environment? controller? policy?).
> Without this check we would have trained the VLA on bad data and blamed the model.

## 8. Design decision for Ch.2: perfect demos only (option A)

The teacher never makes mistakes, so its demos contain no "recover from an error" behavior.
A VLA trained on them may drift into states the teacher never visited and not know how to recover. Errors then accumulate.
This is **covariate shift / compounding error**, the classic problem of imitation learning.

We chose to **record clean demos first**:

- Ch.4 can then show this failure mode with real evidence.
- Ch.5's residual RL then has a concrete job: fix those failures.

---

## Interview check

Try answering in English (3–5 sentences each), then send your answers for feedback.

1. *What is the difference between the policy and the controller in your setup? What does one action actually do?*
2. *Your scripted expert reads the exact cube position. Why is that allowed for the expert but not for the VLA?*
3. *Your expert's success rate was 25% at first. How did you find the cause?*

## Concept recap

- **Observation / Action**: what the robot knows / what it can command
- **Proprioception**: the robot's sense of its own body
- **Privileged information**: available in simulation, not in reality
- **Policy vs Controller**: decide where to go vs how to apply joint torques
- **Joint space vs Cartesian space**
- **Control frequency (20 Hz) vs simulation frequency (500 Hz)**
- **P controller, closed-loop vs open-loop**
- **FSM**
- **Success condition, reward hacking**
- **Covariate shift / compounding error**
