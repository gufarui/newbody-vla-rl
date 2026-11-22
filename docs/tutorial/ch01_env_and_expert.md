# Chapter 1: Make the Arm Move, and Write a Teacher That Grasps Automatically

## By the end of this chapter you can

- write code that moves the robot arm in simulation
- explain clearly "what the robot sees (observation)" and "what the robot can do (action)"
- write a small controller that flies the gripper to a point above the cube
- understand how the project's automatic grasping teacher (scripted expert) works
- answer 3 interview questions

Estimated time: 1.5 – 2 hours. You don't need to do it in one sitting.

## Why does this chapter come first?

In the end we want the VLA to learn to grasp a cube. A VLA learns by **imitation**, so someone has to demonstrate the task first.

In the real world, demonstrations are recorded by a human with a teleoperation device. In simulation, we can write a program to act as the "teacher":
it can read the cube's exact position directly (it cheats) and then grasp automatically. The teacher grasps a few hundred times, we record it, and that becomes the VLA's training material.

Before writing the teacher, you need to know how to make the arm move. So this chapter has two halves: **play with the environment first, then write the teacher**.

---

## Setup

Open a terminal and go to the project folder:

```bash
cd ~/Desktop/robotics学习/newbody-vla-rl
```

Check that the environment works (runs 6 small tests, a few seconds):

```bash
uv run pytest -q
```

> `uv run` means: run the following command with this project's own Python environment (`.venv`),
> so you don't need to activate the virtual environment by hand.

When you see `6 passed`, you're ready. Lots of `[robosuite WARNING]` lines get printed; you can ignore them.

---

## Step 1: Open the environment and see what the robot "sees"

### Do it

Create `playground/step1_look.py` and **type** this code yourself (don't copy-paste; typing it helps you remember):

```python
import numpy as np
import yaml
from newbody.env.lift_env import make_lift_env, reset_with_seed

cfg = yaml.safe_load(open("configs/lift_kinova3.yaml"))
env = make_lift_env(cfg["env"])
obs = reset_with_seed(env, seed=0)

for key, value in obs.items():
    print(f"{key:30s} shape={np.shape(value)}")
print("action_dim =", env.action_dim)
```

Line by line:

- `cfg = ...`: reads the config file. Open `configs/lift_kinova3.yaml` and take a look: robot model, task name, cameras, and so on.
- `make_lift_env`: creates the simulation environment. It's a thin wrapper around robosuite; the code is in `src/newbody/env/lift_env.py`.
- `reset_with_seed(env, seed=0)`: resets the world to an initial state. The same seed always gives the same cube position, so experiments are reproducible.
- `obs`: the observation, a dictionary holding everything the robot "knows" right now.

Run it:

```bash
uv run python playground/step1_look.py
```

### Observe

You'll see about 17 lines, each one a type of information and its shape.

### Explanation

Sort these keys into three groups and they make sense:

**1. The robot's sense of its own body (proprioception)**
Like how you know where your hand is even with your eyes closed.

- `robot0_joint_pos` (7,): the angles of the 7 joints, because Kinova3 has 7 joints (7 DoF, degrees of freedom)
- `robot0_eef_pos` (3,): the x, y, z position of the gripper tip in space (eef = end-effector)
- `robot0_gripper_qpos` (6,): how open or closed the gripper fingers are

**2. Camera images**

- `agentview_image` (256, 256, 3): a camera looking from the front, 256×256 pixels, 3 RGB channels
- `robot0_eye_in_hand_image`: a camera mounted on the wrist

**3. Exact object information**

- `cube_pos` (3,): the exact coordinates of the cube
- `gripper_to_cube_pos`: the vector from the gripper to the cube

Finally, `action_dim = 4` means we only need to give the robot **4 numbers** per step. We'll try that next.

### Think about it

**Q1.** For a real robot in the real world, which of these three groups can you get directly, and which can't you get?

<details><summary>Show answer</summary>

Group 1 (joint angles, gripper position): yes, the motors have encoders.
Group 2 (images): yes, just mount a camera.
Group 3 (the cube's exact coordinates): **no**. In reality nobody tells you where the cube is; you have to infer it from images.

Information that exists in simulation but not in reality is called **privileged information**. It is a hard rule for this whole project: **the teacher may use it, the VLA may not**. The VLA only gets images + proprioception + a text instruction.

</details>

---

## Step 2: Move the arm up

### Do it

Create `playground/step2_move_up.py`:

```python
import numpy as np
import yaml
from newbody.env.lift_env import make_lift_env, reset_with_seed

cfg = yaml.safe_load(open("configs/lift_kinova3.yaml"))
env = make_lift_env(cfg["env"], render_cameras=False)   # no camera rendering, runs faster
obs = reset_with_seed(env, seed=0)

z_start = obs["robot0_eef_pos"][2]
action = np.array([0.0, 0.0, 1.0, -1.0])   # dx, dy, dz, gripper
for t in range(10):
    obs, reward, done, info = env.step(action)
    print(f"step {t}: eef z = {obs['robot0_eef_pos'][2]:.3f}")
print(f"total rise = {obs['robot0_eef_pos'][2] - z_start:.3f} m")
```

What the 4 numbers mean:

- First 3: how far to move the gripper in x, y, z, each in the range -1 to 1
- 4th: the gripper, -1 = open, +1 = close

**A value of 1 corresponds to 5 cm** (set in robosuite's controller config). So `dz = 1.0` means "move up 5 cm".

### Predict first (write it down before running!)

Every step commands "up 5 cm", 10 steps in total. How far do you predict it rises in total?

```bash
uv run python playground/step2_move_up.py
```

### Observe

<details><summary>Run it yourself first, then expand to compare with the reference output</summary>

```
step 0: eef z = 0.964
step 1: eef z = 0.973
...
step 9: eef z = 1.077
total rise = 0.117 m
```

It only rose about **12 cm**, not 50 cm. Roughly 1.3 cm per step.

</details>

### Explanation

Why did we command 5 cm but only get 1.3 cm? You need to separate two things:

**The policy** is the "brain": it decides "where do I want to go". That's the 4 numbers you give.

**The controller** is the "cerebellum + muscles": it turns "I want to go up 5 cm" into how much force each of the 7 joints should apply.
Ours is called **OSC (Operational Space Control)**.

The key point: `dz = 1` does not mean "teleport 5 cm". It means **place a target point 5 cm above the current position, and the controller pulls the hand toward it like a spring**.
50 milliseconds later the next action arrives and a new target point is placed. The arm has mass and inertia, so in 50 ms it only gets pulled a little over 1 cm.

An analogy: you walk a dog and keep throwing a ball ahead, but you throw the next ball before the dog reaches the last one. The dog is always chasing, and its running speed depends on the dog itself.

> This matters a lot later: **the action is not the motion that actually happens**.
> The same action on a different robot or with a different controller has a different effect.
> That is one of the reasons "cross-embodiment" transfer is hard.

Why do we control the end-effector position instead of the 7 joints directly?

- Controlling joints directly (**joint space**): to move forward 1 cm you'd have to compute how many degrees each of the 7 joints must turn. That's **inverse kinematics**, and it's tedious.
- Controlling the end-effector position (**Cartesian space**): just say "forward 1 cm" and the controller works it out for you.

The simpler the action, the easier it is for the VLA and RL to learn later. The cost: we only control position, the gripper always points down, and the wrist can't rotate.

### Think about it

**Q2.** Change `range(10)` to `range(40)` and run again. Does the rise per step get bigger, get smaller, or stay about the same? What does that tell you?

<details><summary>Show answer</summary>

It stays almost constant, about 1.2 cm per step (about 49 cm over 40 steps).
Because the target point is always 5 cm above the hand, the "spring" is always stretched by the same amount, so the arm rises at a steady speed.
That speed is set by the controller's parameters (how stiff the spring is, how much damping), not directly by your action.

</details>

**Q3.** Change `action` to `[0.0, 0.0, 0.5, -1.0]`. About how far does it move per step? Is it proportional to the 1.0 case?

<details><summary>Show answer</summary>
About 0.64 cm/step, roughly half of the 1.0 case (1.2 cm/step). The target point is half as far, the spring is stretched half as much, so the speed is roughly halved.
</details>

---

## Step 3: Two kinds of "frequency"

### Do it

Create `playground/step3_freq.py`:

```python
import yaml
from newbody.env.lift_env import make_lift_env

cfg = yaml.safe_load(open("configs/lift_kinova3.yaml"))
env = make_lift_env(cfg["env"], render_cameras=False)
print("control_freq  =", env.control_freq, "Hz")
print("sim timestep  =", env.model_timestep, "s")
```

```bash
uv run python playground/step3_freq.py
```

### Explanation

You'll see `20 Hz` and `0.002 s`.

- The **physics simulation** computes everything every 0.002 s (2 ms): all forces, collisions, velocities, positions. That's 500 times per second. Physics needs fine time steps, otherwise it computes wrong results (for example, a finger passing through the cube).
- The **policy** only makes 20 decisions per second (one every 50 ms). The brain doesn't need to be that fast.

So every time you call `env.step(action)`, this is what actually happens inside:

```
you give one action
  → the controller sets a target point from the action
  → the physics runs 25 small steps (2 ms each, 50 ms total); in each small step the controller recomputes the joint forces
  → a new observation is returned
```

### Think about it

**Q4.** Why exactly 25 small steps?

<details><summary>Show answer</summary>
1 / 20 Hz = 0.05 s, and 0.05 / 0.002 = 25.
</details>

**Q5.** Later, one VLA inference may take tens of milliseconds. If the VLA had to think again at every step, what's the highest Hz it could run at? Do you think that would be a problem?
(Write down your idea. We come back to it in Chapter 4, which covers **action chunking**: the VLA plans several future steps at once.)

---

## Step 4: Look at the camera image

### Do it

Create `playground/step4_image.py`:

```python
import imageio
import yaml
from newbody.env.lift_env import make_lift_env, reset_with_seed

cfg = yaml.safe_load(open("configs/lift_kinova3.yaml"))
env = make_lift_env(cfg["env"])
obs = reset_with_seed(env, seed=0)

raw = obs["agentview_image"]
print(raw.shape, raw.dtype, raw.min(), raw.max())
imageio.imwrite("playground/raw.png", raw)
imageio.imwrite("playground/flipped.png", raw[::-1])
```

```bash
uv run python playground/step4_image.py
open playground/raw.png playground/flipped.png
```

(`open` is a macOS command that opens files with the default app.)

### Observe

What's the difference between the two images? Which one is correct?

### Explanation

robosuite outputs images **upside down** (because the underlying OpenGL coordinates start at the bottom-left corner). `raw[::-1]` reverses the order of the rows, which flips the image vertically.

`uint8` and `0 ~ 255`: each color channel of each pixel is an integer from 0 to 255.

This looks like a small detail, but it can be fatal:

> SmolVLA was pretrained on normally oriented images. If we feed it upside-down images, it's like seeing the world upside down. Learning becomes much harder, and **nothing raises an error**; the success rate is just mysteriously low.

When we record data in Chapter 2, we'll explicitly check image orientation and color order (RGB vs BGR).

---

## Step 5: Write your own controller to fly above the cube

So far we gave fixed actions. Now you'll write your first **closed-loop controller**: at every step it checks "where am I, where is the target" and then decides where to move.

### Do it

Create `playground/step5_approach.py` and fill in the `TODO` line:

```python
import sys
import numpy as np
import yaml
from newbody.env.lift_env import make_lift_env, reset_with_seed

GAIN = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0

cfg = yaml.safe_load(open("configs/lift_kinova3.yaml"))
env = make_lift_env(cfg["env"], render_cameras=False)
obs = reset_with_seed(env, seed=0)

for t in range(40):
    eef = obs["robot0_eef_pos"]
    cube = obs["cube_pos"]
    target = cube + np.array([0.0, 0.0, 0.10])     # 10 cm directly above the cube
    error = target - eef                            # how far we still are (meters)

    # TODO: compute action_xyz from error (3 numbers, each between -1 and 1)
    # Hint: action 1.0 corresponds to 0.05 m. Large error -> move fast, small error -> move slowly.
    #       np.clip can limit the range.
    action_xyz = ...

    action = np.append(action_xyz, -1.0)           # keep the gripper open
    obs, reward, done, info = env.step(action)
    if t % 4 == 0:
        print(f"t={t:2d}  error(cm) = {np.round(error * 100, 1)}  |error| = {np.linalg.norm(error) * 100:.1f} cm")
```

Think for 5 minutes before you look at the answer.

<details><summary>Reference answer</summary>

```python
action_xyz = np.clip(GAIN * error / 0.05, -1.0, 1.0)
```

- `error / 0.05`: converts "meters" into "action units"
- `GAIN`: an amplification factor
- `np.clip(..., -1, 1)`: cuts off anything outside the range

</details>

Run it with `GAIN` = 1.0:

```bash
uv run python playground/step5_approach.py 1.0
```

### Explanation

What you just wrote is a **P controller (proportional controller)**: the action is proportional to the error.
Far away, it moves fast; close to the target, it slows down.

It's **closed-loop**: it re-measures `eef` at every step, so even if the controller has some offset or the arm has inertia, the next step corrects it automatically.

Compare with Step 2: that was **open-loop**. You just gave actions and never looked at the result.

> Remember this pair of concepts. Chapter 4 meets the "big version" of it:
> does the VLA predict 50 future actions and execute them with its eyes closed (closer to open-loop),
> or does it look again every few steps (closer to closed-loop)?

### Experiment: change GAIN

**Predict first**: with GAIN = 0.3 and GAIN = 5.0, which one arrives faster? Will either overshoot (error becomes negative)? Write it down, then run:

```bash
uv run python playground/step5_approach.py 0.3
uv run python playground/step5_approach.py 5.0
```

<details><summary>Reference output (seed=0, look at the x error column)</summary>

| GAIN | error first < 1 cm | overshoot? |
|---|---|---|
| 0.3 | around t=32~36 | no, approaches slowly |
| 1.0 | around t=16~20 | a little: x error goes to -0.8 cm, then comes back |
| 5.0 | around t=16 | almost none, final error reaches 0.0 |

</details>

### Think about it

**Q6.** With a large GAIN, what is the action most of the time? (Hint: look at `np.clip`.) So what determines the speed when far away?

<details><summary>Show answer</summary>

With a large GAIN, `GAIN * error / 0.05` is much larger than 1 and gets clipped to ±1. So far away it's always "full speed", and the speed limit comes from the action range and the controller.
Only when very close does it enter the "proportional" region and start slowing down.

</details>

**Q7.** Why does GAIN = 1.0 overshoot a little while 5.0 barely does?
There's no single correct answer here; just share your guess. (Hint: remember the dog-and-ball analogy from Step 2: the action places a target point, and the controller pulls like a spring.)

---

## Step 6: See how the project's teacher grasps the cube

You can already "fly above the cube". A full grasp just adds a few more stages:

```
APPROACH  →  DESCEND        →  GRASP         →  LIFT     →  VERIFY
fly above    go down to cube    close gripper    lift it     check success
```

A program that "works in stages and switches to the next stage when a condition is met" is called an **FSM (Finite State Machine)**.

### Do it: read the code

Open `src/newbody/expert/fsm_expert.py` and find the `act` function. Read along and answer:

1. What is `target` in the APPROACH stage? Is it the same as what you wrote in Step 5?
2. Under what condition does it switch from APPROACH to DESCEND?
3. Why does the GRASP stage stay in place for 10 steps? (Look at `grasp_close_steps`.)
4. How does the `_move` function differ from your P controller?

### Do it: run the teacher

```bash
uv run python scripts/run_expert.py --episodes 5 --videos 5 --name my_first_run
open results/expert/my_first_run/seed0_success.mp4
```

In the video, the left side is the front camera and the right side is the wrist camera.

Now look at what the teacher did at each step. `steps_seed0.csv` records the stage, action and heights for every step. The command below prints only the rows where the **stage changes**:

```bash
uv run python -c "
import csv
prev = None
for r in csv.DictReader(open('results/expert/my_first_run/steps_seed0.csv')):
    if r['phase'] != prev:
        print(r['t'], r['phase'], 'grip=', r['grip'], 'cube_height=', r['cube_z_above_table'], 'grasping=', r['grasping'])
        prev = r['phase']
"
```

### Think about it

**Q8.** When the cube rests on the table, `cube_z_above_table` is about 0.02. Why isn't it 0?
<details><summary>Show answer</summary>
It records the height of the cube's **center**. The cube's side is about 4 cm, so its center is about 2 cm above the table.
</details>

**Q9.** At which step does `grasping` become 1? Is it the moment the gripper starts closing?

---

## Step 7: How do we decide "success"?

robosuite's official rule: if **at any single instant** the cube's center is more than 4 cm above the table, it counts as success.

### Think about it (think first, then read on)

**Q10.** What's the loophole in this rule? Can you come up with a case where the cube is "not really grasped and lifted, but still judged a success"?

<details><summary>Show answer</summary>

For example, the gripper **knocks the cube flying** or **bounces it up**. In one frame mid-air it's above 4 cm, so it counts as a success.

</details>

So we wrote a stricter rule: the cube must be lifted **and** actually held by the gripper, for **10 consecutive steps**.
The code is short. Open `src/newbody/evaluation/success.py` and read it (only about 20 lines).

> **Why be so strict?** When we do RL in Chapter 5, the robot will try anything to raise its score.
> If the rule has a loophole, it will learn to exploit the loophole (for example, learn to flick the cube up) instead of learning to grasp.
> This is called **reward hacking**, and it's one of the most common pitfalls in RL.

---

## Step 8: A real bug story

This is a problem we actually hit while building this chapter, and it's the most valuable lesson in it.

**Symptom**: the teacher's first run had only 25% success, and **no error at all**.

**Investigation** (each step is just one small experiment):

1. Watched the failed videos and logs: the gripper closed but didn't hold the cube.
2. The cube's orientation is random each time. Grouped success by orientation: **it only succeeded when the cube was rotated about 45°**.
3. Fixed the cube orientation and tested again: 0° gave 0/10, 45° gave 9/10. The effect reproduced reliably.
4. Checked physical contacts: at 0°, the fingers **never touched the cube at all**.
5. Measured the finger gap when fully closed: **6.3 cm**! The cube is only 4 cm wide, so of course it can't be held. Only at 45° does the cube's diagonal (about 6 cm) just reach the fingers.
6. Switched the MuJoCo version (3.9.0 → 3.3.7): the gap became 1.5 cm and success went to 100%.

**Conclusion**: the newer physics engine is incompatible with this gripper model, **but it raises no error; it just quietly makes the robot worse**.

### Do it

We turned this check into a test, so anyone who upgrades MuJoCo later gets stopped:

```bash
uv run pytest -k gripper -v
```

Open `tests/test_env.py`, find `test_gripper_closes_fully`, and make sure you understand what it tests.

### Think about it

**Q11.** What would have happened if we hadn't investigated, and had used this 25% teacher to record data and train the VLA?

<details><summary>Show answer</summary>

Worst case: after training, the VLA has a low success rate. We'd think "the VLA is bad" or "there isn't enough data", and spend lots of GPU time tuning the model.
In fact the problem was at the very bottom layer, the physics simulation.

**Lesson: when the success rate is low, first use the smallest possible experiments to find which layer the problem is in (environment? controller? policy?), and only then change things.**

</details>

---

## Step 9: Your mini research: how does one parameter affect the teacher?

This is the last exercise of the chapter, done like a mini research project: **state a hypothesis first, then test it with an experiment**.

Pick **one** of these parameters:

- `approach_height`: how high above the cube to stop before descending (default 0.10 m)
- `action_gain`: the teacher's P-controller gain (default 1.0, which is your GAIN from Step 5)

### 1. Write down your hypothesis first

In `playground/step9_notes.md`, write:

```
Parameter I chose:
Values I will try:
I predict the success rate will:
I predict the episode length will:
Reasoning:
```

### 2. Run the experiment

`--set` temporarily changes the teacher's parameters. For example:

```bash
uv run python scripts/run_expert.py --episodes 30 --cube-xy-range 0.10 --videos 1 --set approach_height=0.02 --name ex_ah002
uv run python scripts/run_expert.py --episodes 30 --cube-xy-range 0.10 --videos 1 --set action_gain=0.3 --name ex_gain03
```

### 3. Look at the results

- The last line prints the success rate
- `results/expert/<name>/episodes.csv`: each episode's length (`length`) and failure reason (`fail_reason`, which shows the stage it got stuck in)
- Failed episodes automatically save a video; open it to see what actually happened

### 4. Write your conclusion

Was your prediction right? If not, what do you think the reason is? Write it in `step9_notes.md` and send it to me; we'll discuss it together.

---

## Interview practice

Answer **in English**, out loud or in writing, then send your answers to me and I'll point out anything inaccurate. 3 – 5 sentences per question is enough.

1. *What is the difference between the policy and the controller in your setup? What does one action actually do?*
2. *Your scripted expert reads the exact cube position. Why is that allowed for the expert, but not for the VLA?*
3. *Your expert's success rate was 25% at first. How did you find the cause?*

## Concept recap

These terms only mean something once you've done the steps above:

- **Observation / Action**: what the robot knows / what the robot can do (Steps 1, 2)
- **Proprioception**: the robot's sense of its own body, e.g. joint angles and end-effector position (Step 1)
- **Privileged information**: information that exists in simulation but not in reality (Step 1)
- **Policy vs Controller**: the brain decides where to go; the cerebellum decides how hard to push (Step 2)
- **Joint space vs Cartesian space**: controlling joint angles vs controlling the end-effector position (Step 2)
- **Control frequency vs simulation frequency**: 20 Hz vs 500 Hz (Step 3)
- **P controller / closed-loop / open-loop** (Step 5)
- **FSM**: Finite State Machine (Step 6)
- **Success condition / reward hacking** (Step 7)
