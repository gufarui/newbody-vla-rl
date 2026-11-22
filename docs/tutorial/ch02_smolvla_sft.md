# Chapter 2: SmolVLA — How a VLA Turns Pixels and Words into Actions

## Where this chapter fits

Milestone 1 gave us 50 demonstrations of the teacher doing pick-and-place.
Now we fine-tune **SmolVLA**, a pretrained Vision-Language-Action model, to imitate them.
After training, the VLA must do the task **on its own**, seeing only camera images, robot state, and the instruction.

This chapter has two parts:

- **Part A (this page, before training):** what goes into SmolVLA, what comes out, and how one training step works.
  Every number here comes from one real training step on our data (`scripts/inspect_smolvla_batch.py`).
- **Part B (after GPU training):** training curves, closed-loop success rate, failure analysis. *(to be written)*

---

# Part A — Inside one training step

## 1. The big picture

A VLA is two networks working together:

```
 front camera ─┐
 wrist camera ─┤                                           ┌─ "noisy" action chunk (50 × 4)
 instruction  ─┼─► VLM (vision-language model) ──context──►│  ACTION EXPERT  ──► velocity ──► cleaner action chunk
 robot state  ─┘   "understand the scene"                  └─ (repeat 10×: denoise)
                   350M params, FROZEN                        100M params, TRAINED
```

- The **VLM** (SmolVLM2-500M, cut to 16 layers) is a model pretrained on images and text from the internet.
  It turns the scene and the instruction into a rich internal description.
- The **action expert** is a smaller transformer that reads that description and **generates a chunk of 50 future actions**.

Analogy: the VLM is someone who looks at the table and understands "the can is there, the target bin is there."
The action expert is the hand that knows how to move. Fine-tuning mostly teaches the hand our new robot.

## 2. What goes in (real shapes from our batch, batch size 2)

| Input | Shape | Notes |
|---|---|---|
| `camera1` (our front camera) | 2 × 3 × 256 × 256 | renamed from `agentview` (see below) |
| `camera2` (our wrist camera) | 2 × 3 × 256 × 256 | renamed from `wrist` |
| `observation.state` | 2 × 1 × 5 | normalized: `(x − mean) / std` |
| language tokens | 2 × 48 | our instruction uses 13 tokens; the rest is padding |
| `action` (training target) | 2 × 50 × 4 | the teacher's **next 50 actions**, normalized |

**Why rename the cameras?** SmolVLA was pretrained with cameras called `camera1/2/3`. A pretrained model keeps
its input names, so we map ours onto them with LeRobot's official `rename_map`. We have only 2 cameras;
SmolVLA simply skips the missing `camera3`.

Inside the model:

- **Images** are resized to 512 × 512 (with padding, keeping the aspect ratio) and scaled to [−1, 1], as the vision encoder expects.
- Each image becomes **64 visual tokens**. A token is one vector (960 numbers) describing part of the input.
- The **prefix** the VLM reads: 64 + 64 image tokens + 48 language tokens + 1 state token = **177 tokens**.
- **State and actions are padded to 32 dimensions.** SmolVLA supports many robots, so it uses a fixed max size.
  Our 5-D state and 4-D action fill the first slots; the rest are zeros. The loss **ignores** the padded action dims.

## 3. What comes out: an action chunk

SmolVLA doesn't predict one action. It predicts the **next 50 actions at once** (2.5 s at 20 Hz). This is **action chunking**.

Why?

- **Speed:** one VLA inference takes tens of milliseconds or more. Predicting 50 actions per inference avoids thinking at every 50 ms step.
- **Smoothness and consistency:** a whole chunk is planned together, so consecutive actions fit together.

The risk: executing many actions without looking again is close to **open-loop**. In Milestone 1 we saw this task is
sensitive to tiny action errors. That's why we agreed to compare executing **10 vs 50** actions per chunk in evaluation.

Near the end of an episode there are fewer than 50 future actions left. Those slots are marked `action_is_pad`
(in our batch: 21 of 100 slots) and excluded from the loss.

## 4. How the action expert generates actions: Flow Matching

### The idea, without math

Start from **pure random noise** shaped like an action chunk (50 × 4). Then, in 10 small steps, move it toward a
realistic action chunk. At each step the network answers one question:

> "From this noisy chunk, in which direction should I move to get closer to a good action chunk?"

That direction is called the **velocity**. Following it for 10 steps turns noise into actions.
(Like sculpting: start with a rough block, remove a bit at each step.)

### Training (what we will do on the GPU)

For each training sample:

1. Take the teacher's real action chunk `A` (50 × 4).
2. Draw random noise `ε` of the same shape, and a random time `t` between 0 and 1 (our batch: t = 0.95 and 0.78).
3. Mix them: `x_t = t · ε + (1 − t) · A`. At t = 1 it's pure noise; at t = 0 it's the real actions.
4. The correct direction from noise toward the real actions is `u = ε − A` (the model moves *against* it at inference).
5. The action expert sees `x_t`, `t`, and the VLM context, and predicts a velocity `v`.
6. **Loss = mean squared error between `v` and `u`**, over valid (non-padded) entries.

We checked in the real run that `x_t` and `u` are exactly these formulas. The first-step loss on our data was 1.52.

### Inference (what happens when the robot acts)

Start with noise `x_1 = ε`. Repeat 10 times: `x ← x − 0.1 · v(x, t)`. The result is the action chunk.
The VLM context is computed **once** and cached; only the small action expert runs 10 times.

### Why not just predict the action with MSE regression?

Plain regression learns the **average** of what the teacher did. If, in some situation, the teacher might go
left **or** right, the average goes straight into the obstacle. Flow matching learns a whole **distribution**
of good action chunks and samples one, so it can represent several valid options (**multimodality**).
It also produces smooth, continuous actions, which suits robot control.

> Honest note for our project: our teacher is deterministic (no randomness), so multimodality matters less here
> than with human demonstrations. Flow matching is still what SmolVLA was pretrained with, so we keep it.

## 5. What fine-tuning changes: frozen vs trainable

| Part | Parameters | During our SFT |
|---|---|---|
| VLM (vision encoder + language model, 16 layers) | 350M | **frozen** |
| Action expert transformer | 98M | trained |
| `state_proj` (5-D state → 960) | 31K | trained |
| `action_in_proj` / `action_out_proj` (action ↔ 720) | 23K each | trained |
| time MLPs (mix the flow time `t` into the action tokens) | 1.5M | trained |

**Total: 450M parameters, 100M trained.** This is the official SmolVLA fine-tuning setup (`train_expert_only=True`, `freeze_vision_encoder=True`).

- **Frozen parameters** are fixed: no gradients flow into them, the optimizer never touches them.
- **Backpropagation** computes, for each trainable parameter, how changing it would change the loss; the optimizer then
  nudges every trainable parameter a little in the direction that lowers the loss.

**Why freeze the VLM?** It already understands images and language from huge pretraining data. With only 50
demonstrations, updating 350M more parameters would mostly overfit and could damage that general knowledge.

### Pretraining vs fine-tuning, for our project specifically

- **Pretraining** (done by Hugging Face): many robots and tasks, mostly SO-100 arms, 6-D **joint-position** actions, 3 cameras.
- **Fine-tuning** (us): one robot, one task, 4-D **Cartesian offset** actions, 2 cameras, 50 demos.

The VLM's understanding of scenes transfers. The action meaning does **not**: `state_proj` and `action_in/out_proj`
were trained for joint positions, and now must learn what our numbers mean. Expect the action expert to do most of the adapting.

## 6. Normalization: why every number is rescaled

Our raw numbers live on very different scales (eef position ~0.05 m spread, gripper −1/+1).
Before entering the model, state and actions are normalized per dimension: `(x − mean) / std`, using statistics
from **our** dataset. The model then works with numbers around −2…+2, and the **postprocessor** undoes this
to produce real actions for the robot.

Example from the real batch: a normalized first action was `[0.40, 1.06, 0.34, 0.71]` (the gripper dim 0.71 = "close").
This is also why we removed the near-constant gripper orientation from the state (Chapter 1).

---

## Interview check (Part A)

Answer in English, 3–5 sentences each:

1. *What are the inputs and outputs of SmolVLA in your project, and what shapes do they have?*
2. *Explain flow matching training in simple terms. Why use it instead of MSE regression on actions?*
3. *Which parameters did you fine-tune, which did you freeze, and why?*

## Concept recap

- **VLA = VLM + action expert**; **visual / language / state tokens**; **prefix**
- **Action chunking** (50 steps); open-loop risk
- **Flow matching**: noise → actions via a learned velocity; training target `u = ε − A`; 10 Euler steps at inference
- **SFT**, **loss**, **backpropagation**, **frozen vs trainable parameters**
- **Pretraining vs fine-tuning**; why action meaning doesn't transfer across embodiments
- **Normalization** and **padding** to fixed dimensions
