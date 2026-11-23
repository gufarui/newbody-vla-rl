# Milestone 3 design — RLT-inspired residual SAC on a frozen SmolVLA (agreed 2026-11-23)

**Frozen base policy:** SFT checkpoint 10k (`outputs/sft_v1/checkpoints/010000/pretrained_model`), all weights frozen.
Baseline it must beat (test seeds, full bin, 10 actions/chunk): **120/200 = 60%** [53, 67].

## MDP

- **Decision step = one action chunk = 10 env steps** (VLA re-plans every 10 steps; RL decides at the same boundaries).
- Transition: `(o_k, δ_k, r_k = sum of rewards over the 10 steps, o_{k+1}, done)`, discount per chunk step.
- Start states: can uniform over the **full source bin** (can_x ±0.145, can_y ±0.195), RL seed range **400000+**
  (disjoint from train 0–49, val 100000+, test 200000+). Same distribution as the final test; report inside/outside
  the demo region separately.

## Action (residual)

- 3-D xyz offset per chunk: `a_t = clip(a_ref_t + λ·[δ, 0], -1, 1)` for the 10 steps, δ ∈ [-1, 1]³ (tanh), **λ = 0.2**
  (≤ 1 cm target offset per step). Gripper is never modified.
- Actor initialized so δ ≈ 0 at start → initial policy = SFT baseline.

## Actor observation (deployable only — no privileged state)

1. Frozen VLA features: VLM prefix hidden states from the same forward pass that produces the reference chunk
   (1 pass/chunk, verified), **mean-pooled per camera** (front 64 tokens, wrist 64 tokens) and concatenated -> 1920-D
   -> small learned projection. Evidence (linear probe, 277 pre-grasp decision points): can-minus-gripper xy offset
   R2 0.81/0.96 from front-cam tokens vs 0.67/0.87 from a mean over all tokens and 0.07/0.42 from robot state.
2. The VLA reference chunk for these 10 steps (10 × 4).
3. Proprioceptive state (5-D).

## Critic (asymmetric, training only)

Same inputs as the actor **plus privileged state** (can position, target compartment). Used only to train the actor;
never at deployment/evaluation. Must be stated in the report.

## Reward

- +1 on strict success (can in target compartment, released, 10 consecutive steps) → episode ends.
- Residual penalty `-β‖δ‖²` per decision (β small, to be set so it never outweighs success).
- Can knocked over (tipped) → episode ends with 0 (unrecoverable; saves compute).
- No shaping at first; add potential-based shaping only if learning is too slow (evidence required).

## Algorithm

SAC: small MLP actor, twin Q-critics + target networks, replay buffer, automatic entropy tuning.
(RLT itself uses TD3 with a chunk-level actor and β‖a − ã‖² regularization; we keep SAC per spec.)

## Checks before paid training

- feature hook returns features of the *current* observation, fixed size, no second full VLA pass
- VLA weights byte-identical before/after RL
- residual scale/bounds correct; δ = 0 reproduces the SFT baseline exactly
- small local SAC sanity run
