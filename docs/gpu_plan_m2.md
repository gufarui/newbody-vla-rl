# Milestone 2 GPU plan (for approval)

**Machine:** rented RTX 4090D (24 GB), 18 CPU cores, 60 GB RAM, 50 GB data disk, ¥1.88 / hour.

## What runs where

Everything that needs the VLA runs **on the server**: training, and closed-loop evaluation (MuJoCo + SmolVLA in the
same process, headless, `MUJOCO_GL=egl`). The network only carries code, data (~75 MB), commands and results.

## Steps

| # | Step | Est. time |
|---|---|---|
| 1 | Environment: `uv export` lock -> pip with China mirror; copy code + dataset with rsync; download `smolvla_base` + SmolVLM2 via `HF_ENDPOINT=https://hf-mirror.com` | 40–60 min |
| 2 | GPU smoke: 20 steps at batch 64 -> measure s/step and VRAM; recompute the time/cost estimate | 5 min |
| 3 | SFT: official recipe, batch 64, 20,000 steps, checkpoint every 5,000 (in `tmux`, survives SSH disconnects, resumable) | ~2.5–3.5 h (to be confirmed in step 2) |
| 4 | Checkpoint selection on **val** seeds: 4 checkpoints × 20 episodes, executing 10 actions per chunk | ~15 min |
| 5 | Final test of the chosen checkpoint on **test** seeds: 200 episodes × (10 and 50 actions per chunk) | ~45 min |
| 6 | rsync results + chosen checkpoint (~0.9 GB) to the Mac; **shut the instance down** | 15 min |

**Total ≈ 5–7 hours ≈ ¥10–14.** Stop rule: if the measured speed in step 2 pushes the estimate above ¥20, I stop and ask.

## Training command (step 3)

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=local/kinova3_pickplace_can_v1_train --dataset.root=data/kinova3_pickplace_can_v1/train \
  --rename_map='{"observation.images.agentview": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --batch_size=64 --steps=20000 --save_freq=5000 --log_freq=100 --num_workers=8 --seed=1000 \
  --output_dir=outputs/sft_v1 --job_name=sft_v1
```

## Evaluation commands (steps 4–5)

```bash
MUJOCO_GL=egl python scripts/eval_vla.py --checkpoint outputs/sft_v1/checkpoints/005000/pretrained_model \
  --split val --episodes 20 --n-action-steps 10 --videos 3 --name sft_v1_005000_val_k10
# ... same for 010000, 015000, 020000; pick the best on val, then:
MUJOCO_GL=egl python scripts/eval_vla.py --checkpoint <chosen> --split test --episodes 200 --n-action-steps 10 --videos 10 --name sft_v1_test_k10
MUJOCO_GL=egl python scripts/eval_vla.py --checkpoint <chosen> --split test --episodes 200 --n-action-steps 50 --videos 10 --name sft_v1_test_k50
```

## Why these choices

- **Official recipe** (batch 64, 20k steps): it's the documented SmolVLA fine-tuning setup; we change nothing we can't justify.
- 20k × 64 = 1.28M samples ≈ 200 passes over our 6,413 frames, so **overfitting is possible**. That's why we save every
  5k steps and choose by closed-loop success on **val** seeds, never test seeds (choosing on test = leakage).
- Test seeds are only used once, at the end, for the reported numbers.

## Risks

- China mirrors / Hugging Face download slow or blocked -> fallback: upload the model files from the Mac (~2 GB).
- EGL rendering not available on the image -> fallback: OSMesa (slower CPU rendering).
- Low success rate -> failure analysis first (Milestone 2 rule), no extra training runs without discussion.
