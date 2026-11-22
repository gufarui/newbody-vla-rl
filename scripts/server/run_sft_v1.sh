#!/bin/bash
cd "$(dirname "$0")/../.."  # repo root
source scripts/server/server_env.sh
.venv/bin/lerobot-train \
  --policy.path=lerobot/smolvla_base --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=local/kinova3_pickplace_can_v1_train --dataset.root=data/kinova3_pickplace_can_v1/train \
  --rename_map='{"observation.images.agentview": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --batch_size=64 --steps=20000 --save_freq=5000 --log_freq=100 --num_workers=8 --seed=1000 \
  --output_dir=outputs/sft_v1 --job_name=sft_v1 2>&1 | tee logs/sft_v1.log
echo "TRAINING_EXIT=${PIPESTATUS[0]}" | tee -a logs/sft_v1.log
