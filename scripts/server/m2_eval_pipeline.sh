#!/bin/bash
# Milestone 2 evaluation pipeline (runs on the GPU server, in tmux, after/while training).
# 1) val, full bin: every checkpoint (5k already done) -> 2) pick best on val -> 3) test, full bin, once.
cd "$(dirname "$0")/../.."
source server_env.sh 2>/dev/null || source scripts/server/server_env.sh
PY=.venv/bin/python
WIDE="--can-x-range -0.145 0.145 --can-y-range -0.195 0.195"
for step in 010000 015000 020000; do
  until [ -f outputs/sft_v1/checkpoints/$step/pretrained_model/model.safetensors ]; do sleep 60; done
  sleep 30
  $PY -W ignore scripts/eval_vla.py --checkpoint outputs/sft_v1/checkpoints/$step/pretrained_model --split val \
    --seed-offset 20 --episodes 40 --n-action-steps 10 --videos 4 $WIDE --name sft_v1_${step}_valwide_k10 \
    > logs/eval_${step}_valwide_k10.log 2>&1
done
BEST=$($PY - <<'PY'
import json
best = max(("005000", "010000", "015000", "020000"),
           key=lambda s: (json.load(open(f"results/eval/sft_v1_{s}_valwide_k10/summary.json"))["success"], int(s)))
print(best)
PY
)
echo "BEST_CHECKPOINT=$BEST" | tee logs/m2_best.txt
for k in 10 50; do
  $PY -W ignore scripts/eval_vla.py --checkpoint outputs/sft_v1/checkpoints/$BEST/pretrained_model --split test \
    --episodes 200 --n-action-steps $k --videos 8 $WIDE --name sft_v1_${BEST}_testwide_k$k > logs/eval_${BEST}_testwide_k$k.log 2>&1
done
echo PIPELINE_DONE | tee -a logs/m2_best.txt
