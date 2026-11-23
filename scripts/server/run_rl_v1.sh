#!/bin/bash
# Residual SAC on the frozen 10k SFT SmolVLA (Milestone 3). Resume with: RESUME=1 scripts/server/run_rl_v1.sh
cd "$(dirname "$0")/../.."
source scripts/server/server_env.sh
.venv/bin/python -W ignore scripts/train_residual_sac.py --name rl_v1 ${RESUME:+--resume} 2>&1 | tee -a logs/rl_v1.log
echo "TRAINING_EXIT=${PIPESTATUS[0]}" | tee -a logs/rl_v1.log
