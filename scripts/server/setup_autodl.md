# Remote GPU setup (AutoDL, China region) — what actually worked (2026-11-23)

Server: RTX 4090 D 24 GB, Ubuntu 22.04, 18-core quota, miniconda Python 3.12 at /root/miniconda3, no system python.

```bash
# on the Mac
uv export --frozen --no-hashes --no-emit-project --no-dev -o /tmp/requirements-lock.txt
rsync -az --exclude .venv --exclude outputs --exclude results ./ gpu:/root/autodl-tmp/newbody-vla-rl/   # code + data (~77 MB)
scp /tmp/requirements-lock.txt gpu:/root/autodl-tmp/newbody-vla-rl/

# on the server
apt-get install -y tmux libegl1 libgl1 libglvnd0 libosmesa6 ffmpeg       # EGL loader for headless MuJoCo; ffmpeg for torchcodec
/root/miniconda3/bin/pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple
cd /root/autodl-tmp/newbody-vla-rl
uv venv -p /root/miniconda3/bin/python3.12 .venv
uv pip install -p .venv/bin/python -r requirements-lock.txt \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple --index-strategy unsafe-best-match   # ~6 min, torch cu13
uv pip install -p .venv/bin/python -e . --no-deps
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/autodl-tmp/hf
.venv/bin/hf download lerobot/smolvla_base                     # ~1 min
.venv/bin/hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct  # ~3 min (miniconda's hf CLI crashed on decompression; use the venv's)
source scripts/server/server_env.sh && .venv/bin/pytest -q      # 9 passed with MUJOCO_GL=egl
tmux new-session -d -s sft scripts/server/run_sft_v1.sh
```

Measured: batch 64 -> 0.56 s/step, 14 GB VRAM, GPU util ~99%, data loading negligible after warm-up.

**Pitfall (measured 2026-11-23):** the container has an 18-core CPU quota, but torch sees the host's 256 cores and
defaults to 128 threads. Two eval processes (~152 threads each) were CPU-throttled in 59% of scheduler periods:
80 s/episode with the GPU at 1%. `OMP_NUM_THREADS=8` (in server_env.sh) -> 8 s/episode, 10x faster.
