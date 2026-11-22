export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf
export HF_HUB_OFFLINE=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export UV_CACHE_DIR=/root/autodl-tmp/uv-cache
export OMP_NUM_THREADS=8   # container quota is 18 cores but torch sees 256 host cores -> defaults to 128 threads
export MKL_NUM_THREADS=8   # (measured: 2 evals x 152 threads -> 80 s/episode, CPU throttled; capped -> 8 s/episode)
