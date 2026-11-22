# NewBody-VLA-RL

Cross-embodiment VLA adaptation with residual reinforcement learning, simulation only.

Pipeline (planned): scripted expert on a simulated **Kinova3** → LeRobot dataset →
SFT of `lerobot/smolvla_base` → **frozen VLA + residual SAC** → controlled experiments.

Status: Milestone 0 (environment) and Milestone 1 (scripted expert) done. See [docs/research_log.md](docs/research_log.md).

Hands-on tutorial (follow along step by step): [docs/tutorial/](docs/tutorial/README.md)

## Install

```bash
uv sync            # Python 3.12, pinned robosuite 1.5.2 + mujoco 3.3.7 (see pyproject.toml for why)
uv run pytest -q
```

## Run the scripted expert

```bash
uv run python scripts/run_expert.py --episodes 100 --videos 3                     # cube xy +-3 cm
uv run python scripts/run_expert.py --episodes 100 --cube-xy-range 0.10 --name wide
uv run python scripts/run_expert.py --episodes 20 --set approach_height=0.05       # override an expert param
```

Outputs: `results/expert/<name>/` with `episodes.csv`, `summary.json`, per-step logs and MP4s
(left: agentview, right: wrist camera).

## Layout

```
configs/            env / success / expert config (YAML)
src/newbody/env/        Kinova3 Lift factory, seeded resets, image helpers
src/newbody/expert/     FSM scripted expert (privileged cube state)
src/newbody/evaluation/ stricter hold-success check
scripts/            CLI entry points
tests/              smoke + regression tests
docs/tutorial/      hands-on tutorial, one chapter per milestone
docs/               research log
```
