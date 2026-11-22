# NewBody-VLA-RL

Lightweight RL post-training for cross-embodiment VLA adaptation, simulation only.

Pipeline: scripted expert on a simulated **Kinova3** (robosuite **PickPlaceCan**) → LeRobot dataset →
SFT of `lerobot/smolvla_base` → **frozen VLA + RLT-inspired residual SAC** → SFT-only vs SFT+RL.

- Scope and decisions: [docs/project_spec.md](docs/project_spec.md)
- Experiment log: [docs/research_log.md](docs/research_log.md)
- Learning notes (one chapter per milestone): [docs/tutorial/](docs/tutorial/README.md)

Status: Milestone 1 (simulation + scripted expert + demo dataset) in progress.

## Install

```bash
uv sync            # Python 3.12; robosuite 1.5.2 + mujoco 3.3.7 + lerobot 0.7.0 (see pyproject.toml for pins)
uv run pytest -q
```

## Scripted expert

```bash
uv run python scripts/run_expert.py --episodes 100 --videos 3
uv run python scripts/run_expert.py --episodes 20 --set carry_z=0.97 --name low_carry   # override an expert param
```

Outputs: `results/expert/<name>/` with `episodes.csv`, `summary.json`, per-step logs and MP4s
(left: agentview, right: wrist camera).

## Demonstration dataset

```bash
uv run python scripts/collect_demos.py --name kinova3_pickplace_can_v1        # train + val splits from config
uv run python scripts/validate_dataset.py --name kinova3_pickplace_can_v1 --split train --replay 5
```

Data goes to `data/<name>/` (LeRobot dataset per split + privileged sidecar for checks; not in git).

## Layout

```
configs/                 env / success / expert / data config (YAML)
src/newbody/env/         Kinova3 PickPlaceCan factory, seeded resets, image helpers
src/newbody/expert/      FSM scripted expert (privileged can + target state)
src/newbody/evaluation/  stricter place-success check
src/newbody/data/        dataset schema + LeRobot recorder
scripts/                 CLI entry points
tests/                   smoke + regression tests
docs/                    spec, research log, learning notes
```
