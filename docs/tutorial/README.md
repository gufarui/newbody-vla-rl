# Hands-on Tutorial: Building a VLA + RL Robot Project from Scratch

These notes explain what we built at each milestone, why, and what the results show.
The coding agent writes the code; these notes cover what you need to understand to make design decisions
and to explain the project confidently in an interview.

Each chapter has: where it fits in the project, key concepts (with our real measurements),
design decisions, and a few interview questions to answer in English.

## The project map

What we are building, in one sentence:

> Adapt a pretrained Vision-Language-Action model (SmolVLA) to a robot arm it was not trained on (Kinova3),
> then freeze it and train a small RL policy that refines its actions.

| Milestone | What we do | What you must understand | Interview topic |
|---|---|---|---|
| 1 | Simulated Kinova3 pick-and-place; scripted teacher; demo dataset | observation/action, OSC, privileged info, behavior cloning | where the data comes from |
| 2 | Fine-tune SmolVLA on the demos (remote GPU); closed-loop evaluation | VLM + action expert, flow matching, action chunking, SFT | how a VLA generates actions; why low loss ≠ success |
| 3 | Freeze the VLA; RLT-inspired residual SAC using VLA features + reference actions | MDP, Q-function, actor-critic, SAC, reward design, residual policy | why freeze the VLA; how RL refines it |
| 4 | SFT-only vs SFT + RL on the same held-out start states | baselines, controlled comparison, uncertainty, failure analysis | did RL really help? |

Concept-only reading along the way: π0.5, π*0.6, π0.7, RLT (what problem, how at a high level, relation to this project).

## Chapters

- [Chapter 1: Simulation, scripted teacher, and demonstrations](ch01_env_and_expert.md)
- [Chapter 2: SmolVLA — how a VLA turns pixels and words into actions](ch02_smolvla_sft.md) (Part A done; Part B after training)
