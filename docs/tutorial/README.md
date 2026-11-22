# Hands-on Tutorial: Building a VLA + RL Robot Project from Scratch

These notes explain what we built at each milestone, why, and what the results show.
The coding agent writes the code; these notes cover what you need to understand to make design decisions
and to explain the project confidently in an interview.

Each chapter has: where it fits in the project, key concepts (with our real measurements),
design decisions, and a few interview questions to answer in English.

## The project map

What we are building, in one sentence:

> Teach a large model that has "seen many robots" (SmolVLA) to control a robot arm it has **never seen** (Kinova3),
> then use reinforcement learning (RL) to fix its small mistakes.

The steps come in this order, and each one depends on the previous one:

| Ch. | What we do | What you learn | What you can explain in an interview |
|---|---|---|---|
| 1 | Make the arm move; write a "teacher" that grasps the cube automatically | simulation, control, action space | where the data comes from |
| 2 | Record the teacher's behavior as a dataset | imitation learning, data formats | Behavior Cloning |
| 3 | Fine-tune SmolVLA on that data (remote GPU) | VLA architecture, Flow Matching, SFT | how a VLA outputs actions |
| 4 | Let the VLA grasp on its own and measure success rate | closed-loop evaluation, compounding error | why low loss does not mean the task is solved |
| 5 | Freeze the VLA and train a small RL network to correct it | SAC, Q-functions, reward design | Residual RL |
| 6 | Run comparison experiments | experimental design, statistics | is your conclusion trustworthy |
| 7 | Turn it into a portfolio; mock interview | explaining the whole project in English | everything |

## Chapters

- [Chapter 1: The simulated robot and the scripted teacher](ch01_env_and_expert.md)
