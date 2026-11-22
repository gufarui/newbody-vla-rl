# Hands-on Tutorial: Building a VLA + RL Robot Project from Scratch

This tutorial is meant to be **followed while you build**. Every chapter uses the same rhythm:

1. **Do it**: type the code and run the commands yourself
2. **Predict first**: before running, write down what you expect to happen
3. **Observe**: look at the output or the video
4. **Explanation**: why it happened
5. **Think about it**: a few questions; answers are folded, so think first, then expand

Your own code goes into `playground/`. The project's real code lives in `src/`; the tutorial tells you when to read it.

## The project map

What we are building, in one sentence:

> Teach a large model that has "seen many robots" (SmolVLA) to control a robot arm it has **never seen** (Kinova3),
> then use reinforcement learning (RL) to fix its small mistakes.

The steps come in this order, and each one depends on the previous one:

| Ch. | What you will do | What you learn | What you can explain in an interview |
|---|---|---|---|
| 1 | Make the arm move; write a "teacher" that grasps the cube automatically | simulation, control, action space | where the data comes from |
| 2 | Record the teacher's behavior as a dataset | imitation learning, data formats | Behavior Cloning |
| 3 | Fine-tune SmolVLA on that data (remote GPU) | VLA architecture, Flow Matching, SFT | how a VLA outputs actions |
| 4 | Let the VLA grasp on its own and measure success rate | closed-loop evaluation, compounding error | why low loss does not mean the task is solved |
| 5 | Freeze the VLA and train a small RL network to correct it | SAC, Q-functions, reward design | Residual RL |
| 6 | Run comparison experiments | experimental design, statistics | is your conclusion trustworthy |
| 7 | Turn it into a portfolio; mock interview | explaining the whole project in English | everything |

## Chapters

- [Chapter 1: Make the arm move, and write a teacher that grasps automatically](ch01_env_and_expert.md)
