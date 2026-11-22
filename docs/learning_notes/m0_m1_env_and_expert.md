# Milestone 0 + 1 学习笔记：环境与 Scripted Expert

## 1. Physics Simulation vs Robot Control

MuJoCo 是"物理世界"：每 2 ms 算一次力、碰撞、速度和位置。
Controller 是"机器人小脑"：把你给的高层指令变成关节力矩。
Policy（现在是 expert，以后是 VLA）是"大脑"：看到状态后决定下一步要去哪里。

一次 `env.step(action)` 里发生的事：

```
policy 给出 action=[dx,dy,dz,grip]          (20 Hz，每 50 ms 一次)
  -> OSC 把 dx..dz 乘以 0.05 m，算出末端目标位置
  -> 重复 25 次物理步 (每步 2 ms)：
       OSC 用 Jacobian 把"末端想往哪走"变成 7 个关节的力矩
       MuJoCo 积分一步
  -> 返回新 observation
```

这就是 **Control Frequency (20 Hz) vs Simulation Frequency (500 Hz)**。

## 2. Joint Space vs Cartesian Space，为什么选 OSC_POSITION

- Kinova3 有 7 个关节 (7 DoF)。直接控制关节 (Joint Space) 时，"往前移 1 cm" 需要自己解 **Inverse Kinematics**。
- **Operational Space Control (OSC)** 让我们直接在 Cartesian Space 下命令末端执行器，IK 和动力学由 controller 内部处理。
- `OSC_POSITION` 只控制位置，姿态固定为夹爪朝下，所以 action 只有 4 维。动作越简单，后面 VLA 和 RL 越好学。
- 代价：夹爪不能绕竖直轴旋转，只能靠平行夹爪自己"把方块夹正"。

代码：`src/newbody/env/lift_env.py: make_controller_config`

## 3. Finite State Machine Expert 与 Privileged Information

`src/newbody/expert/fsm_expert.py`

```
APPROACH -> DESCEND -> GRASP -> LIFT -> VERIFY
```

每个阶段都是一个目标点 + 一个 P-controller：

```
action_xyz = clip(gain * (target - eef_pos) / 0.05, -1, 1)
```

这是 **Closed-loop Control**：每一步都重新读取 `eef_pos`，所以 controller 的误差和延迟会被持续修正。

Expert 读取的是 `cube_pos`，也就是仿真器里的"真实答案" (**Privileged Information**)。
VLA 以后只能看图像 + 机器人本体状态 + 语言指令。
"老师作弊、学生不能作弊"正是 **Learning from Demonstrations** 的基本设定。

## 4. Success Condition：为什么要比官方更严格

官方的 `_check_success` 只要求某一帧方块高于桌面 4 cm，方块被弹飞也算成功。
我们的 `HoldSuccess` 要求方块**连续 10 步**保持被抬起，并且夹爪确实夹着它。
如果 success 定义有漏洞，后面的 RL 会学会钻漏洞 (**reward hacking**)。

## 5. 本阶段最重要的工程教训：静默的物理回归

现象：expert 只有 25% 成功率，没有报错。

排查链：失败 episode 的日志 → 按 cube yaw 分桶 → 固定 yaw 做对照 → 打印接触 (contacts) → 测量夹爪闭合间距。

结论：MuJoCo 3.9.0 下 Robotiq85 完全闭合后两指还相距 6.3 cm，夹不住 4 cm 的方块。换到 3.3.7 后是 1.5 cm，成功率达到 100%。

可以带走的方法：**success rate 低时，不要先去调算法，先用最小实验把问题定位到环境、控制器还是 policy。**

---

## My Exercise（15 分钟）

在 approach height 和 action gain 中选一个参数，先**写下预测**，再跑实验：

```bash
uv run python scripts/run_expert.py --episodes 30 --cube-xy-range 0.10 --videos 1 --set approach_height=0.02 --name ex_ah002
uv run python scripts/run_expert.py --episodes 30 --cube-xy-range 0.10 --videos 1 --set action_gain=0.3 --name ex_gain03
uv run python scripts/run_expert.py --episodes 30 --cube-xy-range 0.10 --videos 1 --set action_gain=3.0 --name ex_gain3
```

1. 预测：每个参数改动会让 success rate 和 episode length 怎么变？为什么？
2. 跑完后打开 `results/expert/<name>/episodes.csv` 和视频，验证你的预测。
3. 如果出现失败，失败卡在哪个 FSM phase（看 `fail_reason`）？

提示：approach 太低时，夹爪下降途中可能先碰到方块；gain 太大时，P-controller 会怎样？
