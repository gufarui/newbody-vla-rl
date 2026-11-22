# 第 1 章：让机械臂动起来，并写一个自动抓取的"老师"

## 这一章结束时，你能做到

- 自己写代码让仿真里的机械臂移动
- 说清楚"机器人看到了什么（observation）"和"机器人能做什么（action）"
- 自己写一个小控制器，让机械臂飞到方块上方
- 看懂项目里的自动抓取老师（scripted expert）是怎么工作的
- 回答 3 道面试题

预计时间：1.5 – 2 小时。不用一次做完。

## 为什么要先做这一章？

最终我们想让 VLA 学会抓方块。VLA 是靠**模仿**学的，所以得先有人示范给它看。

真实世界里，示范是人拿着遥控器操作出来的。我们在仿真里，可以写一个程序当"老师"：
它能直接读到方块的精确位置（作弊），然后自动去抓。老师抓几百次，录下来，就是 VLA 的教材。

在写老师之前，你得先会让机械臂动。所以这一章分两半：**先玩环境，再写老师**。

---

## 准备

打开终端，进入项目目录：

```bash
cd ~/Desktop/robotics学习/newbody-vla-rl
```

确认环境能用（会跑 6 个小测试，几秒钟）：

```bash
uv run pytest -q
```

> `uv run` 的意思是：用这个项目自己的 Python 环境（`.venv`）来运行后面的命令，
> 所以你不用手动激活虚拟环境。

看到 `6 passed` 就可以开始了。运行时会打印一堆 `[robosuite WARNING]`，可以忽略。

---

## Step 1：打开环境，看看机器人"看到"了什么

### 动手

新建文件 `playground/step1_look.py`，**自己敲**下面的代码（不要复制，敲一遍记得更牢）：

```python
import numpy as np
import yaml
from newbody.env.lift_env import make_lift_env, reset_with_seed

cfg = yaml.safe_load(open("configs/lift_kinova3.yaml"))
env = make_lift_env(cfg["env"])
obs = reset_with_seed(env, seed=0)

for key, value in obs.items():
    print(f"{key:30s} shape={np.shape(value)}")
print("action_dim =", env.action_dim)
```

逐行说明：

- `cfg = ...`：读配置文件。打开 `configs/lift_kinova3.yaml` 看一眼，里面写了机器人型号、任务名、相机等。
- `make_lift_env`：创建仿真环境。它是我们对 robosuite 的一层薄包装，代码在 `src/newbody/env/lift_env.py`。
- `reset_with_seed(env, seed=0)`：把世界重置到一个初始状态。同一个 seed 永远得到同一个方块位置，这样实验才能复现。
- `obs`：observation，一个字典，就是机器人此刻"知道"的所有信息。

运行：

```bash
uv run python playground/step1_look.py
```

### 观察

你会看到大约 17 行输出，每行是一种信息和它的形状（shape）。

### 讲解

把这些 key 分成三类，你就看懂了：

**1. 机器人对自己身体的感觉（proprioception，本体感觉）**
就像你闭着眼睛也知道自己的手在哪里。

- `robot0_joint_pos` (7,)：7 个关节的角度，因为 Kinova3 有 7 个关节（7 DoF，自由度）
- `robot0_eef_pos` (3,)：夹爪末端在空间中的 x, y, z（eef = end-effector，末端执行器）
- `robot0_gripper_qpos` (6,)：夹爪手指的开合程度

**2. 相机图像**

- `agentview_image` (256, 256, 3)：一个从前方看过来的相机，256×256 像素，RGB 3 个通道
- `robot0_eye_in_hand_image`：装在手腕上的相机

**3. 物体的精确信息**

- `cube_pos` (3,)：方块的精确坐标
- `gripper_to_cube_pos`：夹爪到方块的向量

最后 `action_dim = 4`，说明每一步我们只需要给机器人 **4 个数字**。下一步就来试。

### 想一想

**Q1.** 现实世界里的一台机器人，上面三类信息中，哪些能直接拿到，哪些拿不到？

<details><summary>展开答案</summary>

第 1 类（关节角度、夹爪位置）能拿到：电机上有编码器。
第 2 类（图像）能拿到：装相机就行。
第 3 类（方块精确坐标）**拿不到**：现实中没人告诉你方块在哪，只能从图像里看出来。

这种"仿真里有、现实里没有"的信息叫 **privileged information（特权信息）**。
这是整个项目的一条红线：**老师可以用它，VLA 不能用它**。VLA 只能看图像 + 本体感觉 + 一句文字指令。

</details>

---

## Step 2：让机械臂往上走

### 动手

新建 `playground/step2_move_up.py`：

```python
import numpy as np
import yaml
from newbody.env.lift_env import make_lift_env, reset_with_seed

cfg = yaml.safe_load(open("configs/lift_kinova3.yaml"))
env = make_lift_env(cfg["env"], render_cameras=False)   # 不渲染相机，跑得快
obs = reset_with_seed(env, seed=0)

z_start = obs["robot0_eef_pos"][2]
action = np.array([0.0, 0.0, 1.0, -1.0])   # dx, dy, dz, gripper
for t in range(10):
    obs, reward, done, info = env.step(action)
    print(f"step {t}: eef z = {obs['robot0_eef_pos'][2]:.3f}")
print(f"total rise = {obs['robot0_eef_pos'][2] - z_start:.3f} m")
```

4 个数字的含义：

- 前 3 个：让夹爪往 x, y, z 方向移动多少，范围是 -1 到 1
- 第 4 个：夹爪，-1 = 张开，+1 = 闭合

**数字 1 对应 5 cm**（配置在 robosuite 的 controller 里）。所以 `dz = 1.0` 的意思是"往上走 5 cm"。

### 先预测（写下来再运行！）

每一步都命令"往上 5 cm"，一共 10 步。你预测总共上升多少？

```bash
uv run python playground/step2_move_up.py
```

### 观察

<details><summary>先自己运行，再展开对照参考输出</summary>

```
step 0: eef z = 0.964
step 1: eef z = 0.973
...
step 9: eef z = 1.077
total rise = 0.117 m
```

只上升了约 **12 cm**，不是 50 cm。每一步大约 1.3 cm。

</details>

### 讲解

为什么命令 5 cm，实际只走了 1.3 cm？这里要区分两个东西：

**Policy（策略）** 是"大脑"：决定"我想往哪走"，就是你给的那 4 个数字。

**Controller（控制器）** 是"小脑 + 肌肉"：把"想往上走 5 cm"变成 7 个关节各用多大力。
我们用的叫 **OSC（Operational Space Control，操作空间控制）**。

关键是：`dz = 1` 并不是"瞬移 5 cm"，而是**在当前位置上方 5 cm 处放一个目标点，然后控制器像弹簧一样把手往那里拉**。
50 毫秒后，下一个动作来了，又重新放一个目标点。手臂有质量、有惯性，50 毫秒内只能被拉过去 1 cm 多。

用一个比喻：你牵着一只狗，每隔一会儿就往前扔一个球，但还没等狗跑到，你就又扔了下一个。狗一直在追，跑的速度取决于狗本身。

> 这个结论后面很重要：**action 不等于实际发生的运动**。
> 同一个 action，换一台机器人、换一个控制器，效果就不一样。
> 这正是"cross-embodiment（跨机器人本体）"难的地方之一。

为什么我们选"控制末端位置"，而不是直接控制 7 个关节？

- 直接控制关节（**joint space**）：要往前走 1 cm，你得自己算出 7 个关节各转多少度，这叫 **inverse kinematics（逆运动学）**，很麻烦。
- 控制末端位置（**Cartesian space**）：说"往前 1 cm"就行，控制器替你算。

动作越简单，后面 VLA 和 RL 就越好学。代价是我们只控制位置，夹爪始终朝下，不能转手腕。

### 想一想

**Q2.** 把 `range(10)` 改成 `range(40)`，再运行一次。每一步上升的距离是越来越大、越来越小，还是差不多不变？这说明了什么？

<details><summary>展开答案</summary>

几乎不变，每步约 1.2 cm（40 步共上升约 49 cm）。
因为目标点永远在手的上方 5 cm，"弹簧"一直被拉开同样的长度，所以手臂以一个稳定的速度往上走。
这个速度由控制器的参数（弹簧多硬、阻尼多大）决定，而不是由你的 action 直接决定。

</details>

**Q3.** 把 `action` 改成 `[0.0, 0.0, 0.5, -1.0]`，每步大约走多少？和 1.0 时成比例吗？

<details><summary>展开答案</summary>
约 0.64 cm/步，大约是 1.0 时（1.2 cm/步）的一半。目标点放得近一半，弹簧被拉开一半，速度也大约变成一半。
</details>

---

## Step 3：两种"频率"

### 动手

新建 `playground/step3_freq.py`：

```python
import yaml
from newbody.env.lift_env import make_lift_env

cfg = yaml.safe_load(open("configs/lift_kinova3.yaml"))
env = make_lift_env(cfg["env"], render_cameras=False)
print("control_freq  =", env.control_freq, "Hz")
print("sim timestep  =", env.model_timestep, "s")
```

```bash
uv run python playground/step3_freq.py
```

### 讲解

你会看到 `20 Hz` 和 `0.002 s`。

- **物理仿真**每 0.002 秒（2 毫秒）算一次：所有力、碰撞、速度、位置。也就是 500 次/秒。物理需要算得很细，否则会算错（比如手指穿过方块）。
- **Policy** 每秒只做 20 次决定（每 50 毫秒一次）。大脑不需要那么快。

所以你每调用一次 `env.step(action)`，里面其实发生了：

```
你给一个 action
  → 控制器根据 action 设一个目标点
  → 物理仿真跑 25 小步（每步 2 ms，共 50 ms），每一小步控制器都重新算一次关节力
  → 返回新的 observation
```

### 想一想

**Q4.** 为什么正好是 25 小步？

<details><summary>展开答案</summary>
1 / 20 Hz = 0.05 秒，0.05 / 0.002 = 25。
</details>

**Q5.** 后面 VLA 推理一次可能要几十毫秒。如果 VLA 每一步都要重新想，最多能跑到多少 Hz？你觉得这会是个问题吗？
（先记下你的想法，第 4 章会回来讨论，那里会讲 **action chunking**：VLA 一次想好未来好几步。）

---

## Step 4：看一眼相机画面

### 动手

新建 `playground/step4_image.py`：

```python
import imageio
import yaml
from newbody.env.lift_env import make_lift_env, reset_with_seed

cfg = yaml.safe_load(open("configs/lift_kinova3.yaml"))
env = make_lift_env(cfg["env"])
obs = reset_with_seed(env, seed=0)

raw = obs["agentview_image"]
print(raw.shape, raw.dtype, raw.min(), raw.max())
imageio.imwrite("playground/raw.png", raw)
imageio.imwrite("playground/flipped.png", raw[::-1])
```

```bash
uv run python playground/step4_image.py
open playground/raw.png playground/flipped.png
```

（`open` 是 macOS 命令，用默认看图软件打开图片。）

### 观察

两张图有什么区别？哪一张是对的？

### 讲解

robosuite 输出的图像是**上下颠倒**的（因为底层 OpenGL 的坐标从左下角开始）。`raw[::-1]` 把行的顺序反过来，就是上下翻转。

`uint8` 和 `0 ~ 255`：每个像素的每个颜色通道是 0 到 255 的整数。

这看起来是个小细节，但非常要命：

> SmolVLA 预训练时见到的都是正常方向的图片。如果我们喂给它颠倒的图，它就像倒着看世界，学起来会非常困难，而且**不会报错**，只会成功率莫名其妙地低。

第 2 章录数据时，我们会专门检查图像方向、颜色顺序（RGB 还是 BGR）。

---

## Step 5：自己写一个控制器，飞到方块上方

前面都是"给固定动作"。现在你来写第一个**闭环控制器**：每一步看一眼"我在哪、目标在哪"，然后决定往哪走。

### 动手

新建 `playground/step5_approach.py`，把 `TODO` 那一行补完：

```python
import sys
import numpy as np
import yaml
from newbody.env.lift_env import make_lift_env, reset_with_seed

GAIN = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0

cfg = yaml.safe_load(open("configs/lift_kinova3.yaml"))
env = make_lift_env(cfg["env"], render_cameras=False)
obs = reset_with_seed(env, seed=0)

for t in range(40):
    eef = obs["robot0_eef_pos"]
    cube = obs["cube_pos"]
    target = cube + np.array([0.0, 0.0, 0.10])     # 方块正上方 10 cm
    error = target - eef                            # 还差多远（米）

    # TODO: 根据 error 算出 action_xyz（3 个数，每个都要在 -1 到 1 之间）
    # 提示：动作 1.0 对应 0.05 m。误差大就走快，误差小就走慢。可以用 np.clip 限制范围。
    action_xyz = ...

    action = np.append(action_xyz, -1.0)           # 夹爪保持张开
    obs, reward, done, info = env.step(action)
    if t % 4 == 0:
        print(f"t={t:2d}  error(cm) = {np.round(error * 100, 1)}  |error| = {np.linalg.norm(error) * 100:.1f} cm")
```

先自己想 5 分钟再看答案。

<details><summary>参考答案</summary>

```python
action_xyz = np.clip(GAIN * error / 0.05, -1.0, 1.0)
```

- `error / 0.05`：把"米"换算成"动作单位"
- `GAIN`：放大倍数
- `np.clip(..., -1, 1)`：超出范围就截断

</details>

运行，`GAIN` 用 1.0：

```bash
uv run python playground/step5_approach.py 1.0
```

### 讲解

你刚写的叫 **P 控制器（proportional controller，比例控制）**：动作和误差成正比。
离得远，就走得快；快到了，就慢下来。

它是**闭环（closed-loop）** 的：每一步都重新测量 `eef`，所以就算控制器有偏差、机械臂有惯性，下一步也会自动纠正。

对比一下 Step 2：那是**开环（open-loop）**，你只管给动作，不看结果。

> 记住这对概念。第 4 章会遇到它的"大号版本"：
> VLA 一次预测未来 50 步动作然后闭着眼执行（偏开环），还是每几步就重新看一眼（偏闭环）？

### 动手做实验：改 GAIN

**先预测**：GAIN = 0.3 和 GAIN = 5.0，哪个更快到达？会不会冲过头（误差变成负数）？写下来，然后运行：

```bash
uv run python playground/step5_approach.py 0.3
uv run python playground/step5_approach.py 5.0
```

<details><summary>参考输出（seed=0，看 x 方向误差那一列）</summary>

| GAIN | 误差首次 < 1 cm | 冲过头了吗 |
|---|---|---|
| 0.3 | 约 t=32~36 | 没有，慢慢靠近 |
| 1.0 | 约 t=16~20 | 有一点：x 误差变成 -0.8 cm 再回来 |
| 5.0 | 约 t=16 | 几乎没有，最后误差到 0.0 |

</details>

### 想一想

**Q6.** GAIN 很大时，大部分时间 action 都是多少？（提示：看 `np.clip`。）所以远处的速度由什么决定？

<details><summary>展开答案</summary>

GAIN 大时，`GAIN * error / 0.05` 远大于 1，被 clip 成 ±1。也就是远处都是"全速"，速度上限由 action 范围和控制器决定。
只有离得很近时，才进入"比例"区间，开始减速。

</details>

**Q7.** 为什么 GAIN = 1.0 反而有一点冲过头，而 5.0 几乎没有？
这个问题没有标准答案，说说你的猜测就行。（提示：回想 Step 2 的"牵狗"比喻：action 放的是目标点，控制器像弹簧一样拉。）

---

## Step 6：看项目里的老师是怎么抓方块的

你已经会"飞到方块上方"了。完整的抓取只是再多几个阶段：

```
APPROACH → DESCEND → GRASP → LIFT → VERIFY
飞到上方    下降到方块    闭合夹爪    抬起来     检查是否成功
```

这种"分阶段、满足条件就切换到下一阶段"的程序叫 **FSM（Finite State Machine，有限状态机）**。

### 动手：读代码

打开 `src/newbody/expert/fsm_expert.py`，找到 `act` 函数。对照着读，回答：

1. APPROACH 阶段的 `target` 是什么？和你 Step 5 写的一样吗？
2. 什么条件下从 APPROACH 切换到 DESCEND？
3. GRASP 阶段为什么要原地停 10 步？（看 `grasp_close_steps`）
4. `_move` 函数和你写的 P 控制器有什么区别？

### 动手：运行老师

```bash
uv run python scripts/run_expert.py --episodes 5 --videos 5 --name my_first_run
open results/expert/my_first_run/seed0_success.mp4
```

视频左边是前方相机，右边是手腕相机。

再看看老师每一步做了什么。`steps_seed0.csv` 记录了每一步的阶段、动作、高度。下面的命令只打印**阶段切换**的那几行：

```bash
uv run python -c "
import csv
prev = None
for r in csv.DictReader(open('results/expert/my_first_run/steps_seed0.csv')):
    if r['phase'] != prev:
        print(r['t'], r['phase'], 'grip=', r['grip'], 'cube_height=', r['cube_z_above_table'], 'grasping=', r['grasping'])
        prev = r['phase']
"
```

### 想一想

**Q8.** 方块静止在桌上时，`cube_z_above_table` 大约是 0.02。为什么不是 0？
<details><summary>展开答案</summary>
记录的是方块**中心**的高度。方块边长约 4 cm，中心离桌面约 2 cm。
</details>

**Q9.** 从哪一步开始 `grasping` 变成 1？是夹爪开始闭合的那一刻吗？

---

## Step 7：怎么判断"成功"？

robosuite 官方的规则是：**某一瞬间**方块中心高于桌面 4 cm，就算成功。

### 想一想（先自己想，再往下读）

**Q10.** 这个规则有什么漏洞？能不能想出一种"没有真正抓起来，却被判成功"的情况？

<details><summary>展开答案</summary>

比如夹爪把方块**撞飞**或**弹起来**，方块在空中的某一帧超过了 4 cm，就被算成功了。

</details>

所以我们写了更严格的规则：方块被抬高 **并且** 夹爪确实夹着它，**连续 10 步**都满足才算成功。
代码很短，打开 `src/newbody/evaluation/success.py` 读一下（只有 20 行）。

> **为什么要较真？** 第 5 章做 RL 时，机器人会想尽办法让分数变高。
> 如果规则有漏洞，它就会学会钻漏洞（比如学会把方块拍飞），而不是学会抓。
> 这叫 **reward hacking**，是 RL 里最常见的坑之一。

---

## Step 8：一个真实的 bug 故事

这是我们在做这一章时真实遇到的问题，也是这章最有价值的一课。

**现象**：老师第一次跑，成功率只有 25%，而且**没有任何报错**。

**排查过程**（每一步都只做一个小实验）：

1. 看失败的录像和日志：夹爪闭合了，但没夹住方块。
2. 方块每次的朝向是随机的。按朝向分组统计成功率：**只有方块转了约 45° 时才能成功**。
3. 固定方块朝向再测：0° 时 0/10，45° 时 9/10。现象稳定复现。
4. 检查物理接触：0° 时，手指**根本没碰到方块**。
5. 测量手指完全闭合后的间距：**6.3 cm**！方块只有 4 cm 宽，当然夹不住。只有转 45° 时，方块的对角线（约 6 cm）才刚好被碰到。
6. 换一个 MuJoCo 版本（3.9.0 → 3.3.7）：间距变成 1.5 cm，成功率 100%。

**结论**：新版物理引擎和这个夹爪模型不兼容，**但它不报错，只是悄悄让机器人变笨**。

### 动手

我们把这个检查写成了一个测试，以后谁升级了 MuJoCo 都会被拦住：

```bash
uv run pytest -k gripper -v
```

打开 `tests/test_env.py`，找到 `test_gripper_closes_fully`，读懂它在测什么。

### 想一想

**Q11.** 如果当时没去查，而是直接拿这个 25% 的老师录数据、训练 VLA，会发生什么？

<details><summary>展开答案</summary>

最糟的情况：VLA 训练完成功率很低，我们会以为是"VLA 不行"或"数据不够"，然后花大量 GPU 时间调模型。
其实问题出在最底层的物理仿真上。

**教训：成功率低时，先用最小的实验确定问题出在哪一层（环境？控制器？策略？），再动手改。**

</details>

---

## Step 9：你的小研究：一个参数会怎样影响老师？

这是这章的最后一个练习，像做一个迷你研究：**先提出假设，再做实验验证**。

选下面**一个**参数：

- `approach_height`：在方块上方多高的地方停下再下降（默认 0.10 m）
- `action_gain`：老师的 P 控制器放大倍数（默认 1.0，就是你 Step 5 的 GAIN）

### 1. 先写下你的假设

在 `playground/step9_notes.md` 里写：

```
我选的参数：
我要试的值：
我预测成功率会：
我预测每局步数会：
理由：
```

### 2. 做实验

`--set` 可以临时修改老师的参数。例如：

```bash
uv run python scripts/run_expert.py --episodes 30 --cube-xy-range 0.10 --videos 1 --set approach_height=0.02 --name ex_ah002
uv run python scripts/run_expert.py --episodes 30 --cube-xy-range 0.10 --videos 1 --set action_gain=0.3 --name ex_gain03
```

### 3. 看结果

- 最后一行会打印成功率
- `results/expert/<name>/episodes.csv`：每一局的长度（`length`）、失败原因（`fail_reason`，失败时会显示卡在哪个阶段）
- 失败的局会自动保存视频，打开看看到底发生了什么

### 4. 写结论

预测对了吗？如果不对，你觉得原因是什么？写在 `step9_notes.md` 里，然后发给我，我们一起讨论。

---

## 面试练习

先**用英语**口头或书面回答，再发给我，我会指出哪里不准确。每题 3 – 5 句话即可。

1. *What is the difference between the policy and the controller in your setup? What does one action actually do?*
2. *Your scripted expert reads the exact cube position. Why is that allowed for the expert, but not for the VLA?*
3. *Your expert's success rate was 25% at first. How did you find the cause?*

## 本章概念小结

只有在你做完上面的步骤之后，这些词才有意义：

- **Observation / Action**：机器人知道什么 / 机器人能做什么（Step 1, 2）
- **Proprioception**：本体感觉，关节角度、末端位置（Step 1）
- **Privileged information**：仿真里有、现实里没有的信息（Step 1）
- **Policy vs Controller**：大脑决定去哪，小脑负责怎么用力（Step 2）
- **Joint space vs Cartesian space**：控制关节角度 vs 控制末端位置（Step 2）
- **Control frequency vs simulation frequency**：20 Hz vs 500 Hz（Step 3）
- **P controller / closed-loop / open-loop**（Step 5）
- **FSM**：有限状态机（Step 6）
- **Success condition / reward hacking**（Step 7）
