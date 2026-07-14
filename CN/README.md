可以把训练代码拆成 **环境层、智能体层、训练层、评估层** 四部分。核心思想是：先把现有导弹模型和目标模型包装成一个类似 Gym 的环境，DDQN 只负责“给状态、选动作、拿奖励”。

## 一、推荐的项目结构

```text
missile_ddqn/
├── config.py                  # 所有训练参数
├── train.py                   # 主训练入口
├── evaluate.py                # 测试与基线对比
│
├── env/
│   ├── missile_env.py         # 强化学习环境
│   ├── scenario_generator.py  # 初始态势随机生成
│   ├── maneuver.py            # 目标机动逻辑
│   └── reward.py              # 奖励函数
│
├── models/
│   ├── missile_model.py       # 现有导弹六自由度/质点模型接口
│   └── target_model.py        # 目标飞机模型接口
│
├── rl/
│   ├── q_network.py           # Q网络
│   ├── replay_buffer.py       # 经验回放池
│   └── ddqn_agent.py          # DDQN核心算法
│
├── utils/
│   ├── state_processor.py     # 状态构造与归一化
│   ├── cn_mapper.py           # 动作索引映射为CN
│   ├── logger.py              # TensorBoard、CSV日志
│   └── seed.py                # 随机种子
│
├── checkpoints/               # 模型权重
└── results/                   # 测试结果、轨迹和统计图
```

其中最重要的是三个文件：

```text
missile_env.py
ddqn_agent.py
train.py
```

------

# 二、整体调用关系

训练代码的调用链大致是：

```text
train.py
  ├── 创建 MissileEnv
  ├── 创建 DDQNAgent
  ├── 每回合 env.reset()
  ├── agent.select_action(state)
  ├── env.step(action)
  ├── replay_buffer.add(...)
  ├── agent.update()
  └── 保存模型和训练指标
```

环境内部则是：

```text
MissileEnv.step(action)
  ├── 计算当前弹目距离
  ├── 动作映射为CN
  ├── 在一个决策周期内：
  │     ├── 更新目标飞机
  │     ├── CN传入导弹模型
  │     └── 更新导弹状态
  ├── 判断命中、脱靶或超时
  ├── 计算奖励
  └── 返回 next_state, reward, done, info
```

------

# 三、配置文件

`config.py` 统一存储参数，避免参数散落在各个文件中。

```python
from dataclasses import dataclass


@dataclass
class Config:
    # 状态与动作
    state_dim: int = 12
    action_dim: int = 4

    # Q网络
    hidden_dim1: int = 128
    hidden_dim2: int = 128

    # DDQN
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 128
    replay_capacity: int = 100_000
    warmup_steps: int = 5_000
    target_update_interval: int = 2_000
    gradient_clip: float = 10.0

    # 探索率
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 300_000

    # 仿真
    simulation_dt: float = 0.01
    decision_dt: float = 0.2
    max_flight_time: float = 80.0
    hit_distance: float = 10.0

    # 状态归一化
    position_scale: float = 100_000.0
    velocity_scale: float = 2_000.0

    # 训练
    num_episodes: int = 50_000
    seed: int = 42
    device: str = "cuda"
```

决策周期和仿真周期之间的关系是：

```python
substeps = int(decision_dt / simulation_dt)
```

例如：

```text
仿真步长：0.01秒
决策周期：0.20秒
每次动作执行：20个导弹模型仿真步
```

------

# 四、CN映射模块

```
utils/cn_mapper.py
def get_cn(distance: float, action: int) -> float:
    if distance > 50_000:
        cn_values = [1.0, 1.3, 1.7, 2.0]
    elif distance > 10_000:
        cn_values = [3.0, 3.3, 3.7, 4.0]
    else:
        cn_values = [5.0, 5.3, 5.7, 6.0]

    if not 0 <= action < len(cn_values):
        raise ValueError(f"非法动作索引: {action}")

    return cn_values[action]
```

这里要注意：

DDQN真正学习的动作仍然是：

```text
0、1、2、3
```

但同一个动作在不同距离阶段对应不同CN。

例如动作2：

```text
远程：CN = 1.7
中程：CN = 3.7
近程：CN = 5.7
```

因此，本质上DDQN学习的是：

```text
当前阶段选择偏小、次小、次大还是最大的CN
```

而不是学习全局固定的四个CN。

------

# 五、状态处理模块

```
utils/state_processor.py
import numpy as np


class StateProcessor:
    def __init__(
        self,
        position_scale: float = 100_000.0,
        velocity_scale: float = 2_000.0,
    ):
        self.position_scale = position_scale
        self.velocity_scale = velocity_scale

    def build_state(
        self,
        missile_position: np.ndarray,
        missile_velocity: np.ndarray,
        target_position: np.ndarray,
        target_velocity: np.ndarray,
    ) -> np.ndarray:
        relative_position = target_position - missile_position
        relative_velocity = target_velocity - missile_velocity

        state = np.concatenate(
            [
                relative_position / self.position_scale,
                relative_velocity / self.velocity_scale,
                missile_velocity / self.velocity_scale,
                target_velocity / self.velocity_scale,
            ],
            axis=0,
        )

        return state.astype(np.float32)
```

最终状态排列为：

```text
[
  Δx, Δy, Δz,
  Δvx, Δvy, Δvz,
  vmx, vmy, vmz,
  vtx, vty, vtz
]
```

建议再加入异常检查：

```python
if not np.all(np.isfinite(state)):
    raise RuntimeError("状态中出现NaN或Inf")
```

因为导弹模型在训练过程中可能出现数值发散，一旦把 `NaN` 存入经验池，后面的网络训练也会全部变成 `NaN`。

------

# 六、初始场景生成器

```
env/scenario_generator.py
```

它负责随机生成每个回合的初始条件。

```python
import numpy as np


class ScenarioGenerator:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def sample(self) -> dict:
        distance = self.rng.uniform(20_000, 100_000)
        bearing = self.rng.uniform(-np.pi / 3, np.pi / 3)
        altitude_difference = self.rng.uniform(-5_000, 5_000)

        missile_speed = self.rng.uniform(700, 1_200)
        target_speed = self.rng.uniform(200, 400)

        maneuver_type = self.rng.choice(
            [
                "straight",
                "climb",
                "dive",
                "left_turn",
                "right_turn",
                "left_climb",
                "right_climb",
            ]
        )

        missile_position = np.array([0.0, 0.0, 0.0])

        target_position = np.array(
            [
                distance * np.cos(bearing),
                altitude_difference,
                distance * np.sin(bearing),
            ]
        )

        missile_velocity = np.array(
            [
                missile_speed,
                0.0,
                0.0,
            ]
        )

        target_velocity = np.array(
            [
                -target_speed,
                0.0,
                0.0,
            ]
        )

        return {
            "missile_position": missile_position,
            "missile_velocity": missile_velocity,
            "target_position": target_position,
            "target_velocity": target_velocity,
            "maneuver_type": maneuver_type,
            "maneuver_strength": self.rng.uniform(0.6, 1.2),
            "maneuver_start_time": self.rng.uniform(0.0, 5.0),
        }
```

实际使用时，初始速度方向不能一直固定成正对或迎头，需要继续随机化：

```text
导弹航向角
目标航向角
进入角
高低差
侧向偏差
```

否则网络很容易只适应少数固定态势。

------

# 七、目标机动模块

```
env/maneuver.py
```

每个回合固定一种机动类型，但机动参数可以随机。

```python
class TargetManeuverController:
    def __init__(self, maneuver_type, strength, start_time):
        self.maneuver_type = maneuver_type
        self.strength = strength
        self.start_time = start_time

    def get_command(self, current_time):
        if current_time < self.start_time:
            return {
                "nx": 0.0,
                "nz": 0.0,
            }

        if self.maneuver_type == "straight":
            return {"nx": 0.0, "nz": 0.0}

        if self.maneuver_type == "climb":
            return {"nx": self.strength, "nz": 0.0}

        if self.maneuver_type == "dive":
            return {"nx": -self.strength, "nz": 0.0}

        if self.maneuver_type == "left_turn":
            return {"nx": 0.0, "nz": -self.strength}

        if self.maneuver_type == "right_turn":
            return {"nx": 0.0, "nz": self.strength}

        if self.maneuver_type == "left_climb":
            return {
                "nx": self.strength,
                "nz": -self.strength,
            }

        if self.maneuver_type == "right_climb":
            return {
                "nx": self.strength,
                "nz": self.strength,
            }

        raise ValueError(f"未知机动类型: {self.maneuver_type}")
```

这里的 `nx、nz` 只是示意，最终要严格按照你的目标飞机模型接口填写。

------

# 八、强化学习环境

`env/missile_env.py` 是整个系统中最关键的部分。

## 1. 环境职责

环境需要完成：

```text
初始化导弹和目标
构造状态
接收动作
映射CN
推进仿真
计算奖励
判断终止
返回统计信息
```

## 2. 大体结构

```python
import numpy as np

from utils.cn_mapper import get_cn
from utils.state_processor import StateProcessor


class MissileEnv:
    def __init__(
        self,
        config,
        missile_model_factory,
        target_model_factory,
        scenario_generator,
    ):
        self.config = config
        self.missile_model_factory = missile_model_factory
        self.target_model_factory = target_model_factory
        self.scenario_generator = scenario_generator

        self.state_processor = StateProcessor(
            position_scale=config.position_scale,
            velocity_scale=config.velocity_scale,
        )

        self.substeps = round(
            config.decision_dt / config.simulation_dt
        )

        self.missile = None
        self.target = None

        self.current_time = 0.0
        self.min_distance = float("inf")
        self.previous_distance = None
        self.current_cn = None
        self.cn_change_count = 0

    def reset(self):
        scenario = self.scenario_generator.sample()

        self.missile = self.missile_model_factory()
        self.target = self.target_model_factory()

        self.missile.reset(
            position=scenario["missile_position"],
            velocity=scenario["missile_velocity"],
        )

        self.target.reset(
            position=scenario["target_position"],
            velocity=scenario["target_velocity"],
            maneuver_type=scenario["maneuver_type"],
            maneuver_strength=scenario["maneuver_strength"],
            maneuver_start_time=scenario["maneuver_start_time"],
        )

        self.current_time = 0.0
        self.min_distance = self._get_distance()
        self.previous_distance = self.min_distance
        self.current_cn = None
        self.cn_change_count = 0

        return self._get_state()

    def step(self, action: int):
        distance_before = self._get_distance()
        cn = get_cn(distance_before, action)

        if self.current_cn is not None and cn != self.current_cn:
            self.cn_change_count += 1

        self.current_cn = cn

        hit = False
        invalid = False

        for _ in range(self.substeps):
            self.target.step(self.config.simulation_dt)

            self.missile.step(
                target_position=self.target.position,
                target_velocity=self.target.velocity,
                cn=cn,
                dt=self.config.simulation_dt,
            )

            self.current_time += self.config.simulation_dt

            distance = self._get_distance()
            self.min_distance = min(self.min_distance, distance)

            if distance <= self.config.hit_distance:
                hit = True
                break

            if self._has_invalid_state():
                invalid = True
                break

        distance_after = self._get_distance()

        timeout = self.current_time >= self.config.max_flight_time
        miss = self._check_miss(distance_before, distance_after)

        terminated = hit or miss or invalid
        truncated = timeout and not terminated
        done = terminated or truncated

        reward = self._calculate_reward(
            distance_before=distance_before,
            distance_after=distance_after,
            hit=hit,
            miss=miss,
            timeout=timeout,
            invalid=invalid,
        )

        next_state = self._get_state()

        info = {
            "hit": hit,
            "miss": miss,
            "timeout": timeout,
            "invalid": invalid,
            "distance": distance_after,
            "min_distance": self.min_distance,
            "missile_speed": np.linalg.norm(self.missile.velocity),
            "current_cn": cn,
            "cn_change_count": self.cn_change_count,
            "flight_time": self.current_time,
            "maneuver_type": self.target.maneuver_type,
        }

        self.previous_distance = distance_after

        return next_state, reward, done, info
```

------

# 九、脱靶终止条件

不能只依靠“超过最大飞行时间”判断脱靶，否则很多明显已经飞过目标的轨迹还会继续仿真。

可以设置以下条件：

```text
1. 弹目距离先减小后连续增大；
2. 弹目相对速度已经处于远离状态；
3. 导弹速度过低；
4. 导弹飞出仿真边界；
5. 数值异常；
6. 最大时间到达。
```

示例：

```python
def _check_miss(self, distance_before, distance_after):
    relative_position = self.target.position - self.missile.position
    relative_velocity = self.target.velocity - self.missile.velocity

    range_rate = np.dot(
        relative_position,
        relative_velocity,
    ) / (np.linalg.norm(relative_position) + 1e-6)

    moving_away = range_rate > 0
    passed_nearest_point = (
        distance_after > self.min_distance + 500.0
    )

    missile_speed = np.linalg.norm(self.missile.velocity)
    too_slow = missile_speed < 200.0

    return (
        (moving_away and passed_nearest_point)
        or too_slow
    )
```

这里的 `500 m`、`200 m/s` 都是初始阈值，需要结合模型调整。

------

# 十、奖励函数

可以单独放到 `env/reward.py`，也可以先写在环境内部。

```python
import numpy as np


def compute_reward(
    distance_before,
    distance_after,
    hit,
    miss,
    timeout,
    invalid,
    min_distance,
    missile_speed,
):
    distance_reward = (
        0.02 * (distance_before - distance_after) / 1000.0
    )

    time_penalty = -0.001

    reward = distance_reward + time_penalty

    if hit:
        speed_reward = np.clip(
            (missile_speed - 450.0) / 300.0,
            0.0,
            1.0,
        )
        reward += 5.0 + speed_reward

    elif miss or timeout:
        near_miss_reward = 0.5 * np.clip(
            (2000.0 - min_distance) / 2000.0,
            0.0,
            1.0,
        )
        reward += -2.0 + near_miss_reward

    elif invalid:
        reward -= 3.0

    return float(reward)
```

环境中调用：

```python
reward = compute_reward(
    distance_before=distance_before,
    distance_after=distance_after,
    hit=hit,
    miss=miss,
    timeout=timeout,
    invalid=invalid,
    min_distance=self.min_distance,
    missile_speed=np.linalg.norm(self.missile.velocity),
)
```

建议第一阶段不要加入过多奖励项。否则可能出现：

```text
奖励总和变高了，但命中率没有提高
```

第一版先保留：

```text
距离奖励
时间惩罚
命中奖励
脱靶惩罚
异常惩罚
```

------

# 十一、Q网络

```
rl/q_network.py
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim1: int = 128,
        hidden_dim2: int = 128,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, action_dim),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight,
                    nonlinearity="relu",
                )
                nn.init.zeros_(module.bias)

    def forward(self, state):
        return self.network(state)
```

注意最后一层不要使用：

```python
Softmax()
Sigmoid()
```

因为输出是四个动作的Q值，不是动作概率。

------

# 十二、经验回放池

```
rl/replay_buffer.py
import numpy as np
import torch


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        device: str,
    ):
        self.capacity = capacity
        self.device = device

        self.states = np.zeros(
            (capacity, state_dim),
            dtype=np.float32,
        )
        self.actions = np.zeros(
            capacity,
            dtype=np.int64,
        )
        self.rewards = np.zeros(
            capacity,
            dtype=np.float32,
        )
        self.next_states = np.zeros(
            (capacity, state_dim),
            dtype=np.float32,
        )
        self.dones = np.zeros(
            capacity,
            dtype=np.float32,
        )

        self.position = 0
        self.size = 0

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(
            0,
            self.size,
            size=batch_size,
        )

        states = torch.as_tensor(
            self.states[indices],
            device=self.device,
        )

        actions = torch.as_tensor(
            self.actions[indices],
            device=self.device,
        ).unsqueeze(1)

        rewards = torch.as_tensor(
            self.rewards[indices],
            device=self.device,
        ).unsqueeze(1)

        next_states = torch.as_tensor(
            self.next_states[indices],
            device=self.device,
        )

        dones = torch.as_tensor(
            self.dones[indices],
            device=self.device,
        ).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size
```

对于10万条、12维状态，这种预分配数组比用 Python 的 `deque` 存元组更高效。

------

# 十三、DDQN智能体

```
rl/ddqn_agent.py
import random

import numpy as np
import torch
import torch.nn.functional as F

from rl.q_network import QNetwork
from rl.replay_buffer import ReplayBuffer


class DDQNAgent:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)

        self.online_network = QNetwork(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim1=config.hidden_dim1,
            hidden_dim2=config.hidden_dim2,
        ).to(self.device)

        self.target_network = QNetwork(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim1=config.hidden_dim1,
            hidden_dim2=config.hidden_dim2,
        ).to(self.device)

        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=config.learning_rate,
        )

        self.replay_buffer = ReplayBuffer(
            capacity=config.replay_capacity,
            state_dim=config.state_dim,
            device=config.device,
        )

        self.environment_steps = 0
        self.update_steps = 0

    def get_epsilon(self):
        progress = min(
            self.environment_steps
            / self.config.epsilon_decay_steps,
            1.0,
        )

        epsilon = (
            self.config.epsilon_start
            + progress
            * (
                self.config.epsilon_end
                - self.config.epsilon_start
            )
        )

        return epsilon

    def select_action(self, state, training=True):
        epsilon = self.get_epsilon() if training else 0.0

        if training and random.random() < epsilon:
            return random.randrange(self.config.action_dim)

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.online_network(state_tensor)

        return int(q_values.argmax(dim=1).item())

    def store_transition(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        self.replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            done,
        )

    def update(self):
        if len(self.replay_buffer) < self.config.warmup_steps:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.replay_buffer.sample(
            self.config.batch_size
        )

        current_q = self.online_network(states).gather(
            dim=1,
            index=actions,
        )

        with torch.no_grad():
            # 在线网络选择下一状态的动作
            next_actions = (
                self.online_network(next_states)
                .argmax(dim=1, keepdim=True)
            )

            # 目标网络评价该动作
            next_q = self.target_network(
                next_states
            ).gather(
                dim=1,
                index=next_actions,
            )

            target_q = rewards + (
                self.config.gamma
                * (1.0 - dones)
                * next_q
            )

        loss = F.smooth_l1_loss(
            current_q,
            target_q,
        )

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            self.config.gradient_clip,
        )

        self.optimizer.step()

        self.update_steps += 1

        if (
            self.update_steps
            % self.config.target_update_interval
            == 0
        ):
            self.target_network.load_state_dict(
                self.online_network.state_dict()
            )

        return {
            "loss": float(loss.item()),
            "q_mean": float(current_q.mean().item()),
            "target_q_mean": float(target_q.mean().item()),
            "epsilon": self.get_epsilon(),
        }

    def save(self, path):
        torch.save(
            {
                "online_network": self.online_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "environment_steps": self.environment_steps,
                "update_steps": self.update_steps,
            },
            path,
        )
```

这里的核心DDQN逻辑是：

```python
next_actions = online_network(next_states).argmax(...)
next_q = target_network(next_states).gather(next_actions)
```

不能错误地写成普通DQN形式：

```python
target_network(next_states).max(...)
```

------

# 十四、主训练代码

```
train.py
from pathlib import Path

import numpy as np
import torch

from config import Config
from env.missile_env import MissileEnv
from env.scenario_generator import ScenarioGenerator
from rl.ddqn_agent import DDQNAgent


def train():
    config = Config()

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    rng = np.random.default_rng(config.seed)

    scenario_generator = ScenarioGenerator(rng)

    env = MissileEnv(
        config=config,
        missile_model_factory=create_missile_model,
        target_model_factory=create_target_model,
        scenario_generator=scenario_generator,
    )

    agent = DDQNAgent(config)

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    best_hit_rate = 0.0
    recent_results = []

    for episode in range(1, config.num_episodes + 1):
        state = env.reset()

        episode_reward = 0.0
        episode_steps = 0
        final_info = None

        while True:
            action = agent.select_action(
                state,
                training=True,
            )

            next_state, reward, done, info = env.step(action)

            agent.store_transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )

            agent.environment_steps += 1
            train_info = agent.update()

            state = next_state
            episode_reward += reward
            episode_steps += 1
            final_info = info

            if done:
                break

        recent_results.append(int(final_info["hit"]))

        if len(recent_results) > 100:
            recent_results.pop(0)

        recent_hit_rate = sum(recent_results) / len(
            recent_results
        )

        if episode % 10 == 0:
            print(
                f"Episode={episode}, "
                f"Reward={episode_reward:.3f}, "
                f"Steps={episode_steps}, "
                f"Hit={final_info['hit']}, "
                f"MinDistance={final_info['min_distance']:.2f}, "
                f"HitRate100={recent_hit_rate:.3f}, "
                f"Epsilon={agent.get_epsilon():.3f}"
            )

        if episode % 1000 == 0:
            agent.save(
                checkpoint_dir / f"episode_{episode}.pt"
            )

        if (
            len(recent_results) == 100
            and recent_hit_rate > best_hit_rate
        ):
            best_hit_rate = recent_hit_rate
            agent.save(checkpoint_dir / "best.pt")


if __name__ == "__main__":
    train()
```

这里的 `create_missile_model()` 和 `create_target_model()` 是你对现有模型的封装。

例如：

```python
def create_missile_model():
    return ExistingMissileModelAdapter()


def create_target_model():
    return ExistingTargetModelAdapter()
```

------

# 十五、现有导弹模型的适配层

你现有导弹模型未必是面向强化学习环境设计的，所以建议额外写一个适配器，而不是直接修改原模型。

```python
class MissileModelAdapter:
    def __init__(self):
        self.model = OriginalMissileModel()
        self.position = None
        self.velocity = None

    def reset(self, position, velocity):
        self.model.initialize(
            position=position,
            velocity=velocity,
        )

        self._sync_state()

    def step(
        self,
        target_position,
        target_velocity,
        cn,
        dt,
    ):
        self.model.set_target_state(
            position=target_position,
            velocity=target_velocity,
        )

        self.model.set_guidance_coefficient(cn)

        self.model.run_one_step(dt)

        self._sync_state()

    def _sync_state(self):
        self.position = self.model.get_position()
        self.velocity = self.model.get_velocity()
```

这样强化学习部分只依赖统一接口：

```python
missile.reset(...)
missile.step(..., cn=cn, dt=dt)
missile.position
missile.velocity
```

至于导弹模型内部如何通过CN计算导引指令、过载、姿态和下一状态，全部由原模型负责。

------

# 十六、训练前的随机经验收集

方案中写了初始随机经验5000～10000条，可以采用两种方式。

## 方式一：边交互边训练，但经验不足时不更新

也就是前面代码中的：

```python
if len(replay_buffer) < warmup_steps:
    return None
```

此时 ε 初始为1，前面基本都是随机动作。

## 方式二：先专门收集随机经验

```python
def collect_random_experience(
    env,
    agent,
    num_steps,
):
    state = env.reset()

    for _ in range(num_steps):
        action = np.random.randint(
            agent.config.action_dim
        )

        next_state, reward, done, info = env.step(action)

        agent.store_transition(
            state,
            action,
            reward,
            next_state,
            done,
        )

        state = next_state

        if done:
            state = env.reset()
```

正式训练前：

```python
collect_random_experience(
    env=env,
    agent=agent,
    num_steps=config.warmup_steps,
)
```

第二种逻辑更清晰。

------

# 十七、测试代码结构

`evaluate.py` 中关闭探索：

```python
action = agent.select_action(
    state,
    training=False,
)
```

建议针对每种机动分别测试：

```python
maneuver_types = [
    "straight",
    "climb",
    "dive",
    "left_turn",
    "right_turn",
    "left_climb",
    "right_climb",
]
```

测试结果可以整理成：

```python
{
    "straight": {
        "episodes": 1000,
        "hit_rate": 0.91,
        "mean_min_distance": 13.5,
        "mean_hit_speed": 680.2,
        "mean_flight_time": 24.1,
    },
    ...
}
```

同时还要测试基线：

```python
class FixedCNPolicy:
    def __init__(self, cn):
        self.cn = cn

    def select_cn(self, state, distance):
        return self.cn
```

以及固定分段策略：

```python
def fixed_segment_policy(distance):
    if distance > 50_000:
        return 2.0
    elif distance > 10_000:
        return 4.0
    else:
        return 6.0
```

由于当前环境的 `step()` 接收的是动作索引，评估阶段可以再增加一个接口：

```python
env.step_with_cn(cn)
```

或者将固定CN反向映射为对应动作。

从代码整洁性看，更建议把环境核心接口设计为：

```python
env.step_cn(cn)
```

然后在训练外层做：

```python
cn = get_cn(distance, action)
next_state, reward, done, info = env.step_cn(cn)
```

这样环境不需要知道DDQN的动作映射规则，也能直接测试固定CN基线。

------

# 十八、推荐进一步调整后的分层

实际上，更合理的分层是：

```text
DDQNAgent：输出动作索引
CNPolicy：把动作映射成CN
MissileEnv：只接收CN并推进物理模型
```

训练循环变成：

```python
action = agent.select_action(state)

distance = env.get_distance()

cn = get_cn(
    distance=distance,
    action=action,
)

next_state, reward, done, info = env.step(cn)
```

对应环境：

```python
def step(self, cn: float):
    ...
```

这样的好处是：

```text
环境不与动作空间绑定
固定CN策略更容易测试
以后增加8个动作时不必修改环境
以后使用连续动作算法时也可以复用环境
```

这是我更推荐的最终结构。

------

# 十九、训练日志需要记录什么

每个回合至少记录：

```text
episode_reward
episode_steps
hit
min_distance
final_distance
final_speed
flight_time
maneuver_type
CN变化次数
远中近各阶段动作选择次数
epsilon
loss
平均Q值
```

特别需要监控以下异常：

## 1. Q值不断增大

例如从：

```text
5 → 50 → 500 → 5000
```

通常表示：

```text
奖励尺度过大
终止状态处理错误
学习率过大
数值异常
```

## 2. 奖励提高但命中率不提高

可能是网络在“刷距离奖励”，但无法命中。

## 3. 始终选择同一个动作

需要判断是：

```text
该动作确实最优
动作CN差异太小
探索衰减过快
奖励无法区分动作效果
```

## 4. 每一步都改变CN

可能说明动作选择震荡。可以考虑：

```text
增加动作切换惩罚
延长决策周期
限制连续两次决策的动作变化幅度
```

但第一版不建议立即增加切换惩罚，应先观察。

------

# 二十、最简训练主流程

将所有细节压缩后，核心训练代码其实就是：

```python
for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        action = agent.select_action(state)

        distance = env.get_distance()
        cn = get_cn(distance, action)

        next_state, reward, done, info = env.step(cn)

        agent.replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            done,
        )

        agent.update()

        state = next_state
```

因此整个项目最关键的并不是Q网络，而是：

1. `env.reset()` 能否生成合理、多样的初始态势；
2. `env.step(cn)` 能否稳定推进导弹与目标模型；
3. 命中、脱靶和超时判断是否正确；
4. 奖励是否真的与命中效果一致；
5. 仿真模型是否会出现NaN、状态发散或不合理弹道。

DDQN网络本身只占总代码的一小部分，环境封装和仿真验证才是主要工作。