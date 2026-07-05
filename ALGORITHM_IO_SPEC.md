# Version1 Algorithm IO Specification

本文档记录 `Version1` 当前六类算法的统一输入输出约定。前三个算法已经有 DTO、训练数据和 ONNX 导出链路；后三个算法目前放在 `future_algorithms` 下，先沿用同一套实体、张量和 mask 规则，后续稳定后再考虑是否独立成项目。

## 1. 基础实体约定

统一实体类型定义在 `dto.py` 的 `EntityState`：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | `str` | 实体唯一标识。旧字典输入未传时由调用方默认前缀生成。 |
| `position` | `np.ndarray shape=(3,)` | 位置向量 `[x, y, z]`，单位按当前数据生成逻辑理解为米。 |
| `velocity` | `np.ndarray shape=(3,)` | 速度向量 `[vx, vy, vz]`，单位按当前数据生成逻辑理解为米/秒。 |
| `speed` | `float | None` | 速度模长；未传或与 `velocity` 不一致时，以 `velocity` 模长为准。 |
| `euler_angles` | `np.ndarray shape=(3,)` | 欧拉角，当前前三个算法没有作为训练张量输入。 |

别名：

```text
MissileStateDTO = EntityState
TargetStateDTO = EntityState
AircraftStateDTO = EntityState
```

注意：`future_algorithms/common.py` 中 `HEIGHT_AXIS = 1`，当前威胁评估和航路规划逻辑都把 `position[1]` 当作高度轴使用。

## 2. 固定张量约定

固定维度来自 `shape_config.py`：

| 常量 | 当前值 | 说明 |
| --- | ---: | --- |
| `M_MAX` | `4` | 导弹最大数量。 |
| `A_MAX` | `4` | 友方飞机最大数量。 |
| `T_MAX` | `8` | 目标最大数量。 |
| `FEATURE_DIM` | `6` | 每个实体输入特征数量。 |

基础输入张量：

| 名称 | shape | 特征顺序 |
| --- | --- | --- |
| `X_missile` | `[B, M_MAX, FEATURE_DIM]` | `[x, y, z, vx, vy, vz]` |
| `X_aircraft` | `[B, A_MAX, FEATURE_DIM]` | `[x, y, z, vx, vy, vz]` |
| `X_target` | `[B, T_MAX, FEATURE_DIM]` | `[x, y, z, vx, vy, vz]` |

训练数据中这些张量保存为 `float32`。当前规则算法和 `future_algorithms` 内部通常转为 `float64` 做 NumPy 计算。训练模型内部会在构造 pair 特征时使用：

```text
POSITION_SCALE = 50_000.0
VELOCITY_SCALE = 2_000.0
```

实体数量不足固定上限时，padding 行填 `0`，并通过 mask 区分真实实体和 padding。

## 3. Mask 规则

基础 mask：

| 名称 | shape | 说明 |
| --- | --- | --- |
| `mask_M` | `[B, M_MAX]` | `1/True` 表示真实导弹，`0/False` 表示 padding。 |
| `mask_A` | `[B, A_MAX]` | `1/True` 表示真实友方飞机，`0/False` 表示 padding。 |
| `mask_T` | `[B, T_MAX]` | `1/True` 表示真实目标，`0/False` 表示 padding。 |

pair 级 mask：

```text
joint_mask_MT = mask_M[:, :, None] * mask_T[:, None, :]
joint_mask_AT = mask_A[:, :, None] * mask_T[:, None, :]
```

规则：

- padding 实体不能参与训练 loss、评分统计和最终动作输出。
- padding pair 的预测值应置零或被 mask 掉。
- ONNX 导出只把 batch 维设为动态，`M_MAX/A_MAX/T_MAX/FEATURE_DIM` 必须保持固定。

## 4. 前三个算法 DTO 接口

### 4.1 态势评估

输入：

```text
SAMInputDTO(
    missiles: Tuple[EntityState, ...],
    targets: Tuple[EntityState, ...],
)
```

主接口：

```text
SituationAssessment().compute(SAMInputDTO(...)) -> SAMOutputDTO
```

输出：

| 字段 | shape / 类型 | 说明 |
| --- | --- | --- |
| `situation_matrix` | `[M, T]` | 导弹-目标态势评分，范围 `[0, 1]`。 |
| `pair_metrics` | `Tuple[SAMPairMetricsDTO, ...]` | 可选调试指标，包含距离、接近速度、视线夹角、分量评分等。 |

当前逻辑：沿用 Version1 规则公式，融合接近速度、距离、boresight、导弹能量。

### 4.2 威胁评估

输入：

```text
ThreatInputDTO(
    our_aircrafts: Tuple[EntityState, ...],
    targets: Tuple[EntityState, ...],
)
```

主接口：

```text
ThreatAssessment().compute(ThreatInputDTO(...)) -> ThreatOutputDTO
```

输出：

| 字段 | shape / 类型 | 说明 |
| --- | --- | --- |
| `threat_matrix` | `[A, T]` | 友方飞机-目标威胁评分，范围 `[0, 1]`。 |
| `target_threat_weights` | `[T]` | 按目标聚合并归一化后的威胁权重。 |
| `pair_metrics` | `Tuple[ThreatPairMetricsDTO, ...]` | 可选调试指标，包含距离、LOS、目标速度、高度差、分量评分等。 |

当前逻辑：沿用 Version1 规则公式，融合 LOS 指向、距离、速度比、高度差。`target_threat_weights` 由 `threat_matrix` 按目标平均后归一化得到。

### 4.3 目标分配

输入：

```text
AllocationInputDTO(
    missiles: Tuple[EntityState, ...],
    targets: Tuple[EntityState, ...],
    situation_output: SAMOutputDTO,
    threat_output: ThreatOutputDTO,
)
```

主接口：

```text
TargetAllocation().compute(AllocationInputDTO(...)) -> AllocationOutputDTO
```

输出：

| 字段 | shape / 类型 | 说明 |
| --- | --- | --- |
| `assignments` | `Tuple[AllocationEntryDTO, ...]` | 每枚有效导弹的目标分配结果。 |
| `normalized_matrix` | `[M, T]` | 按导弹逐行 softmax 后的目标概率。 |
| `value_matrix` | `[M, T]` | 融合态势评分和目标威胁权重后的原始价值矩阵。 |
| `target_threat_weights` | `[T]` | 从威胁评估透传的目标威胁权重。 |

当前逻辑：

```text
value = w_advantage * situation + w_threat * target_threat_weight
normalized = row_softmax(value)
target_index = argmax(normalized, axis=target)
```

训练和 ONNX 的分配输出没有单独的 `UNASSIGNED` 类别。有效导弹用 `Y_AllocMask=1` 参与 loss；padding 导弹用 `Y_AllocMask=0` 屏蔽。

## 5. 训练数据 `.npz` 约定

`dataset.py` 要求 `.npz` 中包含以下数组：

| 名称 | shape |
| --- | --- |
| `X_missile` | `[N, M_MAX, FEATURE_DIM]` |
| `X_aircraft` | `[N, A_MAX, FEATURE_DIM]` |
| `X_target` | `[N, T_MAX, FEATURE_DIM]` |
| `mask_M` | `[N, M_MAX]` |
| `mask_A` | `[N, A_MAX]` |
| `mask_T` | `[N, T_MAX]` |
| `Y_S` | `[N, M_MAX, T_MAX]` |
| `Y_Threat` | `[N, A_MAX, T_MAX]` |
| `Y_ThreatW` | `[N, T_MAX]` |
| `Y_AllocIndex` | `[N, M_MAX]` |
| `Y_AllocMask` | `[N, M_MAX]` |

标签来源：

- `Y_S` 来自 `SituationAssessment.compute(...)`。
- `Y_Threat` 和 `Y_ThreatW` 来自 `ThreatAssessment.compute(...)`。
- `Y_AllocIndex` 和 `Y_AllocMask` 来自 `TargetAllocation.compute(...)`。

如果修改 `shape_config.py` 中的固定维度，必须重新生成数据集、重新训练并重新导出 ONNX。

## 6. ONNX 接口约定

当前导出文件和输入输出名称：

| 文件 | 输入 | 输出 |
| --- | --- | --- |
| `situation_model.onnx` | `X_missile, X_target, mask_M, mask_T` | `S_pred, joint_mask_MT` |
| `threat_model.onnx` | `X_aircraft, X_target, mask_A, mask_T` | `Threat_pred, ThreatW_pred, joint_mask_AT` |
| `allocation_model.onnx` | `X_missile, X_aircraft, X_target, mask_M, mask_A, mask_T` | `Alloc_logits, allocation_index, allocation_mask` |
| `version1_aircombat_pipeline.onnx` | `X_missile, X_aircraft, X_target, mask_M, mask_A, mask_T` | `S_pred, Threat_pred, ThreatW_pred, allocation_index, allocation_mask` |

约束：

- `batch_size` 是唯一动态维。
- `allocation_index` 是每枚导弹的目标下标，范围为 `[0, T_MAX - 1]`；有效性由 `allocation_mask` 判断。
- 推理前必须按本规范 padding 并生成 mask。

## 7. 后三个算法当前接口

后三个算法位于 `future_algorithms`，目前是规则/启发式版本，用于先固定接口和调试流程。

### 7.1 目标轨迹预测

当前接口：

```text
predict_target_trajectory(
    X_target_history: [B, T_MAX, HISTORY_STEPS, FEATURE_DIM],
    mask_T: [B, T_MAX] | None,
    dt: float = DEFAULT_DT,
    future_steps: int = FUTURE_STEPS,
) -> TargetTrajectoryPredictionResult
```

当前输出：

| 字段 | shape / 类型 | 说明 |
| --- | --- | --- |
| `predictions` | `[B, T_MAX, future_steps, 3]` | 每个目标未来位置 `[x, y, z]`。padding 目标输出为 `0`。 |
| `method` | `str` | 当前为 `constant_velocity` 或 `average_velocity`。 |
| `dt` | `float` | 时间步长。 |
| `future_steps` | `int` | 预测步数。 |

后续建议：如果要直接接入前三个算法的瞬时 `X_target`，可以保留 `X_target_history` 作为高阶输入，同时提供一个从 `[B, T_MAX, FEATURE_DIM]` 构造最短历史的适配层。

### 7.2 航路规划

当前接口：

```text
plan_routes(
    X_missile: [B, M_MAX, FEATURE_DIM],
    X_target: [B, T_MAX, FEATURE_DIM],
    mask_M: [B, M_MAX] | None,
    mask_T: [B, T_MAX] | None,
) -> RoutePlanningResult
```

当前输出：

| 字段 | shape / 类型 | 说明 |
| --- | --- | --- |
| `route_action` | `[B, M_MAX]` | 每枚有效导弹的航路动作编号。 |
| `decisions` | `Tuple[RouteDecision, ...]` | 逐导弹解释，包含参考目标下标和原因。 |

当前动作：

| action_id | action_name |
| ---: | --- |
| `0` | `KEEP` |
| `1` | `CLIMB_UP` |
| `2` | `CLIMB_DOWN` |
| `3` | `TRACK_MORE` |
| `4` | `TRACK_LESS` |

当前逻辑：每枚有效导弹选择最近有效目标作为参考，根据高度差、距离和接近速度决定动作。

后续建议：稳定接口后，可加入 `allocation_index` 或 `AllocationOutputDTO`，让航路规划优先围绕已分配目标，而不是只选最近目标。

### 7.3 作战使用决策

当前接口：

```text
decide_combat_use(
    friendly_missiles: [M, FEATURE_DIM],
    friendly_aircrafts: [A, FEATURE_DIM],
    enemy_aircrafts: [T_aircraft, FEATURE_DIM],
    enemy_missiles: [T_missile, FEATURE_DIM],
    masks...
) -> CombatDecisionResult
```

当前输出：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `decisions` | `Tuple[CombatDecision, ...]` | 每枚有效友方导弹的攻击、拦截或保持任务决策。 |

当前动作：

| action_id | action_name | target_type |
| ---: | --- | --- |
| `0` | `ATTACK_ENEMY_AIRCRAFT` | `enemy_aircraft` |
| `1` | `INTERCEPT_ENEMY_MISSILE` | `enemy_missile` |
| `2` | `KEEP_CURRENT_TASK` | `none` |

当前逻辑：对敌机攻击收益和敌弹拦截紧迫度分别评分，然后选择更高的动作。

后续建议：这是最应该汇总上游结果的模块，后续可逐步增加以下输入：

```text
situation_output
threat_output
allocation_output
trajectory_prediction_result
route_planning_result
```

在接口稳定前，不建议把它拆到独立项目。

## 8. 修改边界建议

短期内建议保持以下原则：

- 先维护统一实体和张量格式，不引入第二套输入输出规范。
- 后三个算法优先在 `future_algorithms` 中演进，输出尽量保留 `to_dict()` 便于展示和离线检查。
- 新增模型训练或 ONNX 导出前，先补充对应的固定 shape、mask、标签含义和 demo。
- 真正需要独立部署、独立训练、独立版本管理，或依赖环境明显不同的时候，再考虑拆成新项目。
