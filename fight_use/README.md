# 全导弹攻防专家规则

该目录是只依赖 Python 标准库的独立实现，整体移出原项目后仍可运行。系统认为输入的所有友方导弹都刚发射且没有目标，并为每枚导弹选择一架敌机或一枚敌弹。

## 规则约定

- 实体状态为 `[x, y, z, vx, vy, vz]`。
- 位置单位为米，速度单位为米/秒，`y` 轴为高度轴。
- 敌机固定编号为 `0 .. X_max-1`。
- 敌弹固定编号为 `X_max .. X_max+Y_max-1`。
- 目标没有容量限制，多枚友弹可以选择同一个目标。
- 每枚真实友弹默认必须得到目标；不可达时放宽几何门控强制选择。

## 代码结构

- `expert_rules.py`：稳定的公开接口。
- `allocator.py`：`ExpertRuleEngine` 主流程。
- `models.py`：实体、几何结果和分层配置对象。
- `config_loader.py`：JSON配置加载和参数校验。
- `geometry.py`：三维相对运动与CPA计算。
- `scoring.py`：`ScoreModel` 底层攻防评分。
- `threat_evaluation.py`：敌方目标全局威胁评估。
- `candidate_scoring.py`：完整友弹—目标评分矩阵。
- `target_selection.py`：策略偏置、紧急覆盖和兜底选择。
- `expert_rule_config.json`：默认全局配置。
- `example.py`、`test_expert_rules.py`：示例和测试。

## 运行

```powershell
python -B example.py
python -B -m unittest -v
```

## 调用

推荐复用引擎实例，配置文件只在初始化时读取一次：

```python
from expert_rules import ExpertRuleEngine

engine = ExpertRuleEngine(config_path="expert_rule_config.json")
result = engine.assign(
    friendly_missiles=[...],
    friendly_planes=[...],
    enemy_planes=[...],
    enemy_missiles=[...],
)
```

也可以使用函数入口：

```python
from expert_rules import assign_all_missiles

result = assign_all_missiles(
    friendly_missiles,
    friendly_planes,
    enemy_planes,
    enemy_missiles,
)
```

## 攻防倾向

在 `expert_rule_config.json` 中调整：

```json
"strategy": {
  "strategy_bias": 0.0,
  "bias_strength": 0.7,
  "emergency_defense_enabled": true,
  "emergency_overrides_strategy": true
}
```

- `strategy_bias=-1`：强烈倾向攻击。
- `strategy_bias=0`：攻防中性。
- `strategy_bias=1`：强烈倾向防御。
- `bias_strength`：控制倾向对最终比较的影响幅度。
- `emergency_overrides_strategy`：紧急敌弹是否覆盖普通攻防倾向。

底层攻击和防御原始评分不会被策略参数修改。输出同时保留 `raw_expert_scores` 和应用策略后的 `final_expert_scores`，便于解释决策。
