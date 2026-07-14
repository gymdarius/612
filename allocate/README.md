# 目标分配专家规则 V1

公开接口直接接收 `friendly_planes`、`friendly_missiles` 和 `enemy_planes`
三个二维数组，每一行固定为 `[x, y, z, vx, vy, vz]`。

```python
from expert_rule import create_default_expert

result = create_default_expert().assign(
    friendly_planes,
    friendly_missiles,
    enemy_planes,
)
print(result.assignments)
```

规则只使用三维距离、相对接近速度和方向角。敌机对编队的威胁取它对
所有我方飞机威胁的最大值，综合分数为：

```text
综合分数 = 攻击占比 × 攻击优势 + 威胁占比 × 敌机威胁
```

每枚导弹独立选择分数最高的敌机，因此允许多枚导弹攻击同一目标。

## 运行

```powershell
cd 2026-7-1
python example.py
python -m unittest -v
```

所有权重和归一化尺度集中在 `config.yaml`，三组权重必须分别等于 1。
