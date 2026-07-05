"""Version1 训练和 ONNX 推理使用的固定张量尺寸。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.

离线部署时请保持这些常量和训练好的模型一致：
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
- M_MAX/A_MAX/T_MAX 是导弹、友方飞机、目标的最大数量。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
- FEATURE_DIM=6 表示每个实体只输入位置和速度 [x, y, z, vx, vy, vz]。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
- ONNX 只允许 batch 维动态，实体数量维度必须固定为这里的配置。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
"""

M_MAX = 4
A_MAX = 4
T_MAX = 8
FEATURE_DIM = 6

POSITION_SCALE = 50_000.0
VELOCITY_SCALE = 2_000.0
