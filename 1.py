from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from train_situation import SituationMLP


def main() -> None:
    # 1. 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 创建网络结构
    # 必须和训练时的结构一致，当前项目默认 hidden_dim=64
    model = SituationMLP(hidden_dim=64).to(device)

    # 3. 加载 pth 权重
    weight_path = Path("situation_model_200k.pth")
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)

    # 4. 切换到推理模式
    model.eval()

    # 5. 构造输入
    # shape: [B, M_MAX, 6]
    # B=1, M_MAX=4, 6=[x, y, z, vx, vy, vz]
    x_missile = np.zeros((1, 4, 6), dtype=np.float32)

    # 示例：两枚真实导弹
    x_missile[0, 0] = [5000.0, 10000.0, 5000.0, 600.0, -500.0, 0.0]
    x_missile[0, 1] = [5000.0, -10000.0, 5000.0, 600.0, 500.0, 0.0]

    # shape: [B, T_MAX, 6]
    # B=1, T_MAX=8, 6=[x, y, z, vx, vy, vz]
    x_target = np.zeros((1, 8, 6), dtype=np.float32)

    # 示例：三个真实目标
    x_target[0, 0] = [30000.0, 15000.0, 5000.0, -200.0, -50.0, 0.0]
    x_target[0, 1] = [30000.0, -15000.0, 5000.0, -200.0, 50.0, 0.0]
    x_target[0, 2] = [70000.0, 0.0, 6500.0, -250.0, 0.0, -30.0]

    # 6. 构造 mask
    # 1 表示真实实体，0 表示 padding
    mask_m = np.zeros((1, 4), dtype=np.float32)
    mask_t = np.zeros((1, 8), dtype=np.float32)

    # 两枚真实导弹
    mask_m[0, 0] = 1.0
    mask_m[0, 1] = 1.0

    # 三个真实目标
    mask_t[0, 0] = 1.0
    mask_t[0, 1] = 1.0
    mask_t[0, 2] = 1.0

    # 7. 转成 torch.Tensor
    x_missile_tensor = torch.from_numpy(x_missile).to(device)
    x_target_tensor = torch.from_numpy(x_target).to(device)
    mask_m_tensor = torch.from_numpy(mask_m).to(device)
    mask_t_tensor = torch.from_numpy(mask_t).to(device)

    # 8. 前向推理
    with torch.no_grad():
        situation_matrix, joint_mask = model(
            x_missile_tensor,
            x_target_tensor,
            mask_m_tensor,
            mask_t_tensor,
        )

    # 9. 转回 numpy，方便查看
    situation_matrix = situation_matrix.cpu().numpy()
    joint_mask = joint_mask.cpu().numpy()

    # 10. 只取真实导弹和真实目标部分
    real_missile_count = int(mask_m.sum())
    real_target_count = int(mask_t.sum())

    valid_situation = situation_matrix[
        0,
        :real_missile_count,
        :real_target_count,
    ]

    print("situation_matrix shape:", situation_matrix.shape)
    print("joint_mask shape:", joint_mask.shape)
    print()
    print("valid situation matrix [missile_count, target_count]:")
    print(np.array2string(valid_situation, precision=4, suppress_small=True))


if __name__ == "__main__":
    main()