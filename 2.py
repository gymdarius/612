from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from train_threat import ThreatMLP


def main() -> None:
    # 1. 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 创建网络结构
    # 必须和训练时保持一致，当前项目默认 hidden_dim=64
    model = ThreatMLP(hidden_dim=64).to(device)

    # 3. 加载 pth 权重
    weight_path = Path("threat_model_200k.pth")
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)

    # 4. 切换到推理模式
    model.eval()

    # 5. 构造友方飞机输入
    # shape: [B, A_MAX, 6]
    # B=1, A_MAX=4, 6=[x, y, z, vx, vy, vz]
    x_aircraft = np.zeros((1, 4, 6), dtype=np.float32)

    # 示例：两架真实友方飞机
    x_aircraft[0, 0] = [0.0, 10000.0, 5000.0, 250.0, 0.0, 0.0]
    x_aircraft[0, 1] = [0.0, -10000.0, 5000.0, 250.0, 0.0, 0.0]

    # 6. 构造目标输入
    # shape: [B, T_MAX, 6]
    # B=1, T_MAX=8, 6=[x, y, z, vx, vy, vz]
    x_target = np.zeros((1, 8, 6), dtype=np.float32)

    # 示例：三个真实目标
    x_target[0, 0] = [30000.0, 15000.0, 5000.0, -200.0, -50.0, 0.0]
    x_target[0, 1] = [30000.0, -15000.0, 5000.0, -200.0, 50.0, 0.0]
    x_target[0, 2] = [70000.0, 0.0, 6500.0, -250.0, 0.0, -30.0]

    # 7. 构造 mask
    # 1 表示真实实体，0 表示 padding
    mask_a = np.zeros((1, 4), dtype=np.float32)
    mask_t = np.zeros((1, 8), dtype=np.float32)

    # 两架真实友方飞机
    mask_a[0, 0] = 1.0
    mask_a[0, 1] = 1.0

    # 三个真实目标
    mask_t[0, 0] = 1.0
    mask_t[0, 1] = 1.0
    mask_t[0, 2] = 1.0

    # 8. 转成 torch.Tensor
    x_aircraft_tensor = torch.from_numpy(x_aircraft).to(device)
    x_target_tensor = torch.from_numpy(x_target).to(device)
    mask_a_tensor = torch.from_numpy(mask_a).to(device)
    mask_t_tensor = torch.from_numpy(mask_t).to(device)

    # 9. 前向推理
    with torch.no_grad():
        threat_matrix, target_threat_weights, joint_mask = model(
            x_aircraft_tensor,
            x_target_tensor,
            mask_a_tensor,
            mask_t_tensor,
        )

    # 10. 转回 numpy，方便查看
    threat_matrix = threat_matrix.cpu().numpy()
    target_threat_weights = target_threat_weights.cpu().numpy()
    joint_mask = joint_mask.cpu().numpy()

    # 11. 只取真实友机和真实目标部分
    real_aircraft_count = int(mask_a.sum())
    real_target_count = int(mask_t.sum())

    valid_threat = threat_matrix[
        0,
        :real_aircraft_count,
        :real_target_count,
    ]

    valid_target_weights = target_threat_weights[
        0,
        :real_target_count,
    ]

    print("threat_matrix shape:", threat_matrix.shape)
    print("target_threat_weights shape:", target_threat_weights.shape)
    print("joint_mask shape:", joint_mask.shape)
    print()
    print("valid threat matrix [aircraft_count, target_count]:")
    print(np.array2string(valid_threat, precision=4, suppress_small=True))
    print()
    print("valid target threat weights [target_count]:")
    print(np.array2string(valid_target_weights, precision=4, suppress_small=True))


if __name__ == "__main__":
    main()