from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from train_situation import SituationMLP
from train_threat import ThreatMLP
from train_allocation import AllocationCoreTransformer, AllocationPipelineTransformer


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 创建三个网络结构
    situation_model = SituationMLP(hidden_dim=64).to(device)

    threat_model = ThreatMLP(hidden_dim=64).to(device)

    allocation_core = AllocationCoreTransformer(
        d_model=64,
        num_heads=4,
        num_layers=2,
        ff_dim=128,
        dropout=0.0,
    ).to(device)

    # 2. 分别加载三个 pth 权重
    situation_model.load_state_dict(
        torch.load(Path("situation_model_200k.pth"), map_location=device)
    )

    threat_model.load_state_dict(
        torch.load(Path("threat_model_200k.pth"), map_location=device)
    )

    allocation_core.load_state_dict(
        torch.load(Path("allocation_model_200k.pth"), map_location=device)
    )

    # 3. 组装完整目标分配 pipeline
    pipeline = AllocationPipelineTransformer(
        situation_model=situation_model,
        threat_model=threat_model,
        allocation_core=allocation_core,
    ).to(device)

    pipeline.eval()

    # 4. 构造原始输入
    # x_missile: [B, 4, 6]
    # x_aircraft: [B, 4, 6]
    # x_target: [B, 8, 6]
    # 6 = [x, y, z, vx, vy, vz]

    x_missile = np.zeros((1, 4, 6), dtype=np.float32)
    x_aircraft = np.zeros((1, 4, 6), dtype=np.float32)
    x_target = np.zeros((1, 8, 6), dtype=np.float32)

    # 两枚导弹
    x_missile[0, 0] = [5000.0, 10000.0, 5000.0, 600.0, -500.0, 0.0]
    x_missile[0, 1] = [5000.0, -10000.0, 5000.0, 600.0, 500.0, 0.0]

    # 两架友方飞机
    x_aircraft[0, 0] = [0.0, 10000.0, 5000.0, 250.0, 0.0, 0.0]
    x_aircraft[0, 1] = [0.0, -10000.0, 5000.0, 250.0, 0.0, 0.0]

    # 三个目标
    x_target[0, 0] = [30000.0, 15000.0, 5000.0, -200.0, -50.0, 0.0]
    x_target[0, 1] = [30000.0, -15000.0, 5000.0, -200.0, 50.0, 0.0]
    x_target[0, 2] = [70000.0, 0.0, 6500.0, -250.0, 0.0, -30.0]

    # 5. 构造 mask
    mask_m = np.zeros((1, 4), dtype=np.float32)
    mask_a = np.zeros((1, 4), dtype=np.float32)
    mask_t = np.zeros((1, 8), dtype=np.float32)

    mask_m[0, 0] = 1.0
    mask_m[0, 1] = 1.0

    mask_a[0, 0] = 1.0
    mask_a[0, 1] = 1.0

    mask_t[0, 0] = 1.0
    mask_t[0, 1] = 1.0
    mask_t[0, 2] = 1.0

    # 6. 转成 Tensor
    x_missile_tensor = torch.from_numpy(x_missile).to(device)
    x_aircraft_tensor = torch.from_numpy(x_aircraft).to(device)
    x_target_tensor = torch.from_numpy(x_target).to(device)

    mask_m_tensor = torch.from_numpy(mask_m).to(device)
    mask_a_tensor = torch.from_numpy(mask_a).to(device)
    mask_t_tensor = torch.from_numpy(mask_t).to(device)

    # 7. 目标分配推理
    with torch.no_grad():
        allocation_logits = pipeline(
            x_missile_tensor,
            x_aircraft_tensor,
            x_target_tensor,
            mask_m_tensor,
            mask_a_tensor,
            mask_t_tensor,
        )

    allocation_index = torch.argmax(allocation_logits, dim=-1)

    allocation_logits = allocation_logits.cpu().numpy()
    allocation_index = allocation_index.cpu().numpy()

    real_missile_count = int(mask_m.sum())
    real_target_count = int(mask_t.sum())

    valid_logits = allocation_logits[
        0,
        :real_missile_count,
        :real_target_count,
    ]

    valid_allocation_index = allocation_index[
        0,
        :real_missile_count,
    ]

    print("valid allocation logits:")
    print(np.array2string(valid_logits, precision=4, suppress_small=True))
    print()

    print("allocation result:")
    for missile_index, target_index in enumerate(valid_allocation_index):
        print(f"missile {missile_index} -> target {int(target_index)}")


if __name__ == "__main__":
    main()