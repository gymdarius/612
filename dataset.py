from __future__ import annotations

"""Version1 监督学习数据集加载器。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.

generate_dataset.py 会把每个随机场景保存成固定形状的 .npz 数组。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
这个 Dataset 负责校验这些数组的形状，并按训练脚本约定的顺序返回
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
PyTorch Tensor。离线环境中如果更改 shape_config.py，必须重新生成数据集。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
"""

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from shape_config import A_MAX, FEATURE_DIM, M_MAX, T_MAX


REQUIRED_KEYS = (
    "X_missile",
    "X_aircraft",
    "X_target",
    "mask_M",
    "mask_A",
    "mask_T",
    "Y_S",
    "Y_Threat",
    "Y_ThreatW",
    "Y_AllocIndex",
    "Y_AllocMask",
)


class AirCombatDataset(Dataset):
    """读取 Version1 训练数据的 PyTorch Dataset。
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.

    单个样本返回顺序固定为：
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    X_missile, X_aircraft, X_target, mask_M, mask_A, mask_T,
    Y_S, Y_Threat, Y_ThreatW, Y_AllocIndex, Y_AllocMask。
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    """

    def __init__(self, data_path: str | Path) -> None:
        data = np.load(data_path)
        missing = [key for key in REQUIRED_KEYS if key not in data]
        if missing:
            raise KeyError(f"Dataset is missing required arrays: {missing}")

        self.arrays: Dict[str, np.ndarray] = {key: data[key] for key in REQUIRED_KEYS}
        self._validate_shapes()

    def __len__(self) -> int:
        return int(self.arrays["X_missile"].shape[0])

    def __getitem__(self, index: int):
        # 训练脚本直接按这个顺序解包；修改顺序时需要同步所有 train/evaluate 脚本。
        # English note: This comment explains the following implementation detail for offline readability.
        return (
            torch.as_tensor(self.arrays["X_missile"][index], dtype=torch.float32),
            torch.as_tensor(self.arrays["X_aircraft"][index], dtype=torch.float32),
            torch.as_tensor(self.arrays["X_target"][index], dtype=torch.float32),
            torch.as_tensor(self.arrays["mask_M"][index], dtype=torch.float32),
            torch.as_tensor(self.arrays["mask_A"][index], dtype=torch.float32),
            torch.as_tensor(self.arrays["mask_T"][index], dtype=torch.float32),
            torch.as_tensor(self.arrays["Y_S"][index], dtype=torch.float32),
            torch.as_tensor(self.arrays["Y_Threat"][index], dtype=torch.float32),
            torch.as_tensor(self.arrays["Y_ThreatW"][index], dtype=torch.float32),
            torch.as_tensor(self.arrays["Y_AllocIndex"][index], dtype=torch.long),
            torch.as_tensor(self.arrays["Y_AllocMask"][index], dtype=torch.float32),
        )

    def _validate_shapes(self) -> None:
        # 所有数组的 batch 维必须一致，实体数量维必须等于 shape_config.py 中的固定值。
        # English note: This comment explains the following implementation detail for offline readability.
        sample_count = self._common_sample_count(REQUIRED_KEYS)
        expected_shapes = {
            "X_missile": (sample_count, M_MAX, FEATURE_DIM),
            "X_aircraft": (sample_count, A_MAX, FEATURE_DIM),
            "X_target": (sample_count, T_MAX, FEATURE_DIM),
            "mask_M": (sample_count, M_MAX),
            "mask_A": (sample_count, A_MAX),
            "mask_T": (sample_count, T_MAX),
            "Y_S": (sample_count, M_MAX, T_MAX),
            "Y_Threat": (sample_count, A_MAX, T_MAX),
            "Y_ThreatW": (sample_count, T_MAX),
            "Y_AllocIndex": (sample_count, M_MAX),
            "Y_AllocMask": (sample_count, M_MAX),
        }
        for key, shape in expected_shapes.items():
            if self.arrays[key].shape != shape:
                raise ValueError(f"{key} shape mismatch: expected {shape}, got {self.arrays[key].shape}.")

    def _common_sample_count(self, keys: Iterable[str]) -> int:
        counts = {key: int(self.arrays[key].shape[0]) for key in keys}
        unique_counts = set(counts.values())
        if len(unique_counts) != 1:
            raise ValueError(f"Dataset arrays have different sample counts: {counts}")
        return unique_counts.pop()
