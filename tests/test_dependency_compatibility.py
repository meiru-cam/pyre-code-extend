"""Runtime compatibility checks for the numerical stack."""

from __future__ import annotations

import numpy as np
import torch


def test_torch_numpy_bridge_is_available():
    """The supported PyTorch build must be able to consume a NumPy array."""
    array = np.array([1.0, 2.0], dtype=np.float32)

    tensor = torch.from_numpy(array)

    assert tensor.tolist() == [1.0, 2.0]
