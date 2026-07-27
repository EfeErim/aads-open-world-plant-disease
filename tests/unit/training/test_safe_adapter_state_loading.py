from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.training.services.persistence import _load_tensor_state_dict


def test_load_tensor_state_dict_accepts_tensor_mapping(tmp_path: Path) -> None:
    path = tmp_path / "classifier.pth"
    torch.save({"weight": torch.ones(2, 2)}, path)
    state = _load_tensor_state_dict(path, device="cpu")
    assert torch.equal(state["weight"], torch.ones(2, 2))


def test_load_tensor_state_dict_rejects_non_tensor_values(tmp_path: Path) -> None:
    path = tmp_path / "classifier.pth"
    torch.save({"weight": "not-a-tensor"}, path)
    with pytest.raises(ValueError, match="non-tensor"):
        _load_tensor_state_dict(path, device="cpu")
