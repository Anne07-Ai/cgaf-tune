import pytest
import torch

from cgaf_tune.hf import resolve_dtype


def test_resolve_dtype():
    assert resolve_dtype("float32") is torch.float32
    with pytest.raises(ValueError, match="unsupported"):
        resolve_dtype("not_a_dtype")
