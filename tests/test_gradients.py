import pytest
import torch
from torch import nn

from cgaf_tune.gradients import capture_gradients, group_name, parameter_groups, restore_gradients


def test_peft_style_names_group_by_layer():
    assert group_name("base.model.layers.7.self_attn.q_proj.lora_A.default.weight") == "layer.7"
    assert group_name("adapter.weight") == "unscoped"
    assert group_name("anything", "global") == "global"


def test_capture_and_restore_round_trip():
    model = nn.Linear(2, 1, bias=False)
    groups = parameter_groups(model.named_parameters(), "global")
    model(torch.ones(1, 2)).sum().backward()
    captured = capture_gradients(groups)
    model.zero_grad(set_to_none=True)
    restore_gradients(groups, captured)
    assert torch.equal(model.weight.grad, captured["global"][0])


def test_capture_rejects_missing_gradient():
    model = nn.Linear(2, 1, bias=False)
    groups = parameter_groups(model.named_parameters(), "global")
    with pytest.raises(RuntimeError, match="missing gradient"):
        capture_gradients(groups)
