import pytest
import torch
from torch import nn

from cgaf_tune import CGAFConfig, build_step_engine


def make_engine(method):
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    engine = build_step_engine(
        method, model, optimizer, CGAFConfig(), "global", 1.0, anchor_weight=0.5
    )
    return model, engine


@pytest.mark.parametrize("method", ["lora", "rehearsal", "hard_projection", "cgaf"])
def test_each_method_executes_one_update(method):
    model, engine = make_engine(method)
    before = model.weight.detach().clone()
    batch = torch.ones(1, 1)
    result = engine.step(
        lambda: (model(batch) - 1).square().mean(),
        lambda: (model(batch) + 1).square().mean(),
    )
    assert result.method == method
    if method == "hard_projection":
        assert result.projected_groups == 1
        assert torch.allclose(model.weight, before, atol=1e-6, rtol=0)
    else:
        assert not torch.equal(model.weight, before)


def test_unknown_method_is_rejected():
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="unsupported"):
        build_step_engine("unknown", model, optimizer, CGAFConfig(), "global", None)
