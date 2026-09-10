import torch
from torch import nn

from cgaf_tune import CGAFConfig, CGAFStepEngine


def test_engine_updates_parameters_and_reports_metrics():
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.2, -0.1]]))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    engine = CGAFStepEngine(
        model, optimizer, CGAFConfig(temperature=0.05), grouping="global"
    )
    before = model.weight.detach().clone()

    result = engine.step(
        lambda: (model(torch.tensor([[1.0, 0.0]])) - 1.0).square().mean(),
        lambda: (model(torch.tensor([[1.0, 0.0]])) + 1.0).square().mean(),
    )

    assert not torch.equal(model.weight, before)
    assert result.projected_groups == 1
    assert result.groups["global"].cosine < 0
    assert result.mean_gate > 0.5


def test_engine_leaves_aligned_update_unprojected():
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    engine = CGAFStepEngine(model, optimizer, grouping="global")
    batch = torch.tensor([[1.0]])
    result = engine.step(
        lambda: model(batch).square().mean(),
        lambda: model(batch).square().mean(),
    )
    assert result.projected_groups == 0
