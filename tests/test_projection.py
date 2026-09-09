import torch

from cgaf_tune import CGAFConfig, apply_cgaf_projection


def test_aligned_gradient_is_unchanged():
    domain = {"layer.0": [torch.tensor([1.0, 0.0])]}
    anchor = {"layer.0": [torch.tensor([2.0, 0.0])]}

    result, stats = apply_cgaf_projection(domain, anchor)

    assert torch.equal(result["layer.0"][0], domain["layer.0"][0])
    assert not stats["layer.0"].projected


def test_conflicting_component_is_reduced():
    domain = {"layer.0": [torch.tensor([-1.0, 1.0])]}
    anchor = {"layer.0": [torch.tensor([1.0, 0.0])]}

    result, stats = apply_cgaf_projection(
        domain, anchor, CGAFConfig(temperature=0.01)
    )

    assert result["layer.0"][0][0] > domain["layer.0"][0][0]
    assert stats["layer.0"].projected
    assert stats["layer.0"].gate > 0.99


def test_mismatched_groups_fail():
    try:
        apply_cgaf_projection({"a": [torch.ones(1)]}, {"b": [torch.ones(1)]})
    except ValueError:
        return
    raise AssertionError("expected ValueError")
