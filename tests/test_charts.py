from cgaf_tune.charts import write_pareto_svg


def test_pareto_svg_contains_labels(tmp_path):
    summary = {
        "lora": {"domain_gain": {"mean": 0.2}, "mean_forgetting": {"mean": 0.1}},
        "cgaf": {"domain_gain": {"mean": 0.19}, "mean_forgetting": {"mean": 0.03}},
    }
    path = write_pareto_svg(summary, tmp_path / "pareto.svg")
    content = path.read_text(encoding="utf-8")
    assert content.startswith("<svg")
    assert "cgaf" in content and "lora" in content
    assert "lower is better" in content
