from cgaf_tune.reporting import markdown_results_table


def test_markdown_table_contains_intervals():
    metric = {"mean": 0.5, "ci_low": 0.4, "ci_high": 0.6, "examples": 20.0}
    table = markdown_results_table({
        "CGAF": {"overall": {"exact_match": metric, "token_f1": metric}}
    })
    assert "| CGAF | 0.500 [0.400, 0.600]" in table
    assert "| 20 |" in table
