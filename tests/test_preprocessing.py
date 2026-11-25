"""Tests for RedoxPotNet rank preprocessing."""

import pandas as pd

from redoxpotnet.preprocessing import rank_selected_genes_by_fixed_background


def test_rank_scaling_zero_to_one():
    expression = pd.DataFrame(
        [[1.0, 2.0, 3.0]],
        index=["sample_1"],
        columns=["A", "B", "C"],
    )
    ranked = rank_selected_genes_by_fixed_background(
        expression,
        selected_genes=["A", "B", "C"],
        background_genes=["A", "B", "C"],
        scale_range="0_1",
    )
    assert ranked.loc["sample_1", "A"] == 0.0
    assert ranked.loc["sample_1", "B"] == 0.5
    assert ranked.loc["sample_1", "C"] == 1.0
