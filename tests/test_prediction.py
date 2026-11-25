"""Regression test for the bundled RedoxPotNet model."""

from pathlib import Path

import numpy as np
import pandas as pd

from redoxpotnet import RedoxPotNet


def test_example_predictions_match_reference():
    root = Path(__file__).resolve().parents[1]
    predictor = RedoxPotNet()
    observed = predictor.predict_file(root / "examples" / "example_expression_TPM.tsv")
    expected = pd.read_csv(root / "examples" / "example_prediction.csv")
    assert observed["sample"].tolist() == expected["sample"].tolist()
    np.testing.assert_allclose(
        observed["RedoxPotNet_score"].to_numpy(),
        expected["RedoxPotNet_score"].to_numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
