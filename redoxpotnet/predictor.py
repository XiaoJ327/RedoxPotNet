"""High-level inference interface for the fixed RedoxPotNet model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .model import RedoxEncoder
from .preprocessing import calculate_gene_coverage, rank_selected_genes_by_fixed_background


class RedoxPotNet:
    """Load the released RedoxPotNet model and score expression profiles."""

    def __init__(self, model_dir: str | Path | None = None, device: str = "cpu") -> None:
        if model_dir is None:
            model_dir = Path(__file__).resolve().parent / "models" / "redoxpotnet_v1"
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)

        with open(self.model_dir / "model_config.json", encoding="utf-8") as handle:
            self.config = json.load(handle)

        self.redox_genes = pd.read_csv(self.model_dir / "redox_genes.csv")[
            "redox_gene"
        ].astype(str).tolist()
        self.background_genes = pd.read_csv(
            self.model_dir / "rank_background_genes.csv"
        )["rank_background_gene"].astype(str).tolist()

        layer_dims = self.config["architecture"]["layer_dims"]
        self.model = RedoxEncoder(
            input_dim=layer_dims[0],
            hidden_dims=tuple(layer_dims[1:-1]),
            dropout=float(self.config["architecture"]["dropout"]),
        )
        state_dict = torch.load(
            self.model_dir / "redoxpotnet_v1.pt",
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, expression: pd.DataFrame) -> pd.DataFrame:
        """Predict RedoxPotNet scores from a samples-by-genes expression matrix."""
        if not isinstance(expression, pd.DataFrame):
            raise TypeError("expression must be a pandas DataFrame.")
        if expression.empty:
            raise ValueError("expression matrix is empty.")

        expression = expression.copy()
        expression.columns = expression.columns.astype(str)
        coverage = calculate_gene_coverage(
            expression.columns,
            self.redox_genes,
            self.background_genes,
        )

        ranked = rank_selected_genes_by_fixed_background(
            expr_df=expression,
            selected_genes=self.redox_genes,
            background_genes=self.background_genes,
            scale_range=self.config["preprocessing"]["rank_scale_range"],
        )
        tensor = torch.as_tensor(
            ranked.to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )

        with torch.inference_mode():
            scores = self.model(tensor).detach().cpu().numpy().ravel()

        result = pd.DataFrame(
            {
                "sample": ranked.index.astype(str),
                "RedoxPotNet_score": scores,
            }
        )
        for key, value in coverage.items():
            result[key] = value
        return result

    def predict_file(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        sep: str | None = None,
    ) -> pd.DataFrame:
        """Score a genes-by-samples expression file.

        The first column must contain gene symbols. The remaining columns are samples.
        TSV and CSV files are supported.
        """
        input_path = Path(input_path)
        if sep is None:
            sep = "," if input_path.suffix.lower() == ".csv" else "\t"

        expression = pd.read_csv(input_path, sep=sep, index_col=0)
        expression.index = expression.index.astype(str)
        if expression.index.has_duplicates:
            duplicates = expression.index[expression.index.duplicated()].unique().tolist()
            preview = ", ".join(map(str, duplicates[:5]))
            raise ValueError(
                "Input file contains duplicate gene names. "
                f"Aggregate duplicates before prediction. Examples: {preview}"
            )

        result = self.predict(expression.T)
        if output_path is not None:
            result.to_csv(output_path, index=False)
        return result
