"""Preprocessing utilities for RedoxPotNet inference."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def rank_selected_genes_by_fixed_background(
    expr_df: pd.DataFrame,
    selected_genes: Sequence[str],
    background_genes: Sequence[str],
    scale_range: str = "0_1",
) -> pd.DataFrame:
    """Convert expression values to within-sample ranks.

    Parameters
    ----------
    expr_df:
        Expression matrix with samples as rows and genes as columns.
    selected_genes:
        Ordered genes expected by the pretrained RedoxPotNet encoder.
    background_genes:
        Fixed gene universe used for within-sample ranking.
    scale_range:
        Either ``"0_1"`` or ``"-1_1"``.

    Returns
    -------
    pandas.DataFrame
        Matrix with samples as rows and selected RedoxPotNet genes as columns.
        Missing selected genes receive the neutral rank value.
    """
    selected_genes = list(selected_genes)
    background_genes = list(background_genes)

    if expr_df.columns.has_duplicates:
        duplicated = expr_df.columns[expr_df.columns.duplicated()].unique().tolist()
        preview = ", ".join(map(str, duplicated[:5]))
        raise ValueError(
            "Expression matrix contains duplicate gene names. "
            f"Aggregate duplicates before prediction. Examples: {preview}"
        )

    matched_background_genes = [
        gene for gene in background_genes if gene in expr_df.columns
    ]
    matched_selected_genes = [
        gene for gene in selected_genes if gene in expr_df.columns
    ]

    if len(matched_background_genes) < 2:
        raise ValueError(
            "Fewer than two fixed background genes were found in the expression matrix."
        )
    if not matched_selected_genes:
        raise ValueError(
            "None of the RedoxPotNet input genes were found in the expression matrix."
        )

    if scale_range == "0_1":
        fill_value = 0.5
    elif scale_range == "-1_1":
        fill_value = 0.0
    else:
        raise ValueError("scale_range must be either '0_1' or '-1_1'.")

    expression_background = expr_df.loc[:, matched_background_genes].copy()
    rank_background = expression_background.rank(
        axis=1,
        method="average",
        ascending=True,
        na_option="keep",
    )

    genes_per_sample = expression_background.notna().sum(axis=1).astype(float)
    denominator = (genes_per_sample - 1).replace(0, np.nan)

    rank_background = rank_background.sub(1, axis=0)
    rank_background = rank_background.div(denominator, axis=0)

    if scale_range == "-1_1":
        rank_background = rank_background * 2 - 1

    rank_background = rank_background.fillna(fill_value)

    selected_rank = pd.DataFrame(
        fill_value,
        index=expr_df.index,
        columns=selected_genes,
        dtype=float,
    )

    available_selected_genes = [
        gene for gene in selected_genes if gene in rank_background.columns
    ]
    selected_rank.loc[:, available_selected_genes] = rank_background.loc[
        :, available_selected_genes
    ].to_numpy()

    return selected_rank


def calculate_gene_coverage(
    genes_present: Sequence[str],
    selected_genes: Sequence[str],
    background_genes: Sequence[str],
) -> dict[str, float | int]:
    """Return model-gene and background-gene coverage statistics."""
    present = set(genes_present)
    n_redox = sum(gene in present for gene in selected_genes)
    n_background = sum(gene in present for gene in background_genes)
    return {
        "n_redox_genes_found": n_redox,
        "n_redox_genes_total": len(selected_genes),
        "redox_gene_coverage": n_redox / len(selected_genes),
        "n_background_genes_found": n_background,
        "n_background_genes_total": len(background_genes),
        "background_gene_coverage": n_background / len(background_genes),
    }
