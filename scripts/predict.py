#!/usr/bin/env python3
"""Command-line interface for RedoxPotNet prediction."""

from __future__ import annotations

import argparse

from redoxpotnet import RedoxPotNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict RedoxPotNet scores from a genes-by-samples expression matrix."
    )
    parser.add_argument("--input", required=True, help="Input TSV or CSV expression matrix.")
    parser.add_argument("--output", required=True, help="Output CSV file.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device, for example 'cpu' or 'cuda'. Default: cpu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = RedoxPotNet(device=args.device)
    result = predictor.predict_file(args.input, args.output)
    print(f"Predicted {len(result)} samples. Results written to {args.output}")


if __name__ == "__main__":
    main()
