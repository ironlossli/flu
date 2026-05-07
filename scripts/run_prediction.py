#!/usr/bin/env python
"""Inference script for predicting molecular spectra (Absorption/Emission).

Usage:
  python scripts/run_prediction.py --checkpoint <path> --input <csv> --output <csv> [--target abs|em] [--device cuda|cpu]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from predictor.core import SpectraPredictor


def main():
    parser = argparse.ArgumentParser(description="Predict molecular spectra from a trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV (columns: xyz_path, solvent_smiles)")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Path to output CSV")
    parser.add_argument("--target", type=str, default="abs", choices=["abs", "em"], help="Prediction target")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cuda or cpu")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    args = parser.parse_args()

    predictor = SpectraPredictor.from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=args.device,
        target=args.target,
    )

    results = predictor.predict_from_file(
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
    )

    print("\nPrediction complete!")
    print(results.head(10))
    success = (results["status"] == "success").sum()
    print(f"\nSuccess: {success} / {len(results)}")


if __name__ == "__main__":
    main()
