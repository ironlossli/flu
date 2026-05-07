
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm
from glob import glob

from predictor.core import SpectraPredictor
from utils import load_data

def run_ensemble_inference(checkpoints, df, device="cuda"):
    """
    Run inference with multiple models and return mean/std.
    """
    all_preds = []
    
    print(f"Running inference with {len(checkpoints)} models...")
    
    for ckpt in checkpoints:
        predictor = SpectraPredictor.from_checkpoint(ckpt, device=device)
        model_preds = []
        
        # Batch inference is faster, but let's stick to simple loop for clarity in visualization script
        # Or use predict_batch if available.
        try:
            # Prepare lists for batch prediction
            xyz_list = df['xyz_path'].tolist()
            solv_list = df['solvent_smiles'].tolist()
            
            res_df = predictor.predict_batch(xyz_list, solv_list, show_progress=True)
            # res_df has ['prediction', 'status']
            
            # extract predictions, map failures to NaN
            preds = pd.to_numeric(res_df['prediction'], errors='coerce').values
            model_preds = preds
            
        except Exception as e:
            print(f"Error with checkpoint {ckpt}: {e}")
            model_preds = np.full(len(df), np.nan)
            
        all_preds.append(model_preds)
        
    all_preds = np.array(all_preds) # Shape: (n_models, n_samples)
    
    # Compute stats ignoring NaNs
    mean_preds = np.nanmean(all_preds, axis=0)
    std_preds = np.nanstd(all_preds, axis=0)
    
    return mean_preds, std_preds

def plot_parity(y_true, y_pred, y_std, output_path):
    # Filter NaNs
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    y_std = y_std[mask]
    
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    plt.figure(figsize=(8, 8))
    
    # Error bars for uncertainty
    plt.errorbar(y_true, y_pred, yerr=y_std, fmt='o', ecolor='gray', 
                 mec='blue', mfc='blue', alpha=0.5, capsize=2, label='Predictions')
    
    # Reference line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal')
    
    plt.title(f"Parity Plot (MAE={mae:.2f}, R2={r2:.2f})")
    plt.xlabel("Ground Truth (nm)")
    plt.ylabel("Predicted (nm)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(output_path, dpi=300)
    print(f"Saved parity plot to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize Uncertainty (Parity Plot)")
    parser.add_argument("--ckpt_pattern", type=str, required=True, help="Glob pattern for checkpoints (e.g. 'checkpoints/*/best.pt')")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--target_col", type=str, default="y", help="Column name for ground truth")
    parser.add_argument("--output", type=str, default="parity_plot.png", help="Output image path")
    args = parser.parse_args()

    # Find checkpoints
    checkpoints = glob(args.ckpt_pattern)
    if not checkpoints:
        print(f"No checkpoints found matching {args.ckpt_pattern}")
        return
    
    df = load_data(args.input_csv)
    if args.target_col not in df.columns:
        print(f"Error: Target column '{args.target_col}' not found in CSV.")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mean_preds, std_preds = run_ensemble_inference(checkpoints, df, device)
    
    y_true = pd.to_numeric(df[args.target_col], errors='coerce').values
    
    plot_parity(y_true, mean_preds, std_preds, args.output)

if __name__ == "__main__":
    main()
