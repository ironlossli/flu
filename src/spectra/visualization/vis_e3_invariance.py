
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from utils import load_predictor

# datamodule must be imported after utils updates sys.path
from datamodule import GraphBatch, GraphData

def rotate_coordinates(pos_numpy, angle_deg, axis='z'):
    """
    Rotate coordinates by angle_deg around specific axis.
    """
    axis_vec = {'x': [1,0,0], 'y': [0,1,0], 'z': [0,0,1]}[axis]
    r = R.from_rotvec(np.radians(angle_deg) * np.array(axis_vec))
    return r.apply(pos_numpy)

def check_invariance(predictor, xyz_path, solvent_smiles, axis='z', steps=36):
    """
    Rotate molecule and predict.
    """
    record = {
        "xyz_path": xyz_path,
        "solvent_smiles": solvent_smiles,
        "lambda_abs": 0.0,
        "lambda_em": 0.0,
    }
    
    # Base featurization
    base_sample = predictor.featurizer.featurize_record(record)
    if base_sample is None:
        raise ValueError("Featurization failed.")
    
    # Identify the correct position key
    pos_key = 'solute_pos' if 'solute_pos' in base_sample else 'pos'
    if pos_key not in base_sample:
        raise KeyError(f"Could not find position key (expected 'solute_pos' or 'pos') in sample keys: {base_sample.keys()}")

    original_pos = base_sample[pos_key].numpy()
    
    angles = np.linspace(0, 360, steps)
    predictions = []
    
    print(f"Testing rotation invariance around {axis}-axis using key '{pos_key}'...")
    
    for angle in angles:
        # Rotate
        rotated_pos = rotate_coordinates(original_pos, angle, axis)
        
        # Create new sample dict with rotated positions
        sample = base_sample.copy()
        sample[pos_key] = torch.tensor(rotated_pos, dtype=torch.float32)
        
        # Inference
        batch = GraphBatch([GraphData(**sample)], target=predictor.target)
        batch = batch.to(predictor.device)
        
        with torch.no_grad():
            output = predictor.model(batch)
            pred = predictor._extract_pred(output).squeeze().cpu().item()
            predictions.append(pred)
            
    return angles, np.array(predictions)

def plot_invariance(angles, preds, output_path):
    # Calculate deviation from mean
    mean_val = preds.mean()
    deviations = preds - mean_val
    
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(angles, preds, 'b.-')
    plt.title(f"E(3) Invariance Check\nMean Prediction: {mean_val:.4f} nm")
    plt.ylabel("Predicted Wavelength (nm)")
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(angles, deviations, 'r.-')
    plt.title(f"Deviation from Mean (Max Dev: {np.max(np.abs(deviations)):.2e} nm)")
    plt.xlabel("Rotation Angle (degrees)")
    plt.ylabel("Deviation (nm)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved invariance plot to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Verify E(3) Invariance")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--xyz", type=str, required=True, help="Path to input XYZ file")
    parser.add_argument("--solvent", type=str, default="O", help="Solvent SMILES (default: Water)")
    parser.add_argument("--axis", type=str, default="z", choices=['x', 'y', 'z'], help="Rotation axis")
    parser.add_argument("--output", type=str, default="invariance_check.png", help="Output plot path")
    args = parser.parse_args()

    predictor = load_predictor(args.ckpt)
    
    try:
        angles, preds = check_invariance(predictor, args.xyz, args.solvent, args.axis)
        plot_invariance(angles, preds, args.output)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
