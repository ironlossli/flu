
import sys
from pathlib import Path
import torch
import pandas as pd
import numpy as np

# Adjust path to find sibling modules
current_dir = Path(__file__).resolve().parent
src_root = current_dir.parent.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from predictor.core import SpectraPredictor
from spectra.data.build_dataset import DatasetBuilder

def load_predictor(checkpoint_path: str, device: str = "cuda"):
    """
    Load the SpectraPredictor from a checkpoint.
    """
    if not torch.cuda.is_available() and device == "cuda":
        print("Warning: CUDA not available, switching to CPU.")
        device = "cpu"
        
    predictor = SpectraPredictor.from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device
    )
    return predictor

def load_data(csv_path: str):
    """
    Load data from a CSV file. 
    Expected columns: 'xyz_path', 'solvent_smiles', 'y' (optional, true value)
    """
    df = pd.read_csv(csv_path)
    # Ensure xyz paths are absolute or correct relative to execution dir
    # This is a simple heuristic; might need adjustment based on how fix.py was used
    return df
