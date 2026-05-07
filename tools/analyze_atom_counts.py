import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from spectra.engine.preprocessing import parse_xyz

def analyze_atoms(parquet_path):
    print(f"Reading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # All samples
    all_paths = df['xyz_path'].tolist()
    
    # Unique samples
    unique_paths = df['xyz_path'].unique().tolist()
    
    print(f"Total samples: {len(all_paths)}")
    print(f"Unique solutes: {len(unique_paths)}")
    
    # Cache atom counts to avoid re-parsing for same file
    atom_counts_map = {}
    
    print("Parsing XYZ files...")
    for i, path in enumerate(unique_paths):
        try:
            symbols, _ = parse_xyz(path)
            atom_counts_map[path] = len(symbols)
        except Exception as e:
            print(f"Error parsing {path}: {e}")
            atom_counts_map[path] = 0 # Or skip
            
        if (i + 1) % 2000 == 0:
            print(f"Parsed {i + 1}/{len(unique_paths)} unique files...")

    # Statistics for Unique Solutes
    unique_counts = [atom_counts_map[p] for p in unique_paths if p in atom_counts_map and atom_counts_map[p] > 0]
    unique_counts = np.array(unique_counts)
    
    print("\n--- Atom Count Statistics (Unique Solutes) ---")
    print(f"Mean:   {np.mean(unique_counts):.2f}")
    print(f"Median: {np.median(unique_counts):.2f}")
    print(f"Min:    {np.min(unique_counts)}")
    print(f"Max:    {np.max(unique_counts)}")
    print(f"Std:    {np.std(unique_counts):.2f}")

    # Statistics for All Samples (Weighted by occurrence in dataset)
    all_counts = [atom_counts_map[p] for p in all_paths if p in atom_counts_map and atom_counts_map[p] > 0]
    all_counts = np.array(all_counts)
    
    print("\n--- Atom Count Statistics (All Samples / Dataset Distribution) ---")
    print(f"Mean:   {np.mean(all_counts):.2f}")
    print(f"Median: {np.median(all_counts):.2f}")
    print(f"Min:    {np.min(all_counts)}")
    print(f"Max:    {np.max(all_counts)}")
    print(f"Std:    {np.std(all_counts):.2f}")

if __name__ == "__main__":
    analyze_atoms('processed/flu/data_abs.parquet')
