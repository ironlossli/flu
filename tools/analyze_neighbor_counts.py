import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from spectra.engine.preprocessing import parse_xyz, build_edges_by_cutoff

def analyze_neighbors(parquet_path, cutoff=5.0):
    print(f"Reading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Unique XYZ files
    unique_xyz_paths = df['xyz_path'].unique()
    print(f"Found {len(unique_xyz_paths)} unique molecules.")
    
    avg_degrees = []
    
    for i, xyz_path in enumerate(unique_xyz_paths):
        try:
            symbols, pos = parse_xyz(xyz_path)
            num_nodes = len(symbols)
            
            edge_index, _ = build_edges_by_cutoff(pos, cutoff)
            num_edges = edge_index.shape[1]
            
            # Average degree = num_edges / num_nodes
            # Note: edge_index includes both i->j and j->i, so this is the sum of out-degrees
            # effectively 2 * num_pairs / num_nodes
            avg_deg = num_edges / num_nodes if num_nodes > 0 else 0
            avg_degrees.append(avg_deg)
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(unique_xyz_paths)}...")
                
        except Exception as e:
            print(f"Error processing {xyz_path}: {e}")
            
    avg_degrees = np.array(avg_degrees)
    
    print("\n--- Neighbor Analysis (Cutoff = 5.0 Angstrom) ---")
    print(f"Number of molecules analyzed: {len(avg_degrees)}")
    print(f"Mean avg neighbors:   {np.mean(avg_degrees):.2f}")
    print(f"Median avg neighbors: {np.median(avg_degrees):.2f}")
    print(f"Min avg neighbors:    {np.min(avg_degrees):.2f}")
    print(f"Max avg neighbors:    {np.max(avg_degrees):.2f}")
    print(f"Std Dev:              {np.std(avg_degrees):.2f}")

if __name__ == "__main__":
    analyze_neighbors('processed/flu/data_abs.parquet', cutoff=5.0)
