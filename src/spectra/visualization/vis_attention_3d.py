import argparse
import torch
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

from utils import load_predictor, load_data

def compute_saliency(predictor, xyz_path, solvent_smiles):
    """
    Compute gradient-based saliency for the input molecule.
    Returns:
        atom_weights: (N,) numpy array of importance scores (L2 norm of gradient).
        coords: (N, 3) numpy array of atomic coordinates.
        atomic_nums: (N,) numpy array of atomic numbers.
    """
    predictor.model.eval()
    
    # 1. Featurize
    record = {
        "xyz_path": xyz_path,
        "solvent_smiles": solvent_smiles,
        "lambda_abs": 0.0,
        "lambda_em": 0.0,
    }
    sample = predictor.featurizer.featurize_record(record)
    if sample is None:
        raise ValueError("Featurization failed.")
    
    # 2. Build batch & Enable gradients
    from datamodule import GraphBatch, GraphData
    batch = GraphBatch([GraphData(**sample)], target=predictor.target)
    batch = batch.to(predictor.device)
    
    # We need gradients w.r.t coordinates. 
    # GraphBatch uses 'solute_pos' for solute coordinates.
    # To avoid "leaf Variable was used in an in-place operation" errors,
    # we create a leaf variable, but pass a non-leaf (operation result) to the model.
    leaf_pos = batch.solute_pos.detach().clone().to(predictor.device)
    leaf_pos.requires_grad = True
    
    # Pass a non-leaf tensor to the model. 
    # Gradients will flow: Loss -> batch.solute_pos -> leaf_pos
    batch.solute_pos = leaf_pos * 1.0
    
    # 3. Forward pass
    output = predictor.model(batch)
    
    # Use internal helper to find the prediction tensor
    try:
        pred = predictor._extract_pred(output)
    except Exception as e:
        print(f"Keys in output: {output.keys() if isinstance(output, dict) else 'Not a dict'}")
        raise e
    
    # 4. Backward pass
    pred.backward()
    
    # 5. Extract gradients
    # Gradient shape: (N, 3). We take L2 norm per atom to get a scalar importance.
    if leaf_pos.grad is None:
        print("Warning: Gradients are None! Backward pass failed to populate grads.")
        return np.zeros(len(leaf_pos)), leaf_pos.detach().cpu().numpy()

    grads = leaf_pos.grad.detach().cpu().numpy()
    importance = np.linalg.norm(grads, axis=1)
    
    print(f"\n[Debug] Gradient Stats:")
    print(f"  Shape: {grads.shape}")
    print(f"  Raw L2 norms - Min: {importance.min():.2e}, Max: {importance.max():.2e}, Mean: {importance.mean():.2e}, Std: {importance.std():.2e}")

    # Normalize importance (0-1) for better visualization
    # Use Min-Max normalization
    min_val, max_val = importance.min(), importance.max()
    if max_val - min_val > 1e-8:
        importance = (importance - min_val) / (max_val - min_val)
    else:
        print("Warning: Gradient range is too small, weights will be zero.")
        importance = np.zeros_like(importance)
    
    print(f"  Normalized - Min: {importance.min():.2f}, Max: {importance.max():.2f}")
    
    # Extract coordinates. For atomic numbers, we'll read from the xyz file directly 
    # in the main function to be safe, as GraphBatch might transformed features.
    return importance, leaf_pos.detach().cpu().numpy()

def save_to_pdb(output_path, coords, xyz_path, weights):
    """
    Save molecule to PDB format with weights in the B-factor column.
    Reads atomic numbers from the original XYZ file.
    """
    # Read molecule to get atomic numbers and atom symbols
    atomic_info = []
    with open(xyz_path, 'r') as f:
        lines = f.readlines()
        num_atoms = int(lines[0].strip())
        comment = lines[1].strip()
        
        if num_atoms != len(coords):
            print(f"Warning: Atom count mismatch! XYZ file: {num_atoms}, Batch: {len(coords)}. Proceeding with batch atom count.")
            # This might happen if the featurizer filters atoms (e.g. Hydrogens). 
            # We will use the number of atoms from the 'coords' array.
            num_atoms = len(coords)

        for i in range(num_atoms):
            parts = lines[i + 2].split()
            if not parts: continue
            sym = parts[0]
            atomic_info.append({
                'symbol': sym,
                'atomic_num': Chem.GetPeriodicTable().GetAtomicNumber(sym),
                'atom_name': sym.upper() + ('' if len(sym) >= 2 else '  ') # Basic atom naming
            })

    # Fallback if atomic_info is shorter than coords (e.g., if featurizer removed some atoms)
    if len(atomic_info) < len(coords):
        print("Warning: XYZ file has fewer atoms than processed coordinates. Adjusting atomic info.")
        # This is a crude fallback. A proper solution would map indices.
        # For now, just extend with dummy C atoms or use a default.
        while len(atomic_info) < len(coords):
            atomic_info.append({
                'symbol': 'C',
                'atomic_num': 6,
                'atom_name': 'C   '
            })


    print(f"\n[Debug] Weights for PDB - Min: {weights.min():.2e}, Max: {weights.max():.2e}, Mean: {weights.mean():.2e}, Std: {weights.std():.2e}")

    output_lines = []
    # PDB header (minimal)
    output_lines.append("MODEL        1")
    output_lines.append("REMARK   importance values stored in B-factor column")

    for i in range(len(coords)):
        z = atomic_info[i]['atomic_num']
        sym = atomic_info[i]['symbol']
        atom_name = atomic_info[i]['atom_name']
        pos = coords[i]
        w = weights[i]

        # PDB format string for HETATM (simplified):
        # ATOM      1  N   MET A   1      29.982  17.040  -6.305  1.00 20.00           N
        # HETATM  atm_idx atm_name res_name chain_id res_id x_coord y_coord z_coord occ B-factor element
        # Atom index (right justified, width 5)
        # Atom name (left justified, width 4)
        # Residue name (left justified, width 3) - UNL for unliganded
        # Chain ID (width 1)
        # Residue sequence number (right justified, width 4)
        # x, y, z coordinates (right justified, width 8, 3 decimal places)
        # Occupancy (width 6, 2 decimal places) - always 1.00 for us
        # B-factor (width 6, 2 decimal places) - our importance score, 0.00-100.00
        # Element symbol (right justified, width 2)

        # Scale weights to 0-100 and clamp to 0-99.99 for 6.2 format
        bfactor = min(max(w * 100, 0.00), 99.99) 

        atom_line = (
            f"HETATM{i+1:5d} {atom_name: <4} UNL A   1    " # ATOM_ID, ATOM_NAME, RES_NAME, CHAIN_ID, RES_SEQ
            f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}" # X, Y, Z
            f"{1.00:6.2f}{bfactor:6.2f}          {sym:>2}" # OCC, B-FACTOR, ELEMENT
        )
        output_lines.append(atom_line)

    output_lines.append("ENDMDL")
    output_lines.append("END")

    with open(output_path, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"Saved PDB with importance factors to {output_path}")

import traceback

def main():
    parser = argparse.ArgumentParser(description="Visualize Atomic Importance (Saliency Map)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--xyz", type=str, required=True, help="Path to input XYZ file")
    parser.add_argument("--solvent", type=str, required=True, help="Solvent SMILES")
    parser.add_argument("--output", type=str, default="saliency.pdb", help="Output PDB file path")
    args = parser.parse_args()

    predictor = load_predictor(args.ckpt)
    
    try:
        weights, coords = compute_saliency(predictor, args.xyz, args.solvent)
        save_to_pdb(args.output, coords, args.xyz, weights)
        print("\nVisualization Tip: Open the generated PDB in PyMOL/VMD.")
        print("In PyMOL: 'spectrum b, blue_white_red, minimum=0, maximum=100'")
    except Exception as e:
        traceback.print_exc()
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
