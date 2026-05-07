
import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

from utils import load_predictor

def compute_solvent_saliency(predictor, xyz_path, solvent_smiles):
    """
    Compute gradient-based saliency for the solvent.
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
    
    # 2. Build batch
    from datamodule import GraphBatch, GraphData
    batch = GraphBatch([GraphData(**sample)], target=predictor.target)
    batch = batch.to(predictor.device)
    
    # 3. Enable gradients for solvent features
    # Check if solvent_x is float
    if not batch.solvent_x.is_floating_point():
        print(f"Warning: solvent_x is {batch.solvent_x.dtype}, cannot calculate input gradients directly.")
        return None, None

    # Avoid in-place errors
    leaf_solvent_x = batch.solvent_x.detach().clone().to(predictor.device)
    leaf_solvent_x.requires_grad = True
    batch.solvent_x = leaf_solvent_x * 1.0
    
    # 4. Forward
    output = predictor.model(batch)
    pred = predictor._extract_pred(output)
    
    # 5. Backward
    pred.backward()
    
    if leaf_solvent_x.grad is None:
        print("Warning: No gradient for solvent_x.")
        return None, None
        
    # 6. Compute importance
    grads = leaf_solvent_x.grad.detach().cpu().numpy() # [Ns, F]
    importance = np.linalg.norm(grads, axis=1) # [Ns]
    
    # Normalize
    if importance.max() > importance.min():
        importance = (importance - importance.min()) / (importance.max() - importance.min())
    else:
        importance = np.zeros_like(importance)
        
    return importance, solvent_smiles

def draw_solvent_importance(smiles, weights, output_path):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        print(f"Error parsing SMILES: {smiles}")
        return

    # Ensure atom count matches
    if len(weights) != mol.GetNumAtoms():
        # Sometimes implicit Hydrogens handling differs.
        # featurizer might usually include/exclude Hs.
        # Let's try adding Hs if counts don't match.
        mol_h = Chem.AddHs(mol)
        if len(weights) == mol_h.GetNumAtoms():
            mol = mol_h
        else:
            print(f"Atom count mismatch! Weights: {len(weights)}, Mol: {mol.GetNumAtoms()} (Hs added: {mol_h.GetNumAtoms()})")
            return

    # Create color map (Blue -> White -> Red)
    # Using 'coolwarm' or 'bwr' is good. 
    try:
        colormap = plt.get_cmap('coolwarm')
    except:
        colormap = cm.get_cmap('coolwarm')
    
    # Initialize containers
    highlight_atom_colors = {}
    highlight_atoms = []
    
    # Pre-compute colors
    for i, w in enumerate(weights):
        # w is 0-1.
        color = colormap(w)[:3] # RGB tuple (0-1)
        highlight_atom_colors[i] = color
        highlight_atoms.append(i)

    # Draw with improved aesthetics
    output_path = Path(str(output_path))
    is_svg = output_path.suffix.lower() == ".svg"
    if is_svg:
        drawer = rdMolDraw2D.MolDraw2DSVG(600, 600)
    else:
        drawer = rdMolDraw2D.MolDraw2DCairo(600, 600) # Higher resolution
    opts = drawer.drawOptions()
    opts.padding = 0.1
    opts.addStereoAnnotation = True
    opts.bondLineWidth = 2
    opts.highlightBondWidthMultiplier = 1 # Must be int
    opts.scaleBondWidth = True
    
    # Make the base molecule cleaner
    opts.useBWAtomPalette() # Black and white atoms to let highlights pop
    
    # Draw
    drawer.DrawMolecule(
        mol, 
        highlightAtoms=highlight_atoms, 
        highlightAtomColors=highlight_atom_colors,
        highlightBonds=[], # Optional: don't highlight bonds, just atoms
    )
    drawer.FinishDrawing()
    if is_svg:
        svg = drawer.GetDrawingText()
        output_path.write_text(svg)
    else:
        drawer.WriteDrawingText(str(output_path))
    print(f"Saved solvent importance visualization to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize Solvent Importance (2D)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--xyz", type=str, required=True, help="Path to input XYZ file (solute context)")
    parser.add_argument("--solvent", type=str, required=True, help="Solvent SMILES")
    parser.add_argument("--output", type=str, default="solvent_vis.png", help="Output PNG file path")
    args = parser.parse_args()

    predictor = load_predictor(args.ckpt)
    
    try:
        weights, smiles = compute_solvent_saliency(predictor, args.xyz, args.solvent)
        if weights is not None:
            print(f"Solvent Weights Stats - Min: {weights.min():.2f}, Max: {weights.max():.2f}, Mean: {weights.mean():.2f}")
            draw_solvent_importance(smiles, weights, args.output)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
