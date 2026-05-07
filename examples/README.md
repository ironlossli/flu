# Toy Example

This directory contains a minimal example for testing the prediction pipeline.

## Contents

- `input.csv` — example input with solute SMILES and solvent information
- `xyz_files/` — pre-computed XYZ geometries for the example molecules

## Usage

After training a model (see main README), run:

```bash
python scripts/run_prediction.py \
    --checkpoint checkpoints/train/best.pt \
    --input examples/input.csv \
    --output predictions.csv \
    --target abs \
    --device cpu
```

## Input Format

The input CSV must have these columns:
- `xyz_path` — absolute or relative path to XYZ file
- `solvent_smiles` — SMILES string of the solvent

Optional columns:
- `lambda_abs` / `lambda_em` — ground-truth values (for evaluation)
- `solvent_name` — human-readable solvent name (informational)

## Generating XYZ Files

If you have SMILES but no XYZ files, you can use RDKit to generate conformers:

```python
from rdkit import Chem
from rdkit.Chem import AllChem

mol = Chem.MolFromSmiles("c1ccccc1")
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, AllChem.ETKDG())
AllChem.MMFFOptimizeMolecule(mol)

# Write XYZ
Chem.MolToXYZFile(mol, "molecule.xyz")
```
