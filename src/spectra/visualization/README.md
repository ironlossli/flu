# Visualization Toolkit for Spectral Prediction

This folder contains scripts to visualize and analyze the behavior of the spectral prediction models.

## 1. Atomic Importance (Saliency Maps)
Visualizes which atoms contribute most to the prediction using gradient-based saliency.
Generates a `.pdb` file where the **B-factor** column contains the importance score (0-100).

**Usage:**
```bash
python vis_attention_3d.py \
  --ckpt ../../../checkpoints/your_model/best.pt \
  --xyz ../../../predict_data/molecule.xyz \
  --solvent "CCO" \
  --output result.pdb
```
**View:** Open `result.pdb` in PyMOL and run: `spectrum b, blue_white_red, minimum=0, maximum=100`.

## 2. Latent Space (t-SNE)
Extracts solute graph embeddings from a dataset and projects them to 2D using t-SNE. Points are colored by predicted wavelength.

**Usage:**
```bash
python vis_latent_space.py \
  --ckpt ../../../checkpoints/your_model/best.pt \
  --input_csv ../../../predict_data/test_set.csv \
  --output tsne.png
```

## 3. Uncertainty Analysis (Parity Plot)
Uses an **ensemble** of checkpoints (e.g., from cross-validation) to predict mean and standard deviation. Plots a parity plot with error bars.

**Usage:**
```bash
python vis_uncertainty.py \
  --ckpt_pattern "../../../checkpoints/fold_*/best.pt" \
  --input_csv ../../../predict_data/test_set.csv \
  --target_col "abs_max" \
  --output parity.png
```

## 4. E(3) Invariance Check
Verifies that the model's prediction does not change when the input molecule is rotated. Ideally, the deviation should be close to zero.

**Usage:**
```bash
python vis_e3_invariance.py \
  --ckpt ../../../checkpoints/your_model/best.pt \
  --xyz ../../../predict_data/molecule.xyz \
  --axis z
```
