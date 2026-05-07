# Environment-Aware Multiscale Geometric Interaction for Equivariant Molecular Spectral Prediction

This repository contains the official implementation of the paper:

> **Environment-Aware Multiscale Geometric Interaction for Equivariant Molecular Spectral Prediction**
> Haoran Li, Weiran Cui, Minghui Li
> University of Chinese Academy of Sciences
> IJCAI-ECAI 2026

## Overview

We propose a framework for predicting molecular spectra (absorption and emission peak wavelengths) in solution that explicitly models both **global 3D geometry** and **solvent environment effects**.

### Method Overview

**MGIL (Multiscale Geometric Interaction Layer)** enriches equivariant message passing with:

- **Centroid-referenced global anchors** — provide a molecule-level spatial reference to help distinguish globally distinct conformations
- **Moment invariants** — compact, permutation-agnostic local shape statistics computed from neighbor directions
- **Virtual node** — an efficient global communication pathway with linear complexity
- **Radial basis functions (RBF)** — standard distance-based edge features

These components are concatenated into a unified edge feature vector and injected into the backbone's message functions, all while preserving SE(3) invariance and O(|E|) complexity.

**SFM (Solvent Field Modulator)** encodes solvent topology and modulates solute features via:

- **Solvent graph encoder** — a lightweight 2D MPNN producing solvent embeddings from SMILES
- **Cross-attention** — solute atoms query solvent atoms to obtain atom-specific solvent contexts
- **FiLM-style gating** — scalar feature-wise linear modulation from solvent context

### Supported Backbones

- EGNN, SchNet, PaiNN, LEFTNet, LEFTNet V3, Equiformer V1, Equiformer V2, C-GNN

### Tasks

- **Absorption (Abs):** predict peak absorption wavelength λ_max (nm)
- **Emission (EM):** predict peak emission wavelength λ_max (nm)

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU training)

### Setup

```bash
# Create conda environment
conda create -n flu python=3.10 -y
conda activate flu

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio

# Install PyG and related packages
pip install torch_geometric torch_scatter

# Install core dependencies
pip install -r requirements.txt

# Optional: for Equiformer V2 backbone
pip install fairchem-core e3nn
```

## Data Preparation

The model expects data in the following format:

1. A CSV file with columns: `SMILES`, `Solvent`, `NUM` (optional), `Absorption_max_nm`, `Emission_max_nm`
2. XYZ files for 3D solute geometries in a directory

To build the processed dataset:

```bash
python src/spectra/data/build_dataset.py --config configs/data.yaml
```

This generates parquet files and train/val/test splits under `processed/`.

Configuration is in `configs/data.yaml` — adjust paths and column names as needed.

## Training

Train an absorption prediction model with EGNN backbone + MGIL + SFM:

```bash
bash scripts/train_absorption.sh
```

Train an emission prediction model:

```bash
bash scripts/train_emission.sh
```

**Customize the run with environment variables:**

```bash
MODEL_CFG=configs/model/ehc_painn.yaml BATCH_SIZE=64 SEED=42 bash scripts/train_absorption.sh
```

**Available model configs** (under `configs/model/`):

| Config | Description |
|--------|-------------|
| `benchmark_vegnn.yaml` | EGNN + full MGIL + SFM |
| `ehc_egnn_cross_attn.yaml` | EGNN + MGIL + SFM (cross-attention variant) |
| `ehc_egnn_baseline.yaml` | EGNN baseline (concatenation, no cross-attn) |
| `ehc_painn.yaml` | PaiNN + MGIL + SFM |
| `ehc_schnet.yaml` | SchNet + MGIL + SFM |
| `ehc_equiformer.yaml` | Equiformer V1 + MGIL + SFM |
| `ehc_equiformer_v2.yaml` | Equiformer V2 + MGIL + SFM |
| `ehc_leftnet.yaml` | LEFTNet + MGIL + SFM |
| `ablation_v1_no_vector.yaml` | EGNN w/o vector features |
| `ablation_v2_no_global.yaml` | EGNN w/o global anchors |
| `ablation_v3_no_moments.yaml` | EGNN w/o moment invariants |
| `ablation_v4_no_vnode.yaml` | EGNN w/o virtual node |

## Evaluation

```bash
bash scripts/evaluate.sh --checkpoint checkpoints/train/best.pt --target abs --split test
```

## Inference

```bash
python scripts/run_prediction.py \
    --checkpoint checkpoints/train/best.pt \
    --input examples/toy_example/input.csv \
    --output predictions.csv \
    --target abs \
    --device cpu
```

## Reproducing Paper Results

See [docs/reproduce_results.md](docs/reproduce_results.md) for detailed instructions on reproducing each table.

Quick start:

```bash
bash scripts/reproduce_tables.sh
```

## Project Structure

```
├── src/
│   ├── spectra/              # Core library
│   │   ├── models/
│   │   │   ├── backbones/    # Backbone implementations (EGNN, PaiNN, SchNet, etc.)
│   │   │   └── ehc/          # MGIL + SFM adapters, solvent encoder, GBM conditioner
│   │   ├── engine/           # Training loop, data loading, preprocessing
│   │   ├── data/             # Dataset building
│   │   ├── utils/            # I/O utilities
│   │   └── visualization/    # Attention maps, latent space, invariance plots
│   └── predictor/            # Inference interface
├── configs/
│   ├── data.yaml             # Data pipeline configuration
│   ├── train.yaml            # Training configuration
│   └── model/                # Model-specific hyperparameters
├── scripts/                  # Entry-point shell and Python scripts
├── tools/                    # Utility scripts (summarize, analyze, verify)
├── tests/                    # Verification scripts (equivariance, forward pass)
├── examples/                 # Toy examples
└── docs/                     # Extended documentation
```

## Toy Example

See [examples/toy_example/](examples/toy_example/) for a minimal working example with synthetic data.

## Citation

```bibtex
@inproceedings{li2026environment,
  title     = {Environment-Aware Multiscale Geometric Interaction for
               Equivariant Molecular Spectral Prediction},
  author    = {Li, Haoran and Cui, Weiran and Li, Minghui},
  booktitle = {Proceedings of the Thirty-Fifth International Joint Conference
               on Artificial Intelligence (IJCAI-ECAI)},
  year      = {2026},
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

Third-party code notices:
- `src/spectra/models/backbones/equiformer_v2/` contains code adapted from FAIR's EquiformerV2 (CC-BY-NC)
- `src/spectra/models/schnetpack/` contains minimal vendored utilities from schNetPack (MIT)

## Contact

For questions about the paper or code, please open an issue on this repository.
