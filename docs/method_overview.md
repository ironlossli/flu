# Method Overview

## Problem Setting

Given a solute molecule (3D geometry + atom types) and a solvent molecule (SMILES), predict the peak absorption wavelength (λ_abs) or emission wavelength (λ_em) in nanometers.

The prediction must be SE(3)-invariant: rotating or translating the solute should not change the prediction.

## Architecture

```
                     ┌──────────────┐
   Solute (XYZ) ───▶ │   Backbone   │──▶ Node features ──▶ Pool ──▶ MLP ──▶ λ_pred
                     │   (EGNN/     │
   Solvent ─────────▶│   PaiNN/...) │
   (SMILES)     ┌───▶│              │
                │    └──────────────┘
                │
         ┌──────┴──────┐
         │  Solvent     │
         │  Encoder     │──▶ z_s (global embedding)
         │  (MPNN)      │
         └──────┬──────┘
                │
         ┌──────┴──────┐
         │  Cross-Attn  │──▶ z_s,i (per-atom solvent context)
         │  (Q: solute, │
         │   K,V: solv) │
         └──────┬──────┘
                │
                ▼
         ┌──────────────┐
         │  FiLM Gate   │──▶ modulate backbone features per layer
         └──────────────┘
```

## MGIL: Multiscale Geometric Interaction Layer

MGIL enriches each edge (i,j) in the solute graph with a concatenated geometry descriptor:

```
e_ij = [RBF(d_ij), moments_ij, anchors_ij]
```

### Components

**1. Distance RBF (K dimensions)**
- Standard radial basis function expansion of interatomic distances

**2. Moment Invariants (5 dimensions)**
- Computed from neighbor direction statistics under a smooth cutoff
- Captures local anisotropy without enumerating angles/dihedrals
- SE(3)-invariant by construction (inner products of co-rotating vectors)

**3. Global Anchors (2 dimensions)**
- `f_radial = ⟨u_ij, g_i⟩` — directional alignment with centroid
- `f_radius = ‖r_i - c‖` — distance from centroid
- Provides a molecule-level spatial reference to resolve globally distinct conformations

**4. Virtual Node (optional)**
- Graph-level token updated via mean-pool and broadcast each layer
- Enables O(N) long-range communication

All components are O(|E|) in complexity and SE(3)-invariant.

## SFM: Solvent Field Modulator

### Solvent Encoder

A 2D message-passing network processes the solvent molecular graph (atoms as nodes, bonds as edges) to produce:
- `z_s` [B, D] — graph-level solvent embedding
- `h_solvent` [N_s, D] — per-atom solvent embeddings

### Cross-Attention (Selective Interaction)

Each solute atom _i_ queries the solvent atoms _j_:

```
q_i = W_q · h_i         (solute query)
k_j = W_k · s_j         (solvent key)
v_j = W_v · s_j         (solvent value)
α_ij = softmax_j(q_i^T k_j / √d)
z_s,i = Σ_j α_ij · v_j
```

This produces per-atom solvent contexts `z_s,i` — different solute regions attend to different solvent motifs.

### FiLM Modulation

From `z_s,i`, we generate per-channel scale and shift parameters:

```
(γ_i, β_i) = MLP(z_s,i)
h_i' = γ_i ⊙ h_i + β_i
```

FiLM is applied to scalar channels at specified layers. For vector features, scalar gates (without additive bias) preserve equivariance.

## Backbone Coupling

MGIL features enter via the backbone's edge/message functions. SFM modulates intermediate node features via FiLM at configurable layers.

The framework is backbone-agnostic: all supported backbones (EGNN, PaiNN, SchNet, LEFTNet, Equiformer, C-GNN) share a unified `ConditionContext` protocol and adapter interface.

## Training

- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** AdamW with cosine annealing
- **Target conversion:** Physics head optionally converts energy-space predictions to wavelength via `hc/E`
- **Data split:** 70/10/20 (train/val/test), fixed random seed
