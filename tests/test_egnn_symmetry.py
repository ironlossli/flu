import torch
import torch.nn as nn
from spectra.models.backbones.egnn_backbone import EGNNBackbone
import numpy as np

def test_symmetry():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Data
    N = 5
    node_in_dim = 12
    hidden_dim = 64
    node_feats = torch.randn(N, node_in_dim).to(device)
    coords = torch.randn(N, 3).to(device)
    
    # Complete graph for symmetry test
    src, dst = [], []
    for i in range(N):
        for j in range(N):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long).to(device)
    
    # Standard edge_attr: [d, ux, uy, uz]
    diff = coords[edge_index[0]] - coords[edge_index[1]]
    dist = torch.norm(diff, dim=-1, keepdim=True)
    unit = diff / (dist + 1e-8)
    edge_attr = torch.cat([dist, unit], dim=-1) # [E, 4]

    def check_model(model, name):
        print(f"\n--- Testing {name} ---")
        model.eval()
        with torch.no_grad():
            # Original output
            out_orig, _, coords_upd_orig = model(node_feats, coords, edge_index, edge_attr=edge_attr)
            
            # A. Translation Invariance
            t = torch.randn(1, 3).to(device)
            out_trans, _, _ = model(node_feats, coords + t, edge_index, edge_attr=edge_attr)
            diff_trans = (out_orig - out_trans).abs().max().item()
            print(f"Translation Invariance Diff: {diff_trans:.2e}")
            
            # B. Rotation Invariance/Equivariance
            # Random rotation matrix
            from scipy.spatial.transform import Rotation
            R = torch.from_numpy(Rotation.random().as_matrix()).float().to(device)
            
            coords_rot = coords @ R.T
            # Recompute edge_attr for rotated coords (unit vectors will change!)
            diff_rot = coords_rot[edge_index[0]] - coords_rot[edge_index[1]]
            dist_rot = torch.norm(diff_rot, dim=-1, keepdim=True)
            unit_rot = diff_rot / (dist_rot + 1e-8)
            edge_attr_rot = torch.cat([dist_rot, unit_rot], dim=-1)
            
            out_rot, _, coords_upd_rot = model(node_feats, coords_rot, edge_index, edge_attr=edge_attr_rot)
            
            diff_rot_invar = (out_orig - out_rot).abs().max().item()
            print(f"Rotation Invariance Diff: {diff_rot_invar:.2e}")
            
            # Equivariance check for coords update
            # (coords_upd - coords) should rotate with R
            delta_orig = coords_upd_orig - coords
            delta_rot = coords_upd_rot - coords_rot
            diff_rot_equiv = (delta_orig @ R.T - delta_rot).abs().max().item()
            print(f"Rotation Equivariance Diff: {diff_rot_equiv:.2e}")

            # C. Permutation Invariance
            perm = torch.randperm(N)
            node_feats_p = node_feats[perm]
            coords_p = coords[perm]
            
            # Remap edge_index
            rev_perm = torch.zeros(N, dtype=torch.long)
            rev_perm[perm] = torch.arange(N)
            
            # For simplicity with complete graph, we can just rebuild edges
            # or permute the original ones. Let's rebuild.
            src_p, dst_p = [], []
            for i in range(N):
                for j in range(N):
                    if i != j:
                        src_p.append(i)
                        dst_p.append(j)
            edge_index_p = torch.tensor([src_p, dst_p], dtype=torch.long).to(device)
            
            diff_p = coords_p[edge_index_p[0]] - coords_p[edge_index_p[1]]
            dist_p = torch.norm(diff_p, dim=-1, keepdim=True)
            unit_p = diff_p / (dist_p + 1e-8)
            edge_attr_p = torch.cat([dist_p, unit_p], dim=-1)
            
            out_p, _, _ = model(node_feats_p, coords_p, edge_index_p, edge_attr=edge_attr_p)
            diff_perm = (out_orig - out_p).abs().max().item()
            print(f"Permutation Invariance Diff: {diff_perm:.2e}")

    # Test Case 1: Legacy Mode (Should FAIL Rotation Test)
    model_legacy = EGNNBackbone(
        node_in_dim=node_in_dim, hidden_dim=hidden_dim, 
        edge_attr_dim=4, edge_attr_mode="legacy"
    ).to(device)
    check_model(model_legacy, "EGNN Legacy (Non-Invariant)")

    # Test Case 2: Invariant Only Mode (Should PASS all tests)
    model_invar = EGNNBackbone(
        node_in_dim=node_in_dim, hidden_dim=hidden_dim, 
        edge_attr_dim=1, edge_attr_mode="invariant_only"
    ).to(device)
    check_model(model_invar, "EGNN Invariant Only")

    # Test Case 3: Invariant + RBF
    model_rbf = EGNNBackbone(
        node_in_dim=node_in_dim, hidden_dim=hidden_dim,
        edge_attr_dim=1, edge_attr_mode="invariant_only",
        use_rbf=True, n_rbf=20
    ).to(device)
    check_model(model_rbf, "EGNN Invariant + RBF")

    # Test Case 4: Invariant + RBF + FCVM
    model_fcvm = EGNNBackbone(
        node_in_dim=node_in_dim, hidden_dim=hidden_dim,
        edge_attr_dim=1, edge_attr_mode="invariant_only",
        use_rbf=True, n_rbf=20,
        use_fcvm=True, chirality_invariant=True
    ).to(device)
    check_model(model_fcvm, "EGNN Invariant + RBF + FCVM")

if __name__ == "__main__":
    test_symmetry()
