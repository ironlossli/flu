
import sys
import os
import torch
import warnings

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from spectra.models.ehc.backbone_adapter import EGNN_GBM_Adapter
from spectra.models.ehc.backbone_protocol import ConditionContext

def test_config(name, config_override, expected_edge_dim):
    print(f"\n--- Testing Config: {name} ---")
    
    # Base params
    params = dict(
        node_in_dim=12,
        edge_attr_dim=4, # Dummy
        hidden_dim=32,
        num_layers=2,
        z_dim=32,
        n_rbf=20,
        cutoff=5.0,
        use_moments=True,
        use_global=True
    )
    
    # Apply overrides
    params.update(config_override)
    
    # Instantiate
    model = EGNN_GBM_Adapter(**params)
    
    # 1. Check Dimensions
    actual_dim = model.backbone.layers[0].edge_mlp[0].in_features
    # EGNN inputs to Edge MLP: 
    # [h_i, h_j] (2*H) + radial (1) + edge_attr (cached_dim) + ...
    # Wait, let's check cached_edge_dim explicitly from internal attribute if possible,
    # or infer from model.edge_cache
    
    # We can check the adapter's gbm attribute which stores edge_dim?
    # BaseSolventAdapter init sets self.gbm with edge_dim.
    cached_edge_dim = model.gbm.edge_dim
    print(f"Cached Edge Dim: {cached_edge_dim} (Expected: {expected_edge_dim})")
    
    if cached_edge_dim != expected_edge_dim:
        print("FAIL: Edge dimension mismatch!")
        return False

    # 2. Forward Pass Check
    # Dummy Data
    N = 10
    B = 2
    Z = torch.randint(0, 5, (N,))
    pos = torch.randn(N, 3)
    node_batch = torch.tensor([0]*5 + [1]*5).long()
    
    # Build edges (Radius Graph)
    from torch_geometric.nn import radius_graph
    edge_index = radius_graph(pos, r=5.0, batch=node_batch)
    
    # Dummy Condition
    cond = ConditionContext(
        z_s=torch.randn(B, 32),
        z_s_node=torch.randn(N, 32),
        solvent_x=None, solvent_batch=None, film=None
    )
    
    try:
        out = model(Z, pos, node_batch, cond, edge_index=edge_index)
        print("Forward Pass: SUCCESS")
    except Exception as e:
        print(f"Forward Pass: FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

def main():
    # Test Cases
    # Base n_rbf = 20
    
    # 1. Full: M=T, G=T -> Edge=20+5+2=27
    c1 = test_config("Full (SOTA)", 
                     {"use_moments": True, "use_global": True}, 
                     expected_edge_dim=27)
    
    # 2. No-Moments: M=F, G=T -> Edge=20+2=22
    c2 = test_config("Ab-M (No Moments)", 
                     {"use_moments": False, "use_global": True}, 
                     expected_edge_dim=22)

    # 3. No-Global: M=T, G=F -> Edge=20+5=25
    c3 = test_config("Ab-G (No Global)", 
                     {"use_moments": True, "use_global": False}, 
                     expected_edge_dim=25)

    # 4. Base: M=F, G=F -> Edge=20
    c4 = test_config("Base (Pure EGNN)", 
                     {"use_moments": False, "use_global": False}, 
                     expected_edge_dim=20)

    if all([c1, c2, c3, c4]):
        print("\nAll Tests Passed!")
    else:
        print("\nSome Tests Failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
