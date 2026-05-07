import torch
import copy

class MockBatch:
    def __init__(self):
        self.solvent_x = torch.tensor([
            [1.0], [2.0],       # Graph 0
            [3.0], [4.0], [5.0] # Graph 1
        ])
        self.solvent_batch = torch.tensor([0, 0, 1, 1, 1])
        # Edges: 
        # G0: 0->1
        # G1: 2->3, 4->2
        self.solvent_edge_index = torch.tensor([
            [0, 2, 4],
            [1, 3, 2]
        ], dtype=torch.long)
        self.solvent_edge_attr = torch.tensor([[0.1], [0.2], [0.3]])

def permute_solvent_in_batch(batch):
    device = batch.solvent_x.device
    num_nodes = batch.solvent_x.size(0)
    
    unique_batches, counts = torch.unique(batch.solvent_batch, return_counts=True)
    B = len(unique_batches)
    
    global_perm = torch.arange(num_nodes, device=device)
    
    offset = 0
    for b in range(B):
        n = counts[b].item()
        if n > 0:
            local_perm = torch.randperm(n, device=device)
            global_perm[offset : offset+n] = offset + local_perm
        offset += n
        
    old_to_new = torch.empty_like(global_perm)
    old_to_new[global_perm] = torch.arange(num_nodes, device=device)
    
    new_x = batch.solvent_x[global_perm]
    new_edge_index = old_to_new[batch.solvent_edge_index]
    
    new_batch = copy.copy(batch)
    new_batch.solvent_x = new_x
    new_batch.solvent_edge_index = new_edge_index
    
    return new_batch, global_perm

def test_perm():
    batch = MockBatch()
    print("Original X:\n", batch.solvent_x)
    print("Original Edge Index:\n", batch.solvent_edge_index)
    
    new_batch, perm = permute_solvent_in_batch(batch)
    
    print("\nPermutation:", perm)
    print("New X:\n", new_batch.solvent_x)
    print("New Edge Index:\n", new_batch.solvent_edge_index)
    
    # Verification
    # Check if graph structure is preserved
    # G0: Node '1.0' should be connected to Node '2.0' regardless of their new indices.
    
    # Map values to find where they went
    val_to_idx = {v.item(): i for i, v in enumerate(new_batch.solvent_x)}
    
    # Check Edge 0 (originally 0->1, values 1.0->2.0)
    u_new_idx = new_batch.solvent_edge_index[0, 0].item()
    v_new_idx = new_batch.solvent_edge_index[1, 0].item()
    
    u_val = new_batch.solvent_x[u_new_idx].item()
    v_val = new_batch.solvent_x[v_new_idx].item()
    
    print(f"\nEdge 0 connects value {u_val} -> {v_val}")
    
    if u_val == 1.0 and v_val == 2.0:
        print("✓ Edge 0 structural integrity maintained.")
    else:
        print("✗ Edge 0 BROKEN!")

    # Check batch assignment
    if torch.all(new_batch.solvent_batch == batch.solvent_batch):
        print("✓ Batch indices preserved.")
    else:
        print("✗ Batch indices changed (unexpectedly)!")

if __name__ == "__main__":
    test_perm()
