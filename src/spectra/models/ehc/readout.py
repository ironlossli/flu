import torch
import torch.nn as nn
from torch_geometric.utils import softmax

class SpectralFocusReadout(nn.Module):
    """
    Chromophore-Aware Attention Readout (SpectralFocus).
    
    Instead of global mean/sum pooling, this module learns to assign weights 
    to atoms based on their contribution to the spectral property.
    This allows the model to focus on the chromophore (conjugated system) 
    and ignore irrelevant parts (e.g., long alkyl chains).
    """
    def __init__(self, in_channels, hidden_channels=None, gate_nn=None):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else in_channels // 2
        
        # The gating network that computes the attention score (scalar) for each node
        # H_i -> score_i
        if gate_nn is None:
            self.gate_nn = nn.Sequential(
                nn.Linear(in_channels, self.hidden_channels),
                nn.SiLU(),
                nn.Linear(self.hidden_channels, 1)
            )
        else:
            self.gate_nn = gate_nn

    def forward(self, x, batch, return_weights=False):
        """
        Args:
            x (Tensor): Node features [N, in_channels]
            batch (Tensor): Batch indices [N]
            return_weights (bool): If True, return the normalized attention weights.
            
        Returns:
            out (Tensor): Graph features [B, in_channels]
            weights (Tensor, optional): Attention weights [N, 1]
        """
        # 1. Compute raw scores
        # scores: [N, 1]
        scores = self.gate_nn(x)
        
        # 2. Normalize scores across each graph using Softmax
        # weights: [N, 1] where sum(weights) for each graph = 1
        weights = softmax(scores, batch, dim=0)
        
        # 3. Weighted Sum Pooling
        # out: [B, in_channels]
        # x * weights broadcasts to [N, in_channels], then we sum over batch
        weighted_x = x * weights
        
        # Global add pool manually implemented for weighted sum
        # scatter_add is usually used, but here we can just use torch_geometric's global_add_pool
        # on the weighted features.
        from torch_geometric.nn import global_add_pool
        out = global_add_pool(weighted_x, batch)
        
        if return_weights:
            return out, weights
            
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}(in={self.in_channels}, hidden={self.hidden_channels})'
