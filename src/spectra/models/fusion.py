import torch
import torch.nn as nn
from typing import Optional


class ConcatFusion(nn.Module):
    """Concatenate inputs and project."""
    
    def __init__(
        self,
        solute_dim: int,
        solvent_dim: int,
        descriptor_dim: int = 0,
        output_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        input_dim = solute_dim + solvent_dim + descriptor_dim
        
        self.fusion_net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
    
    def forward(
        self,
        solute_emb: torch.Tensor,
        solvent_emb: torch.Tensor,
        descriptors: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if descriptors is not None:
            x = torch.cat([solute_emb, solvent_emb, descriptors], dim=-1)
        else:
            x = torch.cat([solute_emb, solvent_emb], dim=-1)
        
        return self.fusion_net(x)


class AttentionFusion(nn.Module):
    """Self-attention over projected inputs."""
    
    def __init__(
        self,
        solute_dim: int,
        solvent_dim: int,
        descriptor_dim: int = 0,
        output_dim: int = 256,
        num_heads: int = 4
    ):
        super().__init__()
        
        self.solute_proj = nn.Linear(solute_dim, output_dim)
        self.solvent_proj = nn.Linear(solvent_dim, output_dim)
        
        if descriptor_dim > 0:
            self.descriptor_proj = nn.Linear(descriptor_dim, output_dim)
        else:
            self.descriptor_proj = None
        
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.output_net = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(
        self,
        solute_emb: torch.Tensor,
        solvent_emb: torch.Tensor,
        descriptors: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = solute_emb.size(0)
        solute_proj = self.solute_proj(solute_emb).unsqueeze(1)  # [batch, 1, dim]
        solvent_proj = self.solvent_proj(solvent_emb).unsqueeze(1)  # [batch, 1, dim]
        if descriptors is not None and self.descriptor_proj is not None:
            desc_proj = self.descriptor_proj(descriptors).unsqueeze(1)
            seq = torch.cat([solute_proj, solvent_proj, desc_proj], dim=1)  # [batch, 3, dim]
        else:
            seq = torch.cat([solute_proj, solvent_proj], dim=1)  # [batch, 2, dim]
        attn_out, _ = self.attention(seq, seq, seq)  # [batch, seq_len, dim]
        fused = attn_out.mean(dim=1)  # [batch, dim]
        
        return self.output_net(fused)


class GatedFusion(nn.Module):
    """Learned gate between solute/solvent (and optional descriptors)."""
    
    def __init__(
        self,
        solute_dim: int,
        solvent_dim: int,
        descriptor_dim: int = 0,
        output_dim: int = 256
    ):
        super().__init__()
        
        self.solute_proj = nn.Linear(solute_dim, output_dim)
        self.solvent_proj = nn.Linear(solvent_dim, output_dim)
        
        gate_input_dim = solute_dim + solvent_dim
        if descriptor_dim > 0:
            gate_input_dim += descriptor_dim
            self.descriptor_proj = nn.Linear(descriptor_dim, output_dim)
        else:
            self.descriptor_proj = None
        
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, output_dim),
            nn.Sigmoid()
        )
        
        self.output_net = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(
        self,
        solute_emb: torch.Tensor,
        solvent_emb: torch.Tensor,
        descriptors: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        solute_proj = self.solute_proj(solute_emb)
        solvent_proj = self.solvent_proj(solvent_emb)
        if descriptors is not None and self.descriptor_proj is not None:
            gate_input = torch.cat([solute_emb, solvent_emb, descriptors], dim=-1)
            desc_proj = self.descriptor_proj(descriptors)
        else:
            gate_input = torch.cat([solute_emb, solvent_emb], dim=-1)
            desc_proj = 0
        
        gate_weight = self.gate(gate_input)  # [batch, output_dim]
        fused = gate_weight * solute_proj + (1 - gate_weight) * solvent_proj + desc_proj
        
        return self.output_net(fused)


def get_fusion_layer(fusion_type: str, **kwargs) -> nn.Module:
    """Factory for fusion modules."""
    fusions = {
        "concat": ConcatFusion,
        "attention": AttentionFusion,
        "gated": GatedFusion
    }
    
    if fusion_type.lower() not in fusions:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    return fusions[fusion_type.lower()](**kwargs)
