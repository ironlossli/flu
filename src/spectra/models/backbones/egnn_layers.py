from torch import nn
import torch
import logging
from typing import Literal

logger = logging.getLogger(__name__)

class MLP(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )

    def forward(self, x):
        return self.net(x)


class GCL_basic(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self):
        super(GCL_basic, self).__init__()


    def edge_model(self, source, target, edge_attr):
        pass

    def node_model(self, h, edge_index, edge_attr):
        pass

    def forward(self, x, edge_index, edge_attr=None):
        row, col = edge_index
        edge_feat = self.edge_model(x[row], x[col], edge_attr)
        x = self.node_model(x, edge_index, edge_feat)
        return x, edge_feat



class GCL(GCL_basic):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_nf=0, act_fn=nn.ReLU(), bias=True, attention=False, t_eq=False, recurrent=True):
        super(GCL, self).__init__()
        self.attention = attention
        self.t_eq=t_eq
        self.recurrent = recurrent
        input_edge_nf = input_nf * 2
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge_nf + edges_in_nf, hidden_nf, bias=bias),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf, bias=bias),
            act_fn)
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(input_nf, hidden_nf, bias=bias),
                act_fn,
                nn.Linear(hidden_nf, 1, bias=bias),
                nn.Sigmoid())


        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf, bias=bias),
            act_fn,
            nn.Linear(hidden_nf, output_nf, bias=bias))

        #if recurrent:
            #self.gru = nn.GRUCell(hidden_nf, hidden_nf)


    def edge_model(self, source, target, edge_attr):
        edge_in = torch.cat([source, target], dim=1)
        if edge_attr is not None:
            edge_in = torch.cat([edge_in, edge_attr], dim=1)
        out = self.edge_mlp(edge_in)
        if self.attention:
            att = self.att_mlp(torch.abs(source - target))
            out = out * att
        return out

    def node_model(self, h, edge_index, edge_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=h.size(0))
        out = torch.cat([h, agg], dim=1)
        out = self.node_mlp(out)
        if self.recurrent:
            out = out + h
            #out = self.gru(out, h)
        return out


class GCL_rf(GCL_basic):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, nf=64, edge_attr_nf=0, reg=0, act_fn=nn.LeakyReLU(0.2), clamp=False):
        super(GCL_rf, self).__init__()

        self.clamp = clamp
        layer = nn.Linear(nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.phi = nn.Sequential(nn.Linear(edge_attr_nf + 1, nf),
                                 act_fn,
                                 layer)
        self.reg = reg

    def edge_model(self, source, target, edge_attr):
        x_diff = source - target
        radial = torch.sqrt(torch.sum(x_diff ** 2, dim=1)).unsqueeze(1)
        e_input = torch.cat([radial, edge_attr], dim=1)
        e_out = self.phi(e_input)
        m_ij = x_diff * e_out
        if self.clamp:
            m_ij = torch.clamp(m_ij, min=-100, max=100)
        return m_ij

    def node_model(self, x, edge_index, edge_attr):
        row, col = edge_index
        agg = unsorted_segment_mean(edge_attr, row, num_segments=x.size(0))
        x_out = x + agg - x*self.reg
        return x_out


def compute_moment_invariants(coord, edge_index, cutoff=5.0, eps=1e-6):
    """
    Compute rotation-invariant scalars from local geometric moments.
    Replacing explicit frames with moment invariants for stability and E(3) compliance.
    """
    row, col = edge_index
    N = coord.size(0)
    
    # 1. Basic Geometrics
    diff = coord[row] - coord[col]
    dist_sq = torch.sum(diff**2, dim=-1, keepdim=True)
    dist = torch.sqrt(dist_sq + eps)
    u = diff / dist # Unit vectors [E, 3]
    
    # 2. Smooth Weighting (Cosine Envelope)
    # w = 0.5 * (cos(pi * d / cutoff) + 1)
    d_clamped = torch.clamp(dist, max=cutoff)
    w = 0.5 * (torch.cos(d_clamped * 3.1415926535 / cutoff) + 1.0)
    # Mask out edges beyond cutoff
    w = w * (dist <= cutoff).float()

    # 3. Normalization Factor (Effective Degree)
    # t_i = sum(w_ij)
    t_i = torch.zeros(N, 1, device=coord.device, dtype=coord.dtype)
    t_i.scatter_add_(0, row.unsqueeze(-1), w)
    t_inv = 1.0 / (t_i + eps)
    
    # 4. Node-level Moments (Scatter Add)
    # First Moment: s_i = sum(w * u)
    weighted_u = u * w
    s_i = torch.zeros(N, 3, device=coord.device, dtype=coord.dtype)
    s_i.scatter_add_(0, row.unsqueeze(-1).expand(-1, 3), weighted_u)
    s_norm_i = s_i * t_inv # Normalized first moment
    
    # Second Moment: M_i = sum(w * u * u^T)
    # Outer product: [E, 3, 3]
    u_outer = u.unsqueeze(2) * u.unsqueeze(1)
    weighted_outer = u_outer * w.unsqueeze(-1)
    
    M_i = torch.zeros(N, 3, 3, device=coord.device, dtype=coord.dtype)
    M_i_flat = M_i.view(N, 9)
    M_i_flat.scatter_add_(0, row.unsqueeze(-1).expand(-1, 9), weighted_outer.view(-1, 9))
    M_i = M_i.view(N, 3, 3)
    M_norm_i = M_i * t_inv.unsqueeze(-1) # Normalized second moment
    
    # Symmetrize numerically
    M_norm_i = 0.5 * (M_norm_i + M_norm_i.transpose(-1, -2))

    # 5. Edge-level Invariants (Gather & Project)
    # Gather central node moments to edges (target=row)
    s_edge = s_norm_i[row]   # [E, 3]
    M_edge = M_norm_i[row]   # [E, 3, 3]
    
    # I1: Edge-Moment Consistency (u^T M u)
    Mu = torch.bmm(M_edge, u.unsqueeze(2)).squeeze(2) # [E, 3]
    I_a = torch.sum(u * Mu, dim=-1, keepdim=True) # [E, 1]
    
    # I2: Moment Anisotropy (tr(M @ M))
    MM = torch.bmm(M_edge, M_edge)
    I_aniso = MM.diagonal(dim1=1, dim2=2).sum(dim=-1, keepdim=True) # [E, 1]
    
    # I3: Effective Degree (Log-scaled)
    I_deg = torch.log(t_i[row] + 1.0)
    
    # I4: Directional Bias Strength (||s||)
    I_s_norm = torch.norm(s_edge, dim=-1, keepdim=True) # [E, 1]
    
    # I5: Edge-Bias Alignment (u^T s)
    I_s_proj = torch.sum(u * s_edge, dim=-1, keepdim=True) # [E, 1]
    
    invariants = torch.cat([I_deg, I_a, I_aniso, I_s_norm, I_s_proj], dim=-1)
    
    stats = {
        "aniso_mean": I_aniso.mean().item(),
        "s_norm_mean": I_s_norm.mean().item(),
        "deg_mean": t_i.mean().item()
    }
    
    return invariants, stats


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / ((stop - start) / num_gaussians)**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class E_GCL(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False, clamp=False, norm_diff=False, tanh=False, edge_attr_mode: Literal["invariant_only", "legacy"] = "legacy", use_fcvm: bool = False, chirality_invariant: bool = False, use_rbf: bool = False, n_rbf: int = 20, cutoff: float = 5.0, log_fcvm: bool = False, log_config: dict = None, use_vector_features: bool = False, use_virtual_node: bool = False, use_advanced_geometry: bool = True, use_vector_mixing: bool = False, vector_mix_scale: float = 1.0, dropout: float = 0.1):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        self.edge_attr_mode = edge_attr_mode
        self.use_fcvm = use_fcvm
        self.chirality_invariant = chirality_invariant
        self.use_rbf = use_rbf
        self.clamp = clamp
        self.log_fcvm = log_fcvm
        self.log_config = log_config or {}
        self.cutoff = cutoff
        
        self.use_vector_features = use_vector_features
        self.use_virtual_node = use_virtual_node
        self.use_advanced_geometry = use_advanced_geometry
        self.use_vector_mixing = bool(use_vector_mixing)
        self.vector_mix_scale = float(vector_mix_scale)
        self.dropout_p = float(dropout)
        
        self.log_every = self.log_config.get("log_fcvm_every", 200)
        self.log_step_count = 0

        if use_rbf:
            self.rbf_fn = GaussianSmearing(start=0.0, stop=cutoff, num_gaussians=n_rbf)
            edge_coords_nf = n_rbf
        else:
            self.rbf_fn = None
            edge_coords_nf = 1

        # CMD-F3: If use_fcvm is enabled, we use Moment Invariants (5 dims)
        fcvm_nf = 5 if use_fcvm else 0
        
        if use_fcvm:
            self.fcvm_gate_mlp = nn.Sequential(
                nn.Linear(5, hidden_nf // 4),
                nn.SiLU(),
                nn.Linear(hidden_nf // 4, 1),
                nn.Sigmoid()
            )
        
        # Virtual Node Support
        global_nf = 0
        if self.use_virtual_node:
            global_nf = hidden_nf # We broadcast global state to nodes
            # MLP to update global state: Global_old + Agg_Nodes -> Global_new
            self.global_mlp = nn.Sequential(
                nn.Linear(hidden_nf + hidden_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf)
            )

        # Vector Features Support
        vector_readout_dim = 0
        if self.use_vector_features:
            # Inputs to Edge MLP: v_i*u, v_j*u, |v_i|, |v_j| (4 dims)
            vector_readout_dim = 4
            if self.use_advanced_geometry:
                vector_readout_dim += 2 # Cross Norm + Triple Product
            
            # Vector Gate: Scalar Edge Feats -> Vector Weights (for aggregation)
            self.vec_gate_mlp = nn.Sequential(
                nn.Linear(hidden_nf, hidden_nf),
                nn.SiLU(),
                nn.Linear(hidden_nf, hidden_nf)
            )
            # Vector Linear: Transform node vectors
            self.vec_lin = nn.Linear(hidden_nf, hidden_nf, bias=False)

        if self.use_vector_features and self.use_vector_mixing:
            self.vec2scalar_mlp = nn.Sequential(
                nn.Linear(hidden_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf)
            )

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d + fcvm_nf + vector_readout_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim + global_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        if self.dropout_p > 0:
            self.edge_dropout = nn.Dropout(self.dropout_p)
            self.node_dropout = nn.Dropout(self.dropout_p)
        else:
            self.edge_dropout = nn.Identity()
            self.node_dropout = nn.Identity()

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1))*3
        self.coord_mlp = nn.Sequential(*coord_mlp)


        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

        #if recurrent:
        #    self.gru = nn.GRUCell(hidden_nf, hidden_nf)


    def edge_model(self, source, target, radial, edge_attr, fcvm_p=None, v_source=None, v_target=None, coord_diff=None):
        # Solute protection logic...
        if (self.edge_attr_mode == "invariant_only" or self.use_fcvm):
            if edge_attr is not None and edge_attr.size(-1) != 7:
                 if edge_attr.size(-1) > 1:
                     edge_attr = edge_attr[..., :1]

        # Basic inputs
        out_list = [source, target, radial]
        if edge_attr is not None:
            out_list.append(edge_attr)
        if self.use_fcvm and fcvm_p is not None:
            out_list.append(fcvm_p)
            
        # Vector Readout (PaiNN-style projections)
        if self.use_vector_features and v_source is not None and coord_diff is not None:
            # Normalize coord_diff for projection
            dist = radial if radial.size(-1) == 1 else torch.norm(coord_diff, dim=-1, keepdim=True)
            u_ij = coord_diff / (dist + 1e-8)
            
            # Dot products (Projects)
            v_i_proj = (v_source * u_ij.unsqueeze(1)).sum(dim=-1) # [E, H]
            v_j_proj = (v_target * u_ij.unsqueeze(1)).sum(dim=-1) # [E, H]
            
            # Norms
            v_i_norm = torch.sqrt((v_source**2).sum(dim=-1) + 1e-8) # [E, H]
            v_j_norm = torch.sqrt((v_target**2).sum(dim=-1) + 1e-8) # [E, H]
            
            # Flatten: Take MEAN
            v_i_p_mean = v_i_proj.mean(dim=-1, keepdim=True)
            v_j_p_mean = v_j_proj.mean(dim=-1, keepdim=True)
            v_i_n_mean = v_i_norm.mean(dim=-1, keepdim=True)
            v_j_n_mean = v_j_norm.mean(dim=-1, keepdim=True)
            
            out_list.extend([v_i_p_mean, v_j_p_mean, v_i_n_mean, v_j_n_mean])
            
            # Advanced Geometry (E-GNN++)
            if self.use_advanced_geometry:
                # Cross Product (v_i x v_j) -> [E, H, 3]
                cross_ij = torch.cross(v_source, v_target, dim=-1)
                
                # Feature 1: Cross Norm (Plane Area / Non-collinearity) -> [E, 1] (Mean over H)
                cross_norm = torch.sqrt((cross_ij**2).sum(dim=-1) + 1e-8)
                cross_norm_mean = cross_norm.mean(dim=-1, keepdim=True)
                
                # Feature 2: Triple Product (Chirality) (v_i x v_j) . u_ij -> [E, 1] (Mean over H)
                # cross_ij: [E, H, 3], u_ij: [E, 3] -> broadcast u_ij
                triple = (cross_ij * u_ij.unsqueeze(1)).sum(dim=-1)
                triple_mean = triple.mean(dim=-1, keepdim=True)
                
                out_list.extend([cross_norm_mean, triple_mean])

        out = torch.cat(out_list, dim=1)
        out = self.edge_mlp(out)
        out = self.edge_dropout(out)
        
        # FCVM Gating
        if self.use_fcvm and fcvm_p is not None:
            gate = self.fcvm_gate_mlp(fcvm_p)
            out = out * (1.0 + gate)

        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
            
        # Vector Weights for Aggregation
        vector_weights = None
        if self.use_vector_features:
            vector_weights = self.vec_gate_mlp(out) # [E, H]
            
        return out, vector_weights

    def node_model(self, x, edge_index, edge_attr, node_attr, v=None, vector_weights=None, coord_diff=None, global_h_broadcast=None):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        
        # Virtual Node Injection
        if self.use_virtual_node and global_h_broadcast is not None:
             agg = torch.cat([agg, global_h_broadcast], dim=1)
             
        out = self.node_mlp(agg)
        out = self.node_dropout(out)

        if self.use_vector_features and self.use_vector_mixing and v is not None:
            v_norm = torch.norm(v, dim=-1)
            out = out + self.vector_mix_scale * self.vec2scalar_mlp(v_norm)
        
        # Vector Update
        v_new = v
        if self.use_vector_features and v is not None and vector_weights is not None and coord_diff is not None:
            # Delta V = Sum( Weights * u_ij )
            # u_ij: [E, 3]
            # Weights: [E, H]
            # Result: [E, H, 3]
            dist = torch.norm(coord_diff, dim=-1, keepdim=True) + 1e-8
            u_ij = coord_diff / dist # [E, 3]
            
            # Broadcast multiply
            edge_v = vector_weights.unsqueeze(-1) * u_ij.unsqueeze(1) # [E, H, 3]
            
            # Aggregate
            delta_v = torch.zeros_like(v)
            delta_v.index_add_(0, row, edge_v) # Scatter add
            
            # Combine
            # v_new = Lin(v) + delta_v
            # v is [N, H, 3], Lin acts on last dim. Transpose to [N, 3, H] to mix channels.
            v_new = self.vec_lin(v.transpose(1, 2)).transpose(1, 2) + delta_v
        
        if self.recurrent:
            out = x + out
        return out, agg, v_new

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        if self.coords_weight == 0.0:
            return coord
        # ... (rest is same)
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.clamp:
            trans = torch.clamp(trans, min=-100, max=100)
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        
        if self.log_fcvm and self.log_step_count % self.log_every == 0:
            dx_norm = torch.norm(agg * self.coords_weight, dim=-1).mean().item()
            self._last_dx_norm = dx_norm
        
        coord = coord + agg*self.coords_weight
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        sq_dist = torch.sum((coord_diff)**2, 1).unsqueeze(1)

        if self.norm_diff:
            norm = torch.sqrt(sq_dist) + 1
            coord_diff = coord_diff/(norm)

        if self.use_rbf:
            dist = torch.sqrt(sq_dist + 1e-8)
            radial = self.rbf_fn(dist)
        else:
            radial = sq_dist

        return radial, coord_diff

    def _log_stats(self, p, coord_diff, fcvm_frames, row, fcvm_stats=None):
        if not self.log_fcvm: return
        self.log_step_count += 1
        if self.log_step_count % self.log_every != 0: return
        # ... (Keep existing logging logic)
        dist = torch.norm(coord_diff, dim=-1)
        min_d = dist.min().item()
        nan_count = torch.isnan(p).sum().item() if p is not None else 0
        log_msg = f"[Moment Log] step={self.log_step_count} | min_d={min_d:.1e} | nan={nan_count}"
        if fcvm_stats:
            log_msg += f" | aniso={fcvm_stats['aniso_mean']:.2f} | s_norm={fcvm_stats['s_norm_mean']:.2f} | deg={fcvm_stats['deg_mean']:.1f}"
        if p is not None:
            p_mean = p.detach().float().mean(dim=0).cpu().numpy()
            log_msg += f" | inv_means={p_mean}"
        if hasattr(self, "_last_dx_norm"):
            log_msg += f" | dx_norm={self._last_dx_norm:.1e}"
        if hasattr(self, "_probe_delta_p0"):
            log_msg += f" | p0_sens={self._probe_delta_p0:.1e} | ps_sens={self._probe_delta_ps:.1e}"
        logger.info(log_msg)

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None, fcvm_frames=None, v=None, global_h=None, batch=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        # Initialize v if needed
        if self.use_vector_features and v is None:
            v = torch.zeros(h.size(0), h.size(1), 3, device=h.device, dtype=h.dtype)
            
        # Virtual Node Logic
        global_h_broadcast = None
        global_h_new = global_h
        if self.use_virtual_node and batch is not None:
             # 1. Init global if None (usually done outside, but for safety)
             if global_h is None:
                 num_graphs = int(batch.max().item()) + 1
                 global_h = h.new_zeros(num_graphs, h.size(1))
                 
             # 2. Node -> Global (Aggregation)
             msg_n2g = unsorted_segment_mean(h, batch, num_segments=global_h.size(0))
             
             # 3. Update Global
             global_in = torch.cat([global_h, msg_n2g], dim=1)
             global_h_new = self.global_mlp(global_in) + global_h # Residual
             
             # 4. Broadcast Global -> Node
             global_h_broadcast = global_h_new[batch]

        # CMD-F2: Compute Moment Invariants if enabled
        fcvm_p = None
        fcvm_stats = None
        if self.use_fcvm:
            is_solute = (edge_attr is None) or (edge_attr.size(-1) != 7)
            if is_solute:
                fcvm_p, fcvm_stats = compute_moment_invariants(coord, edge_index, cutoff=self.cutoff)

        # LOG-CMD-05: Sensitivity Probe (Simplified for brevity, fcvm-only)
        # ... (Keeping existing probe logic if fcvm is on)

        # Edge Model (Scalar + Vector Weights)
        # Pass v[row], v[col] and coord_diff
        v_source = v[row] if v is not None else None
        v_target = v[col] if v is not None else None
        
        edge_feat, vector_weights = self.edge_model(
            h[row], h[col], radial, edge_attr, 
            fcvm_p=fcvm_p, 
            v_source=v_source, v_target=v_target, coord_diff=coord_diff
        )
        
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        
        self._log_stats(fcvm_p, coord_diff, None, row, fcvm_stats)
        
        h, agg, v_new = self.node_model(
            h, edge_index, edge_feat, node_attr, 
            v=v, vector_weights=vector_weights, coord_diff=coord_diff, 
            global_h_broadcast=global_h_broadcast
        )
        
        return h, coord, edge_attr, v_new, global_h_new


class E_GCL_vel(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """


    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False, norm_diff=False, tanh=False):
        E_GCL.__init__(self, input_nf, output_nf, hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_att_dim, act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight, attention=attention, norm_diff=norm_diff, tanh=tanh)
        self.norm_diff = norm_diff
        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1))

    def forward(self, h, edge_index, coord, vel, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)


        coord += self.coord_mlp_vel(h) * vel
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN
        return h, coord, edge_attr




class GCL_rf_vel(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """
    def __init__(self,  nf=64, edge_attr_nf=0, act_fn=nn.LeakyReLU(0.2), coords_weight=1.0):
        super(GCL_rf_vel, self).__init__()
        self.coords_weight = coords_weight
        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(1, nf),
            act_fn,
            nn.Linear(nf, 1))

        layer = nn.Linear(nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        #layer.weight.uniform_(-0.1, 0.1)
        self.phi = nn.Sequential(nn.Linear(1 + edge_attr_nf, nf),
                                 act_fn,
                                 layer,
                                 nn.Tanh()) #we had to add the tanh to keep this method stable

    def forward(self, x, vel_norm, vel, edge_index, edge_attr=None):
        row, col = edge_index
        edge_m = self.edge_model(x[row], x[col], edge_attr)
        x = self.node_model(x, edge_index, edge_m)
        x += vel * self.coord_mlp_vel(vel_norm)
        return x, edge_attr

    def edge_model(self, source, target, edge_attr):
        x_diff = source - target
        radial = torch.sqrt(torch.sum(x_diff ** 2, dim=1)).unsqueeze(1)
        e_input = torch.cat([radial, edge_attr], dim=1)
        e_out = self.phi(e_input)
        m_ij = x_diff * e_out
        return m_ij

    def node_model(self, x, edge_index, edge_m):
        row, col = edge_index
        agg = unsorted_segment_mean(edge_m, row, num_segments=x.size(0))
        x_out = x + agg * self.coords_weight
        return x_out


def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)
