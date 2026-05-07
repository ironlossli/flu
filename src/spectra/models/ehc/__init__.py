# 导出模型类
from .ehc_model import EHCModel

from .ehc_model import (
    EHCModel,
    SolventConfig,
    BackboneConfig,
    FiLMConfig,
    FBConfig,
    HeadConfig,
)

def _read_mgil_nonlocal(model_cfg, default_scale: float) -> dict:
    return {
        "use_nonlocal": bool(model_cfg.get("mgil_use_nonlocal", False)),
        "nonlocal_heads": model_cfg.get("mgil_nonlocal_heads", 4),
        "nonlocal_features": model_cfg.get("mgil_nonlocal_features", 64),
        "nonlocal_dropout": model_cfg.get("mgil_nonlocal_dropout", 0.0),
        "nonlocal_scale": model_cfg.get("mgil_nonlocal_scale", default_scale),
    }

# EGNN 版
def build_ehc_egnn(model_cfg, data_cfg=None, train_cfg=None):
    m = model_cfg

    solvent_cfg = SolventConfig(
        node_in=m.get("solvent_node_in", 14),
        edge_in=m.get("solvent_edge_in", 7),
        z_dim=m.get("z_dim", 128),
        layers=m.get("solvent_layers", 3),
    )

    # FiLM config：用于 EHCModel 运行时构建 ConditionContext
    film_cfg = FiLMConfig(
        deep_film=False,
        film_every=m.get("film_every", 1),
        film_layers=m.get("film_layers", None),
        film_beta_only=m.get("film_beta_only", False),
        film_scale=m.get("film_scale", 0.7),
    )

    # Backbone adapter kwargs：完全对齐 EGNN_GBM_Adapter 的构造签名
    # 已移除: deep_film, film_every 等运行时策略参数
    egnn_hidden = m.get("egnn_hidden", 256)
    egnn_layers = m.get("egnn_layers", 4)
    
    edge_attr_mode = m.get("edge_attr_mode", "legacy")
    edge_attr_dim = m.get("solute_edge_in", 4)
    # If invariant_only, we strip ux,uy,uz leaving only distance d (1-dim)
    if edge_attr_mode == "invariant_only" and edge_attr_dim > 1:
        edge_attr_dim = 1

    use_backbone_dropout = bool(m.get("use_backbone_dropout", True))
    backbone_dropout = float(m.get("backbone_dropout", 0.1)) if use_backbone_dropout else 0.0

    mgil_nonlocal = _read_mgil_nonlocal(m, default_scale=1.0)
    adapter_kwargs = dict(
        node_in_dim=m.get("solute_node_in", 12),
        edge_attr_dim=edge_attr_dim, # Note: This might be ignored by EGNN_GBM_Adapter now
        hidden_dim=egnn_hidden,
        num_layers=egnn_layers,
        z_dim=solvent_cfg.z_dim,
        use_scalar_film=True,
        use_edge_film=m.get("egnn_use_edge_film", True),
        query_layer_index=m.get("egnn_query_layer_index", 1),
        use_nonlocal=mgil_nonlocal["use_nonlocal"],
        nonlocal_heads=mgil_nonlocal["nonlocal_heads"],
        nonlocal_features=mgil_nonlocal["nonlocal_features"],
        nonlocal_dropout=mgil_nonlocal["nonlocal_dropout"],
        nonlocal_scale=mgil_nonlocal["nonlocal_scale"],
        edge_attr_mode=edge_attr_mode,
        use_fcvm=m.get("use_fcvm", False),
        chirality_invariant=m.get("chirality_invariant", False),
        fcvm_update_frames=m.get("fcvm_update_frames", "once"),
        clamp=m.get("clamp", False),
        coords_weight=m.get("coords_weight", 1.0),
        use_rbf=m.get("egnn_use_rbf", False),
        n_rbf=m.get("egnn_rbf", 20),
        cutoff=m.get("egnn_cutoff", 5.0),
        log_fcvm=m.get("log_fcvm", False),
        log_config=m.get("log_config", {}),
        use_moments=m.get("use_moments", False),
        use_global=m.get("use_global", True),
        use_virtual_node=m.get("use_virtual_node", True),
        use_advanced_geometry=m.get("use_advanced_geometry", True),
        dropout=backbone_dropout,
    )
    
    backbone_cfg = BackboneConfig(
        name="egnn",
        adapter_kwargs=adapter_kwargs,
        hidden_dim=egnn_hidden,
    )

    fb_cfg = FBConfig(
        enabled=m.get("fb_enabled", False),
        roles=m.get("roles", 12),
        d_k=m.get("d_k", 64),
        d_v=m.get("d_v", 64),
        fb_cutoff=m.get("fb_cutoff", None),
        use_updated_coords_for_fb=m.get("use_updated_coords_for_fb", False),
        use_geom=m.get("fb_use_geom", False),
    )

    head_cfg = HeadConfig(
        hidden_dim=m.get("head_hidden", 256),
        energy_space=m.get("energy_space", True),
        use_backbone_readout=m.get("use_backbone_readout", False),
        readout_type=m.get("readout_type", "mean"),
    )

    return EHCModel(
        solvent_cfg=solvent_cfg,
        backbone_cfg=backbone_cfg,
        film_cfg=film_cfg,
        fb_cfg=fb_cfg,
        head_cfg=head_cfg,
        use_concat_baseline=m.get("use_concat_baseline", False),
    )

# SchNet 版
def build_ehc_schnet(model_cfg, data_cfg=None, train_cfg=None):
    m = model_cfg

    solvent_cfg = SolventConfig(
        node_in=m.get("solvent_node_in", 14),
        edge_in=m.get("solvent_edge_in", 7),
        z_dim=m.get("z_dim", 128),
        layers=m.get("solvent_layers", 3),
    )

    film_cfg = FiLMConfig(
        deep_film=False,
        film_every=m.get("film_every", 1),
        film_layers=m.get("film_layers", None),
        film_beta_only=m.get("film_beta_only", False),
        film_scale=m.get("film_scale", 0.7),
    )

    schnet_hidden = m.get("schnet_hidden", 128)

    # 已移除: use_edge_film, use_vector_gate (SchNet 不支持)
    mgil_nonlocal = _read_mgil_nonlocal(m, default_scale=0.1)

    # 已移除: deep_film 等策略参数
    adapter_kwargs = dict(
        z_dim=solvent_cfg.z_dim,
        n_atom_basis=schnet_hidden,
        n_filters=m.get("schnet_filters", 128),
        n_interactions=m.get("schnet_layers", 6),
        n_gaussians=m.get("schnet_gaussians", 100),
        cutoff=m.get("schnet_cutoff", 5.0),
        use_scalar_film=True,
        query_layer_index=m.get("schnet_query_layer_index", 1),
        use_geom=m.get("schnet_use_geom", False),
        geom_n_rbf=m.get("schnet_rbf", 20),
        geom_use_moments=m.get("schnet_use_moments", False),
        geom_use_global=m.get("schnet_use_global", False),
        use_nonlocal=mgil_nonlocal["use_nonlocal"],
        nonlocal_heads=mgil_nonlocal["nonlocal_heads"],
        nonlocal_features=mgil_nonlocal["nonlocal_features"],
        nonlocal_dropout=mgil_nonlocal["nonlocal_dropout"],
        nonlocal_scale=mgil_nonlocal["nonlocal_scale"],
    )
    
    backbone_cfg = BackboneConfig(
        name="schnet",
        adapter_kwargs=adapter_kwargs,
        hidden_dim=schnet_hidden,
    )

    fb_cfg = FBConfig(
        enabled=m.get("fb_enabled", False),
        roles=m.get("roles", 12),
        d_k=m.get("d_k", 64),
        d_v=m.get("d_v", 64),
        fb_cutoff=m.get("fb_cutoff", None),
        use_updated_coords_for_fb=m.get("use_updated_coords_for_fb", False),
        use_geom=m.get("fb_use_geom", False),
    )

    head_cfg = HeadConfig(
        hidden_dim=m.get("head_hidden", 256),
        energy_space=m.get("energy_space", True),
        use_backbone_readout=m.get("use_backbone_readout", False),
    )

    return EHCModel(solvent_cfg, backbone_cfg, film_cfg, fb_cfg, head_cfg)

# PaiNN 版
def build_ehc_painn(model_cfg, data_cfg=None, train_cfg=None):
    m = model_cfg

    solvent_cfg = SolventConfig(
        node_in=m.get("solvent_node_in", 14),
        edge_in=m.get("solvent_edge_in", 7),
        z_dim=m.get("z_dim", 128),
        layers=m.get("solvent_layers", 3),
    )

    film_cfg = FiLMConfig(
        deep_film=False,
        film_every=m.get("film_every", 1),
        film_layers=m.get("film_layers", None),
        film_beta_only=m.get("film_beta_only", False),
        film_scale=m.get("film_scale", 0.7),
    )

    painn_hidden = m.get("painn_hidden", 128)
    use_backbone_dropout = bool(m.get("use_backbone_dropout", True))
    backbone_dropout = float(m.get("backbone_dropout", 0.1)) if use_backbone_dropout else 0.0

    # 已移除: use_edge_film (PaiNN 不支持)
    # 已移除: deep_film 等策略参数
    mgil_nonlocal = _read_mgil_nonlocal(m, default_scale=1.0)
    adapter_kwargs = dict(
        z_dim=solvent_cfg.z_dim,
        hidden=painn_hidden,
        num_interactions=m.get("painn_layers", 6),
        n_rbf=m.get("painn_rbf", 64),
        cutoff=m.get("painn_cutoff", 5.0),
        shared_interactions=False,
        shared_filters=False,
        use_scalar_film=True,
        query_layer_index=m.get("painn_query_layer_index", 1),
        use_virtual_node=m.get("use_virtual_node", True),
        use_nonlocal=mgil_nonlocal["use_nonlocal"],
        nonlocal_heads=mgil_nonlocal["nonlocal_heads"],
        nonlocal_features=mgil_nonlocal["nonlocal_features"],
        nonlocal_dropout=mgil_nonlocal["nonlocal_dropout"],
        nonlocal_scale=mgil_nonlocal["nonlocal_scale"],
        dropout=backbone_dropout,
        residual_scale=m.get("painn_res_scale", 1.0),
        use_geom_gate=m.get("painn_use_geom_gate", False),
        geom_n_rbf=m.get("painn_geom_rbf", 20),
        geom_use_moments=m.get("painn_geom_use_moments", True),
        geom_use_global=m.get("painn_geom_use_global", True),
        geom_edge_scale=m.get("painn_geom_edge_scale", 0.1),
    )
    
    backbone_cfg = BackboneConfig(
        name="painn",
        adapter_kwargs=adapter_kwargs,
        hidden_dim=painn_hidden,
    )

    fb_cfg = FBConfig(
        enabled=m.get("fb_enabled", False),
        roles=m.get("roles", 12),
        d_k=m.get("d_k", 64),
        d_v=m.get("d_v", 64),
        fb_cutoff=m.get("fb_cutoff", None),
        use_updated_coords_for_fb=m.get("use_updated_coords_for_fb", False),
        use_geom=m.get("fb_use_geom", False),
    )

    head_cfg = HeadConfig(
        hidden_dim=m.get("head_hidden", 256),
        energy_space=m.get("energy_space", True),
        use_backbone_readout=m.get("use_backbone_readout", False),
    )

    return EHCModel(
        solvent_cfg=solvent_cfg,
        backbone_cfg=backbone_cfg,
        film_cfg=film_cfg,
        fb_cfg=fb_cfg,
        head_cfg=head_cfg,
        use_concat_baseline=m.get("use_concat_baseline", False),
    )

# Equiformer v1 版
def build_ehc_equiformer(model_cfg, data_cfg=None, train_cfg=None):
    m = model_cfg

    solvent_cfg = SolventConfig(
        node_in=m.get("solvent_node_in", 14),
        edge_in=m.get("solvent_edge_in", 7),
        z_dim=m.get("z_dim", 128),
        layers=m.get("solvent_layers", 3),
    )

    film_cfg = FiLMConfig(
        deep_film=False,
        film_every=m.get("film_every", 1),
        film_layers=m.get("film_layers", None),
        film_beta_only=m.get("film_beta_only", False),
        film_scale=m.get("film_scale", 1.0),
    )

    mgil_nonlocal = _read_mgil_nonlocal(m, default_scale=0.1)

    # 已移除: deep_film 等策略参数
    adapter_kwargs = dict(
        z_dim=solvent_cfg.z_dim,
        variant=m.get("eqf_variant", "gat"),
        max_radius=m.get("eqf_radius", 5.0),
        max_neighbors=m.get("eqf_max_neighbors", 1000),
        num_layers=m.get("eqf_layers", 6),
        number_of_basis=m.get("eqf_num_basis", 128),
        irreps_node_embedding=m.get("eqf_irreps_node_embedding", "128x0e+64x1e+32x2e"),
        irreps_feature=m.get("eqf_irreps_feature", "512x0e"),
        irreps_head=m.get("eqf_irreps_head", "32x0e+16x1o+8x2e"),
        num_heads=m.get("eqf_num_heads", 4),
        norm_layer=m.get("eqf_norm_layer", "layer"),
        basis_type=m.get("eqf_basis_type", "gaussian"),
        fc_neurons=m.get("eqf_fc_neurons", None),
        irreps_mlp_mid=m.get("eqf_irreps_mlp_mid", "128x0e+64x1e+32x2e"),
        class_map=m.get("eqf_class_map", "12"),
        max_atom_type=m.get("eqf_max_atom_type", 12),
        query_layer_index=m.get("eqf_query_layer_index", 1),
        alpha_drop=m.get("eqf_alpha_drop", 0.2),
        proj_drop=m.get("eqf_proj_drop", 0.0),
        out_drop=m.get("eqf_out_drop", 0.0),
        drop_path_rate=m.get("eqf_drop_path_rate", 0.0),
        use_geom_gate=m.get("eqf_use_geom_gate", False),
        geom_n_rbf=m.get("eqf_geom_n_rbf", 20),
        geom_use_moments=m.get("eqf_geom_use_moments", True),
        geom_use_global=m.get("eqf_geom_use_global", True),
        geom_gate_scale=m.get("eqf_geom_gate_scale", 0.1),
        geom_use_ln=m.get("eqf_geom_use_ln", True),
        use_nonlocal=mgil_nonlocal["use_nonlocal"],
        nonlocal_heads=mgil_nonlocal["nonlocal_heads"],
        nonlocal_features=mgil_nonlocal["nonlocal_features"],
        nonlocal_dropout=mgil_nonlocal["nonlocal_dropout"],
        nonlocal_scale=mgil_nonlocal["nonlocal_scale"],
        use_scalar_film=True,
    )
    
    # hidden 维推断暂时保留硬编码或从配置读取
    eq_hidden = 512 # 默认值
    
    backbone_cfg = BackboneConfig(
        name="equiformer",
        adapter_kwargs=adapter_kwargs,
        hidden_dim=eq_hidden,
    )

    fb_cfg = FBConfig(
        enabled=m.get("fb_enabled", False),
        roles=m.get("roles", 12),
        d_k=m.get("d_k", 64),
        d_v=m.get("d_v", 64),
        fb_cutoff=m.get("fb_cutoff", None),
        use_updated_coords_for_fb=m.get("use_updated_coords_for_fb", False),
        use_geom=m.get("fb_use_geom", False),
    )

    head_cfg = HeadConfig(
        hidden_dim=m.get("head_hidden", 256),
        energy_space=m.get("energy_space", True),
        use_backbone_readout=m.get("use_backbone_readout", False),
    )

    return EHCModel(solvent_cfg, backbone_cfg, film_cfg, fb_cfg, head_cfg)

# EquiformerV2 (OC20 风格) 版
def build_ehc_equiformer_v2(model_cfg, data_cfg=None, train_cfg=None):
    m = model_cfg

    def _as_tuple(x, default):
        if x is None:
            return default
        return tuple(x) if isinstance(x, (list, tuple)) else (x,)

    eq2_lmax_list = _as_tuple(m.get("eq2_lmax_list", [6]), (6,))
    eq2_mmax_list = _as_tuple(m.get("eq2_mmax_list", [2]), (2,))

    solvent_cfg = SolventConfig(
        node_in=m.get("solvent_node_in", 14),
        edge_in=m.get("solvent_edge_in", 7),
        z_dim=m.get("z_dim", 128),
        layers=m.get("solvent_layers", 3),
    )

    film_cfg = FiLMConfig(
        deep_film=False,
        film_every=m.get("film_every", 1),
        film_layers=m.get("film_layers", None),
        film_beta_only=m.get("film_beta_only", False),
        film_scale=m.get("film_scale", 1.0),
    )

    # 已移除: use_edge_film, use_vector_gate
    # 已移除: deep_film 等策略参数
    mgil_nonlocal = _read_mgil_nonlocal(m, default_scale=0.1)

    adapter_kwargs = dict(
        num_atoms=m.get("eq2_num_atoms", 0),
        bond_feat_dim=m.get("eq2_bond_feat_dim", 0),
        num_targets=m.get("eq2_num_targets", 1),
        z_dim=solvent_cfg.z_dim,
        cutoff=m.get("eq2_cutoff", 5.0),
        max_num_elements=m.get("eq2_max_num_elements", 100),
        max_neighbors=m.get("eq2_max_neighbors", 100),
        num_layers=m.get("eq2_layers", 12),
        sphere_channels=m.get("eq2_sphere_channels", 128),
        attn_hidden_channels=m.get("eq2_attn_hidden_channels", 128),
        num_heads=m.get("eq2_num_heads", 8),
        attn_alpha_channels=m.get("eq2_attn_alpha_channels", 32),
        attn_value_channels=m.get("eq2_attn_value_channels", 16),
        ffn_hidden_channels=m.get("eq2_ffn_hidden_channels", 512),
        norm_type=m.get("eq2_norm_type", "rms_norm_sh"),
        lmax_list=list(eq2_lmax_list),
        mmax_list=list(eq2_mmax_list),
        grid_resolution=m.get("eq2_grid_resolution", None),
        num_sphere_samples=m.get("eq2_num_sphere_samples", 128),
        edge_channels=m.get("eq2_edge_channels", 128),
        use_atom_edge_embedding=m.get("eq2_use_atom_edge_embedding", True),
        share_atom_edge_embedding=m.get("eq2_share_atom_edge_embedding", False),
        use_m_share_rad=m.get("eq2_use_m_share_rad", False),
        distance_function=m.get("eq2_distance_function", "gaussian"),
        attn_activation=m.get("eq2_attn_activation", "scaled_silu"),
        use_s2_act_attn=m.get("eq2_use_s2_act_attn", False),
        use_attn_renorm=m.get("eq2_use_attn_renorm", True),
        ffn_activation=m.get("eq2_ffn_activation", "scaled_silu"),
        use_gate_act=m.get("eq2_use_gate_act", False),
        use_grid_mlp=m.get("eq2_use_grid_mlp", False),
        use_sep_s2_act=m.get("eq2_use_sep_s2_act", True),
        alpha_drop=m.get("eq2_alpha_drop", 0.1),
        drop_path_rate=m.get("eq2_drop_path_rate", 0.05),
        proj_drop=m.get("eq2_proj_drop", 0.0),
        weight_init=m.get("eq2_weight_init", "normal"),
        use_scalar_film=True,
        query_layer_index=m.get("eq2_query_layer_index", 1),
        use_nonlocal=mgil_nonlocal["use_nonlocal"],
        nonlocal_heads=mgil_nonlocal["nonlocal_heads"],
        nonlocal_features=mgil_nonlocal["nonlocal_features"],
        nonlocal_dropout=mgil_nonlocal["nonlocal_dropout"],
        nonlocal_scale=mgil_nonlocal["nonlocal_scale"],
    )
    
    # hidden 维推断: sphere_channels * len(lmax_list)
    hidden_dim = adapter_kwargs["sphere_channels"] * len(adapter_kwargs["lmax_list"])
    backbone_cfg = BackboneConfig(
        name="equiformer_v2",
        adapter_kwargs=adapter_kwargs,
        hidden_dim=hidden_dim,
    )

    fb_cfg = FBConfig(
        enabled=m.get("fb_enabled", False),
        roles=m.get("roles", 12),
        d_k=m.get("d_k", 64),
        d_v=m.get("d_v", 64),
        fb_cutoff=m.get("fb_cutoff", None),
        use_updated_coords_for_fb=m.get("use_updated_coords_for_fb", False),
        use_geom=m.get("fb_use_geom", False),
    )

    head_cfg = HeadConfig(
        hidden_dim=m.get("head_hidden", 256),
        energy_space=m.get("energy_space", True),
        use_backbone_readout=m.get("use_backbone_readout", False),
    )

    return EHCModel(solvent_cfg, backbone_cfg, film_cfg, fb_cfg, head_cfg)

# LEFTNet 版
def build_ehc_leftnet(model_cfg, data_cfg=None, train_cfg=None):
    m = model_cfg

    solvent_cfg = SolventConfig(
        node_in=m.get("solvent_node_in", 14),
        edge_in=m.get("solvent_edge_in", 7),
        z_dim=m.get("z_dim", 128),
        layers=m.get("solvent_layers", 3),
    )

    film_cfg = FiLMConfig(
        deep_film=False,
        film_every=m.get("film_every", 1),
        film_layers=m.get("film_layers", None),
        film_beta_only=m.get("film_beta_only", False),
        film_scale=m.get("film_scale", 1.0),
    )

    leftnet_hidden = m.get("leftnet_hidden", 256)
    use_backbone_dropout = bool(m.get("use_backbone_dropout", True))
    backbone_dropout = float(m.get("backbone_dropout", 0.1)) if use_backbone_dropout else 0.0

    mgil_nonlocal = _read_mgil_nonlocal(m, default_scale=0.1)

    # 已移除: use_edge_film, use_vector_gate
    # 已移除: deep_film 等策略参数
    adapter_kwargs = dict(
        z_dim=solvent_cfg.z_dim,
        hidden=leftnet_hidden,
        num_layers=m.get("leftnet_layers", 4),
        num_radial=m.get("leftnet_num_radial", 32),
        cutoff=m.get("leftnet_cutoff", 5.0),
        use_scalar_film=True,
        query_layer_index=m.get("leftnet_query_layer_index", 1),
        use_lse=m.get("leftnet_use_lse", True),
        use_fte=m.get("leftnet_use_fte", True),
        use_vector_features=True,
        dropout=backbone_dropout,
        use_geom_gate=m.get("leftnet_use_geom_gate", False),
        geom_n_rbf=m.get("leftnet_geom_n_rbf", None),
        geom_use_moments=m.get("leftnet_geom_use_moments", True),
        geom_use_global=m.get("leftnet_geom_use_global", True),
        geom_gate_scale=m.get("leftnet_geom_gate_scale", 0.1),
        geom_use_ln=m.get("leftnet_geom_use_ln", True),
        use_nonlocal=mgil_nonlocal["use_nonlocal"],
        nonlocal_heads=mgil_nonlocal["nonlocal_heads"],
        nonlocal_features=mgil_nonlocal["nonlocal_features"],
        nonlocal_dropout=mgil_nonlocal["nonlocal_dropout"],
        nonlocal_scale=mgil_nonlocal["nonlocal_scale"],
    )
    
    backbone_cfg = BackboneConfig(
        name="leftnet",
        adapter_kwargs=adapter_kwargs,
        hidden_dim=leftnet_hidden,
    )

    fb_cfg = FBConfig(
        enabled=m.get("fb_enabled", False),
        roles=m.get("roles", 12),
        d_k=m.get("d_k", 64),
        d_v=m.get("d_v", 64),
        fb_cutoff=m.get("fb_cutoff", None),
        use_updated_coords_for_fb=m.get("use_updated_coords_for_fb", False),
        use_geom=m.get("fb_use_geom", False),
    )

    head_cfg = HeadConfig(
        hidden_dim=m.get("head_hidden", 256),
        energy_space=m.get("energy_space", True),
        use_backbone_readout=m.get("use_backbone_readout", False),
    )

    return EHCModel(
        solvent_cfg=solvent_cfg,
        backbone_cfg=backbone_cfg,
        film_cfg=film_cfg,
        fb_cfg=fb_cfg,
        head_cfg=head_cfg,
        use_concat_baseline=m.get("use_concat_baseline", False),
    )

# LEFTNet V3 版
def build_ehc_leftnet_v3(model_cfg, data_cfg=None, train_cfg=None):
    raise NotImplementedError(
        "build_ehc_leftnet_v3 is disabled during strict config unification."
    )
    m = model_cfg

    solvent_cfg = SolventConfig(
        node_in=m.get("solvent_node_in", 14),
        edge_in=m.get("solvent_edge_in", 7),
        z_dim=m.get("z_dim", 128),
        layers=m.get("solvent_layers", 3),
    )

    film_cfg = FiLMConfig(
        deep_film=False,
        film_every=m.get("film_every", 1),
        film_layers=m.get("film_layers", None),
        film_beta_only=m.get("film_beta_only", False),
        film_scale=m.get("film_scale", 1.0),
    )

    leftnet_hidden = m.get("leftnet_hidden", 256)
    use_backbone_dropout = bool(m.get("use_backbone_dropout", True))
    backbone_dropout = float(m.get("backbone_dropout", 0.1)) if use_backbone_dropout else 0.0

    mgil_nonlocal = _read_mgil_nonlocal(m, default_scale=0.1)

    adapter_kwargs = dict(
        z_dim=solvent_cfg.z_dim,
        hidden=leftnet_hidden,
        num_layers=m.get("leftnet_layers", 4),
        num_radial=m.get("leftnet_num_radial", 32),
        cutoff=m.get("leftnet_cutoff", 5.0),
        use_scalar_film=True,
        query_layer_index=m.get("leftnet_query_layer_index", 1),
        use_lse=m.get("leftnet_use_lse", True),
        use_fte=m.get("leftnet_use_fte", True),
        use_vector_features=True,
        use_uvec=m.get("leftnet_use_uvec", True),
        max_num_neighbors=m.get("leftnet_max_num_neighbors", 48),
        cond_dim=m.get("leftnet_cond_dim", None),
        dropout=backbone_dropout,
        use_geom_gate=m.get("leftnet_use_geom_gate", False),
        geom_n_rbf=m.get("leftnet_geom_n_rbf", None),
        geom_use_moments=m.get("leftnet_geom_use_moments", True),
        geom_use_global=m.get("leftnet_geom_use_global", True),
        geom_gate_scale=m.get("leftnet_geom_gate_scale", 0.1),
        geom_use_ln=m.get("leftnet_geom_use_ln", True),
        use_nonlocal=mgil_nonlocal["use_nonlocal"],
        nonlocal_heads=mgil_nonlocal["nonlocal_heads"],
        nonlocal_features=mgil_nonlocal["nonlocal_features"],
        nonlocal_dropout=mgil_nonlocal["nonlocal_dropout"],
        nonlocal_scale=mgil_nonlocal["nonlocal_scale"],
    )
    
    backbone_cfg = BackboneConfig(
        name="leftnet_v3",
        adapter_kwargs=adapter_kwargs,
        hidden_dim=leftnet_hidden,
    )

    fb_cfg = FBConfig(
        enabled=m.get("fb_enabled", False),
        roles=m.get("roles", 12),
        d_k=m.get("d_k", 64),
        d_v=m.get("d_v", 64),
        fb_cutoff=m.get("fb_cutoff", None),
        use_updated_coords_for_fb=m.get("use_updated_coords_for_fb", False),
        use_geom=m.get("fb_use_geom", False),
    )

    head_cfg = HeadConfig(
        hidden_dim=m.get("head_hidden", 256),
        energy_space=m.get("energy_space", True),
        use_backbone_readout=m.get("use_backbone_readout", False),
    )

    return EHCModel(
        solvent_cfg=solvent_cfg,
        backbone_cfg=backbone_cfg,
        film_cfg=film_cfg,
        fb_cfg=fb_cfg,
        head_cfg=head_cfg,
        use_concat_baseline=m.get("use_concat_baseline", False),
    )

# C-GNN (Clifford GNN) 版
def build_ehc_cgnn(model_cfg, data_cfg=None, train_cfg=None):
    raise NotImplementedError(
        "build_ehc_cgnn is disabled during strict config unification."
    )
    m = model_cfg

    solvent_cfg = SolventConfig(
        node_in=m.get("solvent_node_in", 14),
        edge_in=m.get("solvent_edge_in", 7),
        z_dim=m.get("z_dim", 128),
        layers=m.get("solvent_layers", 3),
    )

    film_cfg = FiLMConfig(
        deep_film=False,
        film_every=m.get("film_every", 1),
        film_layers=m.get("film_layers", None),
        film_beta_only=m.get("film_beta_only", False),
        film_scale=m.get("film_scale", 1.0),
    )

    cgnn_hidden = m.get("cgnn_hidden", 128)

    adapter_kwargs = dict(
        z_dim=solvent_cfg.z_dim,
        hidden=cgnn_hidden,
        num_layers=m.get("cgnn_layers", 4),
        use_scalar_film=True,
        query_layer_index=m.get("cgnn_query_layer_index", 1),
    )
    
    # Note: CGNN outputs 4x hidden dim features (s,v,b,t)
    # The adapter returns [N, 4C]. The head should expect 4C or adapter should project?
    # In Adapter, I returned node_feats as is [N, 4C].
    # So hidden_dim for backbone config should be 4 * cgnn_hidden
    
    backbone_cfg = BackboneConfig(
        name="cgnn",
        adapter_kwargs=adapter_kwargs,
        hidden_dim=4 * cgnn_hidden,
    )

    fb_cfg = FBConfig(
        enabled=m.get("fb_enabled", False),
        roles=m.get("roles", 12),
        d_k=m.get("d_k", 64),
        d_v=m.get("d_v", 64),
        fb_cutoff=m.get("fb_cutoff", None),
        use_updated_coords_for_fb=m.get("use_updated_coords_for_fb", False),
        use_geom=m.get("fb_use_geom", False),
    )

    head_cfg = HeadConfig(
        hidden_dim=m.get("head_hidden", 256),
        energy_space=m.get("energy_space", True),
        use_backbone_readout=m.get("use_backbone_readout", False),
    )

    return EHCModel(solvent_cfg, backbone_cfg, film_cfg, fb_cfg, head_cfg)


__all__ = [
    "EHCModel", 
    "build_ehc_egnn", "build_ehc_schnet", "build_ehc_painn",
    "build_ehc_equiformer", "build_ehc_equiformer_v2", "build_ehc_leftnet",
]
