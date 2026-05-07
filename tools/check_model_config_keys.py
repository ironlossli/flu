import argparse
from pathlib import Path

import yaml


COMMON_KEYS = {
    "name",
    "builder",
    "use_backbone_dropout",
    "backbone_dropout",
    "solute_node_in",
    "solute_edge_in",
    "solvent_node_in",
    "solvent_edge_in",
    "z_dim",
    "solvent_layers",
    "film_every",
    "film_layers",
    "film_beta_only",
    "film_scale",
    "head_hidden",
    "energy_space",
    "use_backbone_readout",
    "readout_type",
    "use_concat_baseline",
    "optimizer",
    "scheduler",
    "loss",
    "trainer",
    "loader",
    "fb_enabled",
    "roles",
    "d_k",
    "d_v",
    "fb_cutoff",
    "use_updated_coords_for_fb",
    "fb_use_geom",
    "mgil_use_nonlocal",
    "mgil_nonlocal_heads",
    "mgil_nonlocal_features",
    "mgil_nonlocal_dropout",
    "mgil_nonlocal_scale",
    "log_config",
}

BUILDER_KEYS = {
    "spectra.models.ehc:build_ehc_egnn": {
        "egnn_hidden",
        "egnn_layers",
        "egnn_cutoff",
        "egnn_use_edge_film",
        "egnn_query_layer_index",
        "edge_attr_mode",
        "use_fcvm",
        "chirality_invariant",
        "fcvm_update_frames",
        "clamp",
        "coords_weight",
        "egnn_use_rbf",
        "egnn_rbf",
        "log_fcvm",
        "use_moments",
        "use_global",
        "use_virtual_node",
    },
    "spectra.models.ehc:build_ehc_schnet": {
        "schnet_hidden",
        "schnet_filters",
        "schnet_layers",
        "schnet_gaussians",
        "schnet_cutoff",
        "schnet_query_layer_index",
        "schnet_use_geom",
        "schnet_rbf",
        "schnet_use_moments",
        "schnet_use_global",
    },
    "spectra.models.ehc:build_ehc_painn": {
        "painn_hidden",
        "painn_layers",
        "painn_rbf",
        "painn_cutoff",
        "painn_query_layer_index",
        "painn_res_scale",
        "painn_use_geom_gate",
        "painn_geom_rbf",
        "painn_geom_use_moments",
        "painn_geom_use_global",
        "painn_geom_edge_scale",
        "use_virtual_node",
    },
    "spectra.models.ehc:build_ehc_equiformer": {
        "eqf_variant",
        "eqf_radius",
        "eqf_max_neighbors",
        "eqf_layers",
        "eqf_num_basis",
        "eqf_basis_type",
        "eqf_irreps_node_embedding",
        "eqf_irreps_feature",
        "eqf_irreps_head",
        "eqf_irreps_mlp_mid",
        "eqf_num_heads",
        "eqf_norm_layer",
        "eqf_class_map",
        "eqf_max_atom_type",
        "eqf_query_layer_index",
        "eqf_alpha_drop",
        "eqf_proj_drop",
        "eqf_out_drop",
        "eqf_drop_path_rate",
        "eqf_fc_neurons",
        "eqf_use_geom_gate",
        "eqf_geom_n_rbf",
        "eqf_geom_use_moments",
        "eqf_geom_use_global",
        "eqf_geom_gate_scale",
        "eqf_geom_use_ln",
    },
    "spectra.models.ehc:build_ehc_equiformer_v2": {
        "eq2_num_atoms",
        "eq2_bond_feat_dim",
        "eq2_num_targets",
        "eq2_cutoff",
        "eq2_max_num_elements",
        "eq2_max_neighbors",
        "eq2_layers",
        "eq2_sphere_channels",
        "eq2_attn_hidden_channels",
        "eq2_num_heads",
        "eq2_attn_alpha_channels",
        "eq2_attn_value_channels",
        "eq2_ffn_hidden_channels",
        "eq2_norm_type",
        "eq2_lmax_list",
        "eq2_mmax_list",
        "eq2_grid_resolution",
        "eq2_num_sphere_samples",
        "eq2_edge_channels",
        "eq2_use_atom_edge_embedding",
        "eq2_share_atom_edge_embedding",
        "eq2_use_m_share_rad",
        "eq2_distance_function",
        "eq2_attn_activation",
        "eq2_use_s2_act_attn",
        "eq2_use_attn_renorm",
        "eq2_ffn_activation",
        "eq2_use_gate_act",
        "eq2_use_grid_mlp",
        "eq2_use_sep_s2_act",
        "eq2_alpha_drop",
        "eq2_drop_path_rate",
        "eq2_proj_drop",
        "eq2_weight_init",
        "eq2_query_layer_index",
    },
    "spectra.models.ehc:build_ehc_leftnet": {
        "leftnet_hidden",
        "leftnet_layers",
        "leftnet_num_radial",
        "leftnet_cutoff",
        "leftnet_query_layer_index",
        "leftnet_use_lse",
        "leftnet_use_fte",
        "leftnet_use_geom_gate",
        "leftnet_geom_n_rbf",
        "leftnet_geom_use_moments",
        "leftnet_geom_use_global",
        "leftnet_geom_gate_scale",
        "leftnet_geom_use_ln",
    },
    "spectra.models.ehc:build_ehc_leftnet_v3": {
        "leftnet_hidden",
        "leftnet_layers",
        "leftnet_num_radial",
        "leftnet_cutoff",
        "leftnet_query_layer_index",
        "leftnet_use_lse",
        "leftnet_use_fte",
        "leftnet_use_uvec",
        "leftnet_max_num_neighbors",
        "leftnet_cond_dim",
        "leftnet_use_geom_gate",
        "leftnet_geom_n_rbf",
        "leftnet_geom_use_moments",
        "leftnet_geom_use_global",
        "leftnet_geom_gate_scale",
        "leftnet_geom_use_ln",
    },
    "spectra.models.ehc:build_ehc_cgnn": {
        "cgnn_hidden",
        "cgnn_layers",
        "cgnn_query_layer_index",
    },
}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def find_unknown_keys(cfg: dict) -> list[str]:
    builder = cfg.get("builder")
    if not builder:
        return []
    allowed = set(COMMON_KEYS)
    allowed.update(BUILDER_KEYS.get(builder, set()))
    return sorted(k for k in cfg.keys() if k not in allowed)


def iter_configs(roots: list[Path]) -> list[Path]:
    out = []
    for root in roots:
        if root.exists():
            out.extend(sorted(root.rglob("*.yaml")))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots",
        nargs="*",
        default=[
            "configs/model",
            "configs/_archive/2026-01-08/model",
        ],
    )
    args = parser.parse_args()
    roots = [Path(p) for p in args.roots]
    configs = iter_configs(roots)
    has_unknown = False
    for path in configs:
        cfg = load_yaml(path)
        unknown = find_unknown_keys(cfg)
        if unknown:
            has_unknown = True
            print(f"{path}:")
            for key in unknown:
                print(f"  - {key}")
    return 1 if has_unknown else 0


if __name__ == "__main__":
    raise SystemExit(main())
