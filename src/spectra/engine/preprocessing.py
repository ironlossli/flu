# preprocessing_extended.py
import csv
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None
import torch
from rdkit import Chem
from rdkit.Chem import rdchem

logger = logging.getLogger(__name__)

# ---------- 3D Solute Spec (kept consistent with your original pipeline) ----------
# 12 common elements; keep aligned with original 2D one-hot definition
ATOM_TYPES_3D = ["C", "N", "O", "S", "F", "Cl", "Br", "H", "P", "B", "Si", "I"]

# 2D Solvent atom types (example set; compact 14-dim total with degree+aromatic)
ATOM_TYPES_2D = ["C", "N", "O", "S", "F", "Cl", "Br", "P"]  # 8 dims


def parse_xyz(xyz_path: str) -> Tuple[List[str], np.ndarray]:
    """
    Robustly parse an XYZ file, returning (symbols, positions [N,3]).
    Tries standard XYZ format (first two header lines) then falls back to parse all lines.
    """
    p = Path(xyz_path)
    if not p.exists():
        raise FileNotFoundError(f"XYZ not found: {xyz_path}")
    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]

    def try_block(lines_block: List[str]) -> Tuple[List[str], np.ndarray]:
        symbols, coords = [], []
        for ln in lines_block:
            toks = ln.split()
            if len(toks) < 4:
                continue
            sym = toks[0]
            try:
                x, y, z = float(toks[1]), float(toks[2]), float(toks[3])
            except Exception:
                continue
            symbols.append(sym)
            coords.append([x, y, z])
        if not symbols:
            raise ValueError("No atom lines parsed from XYZ")
        return symbols, np.asarray(coords, dtype=np.float32)

    if len(lines) >= 3:
        try:
            int(lines[0])  # standard XYZ: first line = atom count
            symbols, pos = try_block(lines[2:])
            return symbols, pos
        except Exception:
            pass

    symbols, pos = try_block(lines)
    return symbols, pos


def atom_features_from_symbol(sym: str) -> np.ndarray:
    """
    返回原子序数 Z (int64)，形状为 [1]。
    兼容 SchNet/PaiNN/LEFTNet (需要 Long) 和 EGNN (Adapter会自动处理 Long)。
    """
    # 简单的原子序数映射表
    # ATOM_TYPES_3D = ["C", "N", "O", "S", "F", "Cl", "Br", "H", "P", "B", "Si", "I"]
    z_map = {
        "H": 1,  "B": 5,  "C": 6,  "N": 7,  "O": 8,  "F": 9,
        "Si": 14, "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53
    }
    z = z_map.get(sym, 0) # 0 代表未知
    return np.array([z], dtype=np.int64)


def build_edges_by_cutoff(pos: np.ndarray, cutoff: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Undirected-by-distance -> store as directed pair edges.
    Returns:
      - edge_index: [2, E]
      - edge_attr:  [E, 4] with [dist, ux, uy, uz] (unit vector)
    """
    n = pos.shape[0]
    if n <= 1:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

    edge_src, edge_dst, attrs = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            v = pos[j] - pos[i]
            d = float(np.linalg.norm(v))
            if d <= cutoff:
                if d > 1e-8:
                    u = v / d
                else:
                    u = np.zeros(3, dtype=np.float32)
                # i->j
                edge_src.append(i)
                edge_dst.append(j)
                attrs.append([d, float(u[0]), float(u[1]), float(u[2])])
                # j->i
                edge_src.append(j)
                edge_dst.append(i)
                attrs.append([d, float(-u[0]), float(-u[1]), float(-u[2])])

    if not edge_src:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

    edge_index = np.asarray([edge_src, edge_dst], dtype=np.int64)
    edge_attr = np.asarray(attrs, dtype=np.float32)
    return edge_index, edge_attr


# ---------- 2D Solvent Featurizer (RDKit) ----------
class Solvent2DFromSMILESFeaturizer:
    """
    Convert solvent SMILES to a 2D graph:
      - node: [N, 14]
      - edge_index: [2, E] directed
      - edge_attr: [E, 7]
    Feature schema (example):
      - Atom (14 dims): 8-dim elem one-hot (C,N,O,S,F,Cl,Br,P) + 5-dim degree one-hot (0..4, clipped) + 1 is_aromatic
      - Bond (7 dims): 4-dim bond type one-hot (single,double,triple,aromatic) + 1 conjugated + 1 in_ring + 1 has_stereo
    """

    def __init__(self):
        self.last_error_reason: Optional[str] = None

    @staticmethod
    def _featurize_atom_2d(atom: rdchem.Atom) -> np.ndarray:
        sym = atom.GetSymbol()
        elem_oh = np.zeros(len(ATOM_TYPES_2D), dtype=np.float32)
        if sym in ATOM_TYPES_2D:
            elem_oh[ATOM_TYPES_2D.index(sym)] = 1.0

        deg = atom.GetDegree()  # neighbors (heavy atoms)
        deg = int(max(0, min(4, deg)))
        deg_oh = np.zeros(5, dtype=np.float32)
        deg_oh[deg] = 1.0

        aromatic = 1.0 if atom.GetIsAromatic() else 0.0

        return np.concatenate([elem_oh, deg_oh, np.asarray([aromatic], dtype=np.float32)], axis=0)  # [14]

    @staticmethod
    def _featurize_bond_2d(bond: rdchem.Bond) -> np.ndarray:
        # 4-dim type one-hot
        bt = bond.GetBondType()
        is_arom = bond.GetIsAromatic() or (bt == rdchem.BondType.AROMATIC)

        bt_oh = np.zeros(4, dtype=np.float32)  # [single, double, triple, aromatic]
        if is_arom:
            bt_oh[3] = 1.0
        else:
            if bt == rdchem.BondType.SINGLE:
                bt_oh[0] = 1.0
            elif bt == rdchem.BondType.DOUBLE:
                bt_oh[1] = 1.0
            elif bt == rdchem.BondType.TRIPLE:
                bt_oh[2] = 1.0
            else:
                # leave all-zero for uncommon types
                pass

        conj = 1.0 if bond.GetIsConjugated() else 0.0
        in_ring = 1.0 if bond.IsInRing() else 0.0
        stereo = 1.0 if bond.GetStereo() != rdchem.BondStereo.STEREONONE else 0.0

        return np.concatenate([bt_oh, np.asarray([conj, in_ring, stereo], dtype=np.float32)], axis=0)  # [7]

    def featurize_smiles(self, smiles: str) -> Optional[Dict[str, np.ndarray]]:
        self.last_error_reason = None
        if not isinstance(smiles, str) or not smiles.strip():
            self.last_error_reason = "missing solvent_smiles"
            return None

        try:
            mol = Chem.MolFromSmiles(smiles)
        except Exception as e:
            mol = None
            self.last_error_reason = f"MolFromSmiles failed: {e}"

        if mol is None:
            if self.last_error_reason is None:
                self.last_error_reason = "MolFromSmiles returned None"
            return None

        # Node features
        node_feats = [self._featurize_atom_2d(a) for a in mol.GetAtoms()]
        if not node_feats:
            self.last_error_reason = "no atoms from SMILES"
            return None
        node = np.asarray(node_feats, dtype=np.float32)  # [N,14]

        # Edges: undirected bonds -> directed pairs
        src, dst, eattr = [], [], []
        for b in mol.GetBonds():
            i = b.GetBeginAtomIdx()
            j = b.GetEndAtomIdx()
            attr = self._featurize_bond_2d(b)
            # i -> j
            src.append(i)
            dst.append(j)
            eattr.append(attr)
            # j -> i
            src.append(j)
            dst.append(i)
            eattr.append(attr)

        if len(src) == 0:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0, 7), dtype=np.float32)
        else:
            edge_index = np.asarray([src, dst], dtype=np.int64)
            edge_attr = np.asarray(eattr, dtype=np.float32)

        return {"node": node, "edge_index": edge_index, "edge_attr": edge_attr}


# ---------- Combined Featurizer (3D solute + 2D solvent) ----------
class CombinedSoluteSolventFeaturizer:
    """
    Orchestrates:
      - XYZ -> 3D solute graph (cutoff)
      - SMILES -> 2D solvent graph (RDKit)
      - Extract targets (lambda_abs, lambda_em)
    Produces a single sample dict of tensors suitable for batching.
    """

    def __init__(self, cutoff_3d: float):
        self.cutoff_3d = float(cutoff_3d)
        self.solvent_fzr = Solvent2DFromSMILESFeaturizer()
        self.last_error_reason: Optional[str] = None

    def featurize_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self.last_error_reason = None

        # --- Solute 3D from XYZ ---
        xyz_path = record.get("xyz_path")
        try:
            symbols, pos = parse_xyz(xyz_path)
        except Exception as e:
            self.last_error_reason = f"parse_xyz failed: {e}"
            return None

        solute_list = [atom_features_from_symbol(sym) for sym in symbols]
        solute_x = np.concatenate(solute_list).astype(np.int64)

        eidx, eattr = build_edges_by_cutoff(pos, self.cutoff_3d)  # [2,E], [E,4]

        # --- Solvent 2D from SMILES ---
        solvent_smiles = record.get("solvent_smiles")
        solv = self.solvent_fzr.featurize_smiles(solvent_smiles)
        if solv is None:
            self.last_error_reason = f"solvent featurize failed: {self.solvent_fzr.last_error_reason}"
            return None

        # --- Targets ---
        def _to_float(v: Any) -> float:
            try:
                if v is None:
                    return 0.0
                f = float(v)
                if not np.isfinite(f):
                    return 0.0
                return f
            except Exception:
                return 0.0

        y_abs = _to_float(record.get("lambda_abs"))
        y_em = _to_float(record.get("lambda_em"))

        # --- Compose sample (torch tensors) ---
        sample: Dict[str, Any] = {
            # Solute (3D)
            "solute_x": torch.from_numpy(solute_x).long(),                  # [N,1] (Atomic numbers)
            "solute_pos": torch.from_numpy(pos).float(),                     # [N,3]
            "solute_edge_index": torch.from_numpy(eidx).long(),              # [2,E]
            "solute_edge_attr": torch.from_numpy(eattr).float(),             # [E,4]

            # Solvent (2D) - Now returning node features directly for Cross-Attention
            "solvent_x": torch.from_numpy(solv["node"]).float(),             # [Ns,14]
            "solvent_edge_index": torch.from_numpy(solv["edge_index"]).long(),  # [2,Es]
            "solvent_edge_attr": torch.from_numpy(solv["edge_attr"]).float(),   # [Es,7]

            # Targets
            "lambda_abs": y_abs,
            "lambda_em": y_em,

            # Metadata (optional)
            "xyz_path": xyz_path,
            "solute_smiles": record.get("smiles"),
            "solvent_smiles": solvent_smiles,
            "num_solute_atoms": int(solute_x.shape[0]),
            "num_solvent_atoms": int(solv["node"].shape[0]),
        }
        return sample


# ---------- Dataset Preprocessor with caching ----------
class DatasetPreprocessor:
    """
    Read parquet and build samples (3D solute + 2D solvent) for provided indices.
    Caches list[dict] to avoid repeated heavy preprocessing.
    """

    def __init__(self, parquet_path: str, cutoff_3d: float, max_error_logs: int = 30):
        self.parquet_path = Path(parquet_path)
        logger.info(f"Preprocessor reading: {self.parquet_path}")
        if pd is None:
            raise ImportError("pandas is required for DatasetPreprocessor.")
        self.df = pd.read_parquet(self.parquet_path)
        self.featurizer = CombinedSoluteSolventFeaturizer(cutoff_3d=cutoff_3d)
        self.max_error_logs = int(max_error_logs)

    def preprocess_split(self, indices: List[int], cache_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        cache_file: Optional[Path] = None
        if cache_dir:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            try:
                sig = hashlib.md5(np.asarray(sorted(indices), dtype=np.int64).tobytes()).hexdigest()[:8]
            except Exception:
                sig = f"{len(indices)}"
            cache_file = cache_dir / f"solute3d_solvent2d_cutoff{self.featurizer.cutoff_3d:.1f}_{len(indices)}_{sig}.pt"
            if cache_file.exists():
                logger.info(f"Loading cached: {cache_file}")
                return torch.load(cache_file)

        processed: List[Dict[str, Any]] = []
        failed = 0
        fail_rows: List[Tuple[int, str, str]] = []  # (df_index, xyz_path, reason)

        n = len(indices)
        report_every = max(200, n // 20) if n > 0 else 1

        logger.info(f"Preprocessing {n} samples (3D solute cutoff + 2D solvent RDKit)...")
        for k, idx in enumerate(indices, start=1):
            if idx < 0 or idx >= len(self.df):
                failed += 1
                if len(fail_rows) < 1000:
                    fail_rows.append((idx, "", "index out of bounds"))
                continue

            record = self.df.iloc[idx].to_dict()
            sample = self.featurizer.featurize_record(record)
            if sample is None:
                failed += 1
                reason = self.featurizer.last_error_reason or "unknown error"
                if failed <= self.max_error_logs:
                    logger.warning(f"[{failed}] preprocess failed idx={idx} xyz={record.get('xyz_path')} solvent={record.get('solvent_smiles')}: {reason}")
                if len(fail_rows) < 10000:
                    fail_rows.append((idx, str(record.get("xyz_path")), reason))
                continue

            processed.append(sample)

            if k % report_every == 0 or k == n:
                logger.info(f"  progress {k}/{n} | ok={len(processed)} | fail={failed}")

        logger.info(f"Preprocess done. ok={len(processed)} fail={failed}")

        if cache_dir is not None and fail_rows:
            fail_csv = Path(cache_dir) / "preprocess_failures.csv"
            with open(fail_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["df_index", "xyz_path", "reason"])
                w.writerows(fail_rows)
            logger.info(f"Failure details saved to: {fail_csv}")

        if cache_file is not None:
            torch.save(processed, cache_file)
            logger.info(f"Cached to: {cache_file}")

        return processed
