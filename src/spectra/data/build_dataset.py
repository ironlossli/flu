# build_dataset_random.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import logging
import json
import hashlib
import re
import argparse

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem

try:
    import modify_yaml as _my
    yaml_load = _my.safe_load
except Exception:
    try:
        import yaml as _yaml
        yaml_load = _yaml.safe_load
    except Exception as e:
        raise RuntimeError("Missing YAML loader. Please install: pip install pyyaml") from e

# Optional standardization (guarded import)
try:
    from rdkit.Chem.MolStandardize import rdMolStandardize
    _HAS_STD = True
except Exception:
    _HAS_STD = False

# ---------------------------
# Global policies
# ---------------------------
MIN_ATOMS = 10               # 剔除原子数少于此值的溶质
MIN_ELEMENT_COUNT = 10       # 稀有元素阈值：全局出现少于此次数的元素被视为稀有
FORBIDDEN_ELEMENTS = {'Fe', 'Se', 'Na'}  # 强制剔除包含这些元素的分子 (黑名单)
SOLVENT_MIN_COUNT = 10       # 过滤出现次数少于此值的溶剂（基于 canonical solvent_smiles）

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Single-responsibility builder:
    - Read config (data.yaml), clean and standardize data
    - Ensure/generate solute XYZ files
    - Produce per-task parquet and random split indices
    - All outputs strictly under: [root]/processed/flu/ and [root]/processed/flu/random/
    """

    TARGET_BOUNDS_DEFAULT = (200.0, 1100.0)

    # Base banned elements (will be unioned with FORBIDDEN_ELEMENTS)
    BANNED_ELEMENTS = {
        "Ag", "Ar", "Cd", "Ce", "Cu", "Gd", "Ge", "Hf", "Hg", "In", "K", "Li", "Lu",
        "Mg", "Mo", "Na", "Ni", "Nn", "Pb", "Pr", "Ru", "Se", "Sn", "Te", "Ti", "V",
        "W", "Zn", "Zr"
    }

    def __init__(self, cfg: Dict, config_dir: Path):
        self.cfg = cfg
        self.config_dir = config_dir

        # Resolve project root from cfg["root"] relative to config_dir
        cfg_root = Path(cfg.get("root", "."))
        if cfg_root.is_absolute():
            self.root = cfg_root
        else:
            cand = (self.config_dir / cfg_root)
            self.root = (cand if cand.exists() else (Path.cwd() / cfg_root)).resolve()
        
        self.output_dir = (self.root / "processed" / "flu").resolve()

        # Fixed output locations
        proc = self.cfg.get("processed", {})
        splits_dir_cfg = proc.get("splits_dir")
        if splits_dir_cfg:
            self.splits_dir = self._resolve_path(splits_dir_cfg).resolve()
        else:
            self.splits_dir = (self.output_dir / "random").resolve()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)

        # Will be populated in rename_and_standardize_columns
        self.desc_cols_out: List[str] = []

    # ---------------------------
    # Top-level build
    # ---------------------------
    def build(self):
        logger.info("=" * 60)
        logger.info("开始执行数据集构建流程 (data.yaml -> processed/flu)")
        logger.info("=" * 60)

        base = self.load_raw_and_clean()
        base = self.rename_and_standardize_columns(base)

        # Policies on standardized solute/solvent
        base = self.apply_min_atoms_filter(base)          # 溶质最小原子数
        base = self.apply_rare_element_filter(base)       # 溶质稀有元素
        base = self.apply_solvent_min_count_filter(base)  # 溶剂最小出现次数

        base = self.ensure_xyz_files(base)

        # Splitting config (strategy must be 'random')
        ratios, seed = self._ratios_from_cfg()
        logger.info(f"Split config -> ratios={ratios} seed={seed}")

        # Abs task
        df_abs = self.subset_for_target(base, "lambda_abs")
        self.save_parquet_task(df_abs, "abs")
        self.create_random_split_task(df_abs, "abs", ratios=ratios, seed=seed)

        # Em task
        df_em = self.subset_for_target(base, "lambda_em")
        self.save_parquet_task(df_em, "em")
        self.create_random_split_task(df_em, "em", ratios=ratios, seed=seed)

        logger.info("✓ 两个任务数据集已生成: processed/flu + processed/flu/random")

    # ---------------------------
    # Loading & cleaning
    # ---------------------------
    def load_raw_and_clean(self) -> pd.DataFrame:
        raw = self.cfg["raw"]
        table_path = self._resolve_path(raw["table"])
        if not table_path.exists():
            raise FileNotFoundError(f"未找到原始表: {table_path}")

        df = self._read_csv_auto(
            table_path,
            encoding=raw.get("table_encoding")
        )
        logger.info(f"✓ 读取原始表: {table_path} ({len(df)} 行)")

        # Optional connection file merge (xyz_file)
        conn_file = raw.get("connection_file")
        if conn_file:
            conn_path = self._resolve_path(conn_file)
            if not conn_path.exists():
                raise FileNotFoundError(f"未找到连接表: {conn_path}")
            conn_df = self._read_csv_auto(
                conn_path,
                usecols=["xyz_file"],
                encoding=raw.get("connection_encoding")
            )
            if len(conn_df) != len(df):
                raise ValueError(f"行数不匹配! 原始表 {len(df)} 行, 连接表 {len(conn_df)} 行。")
            df = pd.concat([df.reset_index(drop=True), conn_df.reset_index(drop=True)], axis=1)
            logger.info("✓ 已合并连接表（xyz_file）")

        smiles_col = raw.get("smiles_col", "SMILES")
        solvent_col = raw.get("solvent_col", "Solvent")
        if smiles_col not in df.columns:
            raise KeyError(f"缺失溶质列 {smiles_col}")
        if solvent_col not in df.columns:
            raise KeyError(f"缺失溶剂列 {solvent_col}")

        # Filter invalid solute
        before = len(df)
        df = df[df[smiles_col].apply(self._is_valid_smiles)]
        logger.info(f"过滤无效溶质 SMILES: {before - len(df)} 条")

        # Filter solute that contains banned/forbidden elements (union)
        before = len(df)
        df = df[~df[smiles_col].apply(self._contains_banned_elements)]
        all_banned = sorted(self._all_banned_elements())
        logger.info(f"过滤含禁用元素的溶质: {before - len(df)} 条 (禁用元素并集: {all_banned})")

        # Filter invalid solvent
        before = len(df)
        df = df[df[solvent_col].apply(self._is_valid_smiles)]
        logger.info(f"过滤无效溶剂 SMILES: {before - len(df)} 条")

        # Canonicalize/standardize solute/solvent SMILES
        df["smiles"] = df[smiles_col].apply(self._canonicalize_smiles)
        df["solvent_smiles"] = df[solvent_col].apply(self._canonicalize_smiles)

        return df.reset_index(drop=True)
    
    def _read_csv_auto(self, path: Path, usecols=None, encoding: Optional[str] = None) -> pd.DataFrame:
        """
        尝试多种常见编码读取 CSV；可显式传入 encoding 覆盖。
        """
        if encoding:
            return pd.read_csv(path, usecols=usecols, encoding=encoding)

        for enc in ("utf-8", "utf-8-sig", "gb18030", "cp936", "latin-1"):
            try:
                return pd.read_csv(path, usecols=usecols, encoding=enc)
            except UnicodeDecodeError:
                continue
        # 兜底
        return pd.read_csv(path, usecols=usecols, encoding="latin-1")
    
    def rename_and_standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        raw = self.cfg["raw"]

        # Map target columns -> standardized names
        t_abs = raw.get("target_cols", {}).get("abs")
        t_em = raw.get("target_cols", {}).get("em")
        if t_abs is None or t_em is None:
            raise KeyError("raw.target_cols.abs/em 未配置")

        rename_map: Dict[str, str] = {}
        if t_abs in df.columns:
            rename_map[t_abs] = "lambda_abs"
        if t_em in df.columns:
            rename_map[t_em] = "lambda_em"

        # Normalize descriptor column names to snake_case
        desc_in: List[str] = raw.get("descriptor_cols", [])
        for col in desc_in:
            if col in df.columns:
                rename_map[col] = self._normalize_descriptor_name(col)

        # Preserve NUM and xyz_file if present
        num_col = raw.get("num_col", "NUM")
        xyz_file_col = raw.get("xyz_file_col", "xyz_file")
        if num_col in df.columns:
            rename_map[num_col] = "NUM"
        if xyz_file_col in df.columns:
            rename_map[xyz_file_col] = "xyz_file"

        df = df.rename(columns=rename_map)

        # Record descriptor columns actually present
        self.desc_cols_out = [self._normalize_descriptor_name(c) for c in desc_in if c in df.columns]

        # Prepare xyz_path; if xyz_file present map to abs path under xyz_dir
        xyz_base = self._resolve_path(self.cfg["raw"]["xyz_dir"])
        if "xyz_file" in df.columns:
            df["xyz_path"] = df["xyz_file"].apply(lambda x: str((xyz_base / str(x)).resolve()) if pd.notna(x) else None)
        else:
            df["xyz_path"] = None

        return df

    # ---------------------------
    # Policies application
    # ---------------------------
    def apply_min_atoms_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply MIN_ATOMS policy on solute (canonicalized) smiles.
        """
        if "smiles" not in df.columns:
            return df
        before = len(df)
        df = df[df["smiles"].apply(lambda s: self._num_atoms_in_smiles(s) >= MIN_ATOMS)]
        logger.info(f"过滤原子数少于 {MIN_ATOMS} 的溶质: {before - len(df)} 条")
        return df.reset_index(drop=True)

    def apply_rare_element_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute global element occurrence counts across current solute set,
        mark elements with total count < MIN_ELEMENT_COUNT as rare,
        and drop molecules containing any rare element.
        """
        if "smiles" not in df.columns:
            return df
        counts = self._global_element_counts(df["smiles"])
        rare_elements: Set[str] = {el for el, c in counts.items() if c < MIN_ELEMENT_COUNT}
        if rare_elements:
            before = len(df)
            df = df[~df["smiles"].apply(lambda s: self._contains_any_element(s, rare_elements))]
            logger.info(
                f"过滤含稀有元素(阈值<{MIN_ELEMENT_COUNT}) 的溶质: {before - len(df)} 条；"
                f"稀有元素数: {len(rare_elements)}"
            )
        else:
            logger.info(f"未检测到稀有元素(阈值<{MIN_ELEMENT_COUNT})，不执行剔除。")
        return df.reset_index(drop=True)

    def apply_solvent_min_count_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out rows whose solvent (canonical solvent_smiles) appears fewer than SOLVENT_MIN_COUNT times.
        """
        if "solvent_smiles" not in df.columns:
            return df
        counts = df["solvent_smiles"].value_counts(dropna=True)
        rare_solvents = set(counts[counts < SOLVENT_MIN_COUNT].index)
        if not rare_solvents:
            logger.info(f"未检测到低频溶剂(阈值<{SOLVENT_MIN_COUNT})，不执行剔除。")
            return df
        before = len(df)
        df = df[~df["solvent_smiles"].isin(rare_solvents)].reset_index(drop=True)
        logger.info(
            f"过滤出现次数少于 {SOLVENT_MIN_COUNT} 次的溶剂: {before - len(df)} 条；"
            f"低频溶剂数: {len(rare_solvents)}"
        )
        return df

    # ---------------------------
    # XYZ generation / checking
    # ---------------------------
    def ensure_xyz_files(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure xyz exists; generate missing via RDKit 3D from solute SMILES.
        Output missing list to [root]/processed/flu/missing_xyz_files.csv
        """
        seed = int(self.cfg.get("splitting", {}).get("random_seed", 42))
        xyz_base = self._resolve_path(self.cfg["raw"]["xyz_dir"])
        xyz_base.mkdir(parents=True, exist_ok=True)

        # Assign filenames for rows without xyz_file/path
        need_name = df["xyz_path"].isna()
        if need_name.any():
            def _gen_name(row: pd.Series) -> str:
                # 优先 NUM；否则基于溶质 canonical smiles 做哈希（避免溶剂导致重复几何）
                if ("NUM" in row) and pd.notna(row["NUM"]):
                    try:
                        return f"mol_{int(row['NUM'])}.xyz"
                    except Exception:
                        pass
                s = f"{row.get('smiles','')}"
                h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]
                return f"mol_{h}.xyz"

            df.loc[need_name, "xyz_file"] = df.loc[need_name].apply(_gen_name, axis=1)
            df.loc[need_name, "xyz_path"] = df.loc[need_name, "xyz_file"].apply(lambda fn: str((xyz_base / fn).resolve()))

        exists_flags: List[bool] = []
        n_to_gen = 0
        n_ok = 0

        for _, row in df.iterrows():
            p = Path(row["xyz_path"]) if pd.notna(row["xyz_path"]) else None
            if p is None:
                exists_flags.append(False)
                continue
            if p.exists():
                exists_flags.append(True)
                continue

            # Always attempt generation if missing
            n_to_gen += 1
            ok = self._generate_xyz_from_smiles(row.get("smiles", None), p, seed=seed)
            exists_flags.append(ok)
            if ok:
                n_ok += 1

        df["xyz_exists"] = exists_flags
        logger.info(f"✓ XYZ 检查: 存在={int(df['xyz_exists'].sum())}, 缺失={int((~df['xyz_exists']).sum())}")
        if n_to_gen > 0:
            logger.info(f"尝试生成 XYZ: 目标={n_to_gen}, 成功={n_ok}, 失败={n_to_gen - n_ok}")

        # Save missing/failed list to fixed output
        missing = df[~df["xyz_exists"]][["smiles", "solvent_smiles", "xyz_path"]]
        out_missing = self.output_dir / "missing_xyz_files.csv"
        if len(missing) > 0:
            missing.to_csv(out_missing, index=False)
            logger.info(f"缺失/生成失败列表: {out_missing}")
        else:
            # Ensure previous file doesn't mislead if exists from earlier runs
            try:
                if out_missing.exists():
                    out_missing.unlink()
            except Exception:
                pass

        return df

    # ---------------------------
    # Task subset & export
    # ---------------------------
    def subset_for_target(self, base_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        bounds = self._target_bounds()
        df = base_df[base_df["xyz_exists"]].copy()

        # Numeric target and range filter
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        df = df[df[target_col].notna()]
        df = df[(df[target_col] >= bounds[0]) & (df[target_col] <= bounds[1])]

        # Non-null required features
        keep_desc = [c for c in self.desc_cols_out if c in df.columns]
        must_have = ["smiles", "solvent_smiles", "xyz_path"] + keep_desc
        df = df.dropna(subset=[c for c in must_have if c in df.columns] + [target_col])

        # Optional dedup to be safe
        dedup_cols = [c for c in ["smiles", "solvent_smiles", "xyz_path", target_col] if c in df.columns]
        if dedup_cols:
            df = df.drop_duplicates(subset=dedup_cols)

        # Uniform target name "y"; drop the other task column
        df["y"] = df[target_col]
        other = "lambda_em" if target_col == "lambda_abs" else "lambda_abs"
        if other in df.columns:
            df = df.drop(columns=[other])

        df = df.reset_index(drop=True)
        df["row_id"] = df.index.astype(int)
        return df

    def save_parquet_task(self, df: pd.DataFrame, task: str):
        if task not in ("abs", "em"):
            raise ValueError(f"Unknown task: {task}")
        out = self.output_dir / ("data_abs.parquet" if task == "abs" else "data_em.parquet")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False, compression="snappy")
        logger.info(f"✓[{task}] 数据集保存: {out} (rows={len(df)})")

    def create_random_split_task(self, df: pd.DataFrame, task: str, ratios: List[float], seed: int):
        if not np.isclose(sum(ratios), 1.0):
            s = sum(ratios)
            if s <= 0:
                raise ValueError(f"Invalid ratios: {ratios}")
            ratios = [x / s for x in ratios]
            logger.warning(f"ratios 不规范, 已归一化为: {ratios[0]:.3f}/{ratios[1]:.3f}/{ratios[2]:.3f}")

        # new layout: <splits_dir>/<task>/random_<task>_{train,valid,test}.json
        splits_root = (self.splits_dir / task).resolve()
        splits_root.mkdir(parents=True, exist_ok=True)

        idx = df["row_id"].to_numpy()
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

        n = len(idx)
        n_train = int(n * ratios[0])
        n_valid = int(n * ratios[1])
        train_idx = idx[:n_train]
        valid_idx = idx[n_train:n_train + n_valid]
        test_idx = idx[n_train + n_valid:]

        def _dump(p: Path, arr: np.ndarray):
            with open(p, "w") as f:
                json.dump([int(x) for x in arr.tolist()], f)

        prefix = f"random_{task}"
        _dump(splits_root / f"{prefix}_train.json", train_idx)
        _dump(splits_root / f"{prefix}_valid.json", valid_idx)
        _dump(splits_root / f"{prefix}_test.json",  test_idx)

        logger.info(f"✓[{task}] 随机划分: train={len(train_idx)} valid={len(valid_idx)} test={len(test_idx)} -> {splits_root}")

    # ---------------------------
    # Utilities
    # ---------------------------
    @staticmethod
    def _is_valid_smiles(s: str) -> bool:
        return isinstance(s, str) and Chem.MolFromSmiles(s) is not None

    def _all_banned_elements(self) -> Set[str]:
        # 合并全局禁用与强制黑名单
        return set(self.BANNED_ELEMENTS) | set(FORBIDDEN_ELEMENTS)

    def _contains_banned_elements(self, s: str) -> bool:
        """
        Check if a valid SMILES contains any banned element (solute only).
        Assumes s already passed _is_valid_smiles.
        """
        if not isinstance(s, str):
            return False
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                return False
            ban = self._all_banned_elements()
            for atom in mol.GetAtoms():
                if atom.GetSymbol() in ban:
                    return True
            return False
        except Exception:
            # Fail-closed: do not filter if parsing crashed here (already validated before)
            return False

    @staticmethod
    def _mol_to_canonical_smiles(mol: Chem.Mol) -> str:
        return Chem.MolToSmiles(mol, canonical=True)

    def _canonicalize_smiles(self, s: str) -> Optional[str]:
        if not isinstance(s, str):
            return None
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        if _HAS_STD:
            try:
                mol = rdMolStandardize.Cleanup(mol)
                mol = rdMolStandardize.FragmentParent(mol)  # 去盐
                reion = rdMolStandardize.Reionizer()
                mol = reion.reionize(mol)
            except Exception:
                pass
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
        return self._mol_to_canonical_smiles(mol)

    def _generate_xyz_from_smiles(self, smiles: Optional[str], out_path: Path, seed: int = 42) -> bool:
        if not isinstance(smiles, str) or not smiles:
            return False
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return False
            # Relaxed sanitize
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                try:
                    flags = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                    Chem.SanitizeMol(mol, sanitizeOps=flags)
                    Chem.SetAromaticity(mol)
                except Exception:
                    return False

            mol = Chem.AddHs(mol)

            params = AllChem.ETKDGv3() if hasattr(AllChem, "ETKDGv3") else AllChem.ETKDG()
            params.randomSeed = int(seed)
            params.useRandomCoords = True
            params.maxAttempts = 20
            if hasattr(params, "numThreads"):
                params.numThreads = 0  # all cores

            cid = AllChem.EmbedMolecule(mol, params)
            if cid < 0:
                # fallback: try pure random coords a few times
                for _ in range(5):
                    cid = AllChem.EmbedMolecule(mol, randomSeed=int(seed), useRandomCoords=True)
                    if cid >= 0:
                        break
                if cid < 0:
                    return False

            # Short optimization to avoid stalls
            try:
                if AllChem.MMFFHasAllMoleculeParams(mol):
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
                else:
                    AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            except Exception:
                pass

            conf = mol.GetConformer()
            n = mol.GetNumAtoms()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                f.write(f"{n}\n")
                f.write("generated by RDKit\n")
                for i in range(n):
                    a = mol.GetAtomWithIdx(i)
                    p = conf.GetAtomPosition(i)
                    f.write(f"{a.GetSymbol()} {p.x:.6f} {p.y:.6f} {p.z:.6f}\n")
            return True
        except Exception:
            return False

    def _normalize_descriptor_name(self, name: str) -> str:
        # normalize typical names to snake_case used downstream
        key = re.sub(r"[^A-Za-z0-9]", "", name).lower()
        mapping = {
            "tpsa": "tpsa",
            "mollogp": "mol_logp",
            "molmr": "mol_mr",
            "numhdonors": "num_h_donors",
            "numhacceptors": "num_h_acceptors",
            "molwt": "mol_wt",
            "labuteasa": "labute_asa",
            "fractioncsp3": "fraction_csp3",
            "numrotatablebonds": "num_rotatable_bonds",
        }
        return mapping.get(key, name.lower())

    def _resolve_path(self, p: str | Path) -> Path:
        p = Path(p)
        if p.is_absolute():
            return p
        cand = (self.config_dir / p)
        if cand.exists():
            return cand.resolve()
        return (self.root / p).resolve()

    def _target_bounds(self) -> Tuple[float, float]:
        gen = self.cfg.get("generation", {})
        b = gen.get("target_bounds", None)
        if isinstance(b, (list, tuple)) and len(b) == 2:
            return float(b[0]), float(b[1])
        return self.TARGET_BOUNDS_DEFAULT

    def _ratios_from_cfg(self) -> Tuple[List[float], int]:
        sp = self.cfg.get("splitting", {})
        strat = sp.get("strategy", "random")
        if strat != "random":
            raise ValueError(f"splitting.strategy 必须为 'random' (当前: {strat})")

        train = float(sp.get("train_ratio", 0.7))
        valid = float(sp.get("valid_ratio", 0.2))
        test = sp.get("test_ratio", None)
        if test is None:
            test = max(0.0, 1.0 - train - valid)
        test = float(test)
        total = train + valid + test
        if total <= 0:
            raise ValueError("无效的划分比例: 总和 <= 0")
        if not np.isclose(total, 1.0):
            train, valid, test = [x / total for x in (train, valid, test)]
            logger.warning(f"ratios 不规范, 已归一化为: {train:.3f}/{valid:.3f}/{test:.3f}")

        seed = int(sp.get("random_seed", 42))
        return [train, valid, test], seed

    # ---------------------------
    # New helper utilities
    # ---------------------------
    def _num_atoms_in_smiles(self, s: str) -> int:
        """
        Return RDKit-perceived atom count for a SMILES (typically heavy atoms).
        """
        if not isinstance(s, str):
            return 0
        try:
            mol = Chem.MolFromSmiles(s)
            return int(mol.GetNumAtoms()) if mol is not None else 0
        except Exception:
            return 0

    def _contains_any_element(self, s: str, elements: Set[str]) -> bool:
        """
        Whether the SMILES contains any element in 'elements'.
        """
        if not isinstance(s, str) or not elements:
            return False
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                return False
            elset = elements
            for a in mol.GetAtoms():
                if a.GetSymbol() in elset:
                    return True
            return False
        except Exception:
            return False

    def _global_element_counts(self, smiles_series: pd.Series) -> Dict[str, int]:
        """
        Count element occurrences across all provided SMILES (solute only).
        """
        counts: Dict[str, int] = {}
        for s in smiles_series:
            if not isinstance(s, str):
                continue
            try:
                mol = Chem.MolFromSmiles(s)
                if mol is None:
                    continue
                for a in mol.GetAtoms():
                    sym = a.GetSymbol()
                    counts[sym] = counts.get(sym, 0) + 1
            except Exception:
                continue
        return counts


def main():
    parser = argparse.ArgumentParser(description="Build Abs/Em datasets from data.yaml (outputs to processed/flu)")
    parser.add_argument("--config", "-c", default="configs/data.yaml", help="Path to data.yaml")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml_load(f)
    builder = DatasetBuilder(cfg, config_dir=config_path.parent)
    builder.build()


if __name__ == "__main__":
    main()
