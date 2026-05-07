# src/predictor/core.py
"""
生产级光谱预测器核心模块

Features:
- 自动配置加载与验证
- 多模型集成预测
- 批量推理优化
- 特征可视化接口
- 完善的错误处理
"""

from __future__ import annotations
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import torch
import numpy as np
try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None
from torch.utils.data import DataLoader, Dataset

# 动态路径设置
import sys
ENGINE_DIR = Path(__file__).resolve().parent.parent / "spectra" / "engine"
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

from model_registry import build_model_from_config
from preprocessing import CombinedSoluteSolventFeaturizer
from datamodule import GraphData, make_collate_graph

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)


class SpectraPredictor:
    """
    统一光谱预测器接口
    
    支持功能：
    - 单样本/批量预测
    - 模型集成（多检查点）
    - 特征提取与可视化
    - 不确定性估计（集成方差）
    
    Example:
        >>> predictor = SpectraPredictor.from_checkpoint(
        ...     "checkpoints/ehc_egnn_abs_20251110/best.pt"
        ... )
        >>> result = predictor.predict(
        ...     xyz_path="path/to/molecule.xyz",
        ...     solvent_smiles="CCO"
        ... )
        >>> print(f"预测吸收波长: {result['wavelength']:.2f} nm")
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict[str, Any],
        featurizer: CombinedSoluteSolventFeaturizer,
        device: torch.device,
        target: str = "abs",
    ):
        """
        Args:
            model: 已加载权重的PyTorch模型
            config: 完整配置字典 (data/model/train)
            featurizer: 特征提取器实例
            device: 推理设备
            target: 预测目标 ("abs" 或 "em")
        """
        self.model = model.eval()
        self.config = config
        self.featurizer = featurizer
        self.device = device
        self.target = target
        
        # 缓存配置关键信息
        self.cutoff = float(config["model"].get("cutoff", 5.0))
        self.batch_size = int(config["model"].get("loader", {}).get("batch_size", 32))
    
    # ==================== 工厂方法 ====================
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        config_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        target: Optional[str] = None,
    ) -> SpectraPredictor:
        """
        从检查点文件加载预测器
        
        Args:
            checkpoint_path: 模型权重文件路径 (.pt)
            config_dir: 配置文件目录（默认为检查点同级configs/）
            device: 推理设备 ("cuda"/"cpu"/None=自动)
            target: 覆盖配置中的目标任务
        
        Returns:
            SpectraPredictor实例
        """
        ckpt_path = Path(checkpoint_path).resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"检查点不存在: {ckpt_path}")
        
        # 1. 加载配置
        if config_dir is None:
            config_dir = ckpt_path.parent / "configs"
        config = cls._load_configs(config_dir)
        
        # 2. 确定设备
        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        
        # 3. 构建模型
        model = build_model_from_config(
            config["model"],
            config["data"],
            config["train"]
        )
        
        # 4. 加载权重
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        try:
            if "model" in state:
                model.load_state_dict(state["model"], strict=True)
            elif "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"], strict=True)
            else:
                model.load_state_dict(state, strict=True)
        except RuntimeError as e:
            logger.warning(f"⚠️ 严格加载失败，尝试 strict=False 加载 (部分参数可能未初始化): {e}")
            if "model" in state:
                model.load_state_dict(state["model"], strict=False)
            elif "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"], strict=False)
            else:
                model.load_state_dict(state, strict=False)
        
        model.to(device).eval()
        
        # 5. 初始化特征提取器
        cutoff = float(config["model"].get("cutoff", 5.0))
        featurizer = CombinedSoluteSolventFeaturizer(cutoff_3d=cutoff)
        
        # 6. 确定目标
        if target is None:
            target = config["train"].get("target", "abs")
        
        logger.info(
            f"✓ 预测器加载成功 | 模型={config['model'].get('name')} | "
            f"目标={target} | 设备={device} | cutoff={cutoff:.1f}Å"
        )
        
        return cls(model, config, featurizer, device, target)
    
    @classmethod
    def from_ensemble(
        cls,
        checkpoint_paths: List[Union[str, Path]],
        device: Optional[str] = None,
        target: Optional[str] = None,
    ) -> EnsemblePredictor:
        """
        创建集成预测器（多模型投票）
        
        Args:
            checkpoint_paths: 多个检查点路径列表
            device: 推理设备
            target: 预测目标
        
        Returns:
            EnsemblePredictor实例
        """
        predictors = [
            cls.from_checkpoint(p, device=device, target=target)
            for p in checkpoint_paths
        ]
        return EnsemblePredictor(predictors, device=device or "auto")
    
    # ==================== 预测接口 ====================
    
    @torch.no_grad()
    def predict(
        self,
        xyz_path: str,
        solvent_smiles: str,
        return_features: bool = False,
        return_uncertainty: bool = False,
    ) -> Union[float, Dict[str, Any]]:
        """
        单样本预测
        
        Args:
            xyz_path: 溶质XYZ文件路径
            solvent_smiles: 溶剂SMILES字符串
            return_features: 是否返回中间特征（用于可视化/调试）
            return_uncertainty: 是否返回不确定性（需要集成模型）
        
        Returns:
            float: 预测波长 (nm) [默认]
            dict: 包含详细信息的字典 [return_features=True时]
                - wavelength: 预测波长
                - graph_embedding: 图级表征 [B, H]
                - node_features: 节点特征 [N, H]
                - solvent_embedding: 溶剂表征 [B, z_dim]
                - attention_weights: 注意力权重（如有）
        """
        # 1. 特征化
        record = {
            "xyz_path": xyz_path,
            "solvent_smiles": solvent_smiles,
            "lambda_abs": 0.0,  # dummy
            "lambda_em": 0.0,
        }
        
        sample = self.featurizer.featurize_record(record)
        if sample is None:
            raise ValueError(
                f"特征提取失败: {self.featurizer.last_error_reason}\n"
                f"  XYZ: {xyz_path}\n"
                f"  Solvent: {solvent_smiles}"
            )
        
        # 2. 构建batch
        from datamodule import GraphBatch
        batch = GraphBatch([GraphData(**sample)], target=self.target)
        batch = batch.to(self.device)
        
        # 3. 模型推理
        output = self.model(batch)
        pred = self._extract_pred(output).squeeze().cpu().item()
        
        # 4. 返回格式
        if not return_features:
            return pred
        
        result = {
            "wavelength": pred,
            "graph_embedding": output.get("graph_emb", torch.tensor([])).cpu().numpy(),
            "node_features": output.get("node_scalar", torch.tensor([])).cpu().numpy(),
            "solvent_embedding": output.get("z_s", torch.tensor([])).cpu().numpy(),
            "coords": output.get("coords", torch.tensor([])).cpu().numpy(),
        }
        
        # FB分支预测（如果启用）
        if "y_fb" in output:
            result["fb_prediction"] = output["y_fb"].squeeze().cpu().item()
        
        # 注意力权重（如果有）
        if "backbone_aux" in output:
            bb_aux = output["backbone_aux"]
            if "fb_aux" in bb_aux:
                fb_aux = bb_aux["fb_aux"]
                if "assign" in fb_aux:
                    result["block_assignment"] = fb_aux["assign"].cpu().numpy()
            
            if "attn_weights" in bb_aux and bb_aux["attn_weights"] is not None:
                result["attention_weights"] = bb_aux["attn_weights"].cpu().numpy()
            
            if "z_s_node" in bb_aux and bb_aux["z_s_node"] is not None:
                result["z_s_node"] = bb_aux["z_s_node"].cpu().numpy()
                
            if "solvent_h" in bb_aux and bb_aux["solvent_h"] is not None:
                result["solvent_h"] = bb_aux["solvent_h"].cpu().numpy()
        
        return result
    
    @torch.no_grad()
    def predict_batch(
        self,
        xyz_paths: List[str],
        solvent_smiles_list: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        if pd is None:
            raise ImportError("pandas is required for predict_batch().")
        """
        批量预测（优化效率）
        
        Args:
            xyz_paths: XYZ文件路径列表
            solvent_smiles_list: 溶剂SMILES列表
            batch_size: 批次大小（默认使用配置值）
            show_progress: 是否显示进度条
        
        Returns:
            DataFrame包含列: [xyz_path, solvent_smiles, prediction, status]
        """
        if len(xyz_paths) != len(solvent_smiles_list):
            raise ValueError("xyz_paths和solvent_smiles_list长度不一致")
        
        batch_size = batch_size or self.batch_size
        results = []
        
        # 进度条（可选）
        iterator = range(0, len(xyz_paths), batch_size)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="批量预测")
            except ImportError:
                pass
        
        for i in iterator:
            batch_xyz = xyz_paths[i:i+batch_size]
            batch_smiles = solvent_smiles_list[i:i+batch_size]
            
            # 特征化batch（跳过失败样本）
            samples, valid_indices = [], []
            for j, (xyz, smiles) in enumerate(zip(batch_xyz, batch_smiles)):
                record = {
                    "xyz_path": xyz,
                    "solvent_smiles": smiles,
                    "lambda_abs": 0.0,
                    "lambda_em": 0.0,
                }
                sample = self.featurizer.featurize_record(record)
                if sample:
                    samples.append(GraphData(**sample))
                    valid_indices.append(i + j)
                else:
                    results.append({
                        "xyz_path": xyz,
                        "solvent_smiles": smiles,
                        "prediction": None,
                        "status": f"特征化失败: {self.featurizer.last_error_reason}"
                    })
            
            if not samples:
                continue
            
            # 批量推理
            from datamodule import GraphBatch
            try:
                batch = GraphBatch(samples, target=self.target).to(self.device)
                output = self.model(batch)
                preds = self._extract_pred(output).squeeze().cpu().tolist()
                
                # 确保preds是列表
                if not isinstance(preds, list):
                    preds = [preds]
                
                for idx, pred in zip(valid_indices, preds):
                    results.append({
                        "xyz_path": xyz_paths[idx],
                        "solvent_smiles": solvent_smiles_list[idx],
                        "prediction": pred,
                        "status": "success"
                    })
            except Exception as e:
                logger.error(f"批次推理失败: {e}")
                for idx in valid_indices:
                    results.append({
                        "xyz_path": xyz_paths[idx],
                        "solvent_smiles": solvent_smiles_list[idx],
                        "prediction": None,
                        "status": f"推理失败: {str(e)}"
                    })
        
        return pd.DataFrame(results)
    
    def predict_from_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        batch_size: Optional[int] = None,
    ) -> pd.DataFrame:
        if pd is None:
            raise ImportError("pandas is required for predict_from_file().")
        """
        从CSV/Parquet文件批量预测
        
        Args:
            input_path: 输入文件路径（需包含xyz_path, solvent_smiles列）
            output_path: 输出CSV路径（可选）
            batch_size: 批次大小
        
        Returns:
            包含预测结果的DataFrame
        """
        input_path = Path(input_path)
        
        # 读取输入
        if input_path.suffix == ".csv":
            df = pd.read_csv(input_path)
        elif input_path.suffix == ".parquet":
            df = pd.read_parquet(input_path)
        else:
            raise ValueError(f"不支持的文件格式: {input_path.suffix}")
        
        # 验证列
        required = ["xyz_path", "solvent_smiles"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"输入文件缺少必需列: {missing}")
        
        # 批量预测
        results = self.predict_batch(
            xyz_paths=df["xyz_path"].tolist(),
            solvent_smiles_list=df["solvent_smiles"].tolist(),
            batch_size=batch_size,
            show_progress=True,
        )
        
        # 保存（可选）
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(output_path, index=False)
            logger.info(f"预测结果已保存: {output_path}")
        
        return results
    
    # ==================== 私有方法 ====================
    
    @staticmethod
    def _extract_pred(output: Any) -> torch.Tensor:
        """从模型输出中提取预测值"""
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, dict):
            for k in ("pred", "y_pred", "output"):
                if k in output:
                    return output[k]
        raise ValueError(f"无法从模型输出中提取预测值: {type(output)}")
    
    @staticmethod
    def _load_configs(config_dir: Path) -> Dict[str, dict]:
        """加载data/model/train配置"""
        config_dir = Path(config_dir)
        configs = {}
        for name in ("data", "model", "train"):
            path = config_dir / f"{name}.json"
            if not path.exists():
                raise FileNotFoundError(f"配置文件不存在: {path}")
            with open(path) as f:
                configs[name] = json.load(f)
        return configs
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型元信息"""
        return {
            "model_name": self.config["model"].get("name"),
            "target": self.target,
            "cutoff": self.cutoff,
            "device": str(self.device),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "backbone": self.config["model"].get("builder", "").split(":")[-1],
        }


class EnsemblePredictor:
    """
    集成预测器：多个模型的预测结果进行投票/平均
    
    提供：
    - 预测均值（降低方差）
    - 预测标准差（不确定性估计）
    - 模型间一致性分析
    """
    
    def __init__(self, predictors: List[SpectraPredictor], device: str = "auto"):
        if len(predictors) < 2:
            raise ValueError("集成至少需要2个模型")
        self.predictors = predictors
        self.device = device
        self.target = predictors[0].target
    
    @torch.no_grad()
    def predict(
        self,
        xyz_path: str,
        solvent_smiles: str,
        return_std: bool = True,
    ) -> Union[float, Dict[str, float]]:
        """
        集成预测单样本
        
        Args:
            xyz_path: XYZ文件
            solvent_smiles: 溶剂SMILES
            return_std: 是否返回标准差
        
        Returns:
            float: 预测均值 [默认]
            dict: {"mean": 均值, "std": 标准差, "predictions": 各模型预测} [return_std=True]
        """
        preds = []
        for predictor in self.predictors:
            try:
                p = predictor.predict(xyz_path, solvent_smiles)
                preds.append(p)
            except Exception as e:
                logger.warning(f"模型预测失败: {e}")
        
        if not preds:
            raise RuntimeError("所有模型预测均失败")
        
        preds = np.array(preds)
        mean_pred = float(np.mean(preds))
        
        if not return_std:
            return mean_pred
        
        return {
            "mean": mean_pred,
            "std": float(np.std(preds)),
            "min": float(np.min(preds)),
            "max": float(np.max(preds)),
            "predictions": preds.tolist(),
        }
    
    @torch.no_grad()
    def predict_batch(
        self,
        xyz_paths: List[str],
        solvent_smiles_list: List[str],
        batch_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """集成批量预测"""
        all_results = []
        
        for i, predictor in enumerate(self.predictors):
            logger.info(f"模型 {i+1}/{len(self.predictors)} 预测中...")
            df = predictor.predict_batch(
                xyz_paths, solvent_smiles_list, batch_size, show_progress=False
            )
            df = df.rename(columns={"prediction": f"model_{i}"})
            all_results.append(df)
        
        # 合并结果
        merged = all_results[0][["xyz_path", "solvent_smiles"]].copy()
        pred_cols = []
        for i, df in enumerate(all_results):
            col = f"model_{i}"
            merged[col] = df[col]
            pred_cols.append(col)
        
        # 计算统计量
        merged["prediction_mean"] = merged[pred_cols].mean(axis=1)
        merged["prediction_std"] = merged[pred_cols].std(axis=1)
        merged["prediction_min"] = merged[pred_cols].min(axis=1)
        merged["prediction_max"] = merged[pred_cols].max(axis=1)
        
        return merged
