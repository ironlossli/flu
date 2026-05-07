# src/predictor/cli.py
"""
命令行预测工具

Examples:
    # 单样本预测
    python -m predictor.cli predict \
        --checkpoint checkpoints/best.pt \
        --xyz molecule.xyz \
        --solvent "CCO" \
        --target abs
    
    # 批量预测
    python -m predictor.cli batch \
        --checkpoint checkpoints/best.pt \
        --input data.csv \
        --output predictions.csv \
        --target abs
    
    # 集成预测
    python -m predictor.cli ensemble \
        --checkpoints ckpt1.pt ckpt2.pt ckpt3.pt \
        --input data.csv \
        --output ensemble_predictions.csv
"""

import argparse
import logging
from pathlib import Path
from typing import List

from .core import SpectraPredictor, EnsemblePredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def cmd_predict(args):
    """单样本预测命令"""
    predictor = SpectraPredictor.from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=args.device,
        target=args.target,
    )
    
    result = predictor.predict(
        xyz_path=args.xyz,
        solvent_smiles=args.solvent,
        return_features=args.features,
    )
    
    if isinstance(result, dict):
        print("\n预测结果:")
        print(f"  波长: {result['wavelength']:.2f} nm")
        if "fb_prediction" in result:
            print(f"  FB分支: {result['fb_prediction']:.2f} nm")
        if args.features:
            print(f"  图表征维度: {result['graph_embedding'].shape}")
            print(f"  节点特征维度: {result['node_features'].shape}")
    else:
        print(f"\n预测波长: {result:.2f} nm")


def cmd_batch(args):
    """批量预测命令"""
    predictor = SpectraPredictor.from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=args.device,
        target=args.target,
    )
    
    results = predictor.predict_from_file(
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
    )
    
    # 统计
    success = (results["status"] == "success").sum()
    total = len(results)
    print(f"\n批量预测完成:")
    print(f"  成功: {success}/{total} ({100*success/total:.1f}%)")
    if args.output:
        print(f"  结果已保存: {args.output}")


def cmd_ensemble(args):
    """集成预测命令"""
    predictor = EnsemblePredictor.from_ensemble(
        checkpoint_paths=args.checkpoints,
        device=args.device,
        target=args.target,
    )
    
    results = predictor.predict_batch(
        xyz_paths=None,  # 从文件读取
        solvent_smiles_list=None,
        batch_size=args.batch_size,
    )
    
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"集成预测完成，结果保存至: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="光谱预测命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # ===== 单样本预测 =====
    p_predict = subparsers.add_parser("predict", help="单样本预测")
    p_predict.add_argument("--checkpoint", required=True, help="模型检查点路径")
    p_predict.add_argument("--xyz", required=True, help="XYZ文件路径")
    p_predict.add_argument("--solvent", required=True, help="溶剂SMILES")
    p_predict.add_argument("--target", choices=["abs", "em"], default="abs")
    p_predict.add_argument("--device", default="auto")
    p_predict.add_argument("--features", action="store_true", help="输出详细特征")
    p_predict.set_defaults(func=cmd_predict)
    
    # ===== 批量预测 =====
    p_batch = subparsers.add_parser("batch", help="批量预测")
    p_batch.add_argument("--checkpoint", required=True)
    p_batch.add_argument("--input", required=True, help="输入CSV/Parquet")
    p_batch.add_argument("--output", help="输出CSV路径")
    p_batch.add_argument("--target", choices=["abs", "em"], default="abs")
    p_batch.add_argument("--device", default="auto")
    p_batch.add_argument("--batch_size", type=int, default=None)
    p_batch.set_defaults(func=cmd_batch)
    
    # ===== 集成预测 =====
    p_ensemble = subparsers.add_parser("ensemble", help="集成预测")
    p_ensemble.add_argument("--checkpoints", nargs="+", required=True)
    p_ensemble.add_argument("--input", required=True)
    p_ensemble.add_argument("--output", required=True)
    p_ensemble.add_argument("--target", choices=["abs", "em"], default="abs")
    p_ensemble.add_argument("--device", default="auto")
    p_ensemble.add_argument("--batch_size", type=int, default=None)
    p_ensemble.set_defaults(func=cmd_ensemble)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
