import argparse
import json
import logging
import sys
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

# --- 新增/修改部分开始 ---
# 获取当前脚本所在目录 (.../src/spectra/engine)
current_dir = os.path.dirname(os.path.abspath(__file__))
#由此向上回溯两级，找到 src 目录 (.../src)
# logic: engine -> spectra -> src
src_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

# 将 src 目录加入 Python 搜索路径
if src_root not in sys.path:
    sys.path.insert(0, src_root)
# --- 新增/修改部分结束 ---

# 确保能导入项目模块
sys.path.append(os.getcwd())

from model_registry import build_model_from_config
from datamodule import create_dataloaders, resolve_model_cutoff
from runner import extract_pred

# 设置日志，只显示必要信息
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("eval")

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)

def run_evaluation(run_dir: str, device_name: str = "auto"):
    run_path = Path(run_dir)
    config_dir = run_path / "configs"
    
    if not config_dir.exists():
        raise FileNotFoundError(f"未在 {run_path} 下找到 configs 目录，请检查路径。")

    # 1. 加载训练时保存的配置
    data_cfg = load_json(config_dir / "data.json")
    model_cfg = load_json(config_dir / "model.json")
    train_cfg = load_json(config_dir / "train.json")

    # 2. 准备设备
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    logger.info(f"Using device: {device}")

    # 3. 准备数据加载器参数 (逻辑复刻自 runner.py)
    target = train_cfg.get("target", "abs")
    split_name = data_cfg.get("splitting", {}).get("strategy", "random")
    
    # 构建 train_config 以传递 loader 和 cutoff 参数
    train_cfg_for_data = {}
    if model_cfg.get("loader"):
        train_cfg_for_data["loader"] = model_cfg.get("loader")
    
    # 关键：确保 cutoff 与训练一致
    data_cutoff = resolve_model_cutoff(model_cfg)
    train_cfg_for_data.setdefault("data", {})["cutoff"] = data_cutoff

    logger.info(f"Loading Test Data (Target: {target}, Split: {split_name})...")
    # 只获取 test_loader
    _, _, test_loader = create_dataloaders(
        data_config=data_cfg,
        train_config=train_cfg_for_data,
        split_name=split_name,
        target=target,
        model_config=model_cfg,
        distributed=False 
    )

    # 4. 重建模型并加载权重
    logger.info("Building Model...")
    model = build_model_from_config(model_cfg, data_cfg, train_cfg)
    model.to(device)

    ckpt_path = run_path / "best.pt"
    if not ckpt_path.exists():
        logger.warning(f"best.pt not found, trying last.pt...")
        ckpt_path = run_path / "last.pt"
    
    logger.info(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # 5. 推理 (Inference)
    logger.info("Running Inference...")
    y_trues = []
    y_preds = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            # 使用 runner.py 中的 extract_pred 处理不同模型输出格式
            pred = extract_pred(model(batch))
            
            y_trues.append(batch.y.cpu().numpy())
            y_preds.append(pred.cpu().numpy())

    y_true = np.concatenate(y_trues)
    y_pred = np.concatenate(y_preds)

    # 6. 分段分析
    analyze_intervals(y_true, y_pred)

def analyze_intervals(y_true, y_pred):
    # 定义区间
    mask_head = (y_true >= 300) & (y_true < 600)
    mask_tail = (y_true >= 600) & (y_true <= 900)
    
    # 还可以看更极端的长尾 (>800)
    mask_extreme = (y_true > 800)

    def print_metrics(name, t, p):
        if len(t) == 0:
            print(f"\n--- {name} (N=0) ---")
            print("No samples found.")
            return
        
        mse = np.mean((t - p) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(t - p))
        
        # 计算相对误差 (MAPE)
        mape = np.mean(np.abs((t - p) / t)) * 100

        print(f"\n--- {name} (N={len(t)}, Ratio={len(t)/len(y_true):.1%}) ---")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE : {mae:.4f}")
        print(f"MAPE: {mape:.2f}%")

    print("\n" + "="*40)
    print("       DATA DISTRIBUTION ANALYSIS       ")
    print("="*40)

    # 全局
    print_metrics("Global (All)", y_true, y_pred)

    # 头部 (300-600nm)
    print_metrics("Head (300-600 nm)", y_true[mask_head], y_pred[mask_head])

    # 尾部 (600-900nm)
    print_metrics("Tail (600-900 nm)", y_true[mask_tail], y_pred[mask_tail])
    
    # 极端值
    if np.sum(mask_extreme) > 0:
        print_metrics("Extreme (> 800 nm)", y_true[mask_extreme], y_pred[mask_extreme])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 只需要指定之前运行生成的目录，例如 runs/EGNN_abs_random_20251128_...
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the training run directory")
    parser.add_argument("--device", type=str, default="auto", help="cpu or cuda")
    args = parser.parse_args()

    run_evaluation(args.run_dir, args.device)
