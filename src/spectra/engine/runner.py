import csv
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from datamodule import create_dataloaders, resolve_model_cutoff
from model_registry import build_model_from_config

def seed_everything(seed: int = 42, deterministic: bool = False):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(run_dir: Path, level: int = logging.INFO):
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh = logging.FileHandler(run_dir / "train.log")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logging.getLogger().addHandler(fh)


def pick_device(pref: str = "auto") -> torch.device:
    p = (pref or "auto").lower()
    if p == "cpu":
        return torch.device("cpu")
    if p in ("cuda", "gpu", "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if p == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_autocast_dtype(precision: str) -> Optional[torch.dtype]:
    p = (precision or "").lower()
    if p in ("bf16", "bfloat16"):
        return torch.bfloat16
    if p in ("fp16", "float16", "half"):
        return torch.float16
    return None  # fp32


def extract_pred(output: Any) -> torch.Tensor:
    """
    Accepts common model outputs and returns a 1D or 2D tensor [B] or [B,1].
    - tensor -> tensor
    - dict -> prefers keys: 'pred', 'y_pred', 'output', 'out', 'logits'
    - tuple/list -> first element
    """
    if isinstance(output, torch.Tensor):
        y = output
    elif isinstance(output, dict):
        for k in ("pred", "y_pred", "output", "out", "logits"):
            if k in output and isinstance(output[k], torch.Tensor):
                y = output[k]
                break
        else:
            raise ValueError("Model dict output missing known prediction keys")
    elif isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        y = output[0]
    else:
        raise TypeError(f"Unsupported model output type: {type(output)}")
    if y.ndim == 2 and y.size(-1) == 1:
        y = y.squeeze(-1)
    return y


def build_optimizer(params, cfg: dict):
    kind = (cfg.get("type") or "adamw").lower()
    lr = float(cfg.get("lr", 1e-3))
    wd = float(cfg.get("weight_decay", 0.0))
    eps = float(cfg.get("eps", 1e-8))
    betas = tuple(cfg.get("betas", (0.9, 0.999)))
    if kind == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, eps=eps, betas=betas)
    if kind == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd, eps=eps, betas=betas)
    if kind == "sgd":
        momentum = float(cfg.get("momentum", 0.9))
        nesterov = bool(cfg.get("nesterov", False))
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd, nesterov=nesterov)
    if kind == "rmsprop":
        momentum = float(cfg.get("momentum", 0.0))
        alpha = float(cfg.get("alpha", 0.99))
        return torch.optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=wd, alpha=alpha)
    if kind == "adagrad":
        return torch.optim.Adagrad(params, lr=lr, weight_decay=wd, eps=eps)
    raise ValueError(f"Unsupported optimizer type: {kind}")


import torch
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau, 
    CosineAnnealingLR, 
    StepLR, 
    MultiStepLR, 
    ExponentialLR,
    LinearLR,
    SequentialLR
)

def build_scheduler(optimizer, cfg: dict):
    if not cfg or (cfg.get("type") in (None, "", "none")):
        return None, False
        
    kind = str(cfg.get("type")).lower()
    
    warmup_epochs = int(cfg.get("warmup_epochs", 0)) 
    warmup_factor = float(cfg.get("warmup_start_factor", 0.01)) 

    main_sched = None
    step_on_metric = False
    
    if kind in ("reduce_on_plateau", "plateau"):
        mode = str(cfg.get("mode", "min"))
        factor = float(cfg.get("factor", 0.5))
        patience = int(cfg.get("patience", 10))
        threshold = float(cfg.get("threshold", 1e-4))
        min_lr = float(cfg.get("min_lr", 0.0))
        
        sched = ReduceLROnPlateau(
            optimizer, 
            mode=mode, 
            factor=factor, 
            patience=patience, 
            threshold=threshold, 
            min_lr=min_lr
        )
        # ReduceLROnPlateau can't be wrapped by SequentialLR; return directly.
        return sched, True 

    elif kind == "cosine":
        t_max_default = int(cfg.get("t_max", cfg.get("T_max", 50)))
        t_max = t_max_default - warmup_epochs if t_max_default > warmup_epochs else t_max_default
        
        eta_min = float(cfg.get("eta_min", 0.0))
        main_sched = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        
    elif kind == "step":
        step_size = int(cfg.get("step_size", 30))
        gamma = float(cfg.get("gamma", 0.1))
        main_sched = StepLR(optimizer, step_size=step_size, gamma=gamma)
        
    elif kind == "multistep":
        milestones = list(cfg.get("milestones", [30, 60, 90]))
        gamma = float(cfg.get("gamma", 0.1))
        main_sched = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        
    elif kind == "exponential":
        gamma = float(cfg.get("gamma", 0.99))
        main_sched = ExponentialLR(optimizer, gamma=gamma)
        
    else:
        return None, False

    if warmup_epochs > 0 and main_sched is not None:
        warmup_sched = LinearLR(
            optimizer, 
            start_factor=warmup_factor, 
            end_factor=1.0, 
            total_iters=warmup_epochs
        )
        
        sched = SequentialLR(
            optimizer, 
            schedulers=[warmup_sched, main_sched], 
            milestones=[warmup_epochs]
        )
        return sched, False
    
    return main_sched, False


def build_loss(cfg: dict):
    kind = (cfg.get("type") or "mse").lower()
    if kind in ("l1", "mae"):
        return nn.L1Loss()
    if kind in ("mse", "l2"):
        return nn.MSELoss()
    if kind in ("smoothl1", "huber"):
        beta = float(cfg.get("beta", cfg.get("delta", 1.0)))
        return nn.SmoothL1Loss(beta=beta)
    if kind == "weighted_mse":
        threshold = cfg.get("threshold", 600.0)
        tail_weight = cfg.get("tail_weight", 5.0) # 默认设高一点，比如 5倍
        return WeightedMSELoss(threshold=threshold, tail_weight=tail_weight)    

    raise ValueError(f"Unsupported loss type: {kind}")


def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        diff = y_pred - y_true
        mae = diff.abs().mean().item()
        rmse = math.sqrt(torch.mean(diff.pow(2)).item())
        # R2: 1 - SS_res / SS_tot (add small eps for stability)
        y_mean = torch.mean(y_true)
        ss_res = torch.sum(diff.pow(2))
        ss_tot = torch.sum((y_true - y_mean).pow(2)) + 1e-9
        r2 = (1.0 - (ss_res / ss_tot)).item()
        return {"mae": mae, "rmse": rmse, "r2": r2}


def save_state(path: Path, model: nn.Module, optimizer, scaler, epoch: int, best: Dict[str, Any]):
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best": best,
    }
    torch.save(obj, path)


def load_state(path: Path, model: nn.Module, optimizer=None, scaler=None) -> Tuple[int, Dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("epoch", 0)), dict(ckpt.get("best", {}))


class ExperimentRunner:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.logger = logging.getLogger("runner")

    def run(
        self,
        data_cfg: dict,
        model_cfg: dict,
        train_cfg: dict,
        target_override: Optional[str] = None,
        split_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        # ---------------- Trainer knobs ----------------
        trainer_cfg = model_cfg.get("trainer")
        seed = int(trainer_cfg.get("seed", 42))
        device_pref = str(trainer_cfg.get("device", "auto"))
        amp_enabled = bool(trainer_cfg.get("amp", True))
        precision = str(trainer_cfg.get("precision", "bf16"))
        allow_tf32 = bool(trainer_cfg.get("allow_tf32", True))
        cudnn_benchmark = bool(trainer_cfg.get("cudnn_benchmark", True))
        deterministic = bool(trainer_cfg.get("deterministic", False))
        epochs = int(trainer_cfg.get("epochs", 100))
        grad_clip = trainer_cfg.get("grad_clip_norm", None)
        grad_clip = float(grad_clip) if grad_clip is not None else None
        accumulate = int(trainer_cfg.get("accumulate_steps", 1))
        val_every = int(trainer_cfg.get("val_every", 1))
        log_every = int(trainer_cfg.get("log_every_steps", 50))
        # devices  ֶν    ڼ ¼           DataParallel
        _num_devices_cfg = int(trainer_cfg.get("devices", 1))

        # Early stop/checkpoint
        ckpt_cfg = trainer_cfg.get("checkpoint", {})
        monitor_key = str(ckpt_cfg.get("monitor", "val/loss"))
        monitor_mode = str(ckpt_cfg.get("mode", "min")).lower()
        save_last = bool(ckpt_cfg.get("save_last", True))
        save_top_k = int(ckpt_cfg.get("save_top_k", 1))  # not strictly used; we save 'best' singleton

        es_cfg = trainer_cfg.get("early_stop", {})
        es_enabled = bool(es_cfg.get("enabled", True))
        es_patience = int(es_cfg.get("patience", 15))
        es_min_delta = float(es_cfg.get("min_delta", 0.0))
        es_mode = str(es_cfg.get("mode", monitor_mode)).lower()

        # ---------------- Distributed setup ----------------
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        distributed = world_size > 1
        is_main = (not distributed) or (rank == 0)

        if distributed and not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        # ---------------- Seed and perf ----------------
        seed_everything(seed + rank, deterministic=deterministic)
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)

        if distributed:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = pick_device(device_pref)
        amp_dtype = to_autocast_dtype(precision) if amp_enabled else None

        # ---------------- Target/split ----------------
        target = train_cfg.get("target")
        if target_override is not None:
            target = target_override

        split_name = data_cfg.get("splitting", {}).get("strategy")
        if split_override is not None:
            split_name = split_override

        train_cfg_for_data: Dict[str, Any] = {}
        if model_cfg.get("loader"):
            train_cfg_for_data["loader"] = model_cfg.get("loader")
        data_cutoff = resolve_model_cutoff(model_cfg)
        train_cfg_for_data.setdefault("data", {})["cutoff"] = data_cutoff

        # ---------------- Data ----------------
        train_loader, valid_loader, test_loader = create_dataloaders(
            data_config=data_cfg,
            train_config=train_cfg_for_data,
            split_name=split_name,
            target=target,
            model_config=model_cfg,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )

        # ---------------- Model ----------------
        model = build_model_from_config(model_cfg, data_cfg, train_cfg)
        model.to(device)

        if distributed:
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )

        # === NEW: 拿到 fb_ela.res_scale（注意 DDP 下要用 model.module） ===
        fb_module = model
        if isinstance(model, nn.parallel.DistributedDataParallel):
            fb_module = model.module

        has_res_scale = hasattr(fb_module, "fb_ela") and hasattr(fb_module.fb_ela, "res_scale")
        has_fb_alpha = hasattr(fb_module, "fb_alpha") and (fb_module.fb_alpha is not None)
        # ---------------- Optim/sched/loss ----------------
        optimizer_cfg = model_cfg.get("optimizer")
        optimizer = build_optimizer(model.parameters(), optimizer_cfg)
        scheduler_cfg = model_cfg.get("scheduler")
        scheduler, step_on_val = build_scheduler(optimizer, scheduler_cfg)
        loss_cfg = model_cfg.get("loss")
        loss_fn = build_loss(loss_cfg)
        scaler = GradScaler(enabled=(amp_dtype is not None and amp_dtype == torch.float16))

        # ---------------- Resume ----------------
        resume_path = trainer_cfg.get("resume_from", None)
        start_epoch = 0
        best = {
            "score": float("inf") if monitor_mode == "min" else -float("inf"),
            "epoch": -1,
            "metric": monitor_key,
        }
        if resume_path:
            ep, best_state = load_state(Path(resume_path), model, optimizer=optimizer, scaler=scaler)
            start_epoch = ep + 1
            if best_state:
                best.update(best_state)

        # ---------------- IO: configs + metrics.csv ----------------
        if is_main:
            self._save_configs(data_cfg, model_cfg, train_cfg)
            metrics_csv = self.run_dir / "metrics.csv"
            if not metrics_csv.exists():
                with open(metrics_csv, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["epoch", "split", "loss", "mae", "rmse", "r2", "lr"])
        else:
            metrics_csv = self.run_dir / "metrics.csv"  # dummy path,       ̲ д

        # ---------------- Helpers ----------------
        def is_better(curr: float, ref: float) -> bool:
            return (curr < ref - es_min_delta) if monitor_mode == "min" else (curr > ref + es_min_delta)

        def pick_monitored(m: Dict[str, float]) -> float:
            if monitor_key in m:
                return float(m[monitor_key])
            key_map = {
                "val/loss": "loss",
                "val/mae": "mae",
                "val/rmse": "rmse",
                "val/r2": "r2",
            }
            k = key_map.get(monitor_key, "loss")
            return float(m.get(k, m.get("loss", 0.0)))

        no_improve = 0
        global_step = 0

        # ---------------- Training loop ----------------
        for epoch in range(start_epoch, epochs):
            #  ֲ ʽ sampler ÿ   epoch   һ      
            if distributed and isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            model.train()
            train_loss_sum = 0.0
            train_count = 0
            optimizer.zero_grad(set_to_none=True)

            for i, batch in enumerate(train_loader):
                batch = batch.to(device)
                with autocast(enabled=(amp_dtype is not None), dtype=amp_dtype):
                    model_output = model(batch) # Get full output dict
                    pred = extract_pred(model_output)
                    loss = loss_fn(pred, batch.y)

                    # --- FB-ELA Diversity Loss (if enabled) ---
                    lambda_div = float(train_cfg.get("diversity_loss", {}).get("lambda", 0.0))
                    if lambda_div > 1e-6 and model_cfg.get("fb_enabled", False):
                        fb_aux = model_output.get("fb_aux")
                        if fb_aux and "assign" in fb_aux and fb_aux["assign"].numel() > 0:
                            assign = fb_aux["assign"] # [N, K]
                            # Normalize assign columns
                            assign_norm = F.normalize(assign, p=2, dim=0)
                            # Calculate similarity matrix (K, K)
                            similarity_matrix = torch.matmul(assign_norm.T, assign_norm)
                            # Penalize deviation from identity matrix
                            identity = torch.eye(similarity_matrix.size(0), device=device)
                            diversity_loss = (similarity_matrix - identity).pow(2).sum()
                            
                            # Normalize by num_nodes and num_blocks for better scaling
                            diversity_loss = diversity_loss / (assign.size(0) * assign.size(1)**2)
                            
                            loss += lambda_div * diversity_loss
                    
                    # --- FB-ELA Smoothing Loss ---
                    if model_cfg.get("fb_enabled", False):
                        fb_aux = model_output.get("fb_aux")
                        edge_index = model_output.get("edge_index")
                        if fb_aux and "assign" in fb_aux and edge_index is not None:
                            assign = fb_aux["assign"]
                            src, dst = edge_index
                            if src.numel() > 0:
                                diff = assign[src] - assign[dst]
                                smooth_loss = torch.mean(diff.pow(2))
                                loss += 0.1 * smooth_loss
                    
                    loss_scaled = loss / max(1, accumulate)


                if scaler.is_enabled():
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()

                if (i + 1) % max(1, accumulate) == 0:
                    if grad_clip is not None:
                        if scaler.is_enabled():
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                train_loss_sum += float(loss.item()) * pred.size(0)
                train_count += int(pred.size(0))
                global_step += 1

                if is_main and log_every > 0 and (global_step % log_every == 0):
                    rs_val = float("nan")
                    if has_res_scale:
                        rs_val = float(fb_module.fb_ela.res_scale.detach().cpu().item())

                    alpha_val = float("nan")
                    if has_fb_alpha:
                        alpha_val = float(torch.sigmoid(fb_module.fb_alpha).detach().cpu().item())

                    logging.getLogger("train").info(
                        "epoch %d step %d | loss=%.6f lr=%.2e res_scale=%.4f alpha=%.4f",
                        epoch,
                        global_step,
                        float(loss.item()),
                        optimizer.param_groups[0]["lr"],
                        rs_val,
                        alpha_val,
                    )

            #     ֻ ñ  rank   ͳ     train_loss  DDP  ¸  rank ͨ  һ    
            train_loss = train_loss_sum / max(1, train_count)

            # ---------------- Validation ----------------
            do_val = (epoch % max(1, val_every) == 0) or (epoch == epochs - 1)
            val_stats: Dict[str, float] = {}
            if do_val:
                val_stats = self._eval_one(model, valid_loader, loss_fn, device, amp_dtype)
                mon = pick_monitored({"val/" + k: v for k, v in val_stats.items()})
                improved = is_better(mon, best["score"])
                if improved and is_main:
                    best.update({"score": mon, "epoch": epoch})
                    save_state(self.run_dir / "best.pt", model, optimizer=None, scaler=None, epoch=epoch, best=best)
                    no_improve = 0
                else:
                    no_improve += 1

                if scheduler is not None:
                    if step_on_val:
                        scheduler.step(mon)
                    else:
                        scheduler.step()

            # ---------------- CSV logging (only main rank) ----------------
            lr_val = optimizer.param_groups[0]["lr"]
            if is_main:
                with open(metrics_csv, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([epoch, "train", train_loss, "", "", "", lr_val])
                    if do_val and val_stats:
                        w.writerow(
                            [
                                epoch,
                                "valid",
                                val_stats.get("loss", ""),
                                val_stats.get("mae", ""),
                                val_stats.get("rmse", ""),
                                val_stats.get("r2", ""),
                                lr_val,
                            ]
                        )

                logging.getLogger("epoch").info(
                    "epoch %d/%d | train_loss=%.6f | val=%s | best(%s)=%.6f@%d | lr=%.2e",
                    epoch,
                    epochs,
                    train_loss,
                    (
                        "loss={:.6f} mae={:.4f} rmse={:.4f} r2={:.4f}".format(
                            val_stats.get("loss", float("nan")),
                            val_stats.get("mae", float("nan")),
                            val_stats.get("rmse", float("nan")),
                            val_stats.get("r2", float("nan")),
                        )
                        if val_stats
                        else "n/a"
                    ),
                    monitor_key,
                    best["score"],
                    best["epoch"],
                    lr_val,
                )

            # ---------------- Early stopping ----------------
            if es_enabled and no_improve >= es_patience:
                if is_main:
                    logging.getLogger("earlystop").info(
                        "Early stopping at epoch %d (patience=%d)", epoch, es_patience
                    )
                break

        # ---------------- Save last (only main) ----------------
        if save_last and is_main:
            save_state(
                self.run_dir / "last.pt",
                model,
                optimizer=None,
                scaler=None,
                epoch=best.get("epoch", epochs - 1),
                best=best,
            )

        # ---------------- Test on best (only main) ----------------
        test_stats: Dict[str, float] = {}
        if is_main:
            if (self.run_dir / "best.pt").exists():
                load_state(self.run_dir / "best.pt", model)
            test_stats = self._eval_one(model, test_loader, loss_fn, device, amp_dtype)
            with open(self.run_dir / "test_metrics.json", "w") as f:
                json.dump(test_stats, f, indent=2)

            logging.getLogger("runner").info("Done. Test: %s", test_stats)

        # ---------------- Cleanup ----------------
        if distributed and dist.is_initialized():
            dist.destroy_process_group()

        return {"best": best, "test": test_stats}

    def _eval_one(self, model: nn.Module, loader, loss_fn, device, amp_dtype) -> Dict[str, float]:
        model.eval()
        ys, ps = [], []
        loss_sum, n_sum = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                with autocast(enabled=(amp_dtype is not None), dtype=amp_dtype):
                    pred = extract_pred(model(batch))
                    loss = loss_fn(pred, batch.y)
                loss_sum += float(loss.item()) * pred.size(0)
                n_sum += int(pred.size(0))
                ys.append(batch.y.detach().cpu())
                ps.append(pred.detach().cpu())
        y = torch.cat(ys) if ys else torch.zeros(0)
        p = torch.cat(ps) if ps else torch.zeros(0)
        stats = compute_metrics(y, p)
        stats["loss"] = loss_sum / max(1, n_sum)
        return stats

    def _save_configs(self, data_cfg: dict, model_cfg: dict, train_cfg: dict):
        cfg_dir = self.run_dir / "configs"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        for name, cfg in [("data", data_cfg), ("model", model_cfg), ("train", train_cfg)]:
            with open(cfg_dir / f"{name}.json", "w") as f:
                json.dump(cfg, f, indent=2)


def default_run_dir(
    base_dir: Optional[str],
    model_cfg: dict,
    train_cfg: dict,
    target: str,
    split: Optional[str],
) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = Path(base_dir or train_cfg.get("trainer", {}).get("output_dir") or "runs")
    model_name = (model_cfg.get("name") or model_cfg.get("builder") or "model").split(".")[-1].split(":")[-1]
    split_token = split or (train_cfg.get("run", {}).get("split") or "default")
    run_name = f"{model_name}_{target}_{split_token}_{ts}"
    return base / run_name

class WeightedMSELoss(nn.Module):
    def __init__(self, threshold=600.0, tail_weight=2.0):
        super().__init__()
        self.threshold = threshold
        self.tail_weight = tail_weight
        self.mse = nn.MSELoss(reduction='none') # 必须设为 none 以便逐样本加权

    def forward(self, pred, target):
        loss = self.mse(pred, target) # [B]
        
        # 创建权重向量：默认 1.0
        weights = torch.ones_like(target)
        
        # 针对大于阈值 (600nm) 的样本，赋予更高权重 (例如 3.0)
        mask = target > self.threshold
        weights[mask] = self.tail_weight
        
        # 也可以使用平滑权重，例如 log(target)
        # weights = torch.log(target) / torch.log(torch.tensor(300.0))
        
        return (loss * weights).mean()
