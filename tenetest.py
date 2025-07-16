"""
utils.py · 工具函数与可视化模块
提供 EMA、标签拆分、正对采样、t-SNE 可视化和日志记录等通用工具
"""

from __future__ import annotations
import os
import json
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

# 若你把 config 常量写在 config.py，请取消下一行注释
# from config import *

# ======================== 基础工具 ========================


def update_ema(old_value: torch.Tensor, new_value: torch.Tensor, decay: float) -> torch.Tensor:
    """指数移动平均：new = decay * old + (1-decay) * new"""
    return old_value * decay + new_value * (1.0 - decay)


def split_by_tag(
    z: torch.Tensor,
    tags: List[str]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将批次特征按标签 'A' / 'B' / 'O' 拆成三块
    保证返回张量 device 与输入 z 一致
    """
    device = z.device
    mask_a = torch.tensor([t == 'A' for t in tags], dtype=torch.bool, device=device)
    mask_b = torch.tensor([t == 'B' for t in tags], dtype=torch.bool, device=device)
    mask_o = torch.tensor([t == 'O' for t in tags], dtype=torch.bool, device=device)

    def _select(mask: torch.Tensor) -> torch.Tensor:
        return z[mask] if mask.any() else torch.empty(0, z.shape[1], device=device)

    return _select(mask_a), _select(mask_b), _select(mask_o)


# ======================== 特征采集 & t-SNE 可视化 ========================


def collect_features_and_labels(
    encoder: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
    max_samples: int = 2_000,
) -> Tuple[np.ndarray, List[str]]:
    """前向编码器，最多收集 max_samples 个 (feature, tag)"""
    encoder.eval()
    feats: list[np.ndarray] = []
    lbls: list[str] = []
    collected = 0

    with torch.no_grad():
        for imgs, tags in tqdm(dataloader, desc="Collect features"):
            if collected >= max_samples:
                break

            remain = max_samples - collected
            # 截断超量部分
            if len(tags) > remain:
                imgs, tags = imgs[:remain], tags[:remain]

            imgs = imgs.to(device)
            z = encoder(imgs).cpu().numpy()

            feats.append(z)
            lbls.extend(tags)
            collected += len(tags)

    return np.vstack(feats), lbls


def visualize_tsne(
    encoder: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    save_path: str,
    device: str = "cuda",
    perplexity: Optional[int] = None,
    n_iter: int = 1_000,
    max_samples: int = 2_000,
) -> None:
    """绘制 t-SNE 2D 散点并保存 PNG"""
    # ---- 收集特征
    feats, labels = collect_features_and_labels(
        encoder, dataloader, device=device, max_samples=max_samples
    )
    print(f"[t-SNE] collected {len(feats)} samples, dim={feats.shape[1]}")

    # ---- 动态 perplexity
    if perplexity is None:
        perplexity = min(50, max(5, len(feats) // 100))
    print(f"[t-SNE] using perplexity={perplexity}")

    # ---- t-SNE 降维
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        init="random",
        learning_rate="auto",
        random_state=42,
        verbose=1,
    )
    feats_2d = tsne.fit_transform(feats)

    # ---- 颜色映射
    uniq = sorted(set(labels))
    colors = plt.cm.Set1(np.linspace(0, 1, len(uniq)))
    color_map = dict(zip(uniq, colors))

    # ---- 中文 & 画图
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(12, 10))
    for lb in uniq:
        m = np.array(labels) == lb
        plt.scatter(
            feats_2d[m, 0],
            feats_2d[m, 1],
            c=[color_map[lb]],
            label=f"{lb} ({m.sum()})",
            s=36,
            alpha=0.6,
            edgecolors="none",
        )
    plt.title("t-SNE 可视化：A vs B vs O", fontsize=16, fontweight="bold")
    plt.xlabel("t-SNE 维度1")
    plt.ylabel("t-SNE 维度2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[t-SNE] saved → {save_path}")


# ======================== 日志 & 记录器 ========================


def _setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:  # 避免重复
        return logger

    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


class ExperimentLogger:
    """记录训练指标并生成曲线的简单封装"""

    def __init__(self, exp_name: str, log_dir: str = "./logs"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(log_dir, f"{exp_name}_{self.timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)

        self.logger = _setup_logger(exp_name, os.path.join(self.exp_dir, "exp.log"))
        self.logger.info(f"Experiment dir: {self.exp_dir}")

        self.metrics: Dict[str, list[float]] = {
            "epoch": [],
            "loss_total": [],
            "loss_info": [],
            "loss_ncl": [],
            "loss_adv": [],
            "loss_diff": [],
        }

    # -------- 记录 / 保存 / 绘图 --------
    def log_epoch(self, epoch: int, metric_dict: Dict[str, float]):
        self.metrics["epoch"].append(epoch)
        for k in metric_dict:
            if k in self.metrics:
                self.metrics[k].append(metric_dict[k])
        msg = ", ".join(f"{k}:{v:.4f}" for k, v in metric_dict.items())
        self.logger.info(f"Epoch {epoch:03d} | {msg}")

    def save_metrics(self):
        fp = os.path.join(self.exp_dir, "metrics.json")
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Metrics saved → {fp}")

    def plot_losses(self, fname: str = "loss_curves.png"):
        if not self.metrics["epoch"]:
            return
        ep = self.metrics["epoch"]

        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.figure(figsize=(15, 10))

        names = [
            ("loss_total", "总损失"),
            ("loss_info", "信息损失"),
            ("loss_ncl", "NCL稀疏损失"),
            ("loss_adv", "对抗损失"),
            ("loss_diff", "差异损失"),
        ]
        for i, (key, title) in enumerate(names, 1):
            plt.subplot(2, 3, i)
            plt.plot(ep, self.metrics[key], linewidth=2)
            plt.title(title)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(alpha=0.3)

        # 综合子图
        plt.subplot(2, 3, 6)
        for key, title in names[1:]:
            plt.plot(ep, self.metrics[key], label=title, linewidth=2)
        plt.title("各损失对比")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.exp_dir, fname)
        plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        self.logger.info(f"Loss curves saved → {path}")

    # -------- checkpoint 快捷封装 --------
    def save_ckpt(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        fname: str = "ckpt_latest.pt",
    ):
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "loss": loss,
            "time": datetime.now().isoformat(),
        }
        path = os.path.join(self.exp_dir, fname)
        torch.save(ckpt, path)
        self.logger.info(f"Checkpoint saved → {path}")

    def load_ckpt(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, fname: str = "ckpt_latest.pt") -> int:
        path = os.path.join(self.exp_dir, fname)
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if optimizer:
            optimizer.load_state_dict(ckpt["optim"])
        self.logger.info(f"Loaded checkpoint ← {path}")
        return ckpt["epoch"]


# ======================== 简易自测 ========================
if __name__ == "__main__":
    
