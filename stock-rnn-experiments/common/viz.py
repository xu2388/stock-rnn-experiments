"""可视化与训练辅助。"""
from __future__ import annotations

import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import torch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_fig(fig, path: str, dpi: int = 120) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_loss(train_losses, val_losses, out_path: str, title: str = "Loss"):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_losses, label="train", alpha=0.7)
    # val 可能是按 epoch 采样的
    if len(val_losses) > 0:
        # 把 val 对齐到 epoch 数
        n_epochs = len(val_losses)
        xs = np.linspace(0, len(train_losses), n_epochs, endpoint=False)
        ax.plot(xs, val_losses, label="val", marker="o", markersize=3, color="orange")
    ax.set_xlabel("step"); ax.set_ylabel("loss")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    save_fig(fig, out_path)


def plot_predictions(dates, y_true, y_pred, out_path: str, title: str = "",
                     n_show: int = 200):
    """画真实 vs 预测。dates/y_true/y_pred 长度相同。"""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    dates = np.asarray(dates)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    ax.plot(dates, y_true, label="真实", color="black", lw=1.5)
    ax.plot(dates, y_pred, label="预测", color="red", lw=1, alpha=0.8)
    ax.set_title(title); ax.set_ylabel("收盘价")
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    ax = axes[1]
    ax.plot(dates, y_pred - y_true, color="purple", lw=0.8)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("预测残差"); ax.set_ylabel("残差"); ax.set_xlabel("日期")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    save_fig(fig, out_path)


def plot_scatter(y_true, y_pred, out_path: str, title: str = "预测 vs 真实"):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=5, alpha=0.4)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1, label="y=x")
    ax.set_xlabel("真实值"); ax.set_ylabel("预测值")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    save_fig(fig, out_path)


def plot_metrics_bar(results: dict, out_path: str, metric_names: list = None):
    """results: {model_name: {metric: value, ...}, ...}"""
    if metric_names is None:
        metric_names = ["MAE", "RMSE", "MAPE", "R2", "DA"]
    models = list(results.keys())
    n_metrics = len(metric_names)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]
    for i, m in enumerate(metric_names):
        vals = [results[mod].get(m, float("nan")) for mod in models]
        axes[i].bar(models, vals, color=["steelblue", "darkorange", "seagreen"][:len(models)])
        axes[i].set_title(m)
        axes[i].grid(True, alpha=0.3, axis="y")
        for j, v in enumerate(vals):
            axes[i].text(j, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.suptitle("三模型测试集指标对比")
    save_fig(fig, out_path)
