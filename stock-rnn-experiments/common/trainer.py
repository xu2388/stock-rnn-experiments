"""通用训练/评估循环，三个模型共用。"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from .data import eval_metrics


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    n_epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: torch.device = None,
    model_name: str = "model",
    patience: int = 10,
):
    """通用训练循环。返回 (model, train_losses, val_losses)。
    用 val loss 做早停，保存最佳权重到内存。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    crit = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val = float("inf")
    best_state = None
    no_improve = 0
    t0 = time.time()
    print(f"\n=== 训练 {model_name}（{n_epochs} epochs, device={device}）===")
    for epoch in range(1, n_epochs + 1):
        # train
        model.train()
        ep_train = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = crit(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_losses.append(loss.item())
            ep_train.append(loss.item())
        # val
        model.eval()
        ep_val = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                ep_val.append(crit(pred, yb).item())
        val_mean = float(np.mean(ep_val))
        val_losses.append(val_mean)
        sched.step(val_mean)

        improved = val_mean < best_val
        if improved:
            best_val = val_mean
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        print(f"  epoch {epoch:2d}/{n_epochs}  train={np.mean(ep_train):.5f}  "
              f"val={val_mean:.5f}  {'★best' if improved else f'({no_improve}/{patience})'}  "
              f"lr={opt.param_groups[0]['lr']:.2e}")

        if no_improve >= patience:
            print(f"  早停：{patience} 个 epoch 无改善")
            break

    print(f"  训练耗时 {time.time() - t0:.1f}s, 最佳 val loss = {best_val:.5f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses


@torch.no_grad()
def evaluate_model(model: nn.Module, loader, meta: dict, device: torch.device = None):
    """在 loader 上推理，反标准化回原始价格，返回 (y_true_raw, y_pred_raw, metrics)。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    preds, targets = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        pred = model(xb).cpu().numpy()
        preds.append(pred)
        targets.append(yb.numpy())
    preds = np.concatenate(preds, axis=0).flatten()
    targets = np.concatenate(targets, axis=0).flatten()
    # 反标准化：模型预测的是标准化后的 y
    y_mean, y_std = meta["y_mean"], meta["y_std"]
    preds_raw = preds * y_std + y_mean
    targets_raw = targets * y_std + y_mean
    metrics = eval_metrics(targets_raw, preds_raw)
    return targets_raw, preds_raw, metrics
