"""实验 2: LSTNet 股票预测
=============================
LSTNet = CNN + GRU + Skip-GRU + Highway 自回归。

原始 LSTNet 预测所有 m 个变量（输出 m 维）。这里为了与其他模型对比公平，
把输出头改成 1 维（只预测 close）。

输入: (batch, window, n_features)
输出: (batch, 1)  预测下一个交易日标准化 close

架构（忠实于 LSTNet-master/models/LSTNet.py）：
  1. CNN: Conv2d(1, hidC, kernel=(Ck, m)) 提取局部模式
  2. GRU: 在 CNN 输出序列上跑，取最后隐状态
  3. Skip-GRU: 按 skip 周期分组再跑 GRU，捕捉周期性
  4. Highway: 用最近 hw 个时步的 close 做线性自回归残差
  5. 合并 -> Linear -> 输出
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import set_seed, get_pipeline, train_model, evaluate_model
from common.viz import plot_loss, plot_predictions, plot_scatter


class LSTNetRegressor(nn.Module):
    """LSTNet 回归模型（输出 1 维 close），忠实于原始实现。"""
    def __init__(self, window: int, n_features: int,
                 hidRNN: int = 64, hidCNN: int = 32, hidSkip: int = 16,
                 CNN_kernel: int = 6, skip: int = 5, highway_window: int = 5,
                 dropout: float = 0.2, output_dim: int = 1):
        super().__init__()
        self.P = window
        self.m = n_features
        self.hidR = hidRNN
        self.hidC = hidCNN
        self.hidS = hidSkip
        self.Ck = CNN_kernel
        self.skip = skip
        self.pt = (self.P - self.Ck) // self.skip
        self.hw = highway_window

        self.conv1 = nn.Conv2d(1, hidCNN, kernel_size=(CNN_kernel, n_features))
        self.GRU1 = nn.GRU(hidCNN, hidRNN)
        self.dropout = nn.Dropout(dropout)

        if skip > 0 and self.pt > 0:
            self.GRUskip = nn.GRU(hidCNN, hidSkip)
            self.linear1 = nn.Linear(hidRNN + skip * hidSkip, output_dim)
        else:
            self.linear1 = nn.Linear(hidRNN, output_dim)

        if highway_window > 0:
            self.highway = nn.Linear(highway_window, output_dim)

    def forward(self, x):
        # x: (batch, P, m)
        batch_size = x.size(0)

        # CNN: (B, 1, P, m) -> (B, hidC, P-Ck+1, 1) -> (B, hidC, P-Ck+1)
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)  # (B, hidC, P-Ck+1)

        # GRU: 输入 (seq, batch, dim) = (P-Ck+1, B, hidC)
        r = c.permute(2, 0, 1).contiguous()  # (P-Ck+1, B, hidC)
        _, r = self.GRU1(r)  # (1, B, hidR)
        r = self.dropout(torch.squeeze(r, 0))  # (B, hidR)

        # Skip-GRU
        if self.skip > 0 and self.pt > 0:
            # 取最后 pt*skip 步的 CNN 输出
            s = c[:, :, int(-self.pt * self.skip):].contiguous()  # (B, hidC, pt*skip)
            # 重排成 (pt, B*skip, hidC)
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()  # (pt, B, skip, hidC)
            s = s.view(self.pt, batch_size * self.skip, self.hidC)  # (pt, B*skip, hidC)
            _, s = self.GRUskip(s)  # (1, B*skip, hidS)
            s = s.view(batch_size, self.skip * self.hidS)  # (B, skip*hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)  # (B, hidR + skip*hidS)

        res = self.linear1(r)  # (B, output_dim)

        # Highway: 用 close 的最近 hw 步做线性自回归
        if self.hw > 0:
            z = x[:, -self.hw:, 0]  # (B, hw)
            z = self.highway(z)  # (B, output_dim)
            res = res + z

        return res


# ---------- 主流程 ----------
def main():
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(HERE, "outputs", "exp2_lstnet")
    os.makedirs(out_dir, exist_ok=True)

    # 数据
    train_loader, val_loader, test_loader, meta = get_pipeline(batch_size=64)
    print(f"数据: train={meta['n_train']} val={meta['n_val']} test={meta['n_test']} "
          f"features={meta['n_features']} window={meta['window']}")

    # 模型
    model = LSTNetRegressor(
        window=meta["window"], n_features=meta["n_features"],
        hidRNN=64, hidCNN=32, hidSkip=16,
        CNN_kernel=6, skip=5, highway_window=5,
        dropout=0.2, output_dim=1,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")

    # 训练
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        n_epochs=40, lr=1e-3, weight_decay=1e-5,
        device=device, model_name="LSTNet", patience=10,
    )
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    # 可视化 loss
    plot_loss(train_losses, val_losses, os.path.join(out_dir, "1_loss.png"),
              title="LSTNet 训练 loss")

    # 评估
    y_true, y_pred, metrics = evaluate_model(model, test_loader, meta, device)
    print(f"\n测试集指标: {metrics}")
    with open(os.path.join(out_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    # 预测可视化
    test_dates = meta["test_dates"].values
    plot_predictions(test_dates, y_true, y_pred,
                     os.path.join(out_dir, "2_predictions.png"),
                     title="LSTNet 测试集预测 vs 真实")
    plot_scatter(y_true, y_pred, os.path.join(out_dir, "3_scatter.png"),
                 title="LSTNet 预测 vs 真实")

    # 保存预测 csv
    import pandas as pd
    df_out = pd.DataFrame({"date": test_dates, "true": y_true, "pred": y_pred})
    df_out.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    print(f"结果保存到: {out_dir}")
    return metrics


if __name__ == "__main__":
    main()
