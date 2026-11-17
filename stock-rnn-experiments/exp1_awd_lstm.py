"""实验 1: AWD-LSTM 股票预测
================================
把 AWD-LSTM 语言模型改造成时间序列回归模型：
  - 去掉 Embedding（连续值输入）和分类 decoder
  - 保留 WeightDrop（dropout on weight_hh_l0）+ LockedDropout（层间 dropout）
  - 输入投影 Linear(n_features, ninp) 替代 embedding
  - 取最后时步输出 -> Linear(nhid, 1) 回归头

输入: (batch, window, n_features)
输出: (batch, 1)  预测下一个交易日标准化 close
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import set_seed, get_pipeline, train_model, evaluate_model
from common.viz import plot_loss, plot_predictions, plot_scatter


# ---------- AWD-LSTM 回归模型 ----------
class LockedDropout(nn.Module):
    """Variational dropout: 同一个时间步内所有位置共享同一个 mask。"""
    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        # x: (batch, seq, dim) -> mask 沿 (1, seq, 1) 不对，应该是 (batch, 1, dim) 广播
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        return m * x / (1 - dropout)


class WeightDrop(nn.Module):
    """对指定权重做 dropout（应用于 recurrent weight_hh_l0）。
    使用 PyTorch 官方推荐的 nn.utils.parametrize，兼容新版 PyTorch / GPU / CUDNN。"""
    def __init__(self, module, weights, dropout=0.0):
        super().__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        import torch.nn.utils.parametrize as parametrize
        for name_w in weights:
            parametrize.register_parametrization(
                module, name_w, _WeightDropParametrization(dropout)
            )

    def forward(self, *args):
        return self.module(*args)


class _WeightDropParametrization(nn.Module):
    """Parametrization: 每次访问 weight 时动态应用 dropout。"""
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, X):
        return F.dropout(X, p=self.p, training=self.training)


class AWDLSTMRegressor(nn.Module):
    """AWD-LSTM 改造的回归模型。"""
    def __init__(self, n_features: int, ninp: int = 64, nhid: int = 64,
                 nlayers: int = 2, dropout=0.3, dropouth=0.3, dropouti=0.3,
                 wdrop=0.3):
        super().__init__()
        self.lockdrop = LockedDropout()
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers

        # 输入投影（替代 embedding）
        self.input_proj = nn.Linear(n_features, ninp)

        # 堆叠 LSTM（每层单独，便于 WeightDrop）
        rnns = []
        for l in range(nlayers):
            in_dim = ninp if l == 0 else nhid
            rnn = nn.LSTM(in_dim, nhid, 1, dropout=0, batch_first=True)
            if wdrop:
                rnn = WeightDrop(rnn, ["weight_hh_l0"], dropout=wdrop)
            rnns.append(rnn)
        self.rnns = nn.ModuleList(rnns)

        # 回归头
        self.decoder = nn.Linear(nhid, 1)
        self.dropout = dropout
        self.dropouth = dropouth
        self.dropouti = dropouti

        self.init_weights()

    def init_weights(self):
        self.input_proj.weight.data.uniform_(-0.1, 0.1)
        self.input_proj.bias.data.fill_(0)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        # x: (batch, seq, n_features)
        emb = self.input_proj(x)
        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        for l, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output)
            new_hidden.append(new_h)
            if l != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
        output = self.lockdrop(raw_output, self.dropout)
        # 取最后时步
        last = output[:, -1, :]  # (batch, nhid)
        pred = self.decoder(last)  # (batch, 1)
        return pred


# ---------- 主流程 ----------
def main():
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(HERE, "outputs", "exp1_awd_lstm")
    os.makedirs(out_dir, exist_ok=True)

    # 数据
    train_loader, val_loader, test_loader, meta = get_pipeline(batch_size=64)
    print(f"数据: train={meta['n_train']} val={meta['n_val']} test={meta['n_test']} "
          f"features={meta['n_features']} window={meta['window']}")

    # 模型
    model = AWDLSTMRegressor(
        n_features=meta["n_features"], ninp=64, nhid=64, nlayers=2,
        dropout=0.3, dropouth=0.3, dropouti=0.3, wdrop=0.3,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")

    # 训练
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        n_epochs=40, lr=1e-3, weight_decay=1e-5,
        device=device, model_name="AWD-LSTM", patience=10,
    )
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    # 可视化 loss
    plot_loss(train_losses, val_losses, os.path.join(out_dir, "1_loss.png"),
              title="AWD-LSTM 训练 loss")

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
                     title="AWD-LSTM 测试集预测 vs 真实")
    plot_scatter(y_true, y_pred, os.path.join(out_dir, "3_scatter.png"),
                 title="AWD-LSTM 预测 vs 真实")

    # 保存预测 csv
    import pandas as pd
    df_out = pd.DataFrame({"date": test_dates, "true": y_true, "pred": y_pred})
    df_out.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    print(f"结果保存到: {out_dir}")
    return metrics


if __name__ == "__main__":
    main()
