"""实验 3: xLSTM 股票预测
=============================
xLSTM（Beck et al. 2024）= mLSTM（矩阵记忆）+ sLSTM（指数门控 + 状态混合）。

为了在 CPU 上运行（无 GPU/CUDA），只使用 mLSTM block（纯 PyTorch实现）。
sLSTM 需要设 backend="vanilla" 才能跑 CPU，但 mLSTM 已足够展示 xLSTM 能力。

架构：
  - 输入投影 Linear(n_features, embedding_dim)
  - xLSTMBlockStack（num_blocks 个 mLSTM block，含 pre-LN + 残差）
  - 取最后时步 -> Linear(embedding_dim, 1) 回归头

输入: (batch, window, n_features)
输出: (batch, 1)  预测下一个交易日标准化 close
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 加入 xLSTM 源码路径
XLSTM_ROOT = os.path.normpath(os.path.join(HERE, "..", "xlstm-main"))
sys.path.insert(0, XLSTM_ROOT)

from common import set_seed, get_pipeline, train_model, evaluate_model
from common.viz import plot_loss, plot_predictions, plot_scatter

from xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig
from xlstm.blocks.mlstm.layer import mLSTMLayerConfig
from xlstm.blocks.mlstm.block import mLSTMBlockConfig


class xLSTMRegressor(nn.Module):
    """xLSTM 回归模型（纯 mLSTM block，CPU 友好）。"""
    def __init__(self, n_features: int, embedding_dim: int = 64,
                 num_blocks: int = 2, dropout: float = 0.1, window: int = 20):
        super().__init__()
        # 输入投影
        self.input_proj = nn.Linear(n_features, embedding_dim)

        # 配置 xLSTM block stack（只用 mLSTM）
        # context_length 必须为正数，用于构造 causal mask，设为窗口大小
        # 注意：xLSTMBlockStackConfig.__post_init__ 会用顶层的 context_length
        # 覆盖 mlstm_block.mlstm.context_length，所以必须在顶层设置
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    context_length=window,
                    num_heads=4,
                    proj_factor=2.0,
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                ),
            ),
            context_length=window,  # 顶层也要设
            num_blocks=num_blocks,
            embedding_dim=embedding_dim,
            slstm_at=[],  # 空 -> 全是 mLSTM
            add_post_blocks_norm=True,
            bias=False,
            dropout=dropout,
        )
        self.xlstm = xLSTMBlockStack(cfg)

        # 回归头
        self.head = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        # x: (batch, window, n_features)
        x = self.input_proj(x)        # (B, W, D)
        x = self.xlstm(x)             # (B, W, D)
        last = x[:, -1, :]            # (B, D)
        return self.head(last)        # (B, 1)


# ---------- 主流程 ----------
def main():
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(HERE, "outputs", "exp3_xlstm")
    os.makedirs(out_dir, exist_ok=True)

    # 数据
    train_loader, val_loader, test_loader, meta = get_pipeline(batch_size=64)
    print(f"数据: train={meta['n_train']} val={meta['n_val']} test={meta['n_test']} "
          f"features={meta['n_features']} window={meta['window']}")

    # 模型
    model = xLSTMRegressor(
        n_features=meta["n_features"],
        embedding_dim=64, num_blocks=2, dropout=0.1,
        window=meta["window"],
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")

    # 训练
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        n_epochs=40, lr=1e-3, weight_decay=1e-5,
        device=device, model_name="xLSTM", patience=10,
    )
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    # 可视化 loss
    plot_loss(train_losses, val_losses, os.path.join(out_dir, "1_loss.png"),
              title="xLSTM 训练 loss")

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
                     title="xLSTM 测试集预测 vs 真实")
    plot_scatter(y_true, y_pred, os.path.join(out_dir, "3_scatter.png"),
                 title="xLSTM 预测 vs 真实")

    # 保存预测 csv
    import pandas as pd
    df_out = pd.DataFrame({"date": test_dates, "true": y_true, "pred": y_pred})
    df_out.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    print(f"结果保存到: {out_dir}")
    return metrics


if __name__ == "__main__":
    main()
