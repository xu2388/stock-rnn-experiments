# RNN 系列模型股票预测实验

用 AWD-LSTM、LSTNet、xLSTM 三个 RNN 系列模型，对上证指数（sh000001）做下一交易日收盘价预测，并对比预测效果。

## 数据与任务

- **数据来源**：chinese-stock-dataset（1.5GB，1483 只股票，1990–2025）
- **标的**：上证指数 `sh000001`（8534 个交易日）
- **特征（5 维）**：`close`、`log_ret`、`volume`、`amount`、`amplitude`
- **输入**：过去 20 个交易日；**输出**：下一交易日收盘价
- **标准化**：z-score，仅用训练集统计量
- **划分**：按时间顺序 70% / 15% / 15%，不打乱

## 三个模型

| 模型 | 脚本 | 核心架构 |
|------|------|----------|
| AWD-LSTM | `exp1_awd_lstm.py` | 2 层 LSTM + WeightDrop + LockedDropout |
| LSTNet | `exp2_lstnet.py` | CNN + GRU + 跳跃连接 |
| xLSTM | `exp3_xlstm.py` | sLSTM / mLSTM 结构 |

## 怎么跑

```bash
pip install torch numpy pandas matplotlib
python run_all.py
```

## 说明

- 原始数据集（1.5GB CSV）未随仓库上传，`chinese-poetry/` 目录已排除
- `outputs/` 保留了各模型的预测曲线与误差对比图
