# RNN 系列模型股票预测实验记录

## 实验目标

使用三个 RNN 系列模型对中国股票数据集（上证指数 sh000001）进行下一日收盘价预测，对比三者的预测效果。

## 数据集

- **来源**: `chinese-stock-dataset/chinese-stock-dataset.csv`（1.5GB，1483只股票，1990-2025）
- **标的**: 上证指数 `sh000001`（8534 个交易日）
- **特征（5维）**: `[close, log_ret, volume, amount, amplitude]`
- **窗口**: 过去 20 个交易日
- **目标**: 下一交易日 close
- **标准化**: z-score，仅用训练集统计量
- **划分**: train 70% (5959) / val 15% (1277) / test 15% (1278)，按时间顺序不打乱

## 三个模型

| 模型 | 来源 | 核心架构 | 参数量 | 特点 |
|------|------|---------|--------|------|
| AWD-LSTM | awd-lstm-lm-master | 2层 LSTM + WeightDrop + LockedDropout | ~67K | 语言模型正则化技术（权重dropout、variational