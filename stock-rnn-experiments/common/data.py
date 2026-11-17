"""股票数据加载、特征工程、滑窗、标准化、划分。

为三个 RNN 模型（AWD-LSTM / LSTNet / xLSTM）提供统一的对比基准：
  - 标的：上证指数 sh000001
  - 特征：[close_norm, log_ret, vol_norm, amt_norm, amplitude] 共 5 维
  - 窗口：过去 WINDOW=20 个交易日
  - 目标：预测下一个交易日的 close_norm（标准化后的收盘价）
  - 划分：train 70% / val 15% / test 15%（按时间顺序，不打乱）
  - 标准化：仅用训练集统计量
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

CSV_PATH = r"d:\大三上的资料\假期学习记录\chinese-stock-dataset\chinese-stock-dataset.csv"
STOCK_CODE = "sh000001"
WINDOW = 20
HORIZON = 1
N_FEATURES = 5


def load_index_data(csv_path: str = CSV_PATH, stock_code: str = STOCK_CODE) -> pd.DataFrame:
    """流式读取指定股票代码的行，避免加载 1.5GB 全表。"""
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=500000):
        filtered = chunk[chunk["stock_code"] == stock_code].copy()
        if not filtered.empty:
            chunks.append(filtered)
    df = pd.concat(chunks, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def build_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
    """构造特征矩阵和目标。返回 (X, y, dates)。
    X: (T, N_FEATURES) = [close, log_ret, vol_norm, amt_norm, amplitude]
    y: (T,) = next-day close_norm（已标准化前的 close）
    dates: 对应每个样本（X 的最后一行）的日期
    """
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    amount = df["amount"].values.astype(np.float64)

    log_ret = np.zeros_like(close)
    log_ret[1:] = np.log(close[1:] / close[:-1])
    amplitude = np.zeros_like(close)
    amplitude[1:] = (high[1:] - low[1:]) / close[:-1]

    # 原始特征矩阵（未标准化），每行对应一天
    raw = np.stack([close, log_ret, volume, amount, amplitude], axis=1)
    dates = df["date"].copy()
    # 目标 = 下一日 close
    y_raw = close.copy()
    return raw, y_raw, dates


def make_sequences(raw: np.ndarray, y_raw: np.ndarray, dates: pd.Series,
                   window: int = WINDOW, horizon: int = HORIZON):
    """滑窗构造样本。
    返回:
      X: (N, window, N_FEATURES)
      y: (N,) 目标 = 第 i+window+horizon-1 天的 close
      seq_dates: (N,) 每个样本最后历史日的日期
    """
    T = len(raw)
    X, y, seq_dates = [], [], []
    for i in range(window, T - horizon + 1):
        X.append(raw[i - window:i])
        # 目标：从 X 最后一日往后 horizon 天的 close
        y.append(y_raw[i + horizon - 1])
        seq_dates.append(dates.iloc[i - 1])
    X = np.stack(X, axis=0).astype(np.float32)
    y = np.array(y, dtype=np.float32)
    seq_dates = pd.Series(seq_dates)
    return X, y, seq_dates


def split_data(X: np.ndarray, y: np.ndarray, seq_dates: pd.Series,
               train_ratio: float = 0.7, val_ratio: float = 0.15):
    """按时间顺序划分 train/val/test。返回 6 个 ndarray + 3 个 date Series。"""
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return (
        X[:n_train], y[:n_train], seq_dates.iloc[:n_train],
        X[n_train:n_train + n_val], y[n_train:n_train + n_val], seq_dates.iloc[n_train:n_train + n_val],
        X[n_train + n_val:], y[n_train + n_val:], seq_dates.iloc[n_train + n_val:],
    )


class StockDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test,
                    batch_size: int = 64, shuffle_train: bool = True):
    train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(StockDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(StockDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算 MAE / MSE / RMSE / MAPE / R² / DA(方向准确率)。"""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    # MAPE 用绝对值避免除零
    denom = np.maximum(np.abs(y_true), 1e-8)
    mape = float(np.mean(np.abs(y_true - y_pred) / denom) * 100)
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-12
    r2 = float(1 - ss_res / ss_tot)
    # 方向准确率（涨跌方向对不对）
    if len(y_true) > 1:
        true_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(np.diff(y_pred))
        da = float(np.mean(true_dir == pred_dir) * 100)
    else:
        da = float("nan")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2, "DA": da}


def get_pipeline(batch_size: int = 64, normalize_mode: str = "train_stats"):
    """一键获取数据管道。返回 dataloaders + 标准化参数 + 元信息。
    normalize_mode:
      "train_stats" - 用训练集的 mean/std 标准化（z-score）
      "first_close" - close 用第一个训练 close 归一化，其余特征用各自训练 max
    """
    df = load_index_data()
    raw, y_raw, dates = build_features(df)
    X, y, seq_dates = make_sequences(raw, y_raw, dates)

    # 划分
    (X_train, y_train, d_train,
     X_val, y_val, d_val,
     X_test, y_test, d_test) = split_data(X, y, seq_dates)

    # 标准化（仅用训练集统计量）
    # 对每个特征列分别标准化
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8
    X_train_n = (X_train - mean) / std
    X_val_n = (X_val - mean) / std
    X_test_n = (X_test - mean) / std

    # 目标 y 也标准化（用训练集 y 的 mean/std）
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8
    y_train_n = (y_train - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std
    y_test_n = (y_test - y_mean) / y_std

    train_loader, val_loader, test_loader = get_dataloaders(
        X_train_n, y_train_n, X_val_n, y_val_n, X_test_n, y_test_n, batch_size=batch_size
    )

    meta = {
        "n_features": X_train.shape[-1],
        "window": WINDOW,
        "horizon": HORIZON,
        "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
        "feature_mean": mean, "feature_std": std,
        "y_mean": y_mean, "y_std": y_std,
        "train_dates": d_train, "val_dates": d_val, "test_dates": d_test,
        "y_train_raw": y_train, "y_val_raw": y_val, "y_test_raw": y_test,
    }
    return train_loader, val_loader, test_loader, meta
