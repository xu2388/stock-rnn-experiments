# RNN 系列模型股票价格预测实验报告

**实验日期**：2026-07-10  
**数据集**：[chinese-stock-dataset.csv](file:///d:/大三上的资料/假期学习记录/chinese-stock-dataset/chinese-stock-dataset.csv)  
**实验代码目录**：[stock_rnn_experiments/](file:///d:/大三上的资料/假期学习记录/stock_rnn_experiments/)  
**结果目录**：[outputs/](file:///d:/大三上的资料/假期学习记录/stock_rnn_experiments/outputs/)

---

## 目录

1. [实验任务与数据](#1-实验任务与数据)
2. [三个模型架构详解](#2-三个模型架构详解)
   - 2.1 [AWD-LSTM](#21-awd-lstm)
   - 2.2 [LSTNet](#22-lstnet)
   - 2.3 [xLSTM](#23-xlstm)
3. [训练流程与超参数](#3-训练流程与超参数)
4. [评价指标含义](#4-评价指标含义)
5. [实验结果与对比](#5-实验结果与对比)
6. [模型效果分析](#6-模型效果分析)
7. [结论与建议](#7-结论与建议)
8. [附录：生成文件清单](#8-附录生成文件清单)

---

## 1. 实验任务与数据

### 1.1 任务定义

本实验使用中国股票日线数据，训练三个 RNN 系列模型：**AWD-LSTM**、**LSTNet**、**xLSTM**，预测上证指数（`sh000001`）下一交易日的收盘价。

- **输入**：过去 `window = 20` 个交易日的多维度特征序列
- **输出**：下一交易日的收盘价（经标准化后预测，再反标准化）
- **类型**：单步时间序列回归（one-step ahead regression）

### 1.2 原始数据字段

原始 CSV 字段为：

| 字段 | 含义 |
|------|------|
| `date` | 交易日期，格式 `%Y%m%d` |
| `open` | 开盘价 |
| `high` | 最高价 |
| `low` | 最低价 |
| `close` | 收盘价 |
| `volume` | 成交量 |
| `amount` | 成交额 |
| `stock_code` | 股票代码 |
| `market` | 市场（`sh` / `sz`） |

本实验只选取 `stock_code = sh000001`（上证指数），共 8,534 条日线。

### 1.3 特征工程

在 [common/data.py](file:///d:/大三上的资料/假期学习记录/stock_rnn_experiments/common/data.py) 中构造 5 维特征：

| 特征维度 | 符号 | 计算方式 |
|----------|------|----------|
| `close` | 第 0 维 | 当日收盘价 |
| `log_ret` | 第 1 维 | `log(close_t / close_{t-1})` |
| `volume` | 第 2 维 | 当日成交量 |
| `amount` | 第 3 维 | 当日成交额 |
| `amplitude` | 第 4 维 | `(high_t - low_t) / close_{t-1}` |

因此原始特征矩阵 `raw` 的形状为：

```
raw.shape = (T, 5)
```

其中 `T = 8534` 是交易日总数。

### 1.4 滑窗构造样本

使用 `window = 20`、`horizon = 1`：

```python
for i in range(window, T - horizon + 1):
    X.append(raw[i - window:i])           # (20, 5)
    y.append(y_raw[i + horizon - 1])      # 标量 close
```

生成样本：

- `X.shape = (N, 20, 5)`
- `y.shape = (N,)`
- `N = 8534 - 20 + 1 = 8515`

### 1.5 数据划分

按时间顺序划分（**不打乱**）：

| 子集 | 比例 | 样本数 |
|------|------|--------|
| 训练集 train | 70% | 5,959 |
| 验证集 val | 15% | 1,277 |
| 测试集 test | 15% | 1,278 |

### 1.6 标准化

- **特征标准化**：对 `X` 的每个特征列分别做 z-score，均值和标准差只从训练集计算：

  ```
  X_norm = (X - mean_train) / std_train
  ```

- **目标标准化**：对 `y`（收盘价）也做 z-score：

  ```
  y_norm = (y - y_mean_train) / y_std_train
  ```

模型预测的是 `y_norm`，评估时再反标准化回原始价格：

```python
y_pred_raw = y_pred_norm * y_std + y_mean
```

---

## 2. 三个模型架构详解

三个模型都接收统一输入：

```
X: (batch_size, window=20, n_features=5)
```

输出统一为：

```
y_pred: (batch_size, 1)
```

即预测下一个交易日的标准化收盘价。

---

### 2.1 AWD-LSTM

**代码文件**：[exp1_awd_lstm.py](file:///d:/大三上的资料/假期学习记录/stock_rnn_experiments/exp1_awd_lstm.py)

#### 2.1.1 模型来源

AWD-LSTM（ASGD Weight-Dropped LSTM）原论文：

> Merity et al., "Regularizing and Optimizing LSTM Language Models", ICLR 2018.

原始 AWD-LSTM 是语言模型，包含：

1. `Embedding` 层（离散 token → 连续向量）
2. 多层 `LSTM`，每层之间加 `LockedDropout`
3. 对 `weight_hh_l0` 做 `WeightDrop`
4. `Linear` 解码器输出词表分布

本实验将其改造为回归模型：

- **去掉 `Embedding`**，改为 `Linear` 输入投影，把 5 维特征映射到 `ninp` 维。
- **保留 `WeightDrop` 和 `LockedDropout`**，保持 AWD-LSTM 的正则化特色。
- **输出层改为 `Linear(nhid, 1)`**，取最后一个时间步的隐状态做回归。

#### 2.1.2 各模块维度变化

设输入 `x.shape = (B, S, F)`，其中：

- `B = batch_size`
- `S = window = 20`
- `F = n_features = 5`

本实验超参：`ninp = 64`, `nhid = 64`, `nlayers = 2`。

| 模块 | 输入形状 | 输出形状 | 说明 |
|------|----------|----------|------|
| 输入 `x` | `(B, 20, 5)` | — | 原始特征序列 |
| `input_proj`（`Linear(5, 64)`） | `(B, 20, 5)` | `(B, 20, 64)` | 特征升维到 `ninp` |
| `lockdrop`（`dropouti=0.3`） | `(B, 20, 64)` | `(B, 20, 64)` | variational dropout，mask 形状 `(B, 1, 64)` |
| **LSTM 第 0 层**（`batch_first=True`） | `(B, 20, 64)` | `(output: B, 20, 64; hidden: 1, B, 64)` | 输入维度 64，隐状态维度 64 |
| `lockdrop`（`dropouth=0.3`，层间） | `(B, 20, 64)` | `(B, 20, 64)` | 只对前 `nlayers-1` 层输出加 |
| **LSTM 第 1 层** | `(B, 20, 64)` | `(output: B, 20, 64; hidden: 1, B, 64)` | 同上 |
| `lockdrop`（`dropout=0.3`，最终） | `(B, 20, 64)` | `(B, 20, 64)` | 最后一层输出也加 dropout |
| 取最后时间步 `last = output[:, -1, :]` | `(B, 20, 64)` | `(B, 64)` | 提取序列最后时刻表示 |
| `decoder`（`Linear(64, 1)`） | `(B, 64)` | `(B, 1)` | 回归输出 |

#### 2.1.3 WeightDrop 机制

在原始 AWD-LSTM 中，`WeightDrop` 对 LSTM 的 `weight_hh_l0`（隐状态到隐状态的循环权重）做 dropout，这是一种比输入/输出 dropout 更强的正则化。

本实验使用新版 PyTorch 兼容的实现：通过 `torch.nn.utils.parametrize.register_parametrization`，把 `weight_hh_l0` 参数化：

```python
parametrize.register_parametrization(
    rnn, "weight_hh_l0", _WeightDropParametrization(dropout=0.3)
)
```

每次 LSTM forward 访问 `weight_hh_l0` 时，会自动应用 `F.dropout(raw_weight, p=0.3)`。

#### 2.1.4 参数量

```
模型参数量: 67,009
```

主要参数来源：

- `input_proj`：`5 * 64 + 64 = 448`
- LSTM 第 0 层 `weight_ih_l0`：`64 * 64 * 4 = 16,384`；`weight_hh_l0`：`64 * 64 * 4 = 16,384`；bias：`64 * 4 * 2 = 512`
- LSTM 第 1 层同上
- `decoder`：`64 * 1 + 1 = 65`

---

### 2.2 LSTNet

**代码文件**：[exp2_lstnet.py](file:///d:/大三上的资料/假期学习记录/stock_rnn_experiments/exp2_lstnet.py)

#### 2.2.1 模型来源

LSTNet 原论文：

> Lai et al., "Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks", SIGIR 2018.

LSTNet 是专门为多变量时间序列预测设计的模型，核心思想是：

- **CNN**：提取变量间和局部时间模式。
- **GRU**：在 CNN 特征上提取长程时序依赖。
- **Skip-GRU**：按固定周期 `skip` 重新组织序列，捕捉周期性（如日线中的周模式）。
- **Highway**：用最近 `hw` 步的观测值做线性自回归残差，提升短期预测稳定性。

本实验把原始 LSTNet 的输出头从 `m` 维改为 `1` 维，只预测收盘价。

#### 2.2.2 各模块维度变化

设输入 `x.shape = (B, P, m)`，其中：

- `B = batch_size`
- `P = window = 20`
- `m = n_features = 5`

本实验超参：`hidCNN = 32`, `hidRNN = 64`, `hidSkip = 16`, `CNN_kernel = 6`, `skip = 5`, `highway_window = 5`。

计算 `pt = (P - Ck) // skip = (20 - 6) // 5 = 2`。

| 模块 | 输入形状 | 输出形状 | 说明 |
|------|----------|----------|------|
| 输入 `x` | `(B, 20, 5)` | — | 原始特征序列 |
| `view` 为 4D | `(B, 20, 5)` | `(B, 1, 20, 5)` | Conv2d 需要 4D 输入 `(N,C,H,W)` |
| `conv1`（`Conv2d(1, 32, kernel_size=(6, 5))`） | `(B, 1, 20, 5)` | `(B, 32, 15, 1)` | 时间方向卷积：20-6+1=15；特征方向卷积：5-5+1=1 |
| `squeeze` 去掉最后一维 | `(B, 32, 15, 1)` | `(B, 32, 15)` | CNN 输出序列，时间长度 15 |
| **GRU1**（`GRU(32, 64)`） | `(15, B, 32)` | `(1, B, 64)` | 输入需转置为 `(seq, batch, dim)`，取最后时刻隐状态 |
| `dropout` | `(1, B, 64)` | `(B, 64)` | squeeze + dropout |
| **Skip-GRU** | | | |
| 取 CNN 输出最后 `pt*skip = 10` 步 | `(B, 32, 15)` | `(B, 32, 10)` | |
| `view` 为 4D | `(B, 32, 10)` | `(B, 32, 2, 5)` | `(pt=2, skip=5)` |
| `permute` + `view` | `(B, 32, 2, 5)` | `(2, B*5, 32)` | 每个 skip 组形成独立序列 |
| `GRUskip`（`GRU(32, 16)`） | `(2, B*5, 32)` | `(1, B*5, 16)` | 每组跑 GRU |
| `view` | `(1, B*5, 16)` | `(B, 5*16) = (B, 80)` | 拼接 skip 组状态 |
| `dropout` + `cat` | `(B, 64)` 与 `(B, 80)` | `(B, 144)` | 与 GRU1 输出拼接 |
| `linear1`（`Linear(144, 1)`） | `(B, 144)` | `(B, 1)` | 合并特征回归 |
| **Highway** | | | |
| 取 close 最近 `hw=5` 步 | `(B, 20, 5)` | `(B, 5)` | 取 `x[:, -5:, 0]` |
| `highway`（`Linear(5, 1)`） | `(B, 5)` | `(B, 1)` | 线性自回归残差 |
| 最终输出 | `(B, 1)` + `(B, 1)` | `(B, 1)` | `res = linear1(r) + highway(z)` |

#### 2.2.3 Skip-GRU 的直观解释

原始 CNN 输出序列长度为 15。取最后 10 步，按周期 `skip=5` 切分为 5 个组，每组包含 2 个时间点：

```
时间步: [t-9, t-8, t-7, t-6, t-5, t-4, t-3, t-2, t-1, t]
组 0:  [t-9,       t-4,            ]  （间隔 5）
组 1:  [    t-8,        t-3,       ]
组 2:  [        t-7,        t-2,   ]
组 3:  [            t-6,        t-1]
组 4:  [                t-5,        t]
```

每组独立跑一个 `GRU(32, 16)`，捕捉同一相位（如每周同一天）的演化规律。

#### 2.2.4 参数量

```
模型参数量: 22,359
```

主要参数来源：

- `conv1`：`1 * 32 * 6 * 5 + 32 = 992`
- `GRU1`：`32 * 64 * 3 + 64 * 64 * 3 + 64 * 3 * 2 = 18,816 + 384 = 19,200`（约）
- `GRUskip`：`32 * 16 * 3 + 16 * 16 * 3 + 16 * 3 * 2 = 2,688`
- `linear1`：`144 * 1 + 1 = 145`
- `highway`：`5 * 1 + 1 = 6`

---

### 2.3 xLSTM

**代码文件**：[exp3_xlstm.py](file:///d:/大三上的资料/假期学习记录/stock_rnn_experiments/exp3_xlstm.py)

#### 2.3.1 模型来源

xLSTM 原论文：

> Beck et al., "xLSTM: Extended Long Short-Term Memory", 2024.

xLSTM 提出两种新 LSTM 变体：

- **sLSTM**（scalar LSTM）：引入指数门控和状态归一化，增强长程记忆。
- **mLSTM**（matrix LSTM）：把 LSTM 的标量记忆单元扩展为矩阵记忆，类似线性注意力机制。

本实验由于环境限制（CPU 运行），只使用 **mLSTM block**（纯 PyTorch 实现，无需 CUDA kernel）。

#### 2.3.2 整体架构

```python
class xLSTMRegressor(nn.Module):
    def __init__(self, n_features=5, embedding_dim=64, num_blocks=2, dropout=0.1, window=20):
        self.input_proj = nn.Linear(n_features, embedding_dim)
        self.xlstm = xLSTMBlockStack(...)   # 2 个 mLSTM block
        self.head = nn.Linear(embedding_dim, 1)
```

输入输出：

| 模块 | 输入形状 | 输出形状 | 说明 |
|------|----------|----------|------|
| 输入 `x` | `(B, 20, 5)` | — | 原始特征序列 |
| `input_proj`（`Linear(5, 64)`） | `(B, 20, 5)` | `(B, 20, 64)` | 特征升维到 `embedding_dim` |
| `xLSTMBlockStack`（2 个 block） | `(B, 20, 64)` | `(B, 20, 64)` | 每个 block 含 pre-LN + mLSTM + 残差 + post norm |
| 取最后时间步 | `(B, 20, 64)` | `(B, 64)` | |
| `head`（`Linear(64, 1)`） | `(B, 64)` | `(B, 1)` | 回归输出 |

#### 2.3.3 mLSTM Block 内部维度变化

一个 `mLSTMBlock` 的结构：

```
x = x + mLSTM(LayerNorm(x))
```

`mLSTMLayer` 内部流程（设 `D = embedding_dim = 64`，`proj_factor = 2.0`，所以 `inner_dim = D * 2 = 128`，`num_heads = 4`）：

| 子模块 | 输入形状 | 输出形状 | 说明 |
|--------|----------|----------|------|
| `proj_up`（`Linear(64, 256)`） | `(B, 20, 64)` | `(B, 20, 256)` | 升维到 `2 * inner_dim = 256` |
| `split` | `(B, 20, 256)` | `x_mlstm: (B, 20, 128), z: (B, 20, 128)` | 沿最后一维平分 |
| `conv1d`（因果卷积，kernel=4） | `(B, 20, 128)` | `(B, 20, 128)` | 局部时间建模 |
| `q_proj`（headwise Linear） | `(B, 20, 128)` | `(B, 20, 128)` | Q 投影 |
| `k_proj`（headwise Linear） | `(B, 20, 128)` | `(B, 20, 128)` | K 投影 |
| `v_proj`（headwise Linear） | `(B, 20, 128)` | `(B, 20, 128)` | V 投影 |
| `mlstm_cell` | `q,k,v: (B, 20, 128)` | `(B, 20, 128)` | 矩阵记忆递归计算 |
| `learnable_skip` | `(B, 20, 128)` | `(B, 20, 128)` | 可学习残差 |
| `ogate`：`h_state = (h_tilde + skip) * SiLU(z)` | `(B, 20, 128)` | `(B, 20, 128)` | 输出门控 |
| `proj_down`（`Linear(128, 64)`） | `(B, 20, 128)` | `(B, 20, 64)` | 降维回 `embedding_dim` |
| `dropout` | `(B, 20, 64)` | `(B, 20, 64)` | |

#### 2.3.4 mLSTM Cell 的核心计算

在 [xlstm/blocks/mlstm/cell.py](file:///d:/大三上的资料/假期学习记录/xlstm-main/xlstm/blocks/mlstm/cell.py) 中，mLSTM cell 把 Q/K/V reshape 为多头形式：

```python
q = q.view(B, S, num_heads, head_dim)   # (B, 20, 4, 32)
q = q.transpose(1, 2)                    # (B, 4, 20, 32)
```

然后利用输入门 `i` 和遗忘门 `f` 更新矩阵记忆 `C_t` 和隐状态 `h_t`：

```
C_t = f_t * C_{t-1} + i_t * (k_t^T @ v_t)
h_t = C_t @ q_t / normalization
```

这相当于一种带有门控的线性注意力，理论上能捕捉长程依赖。

#### 2.3.5 参数量

```
模型参数量: 60,817
```

主要参数来源：

- `input_proj`：`5 * 64 + 64 = 448`
- 每个 mLSTM block：
  - `proj_up`：`64 * 128 + 0 = 8,192`（bias=False）
  - Q/K/V headwise 投影：3 * (128 * 128 / 32？具体取决于 headwise 实现) ≈ 几千
  - `conv1d`：`128 * 128 * 4 ≈ 65,536`？不对，是 `kernel_size * in_channels * out_channels`，因果卷积是 `4 * 128 * 128 = 65,536`
  - `proj_down`：`128 * 64 = 8,192`
  - mLSTM cell：`igate` 和 `fgate`：`3 * 128 * 4 = 1,536`
- `head`：`64 * 1 + 1 = 65`

（实际参数量以 PyTorch 统计为准，上述为估算。）

---

## 3. 训练流程与超参数

训练代码统一在 [common/trainer.py](file:///d:/大三上的资料/假期学习记录/stock_rnn_experiments/common/trainer.py) 中。

### 3.1 训练设置

| 超参数 | 取值 | 说明 |
|--------|------|------|
| 优化器 | `AdamW` | 带权重衰减的 Adam |
| 学习率 | `1e-3` | 初始学习率 |
| weight_decay | `1e-5` | L2 正则化 |
| 学习率调度 | `ReduceLROnPlateau` | val loss 不下降 5 个 epoch，学习率乘以 0.5 |
| 损失函数 | `MSELoss` | 均方误差，预测标准化后的目标 |
| batch_size | 64 | |
| 最大 epoch | 40 | |
| 早停 patience | 10 | 10 个 epoch 验证集 loss 不下降则停止 |
| 梯度裁剪 | max_norm=5.0 | 防止 RNN 梯度爆炸 |

### 3.2 早停与最佳模型选择

每个 epoch 结束后计算验证集 loss，保存验证 loss 最低的模型状态到内存。训练结束后加载最佳权重进行评估。这种机制有效防止过拟合。

### 3.3 评估流程

```python
y_pred_norm = model(X_test_norm)
y_pred_raw = y_pred_norm * y_std_train + y_mean_train
y_true_raw = y_test_norm * y_std_train + y_mean_train
metrics = eval_metrics(y_true_raw, y_pred_raw)
```

所有指标都在**原始价格尺度**上计算，保证可解释性。

---

## 4. 评价指标含义

在 [common/data.py](file:///d:/大三上的资料/假期学习记录/stock_rnn_experiments/common/data.py) 中定义了 6 个指标。

### 4.1 MAE（Mean Absolute Error，平均绝对误差）

```
MAE = (1/N) * Σ |y_true_i - y_pred_i|
```

- **含义**：预测值与真实值之间平均差距的绝对值。
- **单位**：与目标变量相同，这里是上证指数点数。
- **特点**：对异常值不如 MSE 敏感，直观反映平均预测偏差。

### 4.2 MSE（Mean Squared Error，均方误差）

```
MSE = (1/N) * Σ (y_true_i - y_pred_i)^2
```

- **含义**：预测误差的平方的平均。
- **单位**：目标变量单位的平方（点数²）。
- **特点**：对大误差惩罚更重，常用于训练损失。

### 4.3 RMSE（Root Mean Squared Error，均方根误差）

```
RMSE = sqrt(MSE)
```

- **含义**：MSE 开根号，使单位与目标变量一致。
- **单位**：上证指数点数。
- **特点**：综合反映预测误差大小，对大误差的惩罚比 MAE 更重。

### 4.4 MAPE（Mean Absolute Percentage Error，平均绝对百分比误差）

```
MAPE = (1/N) * Σ |y_true_i - y_pred_i| / |y_true_i| * 100%
```

- **含义**：预测误差占真实值的百分比。
- **单位**：%。
- **特点**：便于不同量级序列之间的比较。但真实值接近 0 时会很大，本实验用 `max(|y_true|, 1e-8)` 做保护。

### 4.5 R²（决定系数）

```
R² = 1 - SS_res / SS_tot
SS_res = Σ (y_true_i - y_pred_i)^2
SS_tot = Σ (y_true_i - mean(y_true))^2
```

- **含义**：模型解释了多少目标变量的方差。
- **取值范围**：(-∞, 1]。
- **特点**：R² = 1 表示完美预测；R² = 0 表示模型只预测均值；R² < 0 表示模型比均值还差。

### 4.6 DA（Directional Accuracy，方向准确率）

```
DA = (1/(N-1)) * Σ [sign(y_true_{i+1} - y_true_i) == sign(y_pred_{i+1} - y_pred_i)] * 100%
```

- **含义**：模型预测涨跌方向（相对前一天上涨还是下跌）的正确率。
- **单位**：%。
- **特点**：对股票预测非常重要。50% 相当于随机猜测。

---

## 5. 实验结果与对比

### 5.1 测试集指标

| 模型 | MAE | MSE | RMSE | MAPE(%) | R² | DA(%) |
|------|-----|-----|------|---------|----|----|
| **LSTNet** | **32.74** | **1802.85** | **42.46** | **0.99** | **0.9719** | 48.24 |
| AWD-LSTM | 36.04 | 2299.43 | 47.95 | 1.09 | 0.9642 | 48.47 |
| xLSTM | 99.79 | 18901.30 | 137.48 | 2.92 | 0.7056 | 49.73 |

对应 JSON 文件：[outputs/comparison.json](file:///d:/大三上的资料/假期学习记录/stock_rnn_experiments/outputs/comparison.json)

### 5.2 指标解读

**从价格回归角度看**：

- LSTNet 的 MAE 仅 32.74 点，相对于上证指数 3500–4000 点的水平，误差约 0.9%，表现优秀。
- AWD-LSTM 紧随其后，误差约 1.0%。
- xLSTM 误差约 2.9%，明显落后。

**从方差解释角度看**：

- LSTNet 和 AWD-LSTM 的 R² 都在 0.96 以上，说明模型能解释收盘价绝大部分波动。
- xLSTM 的 R² 仅 0.71，说明拟合能力明显不足。

**从方向预测角度看**：

- 三个模型的 DA 都在 48%–50% 之间，接近随机水平。
- 这说明：模型能较好拟合价格水平，但预测涨跌方向非常困难。

### 5.3 可视化结果

所有可视化文件位于 [outputs/](file:///d:/大三上的资料/假期学习记录/stock_rnn_experiments/outputs/) 目录：

| 文件 | 内容 |
|------|------|
| [comparison_metrics.png](file:///d:/大三上的资料/假期学习记录/stock_rnn_experiments/outputs/comparison_metrics.png) | 三模型 MAE / RMSE / MAPE / R² / DA 柱状图 |
| [overlay_predictions.png](file:///d:/大三上的资料/假期学习记录/stock_rnn_experiments/outputs/overlay_predictions.png) | 三模型预测曲线与真实价格叠加对比 |
| `exp1_awd_lstm/1_loss.png` | AWD-LSTM 训练/验证 loss |
| `exp1_awd_lstm/2_predictions.png` | AWD-LSTM 测试集预测 vs 真实 |
| `exp1_awd_lstm/3_scatter.png` | AWD-LSTM 预测散点图 |
| `exp2_lstnet/1_loss.png` | LSTNet 训练/验证 loss |
| `exp2_lstnet/2_predictions.png` | LSTNet 测试集预测 vs 真实 |
| `exp2_lstnet/3_scatter.png` | LSTNet 预测散点图 |
| `exp3_xlstm/1_loss.png` | xLSTM 训练/验证 loss |
| `exp3_xlstm/2_predictions.png` | xLSTM 测试集预测 vs 真实 |
| `exp3_xlstm/3_scatter.png` | xLSTM 预测散点图 |

---

## 6. 模型效果分析

### 6.1 为什么 LSTNet 表现最好？

LSTNet 专为**多变量时间序列预测**设计，其结构天然适合股票这种同时具有局部波动、趋势和周期性的数据：

1. **CNN 提取局部模式**：`Conv2d(kernel=(6, 5))` 同时沿时间和特征方向卷积，能捕捉诸如"连续几根 K 线的组合形态"。
2. **GRU 捕捉趋势**：在 CNN 输出上跑 GRU，综合过去 15 步的信息，得到长期趋势表示。
3. **Skip-GRU 捕捉周期性**：按 `skip=5` 分组，尝试捕捉股市中可能的周模式（虽然日线不严格对应自然周，但周期性结构对预测仍有帮助）。
4. **Highway 自回归残差**：直接用最近 5 天收盘价做线性回归，相当于给模型加了一个"动量"基准，让短期预测更稳定。
5. **正则化适中**：dropout=0.2，参数量最小（22K），最不容易过拟合。

### 6.2 为什么 AWD-LSTM 接近 LSTNet？

AWD-LSTM 是纯 RNN 架构，但通过强正则化弥补了结构上的不足：

1. **WeightDrop** 对循环权重做 dropout，有效正则化 LSTM 的隐状态转移。
2. **LockedDropout** 是变分 dropout，比标准 dropout 更适合序列数据。
3. 多层 LSTM 有足够容量学习股价序列的非线性动态。

但 AWD-LSTM 没有显式的 CNN 和周期建模，所以在 RMSE 上略逊于 LSTNet。

### 6.3 为什么 xLSTM 表现最差？

xLSTM（mLSTM）在这个任务上表现不佳，可能原因有：

1. **任务规模与模型复杂度不匹配**：mLSTM 设计用于大规模语言建模（长序列、高维度），而本任务窗口仅 20 步、特征 5 维，矩阵记忆的优势无法发挥。
2. **超参可能未调优**：mLSTM 对 `num_heads`、`proj_factor`、`conv1d_kernel_size` 等超参较敏感，本实验使用默认配置，未做网格搜索。
3. **缺少 sLSTM**：原论文中 xLSTM 通常混合 mLSTM 和 sLSTM 块，本实验只用了 mLSTM。
4. **预训练与初始化**：xLSTM 在大规模任务上通常需要谨慎的初始化，小数据上容易训练不稳定。
5. **特征交互方式**：mLSTM 把输入映射到 Q/K/V 做注意力式交互，但股价特征之间的交互可能不需要这么复杂的结构。

### 6.4 方向预测为什么困难？

三个模型的 DA 都在 48%–50%，说明：

- 模型学到了**价格水平的平滑趋势**（因为收盘价本身是高度自相关的）。
- 但模型难以预测**价格变化的方向**，因为日收益率接近随机游走，历史信息对未来方向的预测力有限。
- 这一点也符合金融学的有效市场假说：短期价格方向很难被历史价格预测。

### 6.5 关于"预测效果"的客观认识

- **R² 高 ≠ 能赚钱**：R² 高可能是因为模型学会了"明天的价格和今天差不多"这一简单规律，而不是真正预测了波动。
- **DA 接近 50% 是正常且重要的信号**：它提醒我们，该模型不宜直接用于交易信号决策。
- **本实验的价值**：展示了如何将 AWD-LSTM、LSTNet、xLSTM 应用于时间序列回归，并公平比较它们的拟合能力。

---

## 7. 结论与建议

### 7.1 主要结论

1. 在本次上证指数收盘价预测任务中，**LSTNet > AWD-LSTM > xLSTM**。
2. LSTNet 凭借其 CNN + GRU + Skip-GRU + Highway 的结构，最适合这种短窗口多变量时序预测。
3. AWD-LSTM 通过强正则化也能取得接近效果，但参数量更大。
4. xLSTM 在小规模金融时序任务上表现不佳，可能需要更长的上下文、更高的维度或更仔细的超参调优。
5. 三个模型都能较好拟合价格水平（R² > 0.96 for LSTNet/AWD-LSTM），但**无法稳定预测涨跌方向**。

### 7.2 后续改进建议

1. **多步预测**：把 horizon 从 1 改为 5 或 10，评估模型对中期趋势的预测能力。
2. **更多特征**：加入技术指标（MACD、KDJ、布林带宽度）、宏观变量、市场情绪指标。
3. **模型融合**：用 LSTNet 和 AWD-LSTM 的预测做集成，可能进一步降低 RMSE。
4. **xLSTM 调优**：尝试 sLSTM + mLSTM 混合、调整 `proj_factor` 和 `num_heads`、使用更长窗口。
5. **区分 regime**：对牛市/熊市/震荡市分别建模，可能提升方向预测准确率。

---

## 8. 附录：生成文件清单

```
stock_rnn_experiments/outputs/
├── comparison.json                  # JSON 格式三模型指标
├── comparison_metrics.png           # 指标对比柱状图
├── overlay_predictions.png          # 三模型预测叠加图
├── run_log.txt                      # 训练日志
├── exp1_awd_lstm/
│   ├── 1_loss.png
│   ├── 2_predictions.png
│   ├── 3_scatter.png
│   ├── metrics.txt
│   ├── model.pt
│   └── predictions.csv
├── exp2_lstnet/
│   ├── 1_loss.png
│   ├── 2_predictions.png
│   ├── 3_scatter.png
│   ├── metrics.txt
│   ├── model.pt
│   └── predictions.csv
└── exp3_xlstm/
    ├── 1_loss.png
    ├── 2_predictions.png
    ├── 3_scatter.png
    ├── metrics.txt
    ├── model.pt
    └── predictions.csv
```

---

*报告结束。*
