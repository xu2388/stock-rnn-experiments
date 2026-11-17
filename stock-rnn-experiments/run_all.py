"""运行全部三个 RNN 模型的股票预测实验，并生成对比报告。"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import set_seed
from common.viz import plot_metrics_bar


def run_all():
    set_seed(0)
    results = {}

    # 实验 1: AWD-LSTM
    print("\n" + "=" * 60)
    print("实验 1: AWD-LSTM")
    print("=" * 60)
    from exp1_awd_lstm import main as run_awd
    results["AWD-LSTM"] = run_awd()

    # 实验 2: LSTNet
    print("\n" + "=" * 60)
    print("实验 2: LSTNet")
    print("=" * 60)
    from exp2_lstnet import main as run_lstnet
    results["LSTNet"] = run_lstnet()

    # 实验 3: xLSTM
    print("\n" + "=" * 60)
    print("实验 3: xLSTM")
    print("=" * 60)
    from exp3_xlstm import main as run_xlstm
    results["xLSTM"] = run_xlstm()

    # 对比报告
    print("\n" + "=" * 60)
    print("三模型对比")
    print("=" * 60)
    out_dir = os.path.join(HERE, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # 保存 JSON
    with open(os.path.join(out_dir, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 打印对比表
    metrics = ["MAE", "MSE", "RMSE", "MAPE", "R2", "DA"]
    header = f"{'Model':<12}" + "".join(f"{m:>12}" for m in metrics)
    print(header)
    print("-" * len(header))
    for model, mets in results.items():
        row = f"{model:<12}" + "".join(f"{mets.get(m, float('nan')):>12.4f}" for m in metrics)
        print(row)

    # 对比柱状图
    plot_metrics_bar(results, os.path.join(out_dir, "comparison_metrics.png"))

    # 三模型预测叠加图
    try:
        plot_overlay(results, out_dir)
    except Exception as e:
        print(f"叠加图跳过: {e}")

    print(f"\n对比结果保存到: {out_dir}")


def plot_overlay(results, out_dir):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pandas as pd
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = {"AWD-LSTM": "red", "LSTNet": "blue", "xLSTM": "green"}
    for model_name, color in colors.items():
        csv_path = os.path.join(HERE, "outputs",
                                {"AWD-LSTM": "exp1_awd_lstm",
                                 "LSTNet": "exp2_lstnet",
                                 "xLSTM": "exp3_xlstm"}[model_name],
                                "predictions.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, parse_dates=["date"])
            if model_name == "AWD-LSTM":
                ax.plot(df["date"], df["true"], color="black", lw=1.5, label="真实")
            ax.plot(df["date"], df["pred"], color=color, lw=1, alpha=0.8, label=f"{model_name} 预测")
    ax.set_title("三模型测试集预测对比")
    ax.set_xlabel("日期"); ax.set_ylabel("收盘价")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.savefig(os.path.join(out_dir, "overlay_predictions.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_all()
