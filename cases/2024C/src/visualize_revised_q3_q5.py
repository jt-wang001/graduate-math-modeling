"""为修正后的问题 3--5 生成论文/答辩可直接使用的关键图。"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

sys.path.append(str(Path(__file__).parent))
from utils import MODEL_DIR, OUTPUT_DIR

FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
})

BLUE, RED, GREEN, PURPLE, GRAY = "#2878B5", "#C94C4C", "#49A078", "#7868A8", "#707070"


def save(fig: plt.Figure, name: str) -> None:
    fig.subplots_adjust()
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {name}")


def fig19_q3_controlled_effects() -> None:
    df = pd.read_csv(OUTPUT_DIR / "q3_controlled_anova.csv", encoding="utf-8-sig")
    df = df.sort_values("部分eta_sq", ascending=True)
    colors = [BLUE if x in ["频率 log(f)", "磁通密度峰值 log(Bm)"] else RED for x in df["效应"]]
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    bars = ax.barh(df["效应"], df["部分eta_sq"], color=colors, edgecolor="white")
    for bar, value in zip(bars, df["部分eta_sq"]):
        ax.text(value + 0.012, bar.get_y() + bar.get_height()/2, f"{value:.4f}", va="center")
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("部分 $\eta^2$（控制其他变量后的解释强度）")
    ax.set_title("图19  问题三：控制频率与 Bm 后的效应强度", fontweight="bold")
    ax.text(0.02, -0.18, "蓝色：连续物理协变量；红色：题设分类因素及两两交互。", transform=ax.transAxes, color=GRAY)
    save(fig, "fig19_q3_controlled_effects.png")


def fig20_q3_adjusted_heatmaps() -> None:
    df = pd.read_csv(OUTPUT_DIR / "q3_adjusted_conditions.csv", encoding="utf-8-sig")
    waveforms = ["正弦波", "三角波", "梯形波"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    vmin, vmax = np.log10(df["adjusted_pred_loss"]).min(), np.log10(df["adjusted_pred_loss"]).max()
    ims = []
    for ax, waveform in zip(axes, waveforms):
        sub = df[df["waveform"] == waveform]
        pivot = sub.pivot(index="temperature", columns="material", values="adjusted_pred_loss").reindex(index=[25, 50, 70, 90])
        im = ax.imshow(np.log10(pivot), aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)
        ims.append(im)
        ax.set_title(waveform, fontweight="bold")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns)
        ax.set_yticks(range(len(pivot.index)), [f"{x}°C" for x in pivot.index])
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.iloc[i,j]/1e3:.1f}", ha="center", va="center", fontsize=8)
        ax.set_xlabel("材料")
    axes[0].set_ylabel("温度")
    cbar = fig.colorbar(ims[-1], ax=axes, shrink=0.72, pad=0.02)
    cbar.set_label("log10(调整预测损耗)")
    fig.suptitle("图20  问题三：固定 f=158.73 kHz、Bm=0.04992 T 的调整预测损耗（格内：10³ W/m³）", fontweight="bold", y=1.03)
    fig.subplots_adjust(top=0.80, bottom=0.16, left=0.06, right=0.93, wspace=0.08)
    save(fig, "fig20_q3_adjusted_heatmaps.png")


def fig21_q4_validation_and_importance() -> None:
    random_cv = pd.read_csv(OUTPUT_DIR / "q4_cv_results.csv", encoding="utf-8-sig").iloc[0]
    group_cv = pd.read_csv(OUTPUT_DIR / "q4_group_cv_results.csv", encoding="utf-8-sig").iloc[0]
    metrics = pd.DataFrame({
        "验证方案": ["随机 5 折", "组合分组 5 折"],
        "R²": [random_cv["R2_mean"], group_cv["R2_mean"]],
        "MAPE (%)": [random_cv["MAPE_mean"], group_cv["MAPE_mean"]],
    })
    model = CatBoostRegressor(); model.load_model(MODEL_DIR / "q4_catboost.cbm")
    names = ["温度", "log(f)", "log(Bm)", "Bm", "log(f·Bm)", "正弦", "三角", "梯形", "材料1", "材料2", "材料3", "材料4", "峰度", "偏度", "峰峰值", "形状因子", "峰值因子", "平台/缓变比", "最大斜率", "谱熵", "谱质心"]
    imp = pd.Series(model.get_feature_importance(), index=names).sort_values().tail(10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(2); width = 0.35
    axes[0].bar(x - width/2, metrics["R²"], width, color=BLUE, label="R²")
    ax2 = axes[0].twinx()
    ax2.bar(x + width/2, metrics["MAPE (%)"], width, color=RED, label="MAPE (%)")
    axes[0].set_ylim(0.98, 1.0); ax2.set_ylim(0, max(metrics["MAPE (%)"])*1.45)
    axes[0].set_xticks(x, metrics["验证方案"]); axes[0].set_ylabel("R²"); ax2.set_ylabel("MAPE / %")
    axes[0].set_title("(a) 随机与分组验证")
    axes[0].legend(loc="upper left"); ax2.legend(loc="upper right")
    for i, row in metrics.iterrows():
        axes[0].text(i-width/2, row["R²"]+0.0005, f"{row['R²']:.4f}", ha="center", fontsize=9)
        ax2.text(i+width/2, row["MAPE (%)"]+0.15, f"{row['MAPE (%)']:.2f}%", ha="center", fontsize=9)

    axes[1].barh(imp.index, imp.values, color=[BLUE if n in ["log(f·Bm)", "Bm", "log(Bm)", "log(f)", "温度"] else GREEN for n in imp.index])
    axes[1].set_xlabel("CatBoost 特征重要性")
    axes[1].set_title("(b) 前 10 个重要特征")
    fig.suptitle("图21  问题四：验证强度与模型解释", fontweight="bold", y=1.02)
    save(fig, "fig21_q4_validation_importance.png")


def fig22_q5_pareto_decisions() -> None:
    pareto = pd.read_csv(OUTPUT_DIR / "q5_pareto_front.csv", encoding="utf-8-sig")
    decision = pd.read_csv(OUTPUT_DIR / "q5_decisions.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(10, 6.5))
    points = ax.scatter(pareto["core_loss"], pareto["trans_energy"], c=pareto["temperature"], cmap="viridis", s=38, alpha=.75, edgecolor="white", linewidth=.35, label="Pareto 有效解")
    style = {"最低损耗": (GREEN, "o"), "最大磁能": (RED, "s"), "最佳折中": (PURPLE, "*")}
    for _, row in decision.iterrows():
        color, marker = style[row["方案"]]
        ax.scatter(row["core_loss"], row["trans_energy"], s=210, marker=marker, color=color, edgecolor="black", linewidth=1.0, zorder=5, label=row["方案"])
        ax.annotate(f"{row['方案']}\n{row['frequency']/1e3:.0f} kHz, {row['Bm']:.3f} T", (row["core_loss"], row["trans_energy"]), xytext=(7, 7), textcoords="offset points", fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("预测磁芯损耗 / W·m⁻³")
    ax.set_ylabel("传输磁能代理指标 f·Bm")
    ax.set_title("图22  问题五：真实波形模板下的 Pareto 前沿与代表方案", fontweight="bold")
    ax.legend(loc="upper left")
    fig.colorbar(points, ax=ax, label="温度 / °C")
    save(fig, "fig22_q5_pareto_decisions.png")


def main() -> None:
    fig19_q3_controlled_effects()
    fig20_q3_adjusted_heatmaps()
    fig21_q4_validation_and_importance()
    fig22_q5_pareto_decisions()


if __name__ == "__main__":
    main()
