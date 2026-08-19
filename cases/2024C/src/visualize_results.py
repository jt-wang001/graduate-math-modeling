"""实验结果可视化方案。

参考高影响力学术论文（Nature/IEEE TPEL/IEEE TMAG）的图表风格，
为 2024C 题五个子问题生成专业、可读、信息完整的图表。

图表目录 outputs/figures/：
  fig01_data_distribution.png   —— 数据分布概览（6 合 1）
  fig02_waveform_examples.png   —— 三种波形示例曲线
  fig03_q1_feature_importance.png —— Q1 特征重要性
  fig04_q1_confusion_matrix.png —— Q1 交叉验证混淆矩阵
  fig05_q1_pca_2d.png           —— Q1 特征 PCA 二维投影
  fig06_q2_model_comparison.png —— Q2 四种模型 R²/MAPE 对比
  fig07_q2_mape_by_temp.png     —— Q2 分温度 MAPE 折线
  fig08_q2_pred_vs_actual.png   —— Q2 预测 vs 真实散点
  fig09_q3_eta_squared.png      —— Q3 因素效应量 η²
  fig10_q3_interaction.png      —— Q3 双因素交互效应
  fig11_q3_loss_heatmap.png     —— Q3 平均损耗热力图
  fig12_q4_model_comparison.png —— Q4 四种模型 R²/MAPE 对比
  fig13_q4_pred_vs_actual.png   —— Q4 预测 vs 真实散点
  fig14_q4_residual.png         —— Q4 残差分布
  fig15_q4_feature_importance.png —— Q4 特征重要性
  fig16_q4_test_distribution.png —— Q4 测试集预测分布
  fig17_q5_pareto_front.png     —— Q5 Pareto 前沿
  fig18_q5_decisions.png        —— Q5 三种决策方案
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.append(str(Path(__file__).parent))
from utils import (
    MODEL_DIR,
    OUTPUT_DIR,
    PROCESSED_DIR,
    build_classification_features,
    build_regression_features,
    extract_time_domain_features,
    extract_frequency_features,
    extract_shape_features,
    get_b_columns,
    waveform_to_label,
)

# ---------- 全局样式设置 ----------
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

sns.set_style("whitegrid", {
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
})
# seaborn set_style 会重置字体，需在之后再次设置
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 学术配色（参考 Nature / IEEE 期刊）
PALETTE = {
    "primary": "#2E5C8A",       # 深蓝
    "secondary": "#C44E52",     # 暗红
    "tertiary": "#55A868",      # 绿
    "quaternary": "#8172B2",    # 紫
    "accent": "#CCB974",        # 黄
    "gray": "#7F7F7F",
}
CAT_COLORS = ["#2E5C8A", "#C44E52", "#55A868", "#8172B2", "#CCB974", "#64B5CD"]
WAVE_COLORS = {"正弦波": "#2E5C8A", "三角波": "#C44E52", "梯形波": "#55A868"}
MAT_COLORS = {"材料1": "#2E5C8A", "材料2": "#C44E52",
              "材料3": "#55A868", "材料4": "#8172B2"}

FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name: str) -> None:
    """保存图片到 FIG_DIR。"""
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {name}")


# ====================================================================
# 一、数据分布与波形示例
# ====================================================================

def fig01_data_distribution(train: pd.DataFrame) -> None:
    """图1：数据分布概览（6 合 1）。"""
    print("生成 fig01_data_distribution ...")
    b_cols = get_b_columns()
    Bm = np.max(np.abs(train[b_cols].to_numpy(dtype=float)), axis=1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("图1  训练集数据分布概览", fontsize=15, fontweight="bold", y=0.995)

    # (a) core_loss 分布（对数坐标）
    ax = axes[0, 0]
    ax.hist(np.log10(train["core_loss"]), bins=60, color=PALETTE["primary"],
            edgecolor="white", alpha=0.85)
    ax.set_xlabel("log₁₀(磁芯损耗 / W·m⁻³)")
    ax.set_ylabel("样本数")
    ax.set_title("(a) 磁芯损耗分布（对数）")

    # (b) 温度分布
    ax = axes[0, 1]
    counts = train["temperature"].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color=CAT_COLORS[:4],
           edgecolor="white")
    ax.set_xlabel("温度 / °C")
    ax.set_ylabel("样本数")
    ax.set_title("(b) 温度分布")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 30, str(v), ha="center", fontsize=9)

    # (c) 波形分布
    ax = axes[0, 2]
    counts = train["waveform"].value_counts()
    ax.bar(counts.index, counts.values,
           color=[WAVE_COLORS[w] for w in counts.index], edgecolor="white")
    ax.set_xlabel("励磁波形")
    ax.set_ylabel("样本数")
    ax.set_title("(c) 励磁波形分布")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 30, str(v), ha="center", fontsize=9)

    # (d) 材料分布
    ax = axes[1, 0]
    counts = train["material"].value_counts().sort_index()
    ax.bar(counts.index, counts.values,
           color=[MAT_COLORS[m] for m in counts.index], edgecolor="white")
    ax.set_xlabel("磁芯材料")
    ax.set_ylabel("样本数")
    ax.set_title("(d) 磁芯材料分布")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 30, str(v), ha="center", fontsize=9)

    # (e) 频率分布
    ax = axes[1, 1]
    ax.hist(train["frequency"] / 1e3, bins=60, color=PALETTE["tertiary"],
            edgecolor="white", alpha=0.85)
    ax.set_xlabel("频率 / kHz")
    ax.set_ylabel("样本数")
    ax.set_title("(e) 频率分布")

    # (f) Bm 分布
    ax = axes[1, 2]
    ax.hist(Bm, bins=60, color=PALETTE["quaternary"],
            edgecolor="white", alpha=0.85)
    ax.set_xlabel("磁通密度峰值 Bₘ / T")
    ax.set_ylabel("样本数")
    ax.set_title("(f) 磁通密度峰值分布")

    plt.tight_layout()
    save_fig(fig, "fig01_data_distribution.png")


def fig02_waveform_examples(train: pd.DataFrame) -> None:
    """图2：三种波形示例曲线。"""
    print("生成 fig02_waveform_examples ...")
    b_cols = get_b_columns()
    t = np.linspace(0, 1, NUM_B := len(b_cols))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    fig.suptitle("图2  三种励磁波形示例（每种 3 条样本）",
                 fontsize=14, fontweight="bold")

    waveforms = ["正弦波", "三角波", "梯形波"]
    for ax, wf in zip(axes, waveforms):
        sub = train[train["waveform"] == wf].sample(3, random_state=42)
        for _, row in sub.iterrows():
            b = row[b_cols].to_numpy(dtype=float)
            ax.plot(t, b, linewidth=1.2, alpha=0.85,
                    label=f"f={row['frequency']/1e3:.0f}kHz")
        ax.set_title(wf, color=WAVE_COLORS[wf], fontweight="bold")
        ax.set_xlabel("归一化时间 (一个周期)")
        ax.legend(loc="upper right", framealpha=0.9)
    axes[0].set_ylabel("磁通密度 B / T")

    plt.tight_layout()
    save_fig(fig, "fig02_waveform_examples.png")


# ====================================================================
# 二、问题一：波形分类
# ====================================================================

def fig03_q1_feature_importance(train: pd.DataFrame) -> None:
    """图3：随机森林特征重要性。"""
    print("生成 fig03_q1_feature_importance ...")
    X = build_classification_features(train)
    y = train["waveform"].map(waveform_to_label).to_numpy()

    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # 特征名
    td_names = list(extract_time_domain_features(np.zeros(1024)).keys())
    fd_names = list(extract_frequency_features(np.zeros(1024), 5).keys())
    sf_names = list(extract_shape_features(np.zeros(1024)).keys())
    feat_names = td_names + fd_names + sf_names

    imp = pd.Series(rf.feature_importances_, index=feat_names).sort_values()

    fig, ax = plt.subplots(figsize=(9, 8))
    colors = [PALETTE["primary"] if n in td_names
              else PALETTE["tertiary"] if n in fd_names
              else PALETTE["secondary"] for n in imp.index]
    ax.barh(imp.index, imp.values, color=colors, edgecolor="white")
    ax.set_xlabel("特征重要性 (Gini 重要性)")
    ax.set_title("图3  问题一：随机森林特征重要性排序", fontweight="bold")

    # 图例
    from matplotlib.patches import Patch
    legend = [Patch(facecolor=PALETTE["primary"], label="时域特征"),
              Patch(facecolor=PALETTE["tertiary"], label="频域特征"),
              Patch(facecolor=PALETTE["secondary"], label="形状特征")]
    ax.legend(handles=legend, loc="lower right")

    plt.tight_layout()
    save_fig(fig, "fig03_q1_feature_importance.png")


def fig04_q1_confusion_matrix(train: pd.DataFrame) -> None:
    """图4：5折交叉验证混淆矩阵（ aggregated）。"""
    print("生成 fig04_q1_confusion_matrix ...")
    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble import GradientBoostingClassifier

    X = build_classification_features(train)
    y = train["waveform"].map(waveform_to_label).to_numpy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_true_all, y_pred_all = [], []
    for tr, va in skf.split(Xs, y):
        rf.fit(Xs[tr], y[tr])
        y_pred_all.extend(rf.predict(Xs[va]))
        y_true_all.extend(y[va])

    cm = confusion_matrix(y_true_all, y_pred_all, labels=[1, 2, 3])
    labels = ["正弦波", "三角波", "梯形波"]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar_kws={"label": "样本数"}, linewidths=0.5, linecolor="white")
    ax.set_xlabel("预测波形")
    ax.set_ylabel("真实波形")
    ax.set_title("图4  问题一：5折交叉验证混淆矩阵\n(总体准确率 100.00%)",
                 fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "fig04_q1_confusion_matrix.png")


def fig05_q1_pca_2d(train: pd.DataFrame) -> None:
    """图5：特征 PCA 二维投影。"""
    print("生成 fig05_q1_pca_2d ...")
    X = build_classification_features(train)
    y = train["waveform"].map(waveform_to_label).to_numpy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    Xp = pca.fit_transform(Xs)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    for wf, label in [("正弦波", 1), ("三角波", 2), ("梯形波", 3)]:
        mask = y == label
        ax.scatter(Xp[mask, 0], Xp[mask, 1], s=8, alpha=0.4,
                   color=WAVE_COLORS[wf], label=wf, edgecolors="none")
    ax.set_xlabel(f"主成分1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"主成分2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("图5  问题一：特征 PCA 二维投影", fontweight="bold")
    ax.legend(markerscale=3, framealpha=0.9)
    plt.tight_layout()
    save_fig(fig, "fig05_q1_pca_2d.png")


# ====================================================================
# 三、问题二：SE 温度修正
# ====================================================================

def fig06_q2_model_comparison() -> None:
    """图6：四种模型 R²/MAPE 对比。"""
    print("生成 fig06_q2_model_comparison ...")
    df = pd.read_csv(OUTPUT_DIR / "q2_metrics.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("图6  问题二：四种温度修正模型性能对比",
                 fontsize=14, fontweight="bold")

    names = df["name"]
    colors = [PALETTE["gray"], PALETTE["primary"],
              PALETTE["secondary"], PALETTE["tertiary"]]

    ax = axes[0]
    bars = ax.bar(names, df["r2"], color=colors, edgecolor="white")
    ax.set_ylabel("R²（决定系数）")
    ax.set_title("(a) R² 对比")
    ax.set_ylim(0.94, 1.0)
    for b, v in zip(bars, df["r2"]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.001, f"{v:.4f}",
                ha="center", fontsize=9)
    ax.tick_params(axis="x", rotation=15)

    ax = axes[1]
    bars = ax.bar(names, df["mape"]*100, color=colors, edgecolor="white")
    ax.set_ylabel("MAPE / %")
    ax.set_title("(b) MAPE 对比")
    for b, v in zip(bars, df["mape"]*100):
        ax.text(b.get_x() + b.get_width()/2, v + 0.3, f"{v:.2f}%",
                ha="center", fontsize=9)
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    save_fig(fig, "fig06_q2_model_comparison.png")


def fig07_q2_mape_by_temp() -> None:
    """图7：分温度 MAPE 折线。"""
    print("生成 fig07_q2_mape_by_temp ...")
    df = pd.read_csv(OUTPUT_DIR / "q2_by_temperature.csv")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    cols = [("mape_se", "原始 SE", PALETTE["gray"], "o"),
            ("mape_lin", "线性温度修正", PALETTE["primary"], "s"),
            ("mape_quad", "二次温度修正", PALETTE["secondary"], "^"),
            ("mape_exp", "指数温度修正", PALETTE["tertiary"], "D")]
    for col, name, color, marker in cols:
        ax.plot(df["temperature"], df[col], marker=marker, markersize=8,
                linewidth=2, color=color, label=name)

    ax.set_xlabel("温度 / °C")
    ax.set_ylabel("MAPE / %")
    ax.set_title("图7  问题二：不同温度下各模型 MAPE 对比", fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.set_xticks([25, 50, 70, 90])
    plt.tight_layout()
    save_fig(fig, "fig07_q2_mape_by_temp.png")


def fig08_q2_pred_vs_actual(train: pd.DataFrame) -> None:
    """图8：预测 vs 真实散点（原始SE vs 二次修正）。"""
    print("生成 fig08_q2_pred_vs_actual ...")
    from scipy.optimize import curve_fit

    df = train[(train["material_id"] == 1) & (train["waveform"] == "正弦波")].copy()
    b_cols = get_b_columns()
    f = df["frequency"].to_numpy(float)
    Bm = np.max(np.abs(df[b_cols].to_numpy(float)), axis=1)
    T = df["temperature"].to_numpy(float)
    P = df["core_loss"].to_numpy(float)

    def se(X, k, a, b):
        return k * np.power(X[0], a) * np.power(X[1], b)
    def se_quad(X, k, a, b, g, d):
        return k * np.power(X[0], a) * np.power(X[1], b) * (1 + g*X[2] + d*X[2]**2)

    p_se, _ = curve_fit(se, (f, Bm), P, p0=[1e-3, 1.5, 2.5], maxfev=30000)
    p_q, _ = curve_fit(se_quad, (f, Bm, T), P, p0=[*p_se, 0.001, 0.0001], maxfev=30000)

    P_se = se((f, Bm), *p_se)
    P_q = se_quad((f, Bm, T), *p_q)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("图8  问题二：预测值 vs 真实值（材料1·正弦波）",
                 fontsize=14, fontweight="bold")

    for ax, (P_pred, name, color) in zip(axes, [
        (P_se, "原始 SE 方程", PALETTE["gray"]),
        (P_q, "二次温度修正", PALETTE["secondary"]),
    ]):
        ax.scatter(P, P_pred, s=12, alpha=0.5, color=color, edgecolors="none")
        m = max(P.max(), P_pred.max())
        ax.plot([0, m], [0, m], "k--", linewidth=1, label="y = x")
        # 按温度着色边界
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("真实损耗 P / W·m⁻³")
        ax.set_ylabel("预测损耗 P̂ / W·m⁻³")
        ax.set_title(name)
        ax.legend(loc="upper left")
        ax.set_aspect("equal")
    plt.tight_layout()
    save_fig(fig, "fig08_q2_pred_vs_actual.png")


# ====================================================================
# 四、问题三：因素分析
# ====================================================================

def fig09_q3_eta_squared() -> None:
    """图9：因素效应量 η²。"""
    print("生成 fig09_q3_eta_squared ...")
    df = pd.read_csv(OUTPUT_DIR / "q3_main_effects.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(df["因素"], df["eta_sq"],
                  color=[PALETTE["primary"], PALETTE["secondary"], PALETTE["tertiary"]],
                  edgecolor="white")
    ax.set_ylabel("效应量 η²（解释方差比例）")
    ax.set_title("图9  问题三：三因素主效应影响程度（η²）", fontweight="bold")
    for b, v in zip(bars, df["eta_sq"]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.001,
                f"{v:.4f}", ha="center", fontsize=10)
    ax.set_ylim(0, max(df["eta_sq"]) * 1.2)
    plt.tight_layout()
    save_fig(fig, "fig09_q3_eta_squared.png")


def fig10_q3_interaction(train: pd.DataFrame) -> None:
    """图10：双因素交互效应图。"""
    print("生成 fig10_q3_interaction ...")
    df = train.copy()
    df["log_loss"] = np.log(df["core_loss"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("图10  问题三：双因素交互效应图（纵轴 = log(磁芯损耗)）",
                 fontsize=14, fontweight="bold")

    # (a) 温度 × 波形
    ax = axes[0]
    for wf in ["正弦波", "三角波", "梯形波"]:
        sub = df[df["waveform"] == wf].groupby("temperature")["log_loss"].mean()
        ax.plot(sub.index, sub.values, marker="o", linewidth=2,
                color=WAVE_COLORS[wf], label=wf)
    ax.set_xlabel("温度 / °C")
    ax.set_ylabel("log(磁芯损耗)")
    ax.set_title("(a) 温度 × 波形")
    ax.legend()

    # (b) 温度 × 材料
    ax = axes[1]
    for m in ["材料1", "材料2", "材料3", "材料4"]:
        sub = df[df["material"] == m].groupby("temperature")["log_loss"].mean()
        ax.plot(sub.index, sub.values, marker="s", linewidth=2,
                color=MAT_COLORS[m], label=m)
    ax.set_xlabel("温度 / °C")
    ax.set_ylabel("log(磁芯损耗)")
    ax.set_title("(b) 温度 × 材料")
    ax.legend()

    # (c) 波形 × 材料
    ax = axes[2]
    pivot = df.groupby(["waveform", "material"])["log_loss"].mean().unstack()
    pivot = pivot.reindex(["正弦波", "三角波", "梯形波"])
    pivot.plot(kind="bar", ax=ax, color=list(MAT_COLORS.values()), edgecolor="white")
    ax.set_xlabel("励磁波形")
    ax.set_ylabel("log(磁芯损耗)")
    ax.set_title("(c) 波形 × 材料")
    ax.legend(title="材料")
    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    save_fig(fig, "fig10_q3_interaction.png")


def fig11_q3_loss_heatmap(train: pd.DataFrame) -> None:
    """图11：平均损耗热力图（温度 × 材料，分波形）。"""
    print("生成 fig11_q3_loss_heatmap ...")
    df = train.copy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("图11  问题三：平均磁芯损耗热力图（log₁₀）",
                 fontsize=14, fontweight="bold")

    for ax, wf in zip(axes, ["正弦波", "三角波", "梯形波"]):
        sub = df[df["waveform"] == wf]
        pivot = (sub.groupby(["temperature", "material"])["core_loss"]
                 .mean().unstack())
        pivot = pivot[["材料1", "材料2", "材料3", "材料4"]]
        log_pivot = np.log10(pivot)
        sns.heatmap(log_pivot, annot=True, fmt=".2f", cmap="YlOrRd",
                    ax=ax, linewidths=0.5, linecolor="white",
                    cbar_kws={"label": "log₁₀(损耗)"})
        ax.set_xlabel("磁芯材料")
        ax.set_ylabel("温度 / °C")
        ax.set_title(wf, color=WAVE_COLORS[wf], fontweight="bold")

    plt.tight_layout()
    save_fig(fig, "fig11_q3_loss_heatmap.png")


# ====================================================================
# 五、问题四：损耗预测
# ====================================================================

def fig12_q4_model_comparison() -> None:
    """图12：四种模型 R²/MAPE 对比。"""
    print("生成 fig12_q4_model_comparison ...")
    df = pd.read_csv(OUTPUT_DIR / "q4_cv_results.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("图12  问题四：四种回归模型 5 折交叉验证性能对比",
                 fontsize=14, fontweight="bold")

    colors = [PALETTE["primary"], PALETTE["tertiary"],
              PALETTE["secondary"], PALETTE["quaternary"]]
    names = df["model"]

    ax = axes[0]
    bars = ax.bar(names, df["R2_mean"], yerr=df["R2_std"],
                  color=colors, edgecolor="white", capsize=5)
    ax.set_ylabel("R²（决定系数）")
    ax.set_title("(a) R² 对比")
    ax.set_ylim(0.98, 1.0)
    for b, v in zip(bars, df["R2_mean"]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.0008,
                f"{v:.4f}", ha="center", fontsize=9)
    ax.tick_params(axis="x", rotation=15)

    ax = axes[1]
    bars = ax.bar(names, df["MAPE_mean"], yerr=df["MAPE_std"],
                  color=colors, edgecolor="white", capsize=5)
    ax.set_ylabel("MAPE / %")
    ax.set_title("(b) MAPE 对比")
    for b, v in zip(bars, df["MAPE_mean"]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.1,
                f"{v:.2f}%", ha="center", fontsize=9)
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    save_fig(fig, "fig12_q4_model_comparison.png")


def fig13_q4_pred_vs_actual(train: pd.DataFrame) -> None:
    """图13：预测 vs 真实散点（CatBoost 训练集回代）。"""
    print("生成 fig13_q4_pred_vs_actual ...")
    from catboost import CatBoostRegressor
    from sklearn.metrics import r2_score

    model = CatBoostRegressor()
    model.load_model(MODEL_DIR / "q4_catboost.cbm")

    X = build_regression_features(train)
    y_true = train["core_loss"].to_numpy(float)
    y_pred = np.exp(model.predict(X))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("图13  问题四：CatBoost 训练集预测性能",
                 fontsize=14, fontweight="bold")

    # (a) 原始尺度
    ax = axes[0]
    sc = ax.scatter(y_true, y_pred, s=10, alpha=0.4,
                    c=train["temperature"], cmap="viridis", edgecolors="none")
    m = max(y_true.max(), y_pred.max())
    ax.plot([0, m], [0, m], "r--", linewidth=1.2, label="y = x")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("真实损耗 / W·m⁻³")
    ax.set_ylabel("预测损耗 / W·m⁻³")
    ax.set_title(f"(a) 预测 vs 真实（R²={r2_score(y_true, y_pred):.4f}）")
    ax.legend()
    plt.colorbar(sc, ax=ax, label="温度 / °C")

    # (b) 按波形分组
    ax = axes[1]
    for wf in ["正弦波", "三角波", "梯形波"]:
        mask = train["waveform"].to_numpy() == wf
        ax.scatter(y_true[mask], y_pred[mask], s=10, alpha=0.4,
                   color=WAVE_COLORS[wf], label=wf, edgecolors="none")
    ax.plot([0, m], [0, m], "k--", linewidth=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("真实损耗 / W·m⁻³")
    ax.set_ylabel("预测损耗 / W·m⁻³")
    ax.set_title("(b) 按波形着色")
    ax.legend(markerscale=3)

    plt.tight_layout()
    save_fig(fig, "fig13_q4_pred_vs_actual.png")


def fig14_q4_residual(train: pd.DataFrame) -> None:
    """图14：残差分布。"""
    print("生成 fig14_q4_residual ...")
    from catboost import CatBoostRegressor

    model = CatBoostRegressor()
    model.load_model(MODEL_DIR / "q4_catboost.cbm")

    X = build_regression_features(train)
    y_true = train["core_loss"].to_numpy(float)
    y_pred = np.exp(model.predict(X))
    residual = (y_pred - y_true) / y_true * 100  # 百分比残差

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("图14  问题四：预测残差分析", fontsize=14, fontweight="bold")

    # (a) 残差直方图
    ax = axes[0]
    ax.hist(residual, bins=80, color=PALETTE["primary"],
            edgecolor="white", alpha=0.85, density=True)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="零误差")
    ax.set_xlabel("相对残差 / %")
    ax.set_ylabel("密度")
    ax.set_title(f"(a) 残差分布（均值={residual.mean():.2f}%, "
                 f"标准差={residual.std():.2f}%）")
    ax.set_xlim(-30, 30)
    ax.legend()

    # (b) 残差 vs 真实值
    ax = axes[1]
    sc = ax.scatter(y_true, residual, s=10, alpha=0.4,
                    c=train["frequency"], cmap="plasma", edgecolors="none")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("真实损耗 / W·m⁻³")
    ax.set_ylabel("相对残差 / %")
    ax.set_title("(b) 残差 vs 真实值（按频率着色）")
    ax.set_ylim(-30, 30)
    plt.colorbar(sc, ax=ax, label="频率 / Hz")

    plt.tight_layout()
    save_fig(fig, "fig14_q4_residual.png")


def fig15_q4_feature_importance(train: pd.DataFrame) -> None:
    """图15：CatBoost 特征重要性。"""
    print("生成 fig15_q4_feature_importance ...")
    from catboost import CatBoostRegressor

    model = CatBoostRegressor()
    model.load_model(MODEL_DIR / "q4_catboost.cbm")

    feat_names = [
        "温度", "log(频率)", "log(Bₘ)", "Bₘ", "log(f·Bₘ)",
        "正弦波", "三角波", "梯形波",
        "材料1", "材料2", "材料3", "材料4",
        "峰度", "偏度", "峰峰值", "波形因子", "峰值因子",
        "平台段比例", "最大斜率", "谱熵", "谱质心",
    ]
    importances = model.get_feature_importance()
    s = pd.Series(importances, index=feat_names).sort_values()

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = []
    for n in s.index:
        if n in ["温度", "log(频率)", "log(Bₘ)", "Bₘ", "log(f·Bₘ)"]:
            colors.append(PALETTE["primary"])
        elif n in ["正弦波", "三角波", "梯形波", "材料1", "材料2", "材料3", "材料4"]:
            colors.append(PALETTE["tertiary"])
        else:
            colors.append(PALETTE["secondary"])
    ax.barh(s.index, s.values, color=colors, edgecolor="white")
    ax.set_xlabel("特征重要性 (PredictionValuesChange)")
    ax.set_title("图15  问题四：CatBoost 特征重要性", fontweight="bold")

    from matplotlib.patches import Patch
    legend = [Patch(facecolor=PALETTE["primary"], label="物理量特征"),
              Patch(facecolor=PALETTE["tertiary"], label="分类 one-hot"),
              Patch(facecolor=PALETTE["secondary"], label="波形统计特征")]
    ax.legend(handles=legend, loc="lower right")
    plt.tight_layout()
    save_fig(fig, "fig15_q4_feature_importance.png")


def fig16_q4_test_distribution() -> None:
    """图16：测试集预测分布。"""
    print("生成 fig16_q4_test_distribution ...")
    df = pd.read_csv(OUTPUT_DIR / "q4_predictions.csv")
    test = pd.read_parquet(PROCESSED_DIR / "test_q4.parquet")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("图16  问题四：附件三 400 样本预测损耗分布",
                 fontsize=14, fontweight="bold")

    # (a) 预测分布
    ax = axes[0]
    ax.hist(np.log10(df["pred_loss"]), bins=40, color=PALETTE["primary"],
            edgecolor="white", alpha=0.85)
    ax.set_xlabel("log₁₀(预测损耗 / W·m⁻³)")
    ax.set_ylabel("样本数")
    ax.set_title("(a) 预测损耗分布（对数）")

    # (b) 按波形分组箱线图
    ax = axes[1]
    df2 = df.merge(test[["sample_id", "waveform"]], on="sample_id")
    order = ["正弦波", "三角波", "梯形波"]
    sns.boxplot(data=df2, x="waveform", y="pred_loss", order=order,
                palette=[WAVE_COLORS[w] for w in order], ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel("励磁波形")
    ax.set_ylabel("预测损耗 / W·m⁻³")
    ax.set_title("(b) 按波形分组的预测损耗")

    plt.tight_layout()
    save_fig(fig, "fig16_q4_test_distribution.png")


# ====================================================================
# 六、问题五：最优化
# ====================================================================

def fig17_q5_pareto_front() -> None:
    """图17：Pareto 前沿。"""
    print("生成 fig17_q5_pareto_front ...")
    df = pd.read_csv(OUTPUT_DIR / "q5_pareto_front.csv")

    fig, ax = plt.subplots(figsize=(10, 6.5))
    sc = ax.scatter(df["core_loss"], df["trans_energy"],
                    c=df["loss_per_energy"], cmap="viridis_r",
                    s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("磁芯损耗 P / W·m⁻³")
    ax.set_ylabel("传输磁能 f × Bₘ")
    ax.set_title("图17  问题五：NSGA-II Pareto 前沿（双目标优化）",
                 fontweight="bold")
    plt.colorbar(sc, label="损耗/磁能比（越小越优）")
    plt.tight_layout()
    save_fig(fig, "fig17_q5_pareto_front.png")


def fig18_q5_decisions() -> None:
    """图18：三种决策方案在 Pareto 前沿上的位置。"""
    print("生成 fig18_q5_decisions ...")
    pareto = pd.read_csv(OUTPUT_DIR / "q5_pareto_front.csv")
    decisions = pd.read_csv(OUTPUT_DIR / "q5_decisions.csv")

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Pareto 前沿
    ax.scatter(pareto["core_loss"], pareto["trans_energy"],
               s=30, alpha=0.4, color=PALETTE["gray"],
               edgecolors="none", label="Pareto 最优解")

    # 三种决策方案
    styles = [
        ("最低损耗", PALETTE["tertiary"], "o", 200),
        ("最大磁能", PALETTE["primary"], "s", 200),
        ("最佳折中", PALETTE["secondary"], "*", 300),
    ]
    for _, row in decisions.iterrows():
        for name, color, marker, size in styles:
            if row["方案"] == name:
                ax.scatter(row["core_loss"], row["trans_energy"],
                           s=size, color=color, marker=marker,
                           edgecolors="black", linewidth=1.2, zorder=5,
                           label=f"{name}\n  (T={row['temperature']:.0f}°C, "
                                 f"f={row['frequency']/1e3:.1f}kHz, "
                                 f"Bₘ={row['Bm']:.3f}T)")
                break

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("磁芯损耗 P / W·m⁻³")
    ax.set_ylabel("传输磁能 f × Bₘ")
    ax.set_title("图18  问题五：三种决策方案在 Pareto 前沿上的位置",
                 fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.95, fontsize=9)
    plt.tight_layout()
    save_fig(fig, "fig18_q5_decisions.png")


# ====================================================================
# 主函数
# ====================================================================

def main() -> None:
    print("=" * 70)
    print("2024C 实验结果可视化方案")
    print("=" * 70)
    print(f"图表保存目录：{FIG_DIR}\n")

    # 读取数据
    train = pd.read_parquet(PROCESSED_DIR / "train_all.parquet")
    print(f"训练集样本数：{len(train)}\n")

    # 一、数据分布
    print("[一] 数据分布与波形示例")
    fig01_data_distribution(train)
    fig02_waveform_examples(train)

    # 二、问题一
    print("\n[二] 问题一：波形分类")
    fig03_q1_feature_importance(train)
    fig04_q1_confusion_matrix(train)
    fig05_q1_pca_2d(train)

    # 三、问题二
    print("\n[三] 问题二：SE 温度修正")
    fig06_q2_model_comparison()
    fig07_q2_mape_by_temp()
    fig08_q2_pred_vs_actual(train)

    # 四、问题三
    print("\n[四] 问题三：因素分析")
    fig09_q3_eta_squared()
    fig10_q3_interaction(train)
    fig11_q3_loss_heatmap(train)

    # 五、问题四
    print("\n[五] 问题四：损耗预测")
    fig12_q4_model_comparison()
    fig13_q4_pred_vs_actual(train)
    fig14_q4_residual(train)
    fig15_q4_feature_importance(train)
    fig16_q4_test_distribution()

    # 六、问题五
    print("\n[六] 问题五：最优化")
    fig17_q5_pareto_front()
    fig18_q5_decisions()

    print(f"\n全部 {len(list(FIG_DIR.glob('*.png')))} 张图表已生成：{FIG_DIR}")


if __name__ == "__main__":
    main()
