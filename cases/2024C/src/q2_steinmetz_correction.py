"""问题二：斯坦麦茨方程（SE）的温度修正。

思路：
1. 选取附件一材料1、正弦波数据，提取 (f, Bm, T, P)。
2. 拟合原始 SE：P = k · f^α · Bm^β。
3. 构造温度修正方程：
   - 线性修正: P = k · f^α · Bm^β · (1 + γ·T)
   - 二次修正: P = k · f^α · Bm^β · (1 + γ·T + δ·T²)
   - 指数修正: P = k · f^α · Bm^β · exp(γ·T)
4. 分别用 R²、MAPE 评估三种修正方程与原始 SE 在不同温度下的预测效果，
   证明温度修正后误差显著降低。
5. 保存参数与对比报告。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_percentage_error, r2_score

import sys
sys.path.append(str(Path(__file__).parent))
from utils import MODEL_DIR, OUTPUT_DIR, PROCESSED_DIR, get_b_columns


# ---------- 模型定义 ----------

def se_model(X, k, alpha, beta):
    """原始 Steinmetz 方程：P = k · f^α · Bm^β。"""
    f, Bm = X
    return k * np.power(f, alpha) * np.power(Bm, beta)


def se_linear_temp(X, k, alpha, beta, gamma):
    """线性温度修正：P = k · f^α · Bm^β · (1 + γ·T)。"""
    f, Bm, T = X
    return k * np.power(f, alpha) * np.power(Bm, beta) * (1.0 + gamma * T)


def se_quad_temp(X, k, alpha, beta, gamma, delta):
    """二次温度修正：P = k · f^α · Bm^β · (1 + γ·T + δ·T²)。"""
    f, Bm, T = X
    return (k * np.power(f, alpha) * np.power(Bm, beta)
            * (1.0 + gamma * T + delta * T * T))


def se_exp_temp(X, k, alpha, beta, gamma):
    """指数温度修正：P = k · f^α · Bm^β · exp(γ·T)。"""
    f, Bm, T = X
    return k * np.power(f, alpha) * np.power(Bm, beta) * np.exp(gamma * T)


def evaluate(y_true, y_pred, name: str) -> dict:
    """计算 R² 与 MAPE。"""
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"  {name:25s}  R²={r2:.4f}   MAPE={mape*100:.2f}%")
    return {"name": name, "r2": r2, "mape": mape}


def main() -> None:
    print("=" * 70)
    print("问题二：斯坦麦茨方程（SE）温度修正")
    print("=" * 70)

    # 1. 读取训练集，筛选材料1+正弦波
    train = pd.read_parquet(PROCESSED_DIR / "train_all.parquet")
    df = train[(train["material_id"] == 1) & (train["waveform"] == "正弦波")].copy()
    print(f"材料1 正弦波样本数：{len(df)}")
    print(f"温度取值：{sorted(df['temperature'].unique())}")
    print(f"频率范围：{df['frequency'].min():.0f} ~ {df['frequency'].max():.0f}")

    b_cols = get_b_columns()
    f = df["frequency"].to_numpy(dtype=float)
    Bm = df[b_cols].to_numpy(dtype=float)
    Bm = np.max(np.abs(Bm), axis=1)  # 磁通密度峰值
    T = df["temperature"].to_numpy(dtype=float)
    P = df["core_loss"].to_numpy(dtype=float)

    print(f"Bm 范围：{Bm.min():.4f} ~ {Bm.max():.4f}")
    print(f"P  范围：{P.min():.2f} ~ {P.max():.2f}")

    # 2. 拟合原始 SE
    print("\n[1] 原始 SE 方程拟合 (P = k · f^α · Bm^β)：")
    p0 = [1e-3, 1.5, 2.5]
    try:
        popt_se, _ = curve_fit(se_model, (f, Bm), P, p0=p0, maxfev=20000)
    except RuntimeError:
        # 用对数线性回归给出更稳定的初值
        popt_se, _ = curve_fit(se_model, (f, Bm), P, p0=p0, maxfev=50000)
    k_se, a_se, b_se = popt_se
    print(f"  k={k_se:.6e}, α={a_se:.4f}, β={b_se:.4f}")
    P_pred_se = se_model((f, Bm), *popt_se)
    metrics_se = evaluate(P, P_pred_se, "原始 SE")

    # 3. 拟合温度修正方程
    print("\n[2] 线性温度修正 (1 + γ·T)：")
    popt_lin, _ = curve_fit(se_linear_temp, (f, Bm, T), P,
                            p0=[*popt_se, 0.001], maxfev=30000)
    print(f"  k={popt_lin[0]:.6e}, α={popt_lin[1]:.4f}, "
          f"β={popt_lin[2]:.4f}, γ={popt_lin[3]:.6f}")
    P_pred_lin = se_linear_temp((f, Bm, T), *popt_lin)
    metrics_lin = evaluate(P, P_pred_lin, "线性温度修正")

    print("\n[3] 二次温度修正 (1 + γ·T + δ·T²)：")
    popt_quad, _ = curve_fit(se_quad_temp, (f, Bm, T), P,
                             p0=[*popt_se, 0.001, 0.0001], maxfev=30000)
    print(f"  k={popt_quad[0]:.6e}, α={popt_quad[1]:.4f}, "
          f"β={popt_quad[2]:.4f}, γ={popt_quad[3]:.6f}, δ={popt_quad[4]:.8f}")
    P_pred_quad = se_quad_temp((f, Bm, T), *popt_quad)
    metrics_quad = evaluate(P, P_pred_quad, "二次温度修正")

    print("\n[4] 指数温度修正 (exp(γ·T))：")
    popt_exp, _ = curve_fit(se_exp_temp, (f, Bm, T), P,
                            p0=[*popt_se, 0.001], maxfev=30000)
    print(f"  k={popt_exp[0]:.6e}, α={popt_exp[1]:.4f}, "
          f"β={popt_exp[2]:.4f}, γ={popt_exp[3]:.6f}")
    P_pred_exp = se_exp_temp((f, Bm, T), *popt_exp)
    metrics_exp = evaluate(P, P_pred_exp, "指数温度修正")

    # 4. 分温度段对比（更直观看到温度修正的效果）
    print("\n按温度分组的 MAPE 对比（%）：")
    print(f"  {'温度':>6s}  {'样本数':>6s}  "
          f"{'原始SE':>10s}  {'线性修正':>10s}  "
          f"{'二次修正':>10s}  {'指数修正':>10s}")
    by_temp = []
    for t in sorted(df["temperature"].unique()):
        mask = T == t
        m_se = mean_absolute_percentage_error(P[mask], P_pred_se[mask]) * 100
        m_lin = mean_absolute_percentage_error(P[mask], P_pred_lin[mask]) * 100
        m_quad = mean_absolute_percentage_error(P[mask], P_pred_quad[mask]) * 100
        m_exp = mean_absolute_percentage_error(P[mask], P_pred_exp[mask]) * 100
        print(f"  {t:>6.0f}  {mask.sum():>6d}  "
              f"{m_se:>10.2f}  {m_lin:>10.2f}  "
              f"{m_quad:>10.2f}  {m_exp:>10.2f}")
        by_temp.append({"temperature": t, "n": int(mask.sum()),
                        "mape_se": m_se, "mape_lin": m_lin,
                        "mape_quad": m_quad, "mape_exp": m_exp})

    # 5. 选最优模型（按整体 MAPE 最小）
    all_metrics = [metrics_se, metrics_lin, metrics_quad, metrics_exp]
    best = min(all_metrics, key=lambda m: m["mape"])
    print(f"\n最优模型：{best['name']}  (整体 MAPE={best['mape']*100:.2f}%)")
    print(f"相对原始 SE 的 MAPE 降幅："
          f"{(metrics_se['mape'] - best['mape']) / metrics_se['mape'] * 100:.1f}%")

    # 6. 保存参数与对比报告
    np.savez(
        MODEL_DIR / "q2_params.npz",
        se=popt_se,
        linear=popt_lin,
        quad=popt_quad,
        exp=popt_exp,
    )

    summary = pd.DataFrame(all_metrics)
    summary.to_csv(OUTPUT_DIR / "q2_metrics.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(by_temp).to_csv(
        OUTPUT_DIR / "q2_by_temperature.csv",
        index=False, encoding="utf-8-sig",
    )

    print(f"\n参数已保存：{MODEL_DIR / 'q2_params.npz'}")
    print(f"指标已保存：{OUTPUT_DIR / 'q2_metrics.csv'}")
    print(f"分温度指标已保存：{OUTPUT_DIR / 'q2_by_temperature.csv'}")


if __name__ == "__main__":
    main()
