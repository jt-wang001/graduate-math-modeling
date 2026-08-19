"""问题三：控制频率和 Bm 后的磁芯损耗因素分析（ANCOVA）。"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

sys.path.append(str(Path(__file__).parent))
from utils import OUTPUT_DIR, PROCESSED_DIR, get_b_columns


def add_bm_and_covariates(data: pd.DataFrame) -> pd.DataFrame:
    """增加 Bm、对数响应和连续协变量，避免分类因素效应被其混杂。"""
    df = data.copy()
    b = df[get_b_columns()].to_numpy(dtype=float)
    df["Bm"] = np.max(np.abs(b), axis=1)
    df["log_loss"] = np.log(df["core_loss"])
    df["log_frequency"] = np.log(df["frequency"])
    df["log_Bm"] = np.log(df["Bm"])
    return df


def partial_eta_squared(aov: pd.DataFrame) -> pd.DataFrame:
    """计算部分 eta²：SS_effect / (SS_effect + SS_residual)。"""
    table = aov.copy()
    residual_ss = float(table.loc["Residual", "sum_sq"])
    table["partial_eta_sq"] = table["sum_sq"] / (table["sum_sq"] + residual_ss)
    table.loc["Residual", "partial_eta_sq"] = np.nan
    return table


def main() -> None:
    print("=" * 70)
    print("问题三：控制频率与 Bm 的因素及协同效应分析")
    print("=" * 70)
    train = add_bm_and_covariates(pd.read_parquet(PROCESSED_DIR / "train_all.parquet"))
    print(f"样本数：{len(train)}；Bm 范围：{train['Bm'].min():.5f}--{train['Bm'].max():.5f} T")

    # 三个题设因素及其两两协同项；频率、Bm 作为连续协变量。
    formula = (
        "log_loss ~ log_frequency + log_Bm + C(temperature) + C(waveform) + C(material) "
        "+ C(temperature):C(waveform) + C(temperature):C(material) "
        "+ C(waveform):C(material)"
    )
    model = ols(formula, data=train).fit()
    aov = partial_eta_squared(sm.stats.anova_lm(model, typ=2))
    aov.to_csv(OUTPUT_DIR / "q3_controlled_anova_full.csv", encoding="utf-8-sig")

    readable = {
        "log_frequency": "频率 log(f)", "log_Bm": "磁通密度峰值 log(Bm)",
        "C(temperature)": "温度", "C(waveform)": "励磁波形", "C(material)": "磁芯材料",
        "C(temperature):C(waveform)": "温度×励磁波形",
        "C(temperature):C(material)": "温度×磁芯材料",
        "C(waveform):C(material)": "励磁波形×磁芯材料",
    }
    report = []
    for index, row in aov.drop(index="Residual").iterrows():
        report.append({
            "效应": readable.get(index, index), "df": float(row["df"]), "F": float(row["F"]),
            "p值": float(row["PR(>F)"]), "部分eta_sq": float(row["partial_eta_sq"]),
            "显著性(0.05)": "显著" if row["PR(>F)"] < 0.05 else "不显著",
        })
    controlled = pd.DataFrame(report).sort_values("部分eta_sq", ascending=False)
    controlled.to_csv(OUTPUT_DIR / "q3_controlled_anova.csv", index=False, encoding="utf-8-sig")
    print(f"\nANCOVA R²={model.rsquared:.4f}，调整 R²={model.rsquared_adj:.4f}")
    print(controlled.to_string(index=False))

    # 在相同（中位）频率与 Bm 下，比较 48 个分类组合的调整预测损耗。
    med_f = float(train["frequency"].median())
    med_bm = float(train["Bm"].median())
    conditions = pd.MultiIndex.from_product(
        [sorted(train.temperature.unique()), sorted(train.waveform.unique()), sorted(train.material.unique())],
        names=["temperature", "waveform", "material"],
    ).to_frame(index=False)
    conditions["frequency_fixed"] = med_f
    conditions["Bm_fixed"] = med_bm
    pred_input = conditions.rename(columns={"frequency_fixed": "frequency", "Bm_fixed": "Bm"}).copy()
    pred_input["log_frequency"] = np.log(pred_input["frequency"])
    pred_input["log_Bm"] = np.log(pred_input["Bm"])
    conditions["adjusted_log_loss"] = model.predict(pred_input)
    conditions["adjusted_pred_loss"] = np.exp(conditions["adjusted_log_loss"])
    conditions = conditions.sort_values("adjusted_pred_loss").reset_index(drop=True)
    conditions.to_csv(OUTPUT_DIR / "q3_adjusted_conditions.csv", index=False, encoding="utf-8-sig")
    print(f"\n固定频率={med_f:.0f} Hz、Bm={med_bm:.5f} T 时的最优分类组合：")
    print(conditions.head(10).to_string(index=False))

    # 样本内实测最小值，明确区别于基于模型的调整比较。
    observed = train.nsmallest(10, "core_loss")[
        ["temperature", "frequency", "waveform", "material", "Bm", "core_loss"]
    ].reset_index(drop=True)
    observed.to_csv(OUTPUT_DIR / "q3_min_loss_samples.csv", index=False, encoding="utf-8-sig")
    print("\n样本内最小观测值（并非理论全局最优）：")
    print(observed.head(10).to_string(index=False))

    print("\n已输出：q3_controlled_anova.csv、q3_adjusted_conditions.csv、q3_min_loss_samples.csv")


if __name__ == "__main__":
    main()
