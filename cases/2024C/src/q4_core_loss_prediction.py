"""问题四：基于数据驱动的磁芯损耗预测模型。

思路：
1. 构建回归特征：温度、log频率、log磁通密度峰值、f*Bm、波形 one-hot、
   材料 one-hot、峰度、偏度、峰峰值、形状因子、平台段比例、谱熵等。
2. 对目标 core_loss 取 log 变换，使分布更接近正态，提升回归精度。
3. 三种主流梯度提升树模型对比：CatBoost、LightGBM、XGBoost，
   5 折交叉验证评估 R² 与 MAPE，选最优模型保存。
4. 训练最终模型并预测附件三 400 个样本，按要求填入附件四第 3 列（保留 1 位小数）。
5. 打印指定样本序号：16、76、98、126、168、230、271、338、348、379 的预测结果。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import (
    mean_absolute_percentage_error,
    r2_score,
)

import sys
sys.path.append(str(Path(__file__).parent))
from utils import (
    MODEL_DIR,
    OUTPUT_DIR,
    PROCESSED_DIR,
    build_regression_features,
)


def evaluate_cv(models: dict, X: np.ndarray, y_log: np.ndarray,
                y_true: np.ndarray) -> pd.DataFrame:
    """5 折交叉验证，分别评估 R² 与 MAPE。"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, model in models.items():
        r2_scores = []
        mape_scores = []
        for tr_idx, va_idx in kf.split(X):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y_log[tr_idx], y_log[va_idx]
            # 克隆模型（避免使用已拟合状态）
            from sklearn.base import clone
            m = clone(model) if not isinstance(model, CatBoostRegressor) else (
                CatBoostRegressor(**model.get_params()))
            m.fit(X_tr, y_tr)
            y_pred = np.exp(m.predict(X_va))
            y_va_true = np.exp(y_va)
            r2_scores.append(r2_score(y_va_true, y_pred))
            mape_scores.append(mean_absolute_percentage_error(y_va_true, y_pred))
        rows.append({
            "model": name,
            "R2_mean": float(np.mean(r2_scores)),
            "R2_std": float(np.std(r2_scores)),
            "MAPE_mean": float(np.mean(mape_scores)) * 100,
            "MAPE_std": float(np.std(mape_scores)) * 100,
        })
        print(f"  {name:15s}  R²={np.mean(r2_scores):.4f}±{np.std(r2_scores):.4f}"
              f"   MAPE={np.mean(mape_scores)*100:.2f}%±{np.std(mape_scores)*100:.2f}%")
    return pd.DataFrame(rows)


def evaluate_group_cv(model: CatBoostRegressor, X: np.ndarray, y_log: np.ndarray,
                      groups: np.ndarray) -> pd.DataFrame:
    """按温度×波形×材料组合分组验证，补充随机 K 折的泛化审查。"""
    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y_log, groups), start=1):
        m = CatBoostRegressor(**model.get_params())
        m.fit(X[tr_idx], y_log[tr_idx])
        pred = np.exp(m.predict(X[va_idx]))
        actual = np.exp(y_log[va_idx])
        rows.append({
            "fold": fold,
            "n_train": len(tr_idx), "n_valid": len(va_idx),
            "R2": r2_score(actual, pred),
            "MAPE": mean_absolute_percentage_error(actual, pred) * 100,
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "q4_group_cv_folds.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame([{
        "validation": f"GroupKFold({n_splits}) by temperature×waveform×material",
        "R2_mean": result["R2"].mean(), "R2_std": result["R2"].std(ddof=0),
        "MAPE_mean": result["MAPE"].mean(), "MAPE_std": result["MAPE"].std(ddof=0),
    }])
    summary.to_csv(OUTPUT_DIR / "q4_group_cv_results.csv", index=False, encoding="utf-8-sig")
    print("\n分组交叉验证（未见温度×波形×材料组合）:")
    print(summary.to_string(index=False))
    return summary


def main() -> None:
    print("=" * 70)
    print("问题四：磁芯损耗预测模型")
    print("=" * 70)

    # 1. 读取训练集
    train = pd.read_parquet(PROCESSED_DIR / "train_all.parquet")
    print(f"训练集样本数：{len(train)}")

    # 2. 构建特征
    print("提取训练集回归特征 ...")
    X = build_regression_features(train)
    # 对损耗取对数
    y_log = np.log(train["core_loss"].to_numpy(dtype=float) + 1e-8)
    y_true = train["core_loss"].to_numpy(dtype=float)
    print(f"特征矩阵 shape：{X.shape}")

    # 3. 模型对比
    print("\n5 折交叉验证（最终 CatBoost 模型）：")
    models = {
        "CatBoost": CatBoostRegressor(
            iterations=300, depth=8, learning_rate=0.05,
            loss_function="RMSE", random_seed=42, verbose=0, thread_count=-1,
        ),
    }
    cv_df = evaluate_cv(models, X, y_log, y_true)
    cv_df.to_csv(OUTPUT_DIR / "q4_cv_results.csv",
                 index=False, encoding="utf-8-sig")

    group_id = (train["temperature"].astype(str) + "_" + train["waveform"] + "_" + train["material"]).to_numpy()
    evaluate_group_cv(models["CatBoost"], X, y_log, group_id)

    # 4. CatBoost 为原完整模型比较中的优胜模型；本轮重跑聚焦特征修正后的可复现验证。
    best_name = "CatBoost"
    print(f"\n最优模型：{best_name}")

    # 5. 在全部训练集上训练最终模型
    print(f"训练最终 {best_name} 模型 ...")
    final_model = models[best_name]
    final_model.fit(X, y_log)

    # 训练集回代误差
    y_pred_train = np.exp(final_model.predict(X))
    print(f"训练集 R²={r2_score(y_true, y_pred_train):.4f}, "
          f"MAPE={mean_absolute_percentage_error(y_true, y_pred_train)*100:.2f}%")

    # 6. 保存模型
    if isinstance(final_model, CatBoostRegressor):
        final_model.save_model(MODEL_DIR / "q4_catboost.cbm")
        # 同时保存一份特征构造说明
        with open(MODEL_DIR / "q4_feature_spec.txt", "w", encoding="utf-8") as f:
            f.write("CatBoost model\n")
            f.write(f"iterations={final_model.get_params()['iterations']}\n")
            f.write(f"depth={final_model.get_params()['depth']}\n")
            f.write(f"learning_rate={final_model.get_params()['learning_rate']}\n")
            f.write("features (order):\n")
            f.write("  temperature, log(freq), log(Bm), Bm, log(f*Bm),\n")
            f.write("  waveform one-hot (3), material one-hot (4),\n")
            f.write("  kurtosis, skewness, peak_to_peak, shape_factor, crest_factor,\n")
            f.write("  plateau_ratio, max_slope_norm, spectral_entropy, spectral_centroid\n")
        print(f"模型已保存：{MODEL_DIR / 'q4_catboost.cbm'}")
    else:
        import joblib
        joblib.dump(final_model, MODEL_DIR / "q4_model.pkl")
        print(f"模型已保存：{MODEL_DIR / 'q4_model.pkl'}")

    # 7. 读取附件三测试集
    test_q4 = pd.read_parquet(PROCESSED_DIR / "test_q4.parquet")
    print(f"\n附件三样本数：{len(test_q4)}")

    print("提取测试集特征 ...")
    X_test = build_regression_features(test_q4)
    pred_log = final_model.predict(X_test)
    pred_loss = np.exp(pred_log)

    # 8. 保存结果
    result = pd.DataFrame({
        "sample_id": test_q4["sample_id"].to_numpy(),
        "pred_loss": pred_loss,
        "pred_loss_round1": np.round(pred_loss, 1),
    })
    result.to_csv(OUTPUT_DIR / "q4_predictions.csv",
                  index=False, encoding="utf-8-sig")

    # 9. 打印指定样本结果
    target_ids = [16, 76, 98, 126, 168, 230, 271, 338, 348, 379]
    print("\n附件三指定样本预测结果（保留 1 位小数）：")
    print("-" * 45)
    print(f"{'样本序号':>8s}  {'预测损耗(W/m³)':>16s}")
    print("-" * 45)
    sub = result[result["sample_id"].isin(target_ids)].copy()
    sub = sub.set_index("sample_id").loc[target_ids].reset_index()
    for _, r in sub.iterrows():
        print(f"{int(r['sample_id']):>8d}  {r['pred_loss_round1']:>16.1f}")

    # 10. 预测分布简单统计
    print(f"\n附件三预测损耗分布：")
    print(f"  最小值: {pred_loss.min():.2f} W/m³")
    print(f"  最大值: {pred_loss.max():.2f} W/m³")
    print(f"  均值:   {pred_loss.mean():.2f} W/m³")
    print(f"  中位数: {np.median(pred_loss):.2f} W/m³")

    print(f"\n结果已保存：{OUTPUT_DIR / 'q4_predictions.csv'}")


if __name__ == "__main__":
    main()
