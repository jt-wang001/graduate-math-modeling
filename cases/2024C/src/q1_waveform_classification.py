"""问题一：励磁波形分类。

思路：
1. 从 1024 点磁通密度序列中提取时域统计特征、频域主频特征和波形形状特征
   （平台段比例、最大斜率、过零次数等，专门用于区分正弦/三角/梯形波）。
2. 使用随机森林 + SVM + 梯度提升三类分类器，通过交叉验证评估，
   并采用软投票（概率平均）集成，提升分类稳健性。
3. 对附件二 80 个样本预测，按要求填入附件四第 2 列（1=正弦, 2=三角, 3=梯形）。
4. 输出三种波形的数量统计，并打印指定样本序号的分类结果。
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import sys
sys.path.append(str(Path(__file__).parent))
from utils import (
    OUTPUT_DIR,
    PROCESSED_DIR,
    build_classification_features,
    waveform_to_label,
)


def main() -> None:
    print("=" * 70)
    print("问题一：励磁波形分类")
    print("=" * 70)

    # 1. 读取训练集
    train = pd.read_parquet(PROCESSED_DIR / "train_all.parquet")
    print(f"训练集样本数：{len(train)}")

    # 2. 构建特征
    print("提取训练集特征 ...")
    X_train = build_classification_features(train)
    y_train = train["waveform"].map(waveform_to_label).to_numpy()

    # 标准化（主要对 SVM 有用）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 3. 定义基分类器
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=1,
        random_state=42, n_jobs=-1,
    )
    svm = SVC(
        kernel="rbf", C=10, gamma="scale",
        probability=True, random_state=42,
    )
    gbdt = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        random_state=42,
    )

    # 软投票集成
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("svm", svm), ("gbdt", gbdt)],
        voting="soft", n_jobs=-1,
    )

    # 4. 5 折交叉验证
    print("\n5 折交叉验证准确率：")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in [("RandomForest", rf), ("SVM", svm), ("GBDT", gbdt),
                      ("Ensemble(soft)", ensemble)]:
        # SVM/Ensemble 需要标准化特征
        if name in ("SVM", "Ensemble(soft)"):
            scores = cross_val_score(clf, X_train_scaled, y_train,
                                     cv=skf, scoring="accuracy", n_jobs=-1)
        else:
            scores = cross_val_score(clf, X_train, y_train,
                                     cv=skf, scoring="accuracy", n_jobs=-1)
        print(f"  {name:18s}: {scores.mean():.4f} ± {scores.std():.4f}")

    # 5. 在全部训练集上训练最终集成模型
    print("\n训练最终集成模型 ...")
    ensemble.fit(X_train_scaled, y_train)

    # 6. 读取附件二测试集
    test_q1 = pd.read_parquet(PROCESSED_DIR / "test_q1.parquet")
    print(f"附件二样本数：{len(test_q1)}")

    print("提取测试集特征 ...")
    X_test = build_classification_features(test_q1)
    X_test_scaled = scaler.transform(X_test)

    # 7. 预测
    pred = ensemble.predict(X_test_scaled)
    proba = ensemble.predict_proba(X_test_scaled)
    confidence = proba.max(axis=1)

    # 8. 保存结果
    result = pd.DataFrame({
        "sample_id": test_q1["sample_id"].to_numpy(),
        "pred_waveform": pred.astype(int),
        "confidence": confidence,
    })
    result.to_csv(OUTPUT_DIR / "q1_results.csv", index=False, encoding="utf-8-sig")

    # 9. 统计三种波形数量
    counts = Counter(pred)
    print("\n附件二波形数量统计：")
    print(f"  正弦波(1): {counts.get(1, 0)}")
    print(f"  三角波(2): {counts.get(2, 0)}")
    print(f"  梯形波(3): {counts.get(3, 0)}")

    # 10. 打印指定样本结果
    target_ids = [1, 5, 15, 25, 35, 45, 55, 65, 75, 80]
    print("\n附件二指定样本分类结果：")
    print("-" * 50)
    print(f"{'样本序号':>8s}  {'预测波形':>6s}  {'置信度':>6s}")
    print("-" * 50)
    sub = result[result["sample_id"].isin(target_ids)].copy()
    sub = sub.set_index("sample_id").loc[target_ids].reset_index()
    for _, r in sub.iterrows():
        wf = {1: "正弦波", 2: "三角波", 3: "梯形波"}[int(r["pred_waveform"])]
        print(f"{int(r['sample_id']):>8d}  {wf:>6s}({int(r['pred_waveform'])})  "
              f"{r['confidence']:.3f}")

    print(f"\n结果已保存：{OUTPUT_DIR / 'q1_results.csv'}")


if __name__ == "__main__":
    main()
