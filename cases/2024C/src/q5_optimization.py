"""问题五：磁性元件的最优化条件。

思路：
1. 以问题四训练好的 CatBoost 模型作为磁芯损耗预测目标函数。
2. 决策变量：
   - 温度（离散：25, 50, 70, 90 °C）
   - 频率 f（连续：50000 ~ 500000 Hz）
   - 励磁波形（离散：正弦波/三角波/梯形波）
   - 磁通密度峰值 Bm（连续：0.01 ~ 0.32 T）
   - 磁芯材料（离散：材料1~4）
3. 双目标优化：
   - min  磁芯损耗 P_core
   - max  传输磁能 f × Bm   （即 min  -f×Bm）
4. 使用 NSGA-II 多目标进化算法求解 Pareto 前沿。
5. 通过“损耗/磁能比”选出最佳折中解，并打印 Pareto 前沿的关键解。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize as pymoo_minimize

import sys
sys.path.append(str(Path(__file__).parent))
from utils import (
    MODEL_DIR,
    OUTPUT_DIR,
    PROCESSED_DIR,
    build_single_waveform_features,
    get_b_columns,
)


# 离散变量取值
TEMPERATURES = [25.0, 50.0, 70.0, 90.0]
WAVEFORMS = ["正弦波", "三角波", "梯形波"]
MATERIALS = ["材料1", "材料2", "材料3", "材料4"]

# 变量边界（连续部分）
F_MIN, F_MAX = 50000.0, 500000.0
BM_MIN, BM_MAX = 0.01, 0.32


def load_model() -> CatBoostRegressor:
    """加载问题四训练好的 CatBoost 模型。"""
    model = CatBoostRegressor()
    model.load_model(MODEL_DIR / "q4_catboost.cbm")
    return model


def select_waveform_templates(train: pd.DataFrame) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """为每类波形挑选真实、代表性的 1024 点归一化模板。

    以同类样本归一化波形的逐点中位数为中心，选择欧氏距离最小的实测样本。
    这样 Q5 生成的完整波形特征与 Q4 的训练特征保持同一计算机制。
    """
    b_cols = get_b_columns()
    templates: dict[str, np.ndarray] = {}
    rows = []
    for waveform in WAVEFORMS:
        sub = train.loc[train["waveform"] == waveform].copy()
        arrays = sub[b_cols].to_numpy(dtype=float)
        normalized = arrays / np.maximum(np.max(np.abs(arrays), axis=1, keepdims=True), 1e-12)
        center = np.median(normalized, axis=0)
        pos = int(np.argmin(np.mean((normalized - center) ** 2, axis=1)))
        selected = sub.iloc[pos]
        template = normalized[pos]
        templates[waveform] = template
        rows.append({
            "waveform": waveform, "source_row": int(sub.index[pos]),
            "source_temperature": selected["temperature"],
            "source_frequency": selected["frequency"], "source_material": selected["material"],
            "source_Bm": float(np.max(np.abs(arrays[pos]))),
            "template_peak": float(np.max(np.abs(template))),
            "mean_sq_distance_to_median_template": float(np.mean((template - center) ** 2)),
        })
    return templates, pd.DataFrame(rows)


def prepare_template_features(templates: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """缓存真实模板的尺度不变特征，避免 NSGA-II 每次创建 DataFrame。

    先严格调用一次训练期同一特征函数。对于 B(t)=Bm*template(t)，峰度、偏度、
    形状/频域特征及归一化斜率均不随 Bm 改变；仅 Bm、log(Bm)、log(fBm)、
    峰峰值需要在每次评价时更新，故此缓存与逐条重算数学等价。
    """
    prepared = {}
    for waveform, template in templates.items():
        prepared[waveform] = build_single_waveform_features(
            temperature=25.0, frequency=50000.0, waveform=waveform,
            Bm=1.0, material=MATERIALS[0], waveform_template=template,
        )
    return prepared


class CoreLossProblem(ElementwiseProblem):
    """多目标优化问题定义。

    决策变量（5 维，全部为浮点，离散变量在 evaluate 中四舍五入取整）：
        x[0] : 温度索引 [0, 3]
        x[1] : 频率 f   [F_MIN, F_MAX]
        x[2] : 波形索引 [0, 2]
        x[3] : Bm      [BM_MIN, BM_MAX]
        x[4] : 材料索引 [0, 3]

    目标（2 个）：
        F[0] : 磁芯损耗 P_core（越小越好）
        F[1] : -传输磁能 f*Bm（越小越好 => 传输磁能越大越好）
    """

    def __init__(self, model: CatBoostRegressor, waveform_templates: dict[str, np.ndarray]):
        super().__init__(
            n_var=5,
            n_obj=2,
            n_constr=0,
            xl=np.array([0.0, F_MIN, 0.0, BM_MIN, 0.0]),
            xu=np.array([3.0, F_MAX, 2.0, BM_MAX, 3.0]),
            vtype=float,
        )
        self.model = model
        self.waveform_templates = waveform_templates
        self.template_features = prepare_template_features(waveform_templates)

    def _evaluate(self, x, out, *args, **kwargs):
        # 离散变量四舍五入取整
        t_idx = int(round(x[0]))
        f = float(x[1])
        w_idx = int(round(x[2]))
        Bm = float(x[3])
        m_idx = int(round(x[4]))

        # 边界保护
        t_idx = max(0, min(3, t_idx))
        w_idx = max(0, min(2, w_idx))
        m_idx = max(0, min(3, m_idx))

        temperature = TEMPERATURES[t_idx]
        waveform = WAVEFORMS[w_idx]
        material = MATERIALS[m_idx]

        # 从真实模板缓存的尺度不变特征构造输入；见 prepare_template_features。
        feat = self.template_features[waveform].copy()
        # 模板缓存使用 Bm=1；峰峰值随 Bm 线性缩放，先保存其基准值。
        peak_to_peak_unit = feat[14]
        feat[0] = temperature
        feat[1] = np.log(f + 1.0)
        feat[2] = np.log(Bm + 1e-8)
        feat[3] = Bm
        feat[4] = np.log(f * Bm + 1e-8)
        feat[5:8] = [1.0 if waveform == w else 0.0 for w in WAVEFORMS]
        feat[8:12] = [1.0 if material == m else 0.0 for m in MATERIALS]
        feat[14] = peak_to_peak_unit * Bm  # 峰峰值是唯一随 Bm 线性缩放的统计量。
        # CatBoost 预测 log(P)，需要 exp 还原
        log_p = float(self.model.predict([feat])[0])
        loss = float(np.exp(log_p))
        trans_energy = f * Bm

        out["F"] = np.array([loss, -trans_energy])


def main() -> None:
    print("=" * 70)
    print("问题五：磁性元件的最优化条件")
    print("=" * 70)

    # 1. 加载模型
    print("加载问题四 CatBoost 模型 ...")
    model = load_model()

    # 2. 从训练集构造真实波形模板，再定义优化问题。
    train = pd.read_parquet(PROCESSED_DIR / "train_all.parquet")
    waveform_templates, template_info = select_waveform_templates(train)
    template_info.to_csv(OUTPUT_DIR / "q5_waveform_templates.csv", index=False, encoding="utf-8-sig")
    print("使用的真实代表性波形模板：")
    print(template_info.to_string(index=False))
    problem = CoreLossProblem(model=model, waveform_templates=waveform_templates)

    # 3. NSGA-II 配置
    algorithm = NSGA2(
        pop_size=120,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.1, eta=20),
    )

    # 4. 执行优化
    print("运行 NSGA-II 多目标优化（180 代）...")
    res = pymoo_minimize(
        problem,
        algorithm,
        ("n_gen", 180),
        seed=42,
        verbose=False,
    )

    pareto_X = res.X
    pareto_F = res.F
    n_pareto = len(pareto_X)
    print(f"\nPareto 最优解个数：{n_pareto}")

    # 5. 解析 Pareto 解
    rows = []
    for i, (x, f_val) in enumerate(zip(pareto_X, pareto_F)):
        t_idx = int(round(x[0]))
        w_idx = int(round(x[2]))
        m_idx = int(round(x[4]))
        rows.append({
            "pareto_id": i,
            "temperature": TEMPERATURES[t_idx],
            "frequency": float(x[1]),
            "waveform": WAVEFORMS[w_idx],
            "Bm": float(x[3]),
            "material": MATERIALS[m_idx],
            "core_loss": float(f_val[0]),
            "trans_energy": float(-f_val[1]),
        })
    pareto_df = pd.DataFrame(rows)
    # NSGA-II 在离散变量四舍五入后可能产生同一有效解，输出前去重。
    before_dedup = len(pareto_df)
    pareto_df["_freq_key"] = pareto_df["frequency"].round(6)
    pareto_df["_bm_key"] = pareto_df["Bm"].round(9)
    pareto_df = pareto_df.drop_duplicates(
        subset=["temperature", "_freq_key", "waveform", "_bm_key", "material"]
    ).drop(columns=["_freq_key", "_bm_key"])
    pareto_df = pareto_df.reset_index(drop=True)
    print(f"Pareto 有效解去重：{before_dedup} -> {len(pareto_df)}")
    # 按“损耗/磁能比”排序（越小越优）
    pareto_df["loss_per_energy"] = (
        pareto_df["core_loss"] / (pareto_df["trans_energy"] + 1e-8)
    )
    pareto_df = pareto_df.sort_values("loss_per_energy").reset_index(drop=True)

    print("\nPareto 前沿（按 损耗/磁能比 升序，前 15 个解）：")
    print(pareto_df.head(15).to_string(index=False))

    # 6. 三种决策方案
    print("\n" + "=" * 70)
    print("三种决策方案对比")
    print("=" * 70)

    # 6.1 最低损耗解
    min_loss_row = pareto_df.loc[pareto_df["core_loss"].idxmin()]
    print("\n[1] 最低磁芯损耗方案：")
    print(f"  温度={min_loss_row['temperature']:.0f}°C, "
          f"频率={min_loss_row['frequency']:.0f}Hz, "
          f"波形={min_loss_row['waveform']}, "
          f"Bm={min_loss_row['Bm']:.4f}T, "
          f"材料={min_loss_row['material']}")
    print(f"  磁芯损耗={min_loss_row['core_loss']:.2f} W/m³, "
          f"传输磁能={min_loss_row['trans_energy']:.2f}")

    # 6.2 最大传输磁能解
    max_energy_row = pareto_df.loc[pareto_df["trans_energy"].idxmax()]
    print("\n[2] 最大传输磁能方案：")
    print(f"  温度={max_energy_row['temperature']:.0f}°C, "
          f"频率={max_energy_row['frequency']:.0f}Hz, "
          f"波形={max_energy_row['waveform']}, "
          f"Bm={max_energy_row['Bm']:.4f}T, "
          f"材料={max_energy_row['material']}")
    print(f"  磁芯损耗={max_energy_row['core_loss']:.2f} W/m³, "
          f"传输磁能={max_energy_row['trans_energy']:.2f}")

    # 6.3 最佳折中解：归一化后距理想点 (最低损耗、最大磁能) 最近。
    # 单纯最小化损耗/磁能比会退化为接近最低损耗端，不能体现“双目标折中”。
    loss_span = max(pareto_df["core_loss"].max() - pareto_df["core_loss"].min(), 1e-12)
    energy_span = max(pareto_df["trans_energy"].max() - pareto_df["trans_energy"].min(), 1e-12)
    pareto_df["utopia_distance"] = np.sqrt(
        ((pareto_df["core_loss"] - pareto_df["core_loss"].min()) / loss_span) ** 2
        + ((pareto_df["trans_energy"].max() - pareto_df["trans_energy"]) / energy_span) ** 2
    )
    best_row = pareto_df.loc[pareto_df["utopia_distance"].idxmin()]
    print("\n[3] 最佳折中方案（归一化理想点距离最小）：")
    print(f"  温度={best_row['temperature']:.0f}°C, "
          f"频率={best_row['frequency']:.0f}Hz, "
          f"波形={best_row['waveform']}, "
          f"Bm={best_row['Bm']:.4f}T, "
          f"材料={best_row['material']}")
    print(f"  磁芯损耗={best_row['core_loss']:.2f} W/m³, "
          f"传输磁能={best_row['trans_energy']:.2f}, "
          f"理想点距离={best_row['utopia_distance']:.6f}")

    # 7. 保存结果
    pareto_df.to_csv(OUTPUT_DIR / "q5_pareto_front.csv",
                     index=False, encoding="utf-8-sig")
    np.save(OUTPUT_DIR / "q5_pareto_X.npy", pareto_X)
    np.save(OUTPUT_DIR / "q5_pareto_F.npy", pareto_F)

    # 保存决策方案
    decision = pd.DataFrame([
        {"方案": "最低损耗", **min_loss_row.to_dict()},
        {"方案": "最大磁能", **max_energy_row.to_dict()},
        {"方案": "最佳折中", **best_row.to_dict()},
    ])
    decision.to_csv(OUTPUT_DIR / "q5_decisions.csv",
                    index=False, encoding="utf-8-sig")

    print(f"\n结果已保存：")
    print(f"  {OUTPUT_DIR / 'q5_pareto_front.csv'}")
    print(f"  {OUTPUT_DIR / 'q5_decisions.csv'}")


if __name__ == "__main__":
    main()
