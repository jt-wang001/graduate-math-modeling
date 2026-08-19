"""通用工具模块：特征提取、路径定义、通用辅助函数。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

# 项目根目录：cases/2024C
PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"

for _d in (OUTPUT_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# 每条波形的采样点数
NUM_B_POINTS = 1024


def get_b_columns() -> list[str]:
    """统一生成 1024 个磁通密度字段名 B_0000 ... B_1023。"""
    return [f"B_{i:04d}" for i in range(NUM_B_POINTS)]


def waveform_to_label(waveform: str) -> int:
    """正弦波->1，三角波->2，梯形波->3。"""
    return {"正弦波": 1, "三角波": 2, "梯形波": 3}[waveform]


def label_to_waveform(label: int) -> str:
    """1->正弦波，2->三角波，3->梯形波。"""
    return {1: "正弦波", 2: "三角波", 3: "梯形波"}[int(label)]


def extract_time_domain_features(b_series: np.ndarray) -> dict:
    """提取时域统计特征。

    包括均值、标准差、最大/最小值、峰峰值、峰度、偏度、
    峰值因子、波形因子、裕度因子等。
    """
    b = np.asarray(b_series, dtype=float).ravel()
    rms = np.sqrt(np.mean(b ** 2) + 1e-12)
    abs_mean = np.mean(np.abs(b)) + 1e-12
    peak = np.max(np.abs(b))
    return {
        "mean": float(np.mean(b)),
        "std": float(np.std(b)),
        "max": float(np.max(b)),
        "min": float(np.min(b)),
        "peak_to_peak": float(np.max(b) - np.min(b)),
        "kurtosis": float(kurtosis(b, fisher=True)),
        "skewness": float(skew(b)),
        "peak_factor": float(peak / rms),       # 峰值因子
        "shape_factor": float(rms / abs_mean),  # 波形因子
        "margin_factor": float(peak / (np.mean(np.sqrt(np.abs(b))) + 1e-12) ** 2),
        "crest_factor": float(peak / rms),
        "rms": float(rms),
    }


def extract_frequency_features(b_series: np.ndarray, n_top: int = 5) -> dict:
    """提取频域特征：前 n_top 个主频的幅值与对应频率位置（归一化）。

    同时返回频谱熵、谱质心等统计量。
    """
    b = np.asarray(b_series, dtype=float).ravel()
    n = len(b)
    fft_vals = np.fft.rfft(b)
    mag = np.abs(fft_vals)
    # 去除直流分量
    mag[0] = 0.0
    total_energy = np.sum(mag) + 1e-12

    # 取前 n_top 个主频
    top_idx = np.argsort(mag)[-n_top:][::-1]
    top_mags = mag[top_idx]
    top_freqs = top_idx / n  # 归一化频率
    feat: dict = {}
    for i in range(n_top):
        feat[f"mag_{i}"] = float(top_mags[i]) if i < len(top_mags) else 0.0
        feat[f"freq_{i}"] = float(top_freqs[i]) if i < len(top_freqs) else 0.0

    # 频谱熵（归一化）
    psd = mag / total_energy
    psd = psd[psd > 0]
    spectral_entropy = -np.sum(psd * np.log(psd + 1e-12))
    spectral_entropy /= np.log(len(mag)) + 1e-12
    feat["spectral_entropy"] = float(spectral_entropy)

    # 谱质心
    freqs = np.arange(len(mag)) / n
    feat["spectral_centroid"] = float(np.sum(freqs * mag) / total_energy)
    return feat


def extract_shape_features(b_series: np.ndarray) -> dict:
    """提取反映波形“形状”的特征，专门用于区分正弦/三角/梯形波。

    - 正弦波：平滑，过零点附近变化率中等，无平台
    - 三角波：分段线性，过零点附近变化率最大，无平台
    - 梯形波：有平台段（导数≈0），且上升/下降沿陡峭
    """
    b = np.asarray(b_series, dtype=float).ravel()
    # 一阶差分（近似导数）
    diff = np.diff(b)
    abs_diff = np.abs(diff)
    # 平台/缓变段比例：阈值应相对相邻采样点的变化尺度，而非波形幅值。
    # 原阈值 0.05*peak 远大于相邻差分，使该特征在训练集中恒为 1。
    peak = np.max(np.abs(b)) + 1e-12
    threshold = 0.15 * (np.max(abs_diff) + 1e-12)
    plateau_ratio = float(np.mean(abs_diff < threshold))

    # 最大变化率（归一化）
    max_slope = float(np.max(abs_diff) / peak)
    # 变化率的标准差（梯形波因有平台+陡沿，std 较大）
    slope_std = float(np.std(abs_diff) / (peak + 1e-12))

    # 二阶差分（曲率）：正弦波平滑，三角波二阶差分集中在转折点
    diff2 = np.diff(b, n=2)
    cur_std = float(np.std(diff2) / (peak + 1e-12))

    # 过零次数
    sign = np.sign(b)
    sign_changes = np.sum(np.abs(np.diff(sign)) > 1)

    # 波形“线性度”：对相邻点拟合误差（正弦/三角 vs 梯形）
    return {
        "plateau_ratio": plateau_ratio,
        "max_slope_norm": max_slope,
        "slope_std_norm": slope_std,
        "curvature_std_norm": cur_std,
        "zero_crossings": float(sign_changes),
        "slope_kurtosis": float(kurtosis(abs_diff, fisher=True)),
    }


def build_classification_features(df: pd.DataFrame) -> np.ndarray:
    """为分类任务构建特征矩阵：时域+频域+形状特征。"""
    b_cols = get_b_columns()
    rows: list[list[float]] = []
    for _, row in df.iterrows():
        b = row[b_cols].to_numpy(dtype=float)
        td = extract_time_domain_features(b)
        fd = extract_frequency_features(b, n_top=5)
        sf = extract_shape_features(b)
        feat = list(td.values()) + list(fd.values()) + list(sf.values())
        rows.append(feat)
    return np.array(rows, dtype=float)


def build_regression_features(df: pd.DataFrame) -> np.ndarray:
    """为回归任务构建特征矩阵。

    特征包含：温度、log频率、log磁通密度峰值、波形 one-hot、
    材料 one-hot、峰度、偏度、峰峰值、谱熵等。
    与 q5 优化时的特征构造保持一致。
    """
    b_cols = get_b_columns()
    waveforms = ["正弦波", "三角波", "梯形波"]
    materials = ["材料1", "材料2", "材料3", "材料4"]
    rows: list[list[float]] = []
    for _, row in df.iterrows():
        b = row[b_cols].to_numpy(dtype=float)
        td = extract_time_domain_features(b)
        fd = extract_frequency_features(b, n_top=3)
        sf = extract_shape_features(b)
        Bm = float(np.max(np.abs(b)))
        feat = [
            float(row["temperature"]),
            float(np.log(row["frequency"] + 1.0)),
            float(np.log(Bm + 1e-8)),
            Bm,
            float(np.log(row["frequency"] * Bm + 1e-8)),  # f*Bm 对数
        ]
        # 波形 one-hot（若存在该列）
        wf = row.get("waveform", None)
        if isinstance(wf, str):
            feat.extend([1.0 if wf == w else 0.0 for w in waveforms])
        else:
            feat.extend([0.0, 0.0, 0.0])
        # 材料 one-hot
        mat = row.get("material", None)
        if isinstance(mat, str):
            feat.extend([1.0 if mat == m else 0.0 for m in materials])
        else:
            feat.extend([0.0, 0.0, 0.0, 0.0])
        # 额外统计特征
        feat.extend([
            td["kurtosis"],
            td["skewness"],
            td["peak_to_peak"],
            td["shape_factor"],
            td["crest_factor"],
            sf["plateau_ratio"],
            sf["max_slope_norm"],
            fd["spectral_entropy"],
            fd["spectral_centroid"],
        ])
        rows.append(feat)
    return np.array(rows, dtype=float)


def build_single_regression_features(
    temperature: float,
    frequency: float,
    waveform: str,
    Bm: float,
    material: str,
) -> list[float]:
    """为单条样本构建回归特征（用于优化问题中调用模型预测）。

    必须与 build_regression_features 中使用的特征顺序保持一致。
    缺失的时域/频域统计特征使用经验值近似（峰度等用典型波形值代替）。
    """
    waveforms = ["正弦波", "三角波", "梯形波"]
    materials = ["材料1", "材料2", "材料3", "材料4"]

    # 各波形的典型峰度/偏度/形状特征近似值（基于训练集统计）
    typical_stats = {
        "正弦波": dict(kurt=-1.5, skew=0.0, shape=1.1107, crest=1.4142,
                      plateau=0.05, slope=6.28 / 1024, se=0.55, centroid=0.001),
        "三角波": dict(kurt=-1.2, skew=0.0, shape=1.155, crest=1.7320,
                      plateau=0.02, slope=4.0 / 1024, se=0.62, centroid=0.001),
        "梯形波": dict(kurt=0.5, skew=0.0, shape=1.10, crest=1.30,
                      plateau=0.25, slope=8.0 / 1024, se=0.45, centroid=0.001),
    }
    s = typical_stats.get(waveform, typical_stats["正弦波"])

    feat = [
        float(temperature),
        float(np.log(frequency + 1.0)),
        float(np.log(Bm + 1e-8)),
        float(Bm),
        float(np.log(frequency * Bm + 1e-8)),
    ]
    feat.extend([1.0 if waveform == w else 0.0 for w in waveforms])
    feat.extend([1.0 if material == m else 0.0 for m in materials])
    feat.extend([
        s["kurt"], s["skew"], 2.0 * Bm, s["shape"], s["crest"],
        s["plateau"], s["slope"], s["se"], s["centroid"],
    ])
    return feat


def build_single_waveform_features(
    temperature: float,
    frequency: float,
    waveform: str,
    Bm: float,
    material: str,
    waveform_template: np.ndarray,
) -> np.ndarray:
    """以真实 1024 点归一化波形模板构造单条回归特征。

    用于 Q5：将模板缩放到给定 Bm 后，复用训练阶段完全相同的
    ``build_regression_features``，避免人工指定波形统计量造成特征失配。
    """
    template = np.asarray(waveform_template, dtype=float).ravel()
    if template.size != NUM_B_POINTS:
        raise ValueError(f"waveform_template must have {NUM_B_POINTS} points")
    scale = np.max(np.abs(template))
    if scale <= 0:
        raise ValueError("waveform_template must not be all zeros")
    b = template / scale * float(Bm)
    row = {
        "temperature": float(temperature),
        "frequency": float(frequency),
        "waveform": waveform,
        "material": material,
    }
    row.update(dict(zip(get_b_columns(), b)))
    return build_regression_features(pd.DataFrame([row]))[0]


if __name__ == "__main__":
    # 自检
    print("PROJECT_DIR =", PROJECT_DIR)
    print("PROCESSED_DIR =", PROCESSED_DIR)
    print("OUTPUT_DIR =", OUTPUT_DIR)
    print("MODEL_DIR =", MODEL_DIR)
    print("B columns 数量：", len(get_b_columns()))
