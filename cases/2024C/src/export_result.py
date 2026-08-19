"""结果导出：填充附件四（Excel 表）。

附件四结构（400 行）：
- 第 1 列：序号（1 ~ 400）
- 第 2 列：附件二（80 个样品）励磁波形分类结果（仅前 80 行）
- 第 3 列：附件三（400 个样品）磁芯损耗预测结果（全部 400 行，保留 1 位小数）

填充后保留原文件名“附件四（Excel表）.xlsx”，保存到 outputs/ 目录。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).parent))
from utils import OUTPUT_DIR, RAW_DIR


def main() -> None:
    print("=" * 70)
    print("结果导出：填充附件四")
    print("=" * 70)

    # 1. 读取问题一与问题四结果
    q1 = pd.read_csv(OUTPUT_DIR / "q1_results.csv")
    q4 = pd.read_csv(OUTPUT_DIR / "q4_predictions.csv")

    print(f"问题一分类结果数：{len(q1)}（应=80）")
    print(f"问题四预测结果数：{len(q4)}（应=400）")

    # 2. 读取附件四模板
    template_path = RAW_DIR / "附件四（Excel表）.xlsx"
    template = pd.read_excel(template_path, engine="openpyxl")
    print(f"附件四模板：{template.shape}")
    print(f"附件四列名：{template.columns.tolist()}")

    # 3. 填写第 2 列（问题一分类结果，1=正弦 2=三角 3=梯形）
    # 仅前 80 行（序号 1~80）；序号 81~400 留空
    col_q1 = template.columns[1]
    q1_map = dict(zip(q1["sample_id"], q1["pred_waveform"].astype(int)))
    # 先用 object 列承载 int 或 None，避免被 pandas 自动转成浮点
    template[col_q1] = template["序号"].map(
        lambda x: int(q1_map[int(x)]) if int(x) in q1_map else None
    )

    # 4. 填写第 3 列（问题四预测损耗，保留 1 位小数）
    col_q4 = template.columns[2]
    q4_map = dict(zip(q4["sample_id"], q4["pred_loss_round1"]))
    template[col_q4] = template["序号"].map(
        lambda x: round(float(q4_map.get(int(x), 0.0)), 1)
    )

    # 5. 保存结果（保留原文件名）
    # 用 openpyxl 直接写入，确保第 2 列写为整数而非 1.0/2.0/3.0
    from openpyxl import Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows

    out_path = OUTPUT_DIR / "附件四（Excel表）.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    for r_idx, row in enumerate(
        dataframe_to_rows(template, index=False, header=True), start=1
    ):
        for c_idx, val in enumerate(row, start=1):
            # 第 2 列（c_idx==2）若是 None 则不写入（保持空单元格）
            if c_idx == 2 and val is None:
                continue
            ws.cell(row=r_idx, column=c_idx, value=val)
    wb.save(out_path)

    print(f"\n附件四已生成：{out_path}")
    print("\n前 10 行预览：")
    print(template.head(10).to_string(index=False))
    print("\n第 75~85 行（跨过 Q1 边界）预览：")
    print(template.iloc[74:85].to_string(index=False))
    print("\n最后 5 行预览：")
    print(template.tail(5).to_string(index=False))

    # 6. 检查覆盖率
    n_q1_filled = template[col_q1].notna().sum()
    n_q4_filled = template[col_q4].notna().sum()
    print(f"\n第 2 列（Q1分类）已填：{n_q1_filled} 行（应=80）")
    print(f"第 3 列（Q4预测）已填：{n_q4_filled} 行（应=400）")


if __name__ == "__main__":
    main()
