from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

NUM_B_POINTS = 1024


def find_excel(keyword: str) -> Path:
    """根据附件名称寻找 Excel 文件。"""
    files = [
        file
        for file in RAW_DIR.glob("*.xlsx")
        if keyword in file.name
    ]

    if len(files) == 0:
        raise FileNotFoundError(
            f"没有找到包含“{keyword}”的 Excel 文件"
        )

    if len(files) > 1:
        raise RuntimeError(
            f"找到多个包含“{keyword}”的 Excel 文件：{files}"
        )

    return files[0]


def get_b_columns() -> list[str]:
    """统一生成 1024 个磁通密度字段名。"""
    return [
        f"B_{i:04d}"
        for i in range(NUM_B_POINTS)
    ]


def clean_basic(df: pd.DataFrame) -> pd.DataFrame:
    """
    基础清理。

    当前阶段仅：
    1. 删除完全空白行
    2. 删除完全空白列
    3. 重置索引

    不修改任何真实测量数据。
    """
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    return df.reset_index(drop=True)


def process_attachment_1() -> pd.DataFrame:
    """处理附件一训练集。"""

    file_path = find_excel("附件一")

    excel = pd.ExcelFile(
        file_path,
        engine="openpyxl",
    )

    if len(excel.sheet_names) != 4:
        raise ValueError(
            f"附件一应有4个Sheet，当前发现{len(excel.sheet_names)}个"
        )

    all_materials = []

    for material_id, sheet_name in enumerate(
        excel.sheet_names,
        start=1,
    ):
        print(f"处理训练集：{sheet_name}")

        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            engine="openpyxl",
        )

        df = clean_basic(df)

        if df.shape[1] != 1028:
            raise ValueError(
                f"{sheet_name} 应为1028列，实际为{df.shape[1]}列"
            )

        # 根据字段位置统一名称
        df.columns = [
            "temperature",
            "frequency",
            "core_loss",
            "waveform",
            *get_b_columns(),
        ]

        # 插入材料信息
        df.insert(
            4,
            "material",
            f"材料{material_id}",
        )

        df.insert(
            5,
            "material_id",
            material_id,
        )

        # 清理文字
        df["waveform"] = (
            df["waveform"]
            .astype(str)
            .str.strip()
        )

        # 保存每种材料
        csv_path = (
            PROCESSED_DIR
            / f"train_{material_id}.csv"
        )

        parquet_path = (
            PROCESSED_DIR
            / f"train_{material_id}.parquet"
        )

        df.to_csv(
            csv_path,
            index=False,
            encoding="utf-8-sig",
        )

        df.to_parquet(
            parquet_path,
            index=False,
        )

        print(
            f"  样本数：{len(df)}"
        )

        all_materials.append(df)

    train_all = pd.concat(
        all_materials,
        ignore_index=True,
    )

    train_all.to_csv(
        PROCESSED_DIR / "train_all.csv",
        index=False,
        encoding="utf-8-sig",
    )

    train_all.to_parquet(
        PROCESSED_DIR / "train_all.parquet",
        index=False,
    )

    print(
        f"\n训练集总样本数：{len(train_all)}"
    )

    return train_all


def process_attachment_2() -> pd.DataFrame:
    """处理附件二，问题一测试集。"""

    file_path = find_excel("附件二")

    df = pd.read_excel(
        file_path,
        sheet_name="测试集",
        engine="openpyxl",
    )

    df = clean_basic(df)

    if df.shape[1] != 1028:
        raise ValueError(
            f"附件二应为1028列，实际为{df.shape[1]}列"
        )

    df.columns = [
        "sample_id",
        "temperature",
        "frequency",
        "material",
        *get_b_columns(),
    ]

    df["material"] = (
        df["material"]
        .astype(str)
        .str.strip()
    )

    df.to_csv(
        PROCESSED_DIR / "test_q1.csv",
        index=False,
        encoding="utf-8-sig",
    )

    df.to_parquet(
        PROCESSED_DIR / "test_q1.parquet",
        index=False,
    )

    print(
        f"附件二样本数：{len(df)}"
    )

    return df


def process_attachment_3() -> pd.DataFrame:
    """处理附件三，问题四测试集。"""

    file_path = find_excel("附件三")

    df = pd.read_excel(
        file_path,
        sheet_name="测试集",
        engine="openpyxl",
    )

    df = clean_basic(df)

    if df.shape[1] != 1029:
        raise ValueError(
            f"附件三应为1029列，实际为{df.shape[1]}列"
        )

    df.columns = [
        "sample_id",
        "temperature",
        "frequency",
        "material",
        "waveform",
        *get_b_columns(),
    ]

    df["material"] = (
        df["material"]
        .astype(str)
        .str.strip()
    )

    df["waveform"] = (
        df["waveform"]
        .astype(str)
        .str.strip()
    )

    df.to_csv(
        PROCESSED_DIR / "test_q4.csv",
        index=False,
        encoding="utf-8-sig",
    )

    df.to_parquet(
        PROCESSED_DIR / "test_q4.parquet",
        index=False,
    )

    print(
        f"附件三样本数：{len(df)}"
    )

    return df


def verify_data(
    train: pd.DataFrame,
    test_q1: pd.DataFrame,
    test_q4: pd.DataFrame,
) -> None:
    """做最基础的数据完整性验证。"""

    print("\n" + "=" * 60)
    print("数据完整性检查")
    print("=" * 60)

    assert len(train) == 12400
    assert len(test_q1) == 80
    assert len(test_q4) == 400

    assert len(
        [
            c
            for c in train.columns
            if c.startswith("B_")
        ]
    ) == 1024

    assert len(
        [
            c
            for c in test_q1.columns
            if c.startswith("B_")
        ]
    ) == 1024

    assert len(
        [
            c
            for c in test_q4.columns
            if c.startswith("B_")
        ]
    ) == 1024

    assert train.isna().sum().sum() == 0
    assert test_q1.isna().sum().sum() == 0
    assert test_q4.isna().sum().sum() == 0

    print("训练集：12400 条 ✓")
    print("附件二：80 条 ✓")
    print("附件三：400 条 ✓")
    print("每条波形：1024 点 ✓")
    print("缺失值检查：通过 ✓")


def main():

    print("=" * 60)
    print("2024C 数据标准化整理")
    print("=" * 60)

    train = process_attachment_1()

    print()
    test_q1 = process_attachment_2()

    print()
    test_q4 = process_attachment_3()

    verify_data(
        train,
        test_q1,
        test_q4,
    )

    print("\n处理结果保存在：")
    print(PROCESSED_DIR)


if __name__ == "__main__":
    main()