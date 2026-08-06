from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"


def find_excel(keyword: str) -> Path:
    files = [
        file
        for file in RAW_DIR.glob("*.xlsx")
        if keyword in file.name
    ]

    if len(files) == 0:
        raise FileNotFoundError(
            f"没有找到包含 {keyword} 的 Excel 文件"
        )

    if len(files) > 1:
        raise RuntimeError(
            f"找到多个包含 {keyword} 的文件：{files}"
        )

    return files[0]


def inspect_excel(keyword: str) -> None:
    file_path = find_excel(keyword)

    print("\n" + "=" * 70)
    print(f"检查：{file_path.name}")
    print("=" * 70)

    excel = pd.ExcelFile(
        file_path,
        engine="openpyxl",
    )

    print("Sheet 数量：", len(excel.sheet_names))
    print("Sheet 名称：", excel.sheet_names)

    for sheet_name in excel.sheet_names:
        print("\n" + "-" * 70)
        print("Sheet：", sheet_name)

        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            engine="openpyxl",
        )

        print("shape：", df.shape)

        print("\n前 10 个列名：")
        print(df.columns[:10].tolist())

        print("\n后 5 个列名：")
        print(df.columns[-5:].tolist())

        print("\n前 3 行前 8 列：")
        print(
            df.iloc[:3, :8]
            .to_string(index=False)
        )

        print("\n总缺失值：")
        print(df.isna().sum().sum())


def main():
    inspect_excel("附件一")
    inspect_excel("附件二")
    inspect_excel("附件三")
    inspect_excel("附件四")


if __name__ == "__main__":
    main()