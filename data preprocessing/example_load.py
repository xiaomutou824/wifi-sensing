"""数据加载示例：从原始 CSV 到预处理后的 numpy 数组."""

from pathlib import Path

import yaml

from preprocess import preprocess_pipeline, save_processed


def main():
    # 加载配置
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 搜索所有 CSV 文件
    import glob
    csv_pattern = config["data"]["csv_paths"][0]
    csv_files = sorted(glob.glob(csv_pattern))
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {f}")

    if not csv_files:
        print("No CSV files found. Please check csv_paths in config.yaml.")
        return

    # 示例：只取一个文件做快速测试
    test_files = csv_files[:1]
    print(f"\nProcessing {len(test_files)} file(s) for demo...")

    # 执行预处理
    result = preprocess_pipeline(
        csv_paths=test_files,
        config=config,
        fit_norm=True,
    )

    # 查看结果
    windows = result["windows"]  # list of [T, C]
    labels = result["labels"]    # list of str
    norm_stats = result["norm_stats"]

    print(f"\n{'='*50}")
    print(f"Preprocessing Result:")
    print(f"  Total windows: {len(windows)}")
    print(f"  Window shape: {windows[0].shape}")
    print(f"  Unique labels: {set(labels)}")
    print(f"  Norm stats: {norm_stats}")

    # 保存
    output_dir = config["data"]["output_dir"]
    save_processed(output_dir, result, split_name="demo")
    print(f"\nSaved to {output_dir}/")


if __name__ == "__main__":
    main()
