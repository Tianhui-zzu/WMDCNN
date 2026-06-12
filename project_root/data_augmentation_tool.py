import os
import shutil
import random
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter

# ==========================================
# ⚙️ 核心配置区域
# ==========================================
INPUT_DIR = "./data/original_csvs"
OUTPUT_DIR = "./data/augmented_csvs"
TARGET_TOTAL = 600  # 最终扩充到的总数量

HEADER_ROW = None  # CSV无表头填 None
INDEX_COL_EXISTS = True  # 第1列是否是时间步/序号？(是的话，增强时将忽略并保护此列)


# ==========================================
# 🔍 增强方式底层逻辑定义
# ==========================================
def add_gaussian_noise(sensor_data, mean=0, std=0.015):
    noise = np.random.normal(mean, std, sensor_data.shape)
    return sensor_data + noise


def random_scaling(sensor_data, scale_range=(0.95, 1.05)):
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    return sensor_data * scale_factor


def time_shifting(sensor_data, shift_limit=5):
    shift_steps = np.random.randint(-shift_limit, shift_limit + 1)
    if shift_steps == 0: shift_steps = 1
    shifted_data = np.empty_like(sensor_data)
    if shift_steps > 0:
        shifted_data[:shift_steps, :] = sensor_data[0, :]
        shifted_data[shift_steps:, :] = sensor_data[:-shift_steps, :]
    else:
        shift_steps = abs(shift_steps)
        shifted_data[-shift_steps:, :] = sensor_data[-1, :]
        shifted_data[:-shift_steps, :] = sensor_data[shift_steps:, :]
    return shifted_data


# ==========================================
# 🚀 增强策略分发路由
# ==========================================
AUG_METHODS = {
    "Jittering": lambda data: add_gaussian_noise(data, std=0.02),
    "Scaling": lambda data: random_scaling(data, scale_range=(0.90, 1.10)),
    "TimeShift": lambda data: time_shifting(data, shift_limit=10)
}


def get_category(filename):
    """从文件名解析气体类别，并将变体规范化"""
    # 匹配规则：遇到第一个 _数字_数字_ 前的部分即为类别前缀
    match = re.match(r'^([A-Za-z0-9_]+?)_\d{2}_\d{2}_', filename)
    if not match:
        return "UNKNOWN"
    cat = match.group(1)
    # 规范化清理 (比如把 NH311, NH312, NH3_11 统一为 NH3 类别)
    cat = cat.replace("NH311", "NH3").replace("NH312", "NH3")
    return cat


def setup_directories():
    if os.path.exists(OUTPUT_DIR):
        print(f"[*] 清理旧的输出目录: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    print(f"[*] 创建输出目录: {OUTPUT_DIR}")


def main():
    setup_directories()

    csv_paths = [p for p in Path(INPUT_DIR).iterdir() if p.suffix.lower() == '.csv']
    if not csv_paths:
        print("[!] 错误：未在输入目录找到 CSV 文件。")
        return

    # 1. 对原始文件进行解析和分组
    category_map = defaultdict(list)
    for p in csv_paths:
        cat = get_category(p.name)
        category_map[cat].append(p)

    num_classes = len(category_map)
    target_per_class = TARGET_TOTAL // num_classes
    method_names = list(AUG_METHODS.keys())

    print(
        f"[*] 发现 {num_classes} 种气体类别。为了达到总数 {TARGET_TOTAL}，每类目标将被严格扩充至 {target_per_class} 个。")
    print("-" * 50)

    # 2. 按类别平衡增强
    for cat, paths in category_map.items():
        original_count = len(paths)
        print(f"[>] 类别: {cat.ljust(10)} | 现有: {original_count} | 需增强: {target_per_class - original_count}")

        # 2.1 无论如何，先把该类别下所有的原图原样保存到输出目录
        for p in paths:
            df = pd.read_csv(p, header=HEADER_ROW)
            original_output_path = os.path.join(OUTPUT_DIR, p.name)
            df.to_csv(original_output_path, index=False, header=False if HEADER_ROW is None else True)

        # 2.2 计算缺口，并在原数据中进行随机重采样以生成增强数据
        augment_needed = target_per_class - original_count
        if augment_needed > 0:
            sampled_paths = random.choices(paths, k=augment_needed)  # 允许重复采样

            for i, p in enumerate(sampled_paths):
                df = pd.read_csv(p, header=HEADER_ROW)

                if INDEX_COL_EXISTS:
                    index_data = df.iloc[:, 0].values.reshape(-1, 1)
                    sensor_data = df.iloc[:, 1:].values
                else:
                    sensor_data = df.values

                # 随机选择增强方法
                chosen_method_name = random.choice(method_names)
                augment_func = AUG_METHODS[chosen_method_name]
                augmented_sensor_data = augment_func(sensor_data)

                if INDEX_COL_EXISTS:
                    final_data = np.hstack((index_data, augmented_sensor_data))
                else:
                    final_data = augmented_sensor_data

                df_augmented = pd.DataFrame(final_data).round(6)

                # 2.3 生成带有增强策略标记的文件名 (加入循环序号 i 防止同一文件被抽中多次导致重名覆盖)
                filename = p.stem
                ext = p.suffix
                aug_name = f"{filename}_aug_{chosen_method_name}_{i}{ext}"
                aug_output_path = os.path.join(OUTPUT_DIR, aug_name)

                df_augmented.to_csv(aug_output_path, index=False, header=False if HEADER_ROW is None else True)

    print("-" * 50)
    # 统计校验最终结果
    final_files = list(Path(OUTPUT_DIR).iterdir())
    final_categories = Counter([get_category(p.name) for p in final_files])
    print(f"[+] 处理完成！当前目录总文件数: {len(final_files)}")
    print(f"[+] 扩充后各类别数量分布: {dict(final_categories)}")


if __name__ == "__main__":
    main()