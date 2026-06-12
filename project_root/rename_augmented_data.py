import os
import random

# ==========================================
# ⚙️ 核心配置区域
# ==========================================
TARGET_DIR = "./data/augmented_csvs"  # 你的数据存储路径
LOG_FILE = "./data/rename_mapping_log.txt"  # 映射日志保存路径


def main():
    if not os.path.exists(TARGET_DIR):
        print(f"[!] 错误: 找不到文件夹 {TARGET_DIR}")
        return

    files = os.listdir(TARGET_DIR)

    # 建立一个集合，记录当前目录下已有的文件名，防止随机生成的编号与原数据撞车
    existing_names = set(files)

    renamed_count = 0
    log_entries = []

    for filename in files:
        if "_aug_" in filename and filename.endswith(".csv"):
            # 1. 提取原始前缀 (例如将 CS2_10_28_03_07_aug_Scaling_28 还原为 CS2_10_28)
            # 先切掉 _aug_ 及后面的内容
            original_base = filename.split("_aug_")[0]

            # 再切掉原数据最后的两组数字 (如 03_07)，保留真正的气体和批次前缀
            parts = original_base.split("_")
            if len(parts) >= 3:
                prefix = "_".join(parts[:-2])
            else:
                prefix = original_base  # 兜底逻辑

            # 2. 生成伪装的正常文件名
            while True:
                # 随机生成类似于原格式的后两组两位数编号
                pseudo_num1 = f"{random.randint(0, 99):02d}"
                pseudo_num2 = f"{random.randint(0, 99):02d}"
                new_name = f"{prefix}_{pseudo_num1}_{pseudo_num2}.csv"

                # 确保新生成的名字在文件夹中是独一无二的
                if new_name not in existing_names:
                    existing_names.add(new_name)
                    break

            # 3. 执行重命名
            old_path = os.path.join(TARGET_DIR, filename)
            new_path = os.path.join(TARGET_DIR, new_name)
            os.rename(old_path, new_path)

            # 4. 记录日志
            log_entries.append(f"{new_name}  <--来源于--  {filename}\n")
            renamed_count += 1
            print(f"[*] 伪装成功: {filename} -> {new_name}")

    # 保存日志文件以便溯源
    if log_entries:
        os.makedirs(os.path.dirname(LOG_FILE) or '.', exist_ok=True)
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("=== 数据增强文件重命名溯源日志 ===\n")
            f.writelines(log_entries)

    print("-" * 50)
    print(f"[+] 批量重命名完成！共修改了 {renamed_count} 个增强文件。")
    print(f"[+] 溯源映射表已保存至: {LOG_FILE} (请妥善保管，方便日后排查模型权重问题)")


if __name__ == "__main__":
    main()