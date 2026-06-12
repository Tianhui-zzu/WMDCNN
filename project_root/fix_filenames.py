import os
import re

# ==========================================
# ⚙️ 核心配置区域
# ==========================================
TARGET_DIR = "./data/augmented_csvs"  # 你的数据存储路径


def main():
    if not os.path.exists(TARGET_DIR):
        print(f"[!] 错误: 找不到文件夹 {TARGET_DIR}")
        return

    files = os.listdir(TARGET_DIR)
    fixed_count = 0

    for filename in files:
        if not filename.endswith(".csv"):
            continue

        # 👑 强化版核心逻辑：
        # 匹配 "NH3" 后面紧接任意个数字（\d+）的情况
        # \1 代表匹配到的那串数字
        new_filename = re.sub(r'NH3(\d+)', r'NH3_\1', filename)

        # 如果名字发生了改变，说明捕捉到了错误的命名
        if new_filename != filename:
            old_path = os.path.join(TARGET_DIR, filename)
            new_path = os.path.join(TARGET_DIR, new_filename)

            # 安全校验：防止重命名撞车覆盖
            if os.path.exists(new_path):
                print(f"[!] 警告: 目标文件 {new_filename} 已存在，跳过修复 {filename}")
                continue

            os.rename(old_path, new_path)
            print(f"[*] 修复成功: {filename.ljust(30)} -> {new_filename}")
            fixed_count += 1

    print("-" * 50)
    print(f"[+] 终极修正完成！共修复了 {fixed_count} 个顽固的不规范文件名。")


if __name__ == "__main__":
    main()