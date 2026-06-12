import os
import re
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from sklearn.model_selection import train_test_split


def get_category(filename):
    """通过正则提取 CSV 文件名中的气体类别，并统一格式"""
    match = re.match(r'^([A-Za-z0-9_]+?)_\d{2}_\d{2}_', filename)
    if not match:
        return "UNKNOWN"
    cat = match.group(1).replace("NH311", "NH3").replace("NH312", "NH3").replace("NH3_11", "NH3")
    return cat


def load_dataset(data_dir, batch_size=5, test_size=0.2):
    """
    加载文件夹中所有的 CSV 数据，自动打标签，并划分为训练集和验证集
    """
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    # 1. 自动扫描并映射所有的气体类别
    categories = sorted(list(set([get_category(f) for f in csv_files])))
    label_map = {cat: i for i, cat in enumerate(categories)}
    print(f"[*] 成功映射 {len(categories)} 种气体类别: {label_map}")

    X, Y1 = [], []

    for file in csv_files:
        path = os.path.join(data_dir, file)
        # 根据你之前数据的特征，假设无表头，用 None
        df = pd.read_csv(path, header=None)

        # 假设第一列是序号列，我们剥离它，只取后面的传感器浮点数据
        sensor_data = df.iloc[:, 1:].values.astype(np.float32)

        # 因为你的网络入口是 1D 卷积，我们需要把 511x9 的矩阵展平(flatten)成一个长一维数组
        x_flat = sensor_data.flatten()
        X.append(x_flat)

        # 根据文件名打上分类标签
        cat_str = get_category(file)
        Y1.append(label_map.get(cat_str, 0))

    X = np.array(X)
    Y1 = np.array(Y1)

    # 你原代码需要 Y2 (浓度回归标签)，由于目前 CSV 名字里没有体现浓度，先用 0 占位防报错
    Y2 = np.zeros_like(Y1, dtype=np.float32)

    # 2. 按照 8:2 随机划分训练集和测试集 (stratify 保证各类气体比例均匀)
    x_train, x_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
        X, Y1, Y2, test_size=test_size, random_state=42, stratify=Y1
    )

    def build_loader(x, y1, y2):
        tx = torch.from_numpy(x)
        ty1 = torch.from_numpy(y1).long()
        ty2 = torch.from_numpy(y2).float()

        data1 = Data.TensorDataset(tx, ty1)
        data2 = Data.TensorDataset(tx, ty2)

        # 注意：Windows 下如果是多进程读取可能会报错，建议设为 num_workers=0
        loader1 = Data.DataLoader(dataset=data1, batch_size=batch_size, shuffle=True, num_workers=0)
        loader2 = Data.DataLoader(dataset=data2, batch_size=batch_size, shuffle=True, num_workers=0)
        return [loader1, loader2]

    print(f"[*] 训练集样本数: {len(x_train)} | 测试集样本数: {len(x_test)}")

    train_loader = build_loader(x_train, y1_train, y2_train)
    test_loader = build_loader(x_test, y1_test, y2_test)

    return train_loader, test_loader, len(categories)