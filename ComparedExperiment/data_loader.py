import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as op
from sklearn.model_selection import train_test_split

path01 = 'test.txt'
def data_load01():
    df = pd.read_csv(path01,sep=',',names=[i for i in range(0,57)])
    data = df.to_numpy()[:,3:]
    label = df.to_numpy()[:,0] # 分类问题,数据集标签在第一列
    return data,label


data_path = 'segment_data_1hz_version_05_normalized_filter'
def data_load02():
    ls_data = []
    ls_label = []
    files = []
    dict_name = {'CS2_H2S': 5, 'H2S_NH3': 4, 'CS2_NH3': 3, 'CS2': 2, 'HS2': 1, 'NH3': 0}
    for file_name in os.listdir(data_path):
        if (len(file_name) == 19):
            label = dict_name[file_name[0:3]]
        else:
            label = dict_name[file_name[0:7]]

        data = pd.read_csv(op.join(data_path, file_name), sep=',', names=[i for i in range(0, 10)])
        print(data.to_numpy().shape[0])
        if data.to_numpy().shape[0] >= 512:
            data = data.to_numpy()[0:256, 1:10]
            # print(op.join(data_path,file_name),data.shape)
            data = data.tolist()
            ls_data.append(data)
            ls_label.append(label)
            files.append(file_name)

    return np.array(ls_data), np.array(ls_label)
print_folder_path01 = '../handled_data1_1hz_expand_t1+t2_x-min'
def data_load03():
    ls_data = []
    ls_label = []
    dict_name = {'CO': 0, 'Ey': 1, 'Ea': 2, 'Me': 3}
    CO, Ey, Ea, Me = 0, 0, 0, 0
    count_num = [CO, Ey, Ea, Me]
    for file_name in os.listdir(print_folder_path01):
        df = pd.read_csv(os.path.join(print_folder_path01, file_name), sep=',', names=[i for i in range(0, 9)])
        # 获取对应txt文件中记录的气体名字
        index = str(file_name[4:6])
        batch_no = str(file_name[0:2])
        turn_no = str(file_name[-2:])
        loc = dict_name[index]
        if turn_no == 'T1':
            ls_data.append(df.to_numpy()[0:10, 1:9].tolist())
            # 利用dict_name进行气体名字和0、1、2、3对换
            # 0:5000代表 每个样本只保留前500s,在时间维度上保持一致,1:9代表去除时间这一列
            ls_label.append(loc)
            count_num[loc] += 1

    # ls_data = np.array(ls_data)
    # ls_label = np.array(ls_label)

    X_train, X_test, y_train, y_test = train_test_split(ls_data, ls_label, test_size=0.2, random_state=42)

    return np.array(X_train), np.array(X_test), np.array(y_train),np.array(y_test)
if __name__ == '__main__':
    data_load02()