import os

import numpy as np
import pandas as pd
import os.path as op


def handle_results(path = 'extract_x_f-b1.csv'):
    df = pd.read_csv(path, sep=',', header=None)
    # df.loc[:, 0] = df[0] - min(df[0]) / (max(df[0]) - min(df[0]))
    # print((df.values -min(df.values))/(max(df.values)-min(df.values)))
    nd1 = (df.values - min(df.values)) / (max(df.values) - min(df.values))
    nd1 = np.around(nd1, decimals=2)
    df = pd.DataFrame(nd1)
    des = 'final_predict_results.csv'
    if op.exists(des):
        print('结果文件已存在,正在删除,保存新的文件')
        os.remove(des)
        df.to_csv(des, index=False, header=None)
        print('输出文件成功')
    else:
        print('结果文件未存在,直接保存新文件')
        df.to_csv(des, index=False, header=None)
        print('输出文件成功')


