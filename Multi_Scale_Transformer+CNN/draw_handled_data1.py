'绘制handled_data1的数据'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as op

path = '../handled_data1'
file_name = 'B1_GCO_F010_R1.txt'


def draw():
    df = pd.read_csv(op.join(path, file_name), sep='', names=[i for i in range(0, 9)])
    for i in range(1, 9):
        plt.plot(df[0], df[i], linestyle='-',label='sensor'+str(i))
    plt.legend(loc='upper right')
    plt.xlabel('time/s')
    plt.ylabel('response/kΩ')
    plt.show()

draw()