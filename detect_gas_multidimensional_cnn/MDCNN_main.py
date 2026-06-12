import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
from Train import train
from Func import get_trainData, get_testData, get_data
from model import MDCNN
from vision_data import vision_data
from load_trained_model_para import load_model
import torch.onnx

'用自搭建卷积神经网络检测气体类别'
epochs = 15
G = 6  # 要预测的气体的类别个数
batch_size = 20
predict_time = 4
lr = 0.001
# os.environ["CUDA_VISIBLE_DEVICES"]='cuda:1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'用batch2_rebuild作为验证集,batch3_rebuild作为训练集'
print_batch = 10


def draw(df):
    x = np.arange(0, df.shape[0])
    plt.plot(x, df['loss'], linestyle='-', label='value of loss')
    # plt.plot(x,df['test acr'],linestyle='-',label='value of loss')
    # plt.plot(x,df['train acr'],linestyle='-',label='value of loss')
    plt.legend(loc='upper right')
    plt.xlabel('batch num')
    plt.ylabel('value of loss')
    plt.title('value of loss')
    plt.show()
    plt.xlabel('batch num')
    plt.ylabel('value of acr')
    plt.title('value of acr on trained data')
    plt.plot(x, df['train acr'], linestyle='-', label='value of acr', color='red')
    plt.show()
    plt.xlabel('batch num')
    plt.ylabel('value of acr')
    plt.title('value of acr on tested data')
    plt.plot(x, df['test acr'], linestyle='-', label='value of acr', color='red')
    plt.show()


if __name__ == '__main__':
    # get_trainData()
    # 确定file_name
    file_name = 'batch1_rebuild.txt'
    print(torch.__version__)
    model = MDCNN()
    # model.load_state_dict(torch.load('MDCNN.pth'))
    # 生成损失函数
    loss_func = nn.CrossEntropyLoss()
    # 生成优化器
    optim = optim.Adam(model.parameters(), lr=lr)
    X_train, X_test, y_train, y_test = get_data(file_name=file_name)
    test_loader = get_testData(batch_size=batch_size, data_x=X_test, data_y=y_test)
    train_loader = get_trainData(batch_size=batch_size, data_x=X_train, data_y=y_train)
    # train_model(optim, loss_func, model, train_loader, test_loader)
    loss_list_epoch = train(model, device, train_loader, test_loader, optim, loss_func, epochs, print_batch)
    draw(pd.DataFrame(loss_list_epoch))
    # train_x = train_x.re11hape(train_x.shape[0], 1, 128)
    # x = torch.randn(batch_size, 1, 128)
    # model_path = 'demo.onnx'
    # torch.onnx.export(model, x, model_path)

    path = 'MDCNN.pth'
    torch.save(model.state_dict(), path)
    # netron.start(model_onnx)  # 输出网络结构
    # 返回生成的文件目录
    extract_x, extract_y = load_model(file_name=file_name)
    print(extract_x,extract_y)
    vision_data(extract_x, extract_y)
