import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from func import  *
from H_CNN import H_CNN





# 超参数
epochs = 70
batch_size = 15
predict_time = 4
lr = 0.0004
print_batch = 5
# os.environ["CUDA_VISIBLE_DEVICES"]='cuda:1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data()
    train_loader = get_trainData(batch_size=batch_size,data_x=X_train,data_y=y_train)
    test_loader = get_testData(batch_size=batch_size,data_x=X_train,data_y=y_train)

    model = H_CNN()
    # model.load_state_dict(torch.load('MST_AAMN_0.89.pth'))
    # 生成损失函数
    loss_func = nn.CrossEntropyLoss()

    # # 生成优化器
    optim = optim.Adam(model.parameters(), lr=lr)
    loss_list_epoch = train(model, device, train_loader, test_loader, optim, loss_func, epochs, print_batch)
    draw(pd.DataFrame(loss_list_epoch))

    # 模型保存
    path = 'ARTICLE_ONE.pth'
    torch.save(model.state_dict(), path)

