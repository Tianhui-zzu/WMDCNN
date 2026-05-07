from train import train
from Func import get_trainData, get_testData, draw, get_data
from model import MST_AAMN
from additive_angular_margin_loss import aam_loss
# from torch.utils.tensorboard import SummaryWriter
from transformers import optimization
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from model_save import save
# import netron
import sklearn

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
bs = 10
epochs = 60
print_batch = 36
lr = 0.001
warmup_steps = 1000  # warmup_steps扩大w倍，要保持整个过程学习率  极值不改变，对应learnner中的factor要扩大w**0.5倍

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # get_trainData()
    print(torch.__version__)
    # 模型
    model = MST_AAMN()

    # model.load_state_dict(torch.load('MST_AAMN.pth'))
    X_train, X_test, y_train, y_test = get_data()
    test_loader = get_testData(batch_size=bs, data_x=X_test, data_y=y_test)
    train_loader = get_trainData(batch_size=bs, data_x=X_train, data_y=y_train)
    steps = len(train_loader.dataset) * epochs / bs
    # 生成损失函数
    # loss_func = nn.CrossEntropyLoss()
    loss_func = aam_loss()

    # 生成优化器
    # 总训练step数=epochs*样本数/batch_size
    steps = len(train_loader.dataset) * epochs / bs
    optim = optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    # scheduler = get_customized_schedule_with_warmup(
    #     optim,
    #     num_warmup_steps=warmup_steps,
    #     d_model=112
    # )
    scheduler = optimization.get_constant_schedule(optim, last_epoch=-1)
    # scheduler = optimization.get_polynomial_decay_schedule_with_warmup(
    #     optim,
    #     num_warmup_steps=2000,
    #     num_training_steps=steps,
    #     lr_end=1e-7,
    #     power=3
    # )
    # 获得返回损失和预测准确率
    loss_list_epoch, lr_list = train(model, device, train_loader, test_loader, optim, loss_func, epochs, print_batch,
                                     scheduler)
    draw(pd.DataFrame(loss_list_epoch), pd.DataFrame(lr_list))

    # 模型保存
    path = 'MST_AAMN.pth'
    torch.save(model.state_dict(), path)
