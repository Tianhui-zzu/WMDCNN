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
lr = 0.0001
warmup_steps = 1000  # warmup_steps扩大w倍，要保持 整个过程学习率  极值不改变，对应learnner中的factor要扩大w**0.5倍

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # get_trainData()
    print(torch.__version__)
    # 模型
    model = MST_AAMN()
    model_best = MST_AAMN()
    # model.load_state_dict(torch.load('MST_AAMN_before.pth'))
    # model_best.load_state_dict(torch.load('MST_AAMN_best.pth'))
    # model.load_state_dict(torch.load('MST_AAMN.pth'))
    X_Source, Y_Source, X_Target, Y_Target = get_data(S_batch_no='B1',T_batch_no='B4')
    source_loader = get_trainData(batch_size=bs, data_x=X_Source, data_y= Y_Source)
    target_loader = get_trainData(batch_size=bs, data_x=X_Target ,data_y=Y_Target )
    steps = len(source_loader.dataset) * epochs / bs
    # 生成损失函数
    # loss_func = nn.CrossEntropyLoss()
    # loss_func = nn.KLDivLoss()

    # 生成优化器
    # 总训练step数=epochs*样本数/batch_size
    loss_func01 = nn.KLDivLoss(reduction='mean', log_target=False)
    # loss_func02 = nn.CrossEntropyLoss()
    loss_func02 = aam_loss()

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
    loss_list_epoch, lr_list = train(model, model_best,device, source_loader, target_loader, optim, epochs, print_batch,
                                     scheduler,loss_func01,loss_func02)
    draw(pd.DataFrame(loss_list_epoch), pd.DataFrame(lr_list))

    # 模型保存
    path = 'MST_AAMN.pth'
    torch.save(model.state_dict(), path)
