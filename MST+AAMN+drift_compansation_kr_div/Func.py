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

print_folder_path01 = '../handled_data1_1hz_expand_t1+t2_x-min'


def get_list_data(turn_no='T1',batch_no='B1'):
    ls_data = []
    ls_label = []
    dict_name = {'CO': 0, 'Ey': 1, 'Ea': 2, 'Me': 3}
    CO, Ey, Ea, Me = 0, 0, 0, 0
    for file_name in os.listdir(print_folder_path01):
        df = pd.read_csv(os.path.join(print_folder_path01, file_name), sep=',', names=[i for i in range(0, 9)])
        # 获取对应txt文件中记录的气体名字
        index = str(file_name[4:6])
        b_n = str(file_name[0:2])
        t_n = str(file_name[-2:])
        # print(turn_no)
        loc = dict_name[index]
        # print(loc)
        # 保证五个类别的样本数量均是36
        # if count_num[loc] < 36:
        # if turn_no == 'T2' and (index == 'Me' or index == 'Ea'):
        #     # if # and (index == 'Ey' or index == 'CO')
        #     ls_data.append(df.to_numpy()[0:512, 1:9].tolist())
        #     # 利用dict_name进行气体名字和0、1、2、3对换
        #     # 0:5000代表 每个样本只保留前500s,在时间维度上保持一致,1:9代表去除时间这一列
        #     ls_label.append(loc)
        #     count_num[loc] += 1
        if t_n == turn_no and b_n == batch_no:
            ls_data.append(df.to_numpy()[0:256, 1:9].tolist())
            # 利用dict_name进行气体名字和0、1、2、3对换
            # 0:5000代表 每个样本只保留前500s,在时间维度上保持一致,1:9代表去除时间这一列
            ls_label.append(loc)
    return  ls_data,ls_label

def get_data(S_batch_no = 'B1',T_batch_no = 'B2'):
    # 获取源域(带标签数据)数据
    ls_data,ls_label = get_list_data(batch_no=S_batch_no)
    X_Source = np.array(ls_data)
    Y_Source = np.array(ls_label)
    # 获取目标域(无标签数据)数据
    if T_batch_no == 'B4' or T_batch_no == 'B5': # 'B4','B5'均为80,共同组成目标域160个样本和'B1' ,'B2','B3'均为160个样本匹配
        ls_data01, ls_label01 = get_list_data(batch_no='B4')
        ls_data02, ls_label02 = get_list_data(batch_no='B5')
        X_Target = np.array(ls_data01+ls_data02)
        Y_Target = np.array(ls_label01+ls_label02)  # 这个只是验证准确率时候用到,在训练的过程中该数据源就是只用到X而不需要标签信息
    else:
        ls_data,ls_label  = get_list_data(batch_no=T_batch_no)
        X_Target = np.array(ls_data)
        Y_Target = np.array(ls_label) # 这个只是验证准确率时候用到,在训练的过程中该数据源就是只用到X而不需要标签信息


    # X_train, X_test, y_train, y_test = train_test_split(ls_data, ls_label, test_size=0.2, random_state=42)

    return X_Source,Y_Source,X_Target,Y_Target


def get_trainData(batch_size=20, data_x=None, data_y=None):
    # data_x= np.delete(data_x, 0, axis=1)
    # train_x = torch.from_numpy(data_x.astype(np.float32))
    # train_y = torch.from_numpy(data_y.astype(np.int64))
    train_x = torch.as_tensor(data_x, dtype=torch.float32)
    train_y = torch.as_tensor(data_y, dtype=int)
    train_data = Data.TensorDataset(train_x, train_y)
    train_dataloader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    # for no, (x, y) in enumerate(train_dataloader):
    #     print('no', no)
    #     print('x', x)
    #     print('y', y)
    #     break
    # for x1 ,x2 in zip(enumerate(train_dataloader),enumerate(train_dataloader)):
    #     batch_no = x1[0]
    #     (x_s,y_s) = x1[1]
    #     (x_t,y_t) = x2[1]
    #     print('batch_no',batch_no)
    #     print('(x_s,y_s)',x_s,y_s)
    #     print('(x_t,y_t)',x_t,y_t)
    return train_dataloader


def get_testData(batch_size=20, data_x=None, data_y=None):
    test_x = torch.as_tensor(data_x, dtype=torch.float32)
    test_y = torch.as_tensor(data_y, dtype=int)
    test_data = Data.TensorDataset(test_x, test_y)
    test_dataloader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    return test_dataloader


def normalize(x):
    # print('归一化之前:',x)
    x_mean = torch.mean(x, dim=1, keepdim=True)
    # print('x_mean',x_mean)
    x_std = torch.std(x, dim=1, keepdim=True)
    # print('x_std',x_std)
    # # 数据标准化
    x1 = (x - x_mean) / x_std
    return x1


def accuracy(output, labels):
    predictions = torch.max(output, 1)[1]
    labels = labels.data.view_as(predictions)
    right_num = predictions.eq(labels).sum()
    return right_num / len(labels)


def predict(model, device, test_loader):
    all_accuracy = 0
    batch_num = 0
    for batch_size, (test_x, test_y) in enumerate(test_loader):
        # 输入第一层1d卷积要将数据重塑为(batch_size,channel,size1)
        test_x = test_x.view(test_x.shape[0], test_x.shape[2], test_x.shape[1])
        test_x = test_x.to(device)
        labels = test_y.to(device)
        output = model.forward(test_x)
        # 直接接收预测准确率
        all_accuracy = all_accuracy + accuracy(output, labels)
        batch_num = batch_num + 1
    return round((all_accuracy.detach().cpu().numpy() / batch_num), 4)


def draw(df1, df2):
    x1 = np.arange(0, df1.shape[0])
    x2 = np.arange(0, df2.shape[0])
    plt.plot(x1, df1['loss'], linestyle='-', label='value of loss')
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
    plt.plot(x1, df1['train acr'], linestyle='-', label='value of acr', color='red')
    plt.show()
    plt.xlabel('batch num')
    plt.ylabel('value of acr')
    plt.title('value of acr on tested data')
    plt.plot(x1, df1['test acr'], linestyle='-', label='value of acr', color='red')
    plt.show()
    plt.xlabel('step num')
    plt.ylabel('value of learning_rate')
    plt.title('value of learning_rate')
    plt.plot(x2, df2['learning_rate'], linestyle='-', label='value of acr', color='red')
    plt.show()


def adjust_learning_rate(optimizer, epoch, init_lr=0.0001):
    """
    optimizer: 优化器
    epoch: 训练轮数，也可以根据需要加入其它参数
    init_lr：初始学习率，也可以设置为全局变量
    """
    lr = init_lr
    if init_lr > 0.0001:
        lr = init_lr * (0.1 ** (epoch // 150))  # 每35轮下降一次学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    # X_train, X_test, y_train, y_test = get_data()
    X_Source,Y_Source,X_Target,Y_Target = get_data(T_batch_no='B4',S_batch_no='B1')
    label_Source = np.array(Y_Source)
    data_Source = np.array(X_Source)
    label_Target = np.array(Y_Target)
    data_Target = np.array(X_Target)
    print('data_Source.shape',data_Source.shape)
    print('data_Target.shape',data_Target.shape)
    source_dataloader= get_trainData(data_x=data_Source,data_y=label_Source)
    target_dataloader= get_trainData(data_x=data_Target,data_y=label_Target)