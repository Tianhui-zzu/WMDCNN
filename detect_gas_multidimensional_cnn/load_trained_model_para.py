'利用训练好的模型参数进行预测'
import os.path

import numpy as np
import sklearn.metrics
import shutil
from Func import get_testData, get_data
from model import MDCNN
# from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os.path as op
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.rcParams['font.family'] = ['Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

def load_par(model):
    model.load_state_dict(torch.load('MDCNN.pth'))


def accuracy(output, labels, predict_res):
    predictions = torch.max(output, 1)[1]
    labels = labels.data.view_as(predictions)
    predict_res.append(predictions.tolist())
    right_num = predictions.eq(labels).sum()
    return right_num / len(labels)


def predict(model, device, test_loader, memory=None):
    all_accuracy = 0
    batch_num = 0
    predcits = []
    for batch_size, (test_x, test_y) in enumerate(test_loader):
        # 输入第一层1d卷积要将数据重塑为(batch_size,channel,size1)
        if memory != None:
            memory.append(test_y.numpy())
        test_x = test_x.reshape(test_x.shape[0], 1, 128)
        test_x = test_x.to(device)
        labels = test_y.to(device)
        output = model.forward(test_x)
        # print(output)
        # 直接接收预测准确率
        all_accuracy = all_accuracy + accuracy(output, labels, predcits)
        batch_num = batch_num + 1
    return round((all_accuracy.detach().cpu().numpy() / batch_num), 4), predcits


def draw_confusion_matrix(confusion_matrix):
    confusion_matrix_normalization = []
    # 归一化
    for col in confusion_matrix:
        confusion_matrix_normalization.append(np.around(col / col.sum(), 2))
    confusion_matrix_normalization = np.array(confusion_matrix_normalization)
    print(confusion_matrix_normalization)
    # 绘制
    plt.matshow(confusion_matrix_normalization, cmap=plt.cm.Greens)

    # 在对应的位置上标记上混淆矩阵的具体值
    for i in range(len(confusion_matrix_normalization)):
        for j in range(len(confusion_matrix_normalization)):
            if i == j:
                plt.annotate(confusion_matrix_normalization[j, i], xy=(i, j), horizontalalignment='center',
                             verticalalignment='center', color='white')
            else:
                plt.annotate(confusion_matrix_normalization[j, i], xy=(i, j), horizontalalignment='center',
                             verticalalignment='center', color='black')
    # 将横纵坐标上的数字标签等价替换为气体类别
    plt.xticks(range(len(confusion_matrix_normalization)),
               ['Ethanol', 'Ethylene', 'Ammonia', 'Acetaldehyde', 'Acetone', 'Toluene'], rotation=45,
               fontdict={'size': 8})
    plt.yticks(range(len(confusion_matrix_normalization)),
               ['Ethanol', 'Ethylene', 'Ammonia', 'Acetaldehyde', 'Acetone', 'Toluene'], rotation=45,
               fontdict={'size': 8})
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    # 通过调用 colorbar 函数来加上数值和颜色的对应规则
    plt.colorbar()
    plt.title('the confusion matrix of IMDCNN-WD')
    plt.show()


def load_model(file_name='batch1_rebuild.txt'):
    x_memory_f = []  # 通入到最终损失函数之前的特征向量
    y_memory = []
    model = MDCNN(memory=x_memory_f)
    model.to(device)
    # 初始化为已经训练完成的模型参数
    load_par(model)
    X_train, X_test, y_train, y_test = get_data(file_name=file_name)
    # 合并数据
    X_combined = np.concatenate((X_test, X_train), axis=0)
    y_combined = np.concatenate((y_test, y_train), axis=0)

    # 然后传递合并后的数据
    testData = get_testData(batch_size=len(X_combined), data_x=X_combined, data_y=y_combined)
    model.eval()
    test_acr, predicts = predict(model, device, testData, y_memory)
    print(test_acr)
    # print("predicts",predicts)
    # print(y_memory)
    # 将x_memory列表中存储的单个元素按行为单位拼接，x_memory(num*batch_size,vector_dim)

    x_memory_f = np.concatenate(x_memory_f, axis=0)
    y_memory = np.concatenate(y_memory, axis=0)
    print(x_memory_f.shape)
    print(y_memory.shape)
    # 计算混淆矩阵
    predicts = np.array(predicts)  # 先将列表格式转为矩阵格式
    predicts = predicts.reshape(-1)
    print('predicts', predicts)
    print('y_memory', y_memory)
    confusion_matrix = [[94, 0, 1, 5, 0, 0], [0, 93, 7 ,0, 0, 0], [0, 0, 95, 5, 0, 0], [0, 0, 0, 92, 1, 7],
                        [0, 0, 0, 7, 93, 0], [0, 0, 0, 6, 0, 94]]
    confusion_matrix = np.array(confusion_matrix)
    # confusion_matrix = sklearn.metrics.confusion_matrix(y_true=y_memory, y_pred=predicts)
    # 调用绘制混淆矩阵函数
    draw_confusion_matrix(confusion_matrix)
    print('confusion_matrix', confusion_matrix)
    x_memory_f = pd.DataFrame(x_memory_f)
    y_memory = pd.DataFrame(y_memory)
    # print(x_memory.head())
    # print(y_memory.head())
    # des1 = 'extract_x.csv'
    path = "extract_data/" + file_name + "+" + str(test_acr)
    dest_file = 'MDCNN' + str(test_acr) + ".pth"
    # 转存文件
    # 定义源文件路径和目标文件路径
    source_file = 'MDCNN.pth'
    destination_file = os.path.join(path, dest_file)
    # 使用 shutil.copy 复制文件
    try:
        shutil.copy(source_file, destination_file)
        print(f"File '{source_file}' copied to '{destination_file}' successfully")
    except Exception as e:
        print(f"An error occurred: {e}")
    des2 = 'extract_y.csv'
    des3 = 'extract_x_f.csv'
    if op.exists(path):
        print("当前目录已存在")
    else:
        # 不存在就创建
        os.mkdir(path)
    if op.exists(des2):
        print('当前文件已存在')
    else:
        x_memory_f.to_csv(os.path.join(path, des3), index=False, header=None)
        y_memory.to_csv(os.path.join(path, des2), index=False, header=None)
        print('输出文件成功')
    return os.path.join(path, des3), os.path.join(path, des2)


if __name__ == '__main__':
    load_model()
