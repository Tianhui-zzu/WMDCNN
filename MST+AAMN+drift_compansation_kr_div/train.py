import pandas as pd
from tqdm import tqdm
from Func import adjust_learning_rate
import time
import torch
import torch.nn as nn
final_acr = 0.92
target_path = 'target_data/extract_x_f-b1.csv'




def normalize(x):
    # print('归一化之前:',x)
    x_mean = torch.mean(x, dim=1, keepdim=True)
    # print('x_mean',x_mean)
    x_std = torch.std(x, dim=1, keepdim=True)
    # print('x_std',x_std)
    # # 数据标准化
    x1 = (x - x_mean) / x_std
    # print('归一化之后:', x1)
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

# 读取漂移前的概率分布target,也是kl散度中target概率分布,
# 拟合的目标就是通过修改现有模型(以漂移前数据进行训练得出的模型)使得漂移后的概率分布output尽可能贴近target
def get_target(path):
    target = pd.read_csv(path,sep=',').to_numpy()
    return  target

def train(model,model_best, device, target_loader, source_loader, optimizer, epochs, print_batch, scheduler,loss_func01,loss_func02):


    loss_list_epoch = {'index': [], 'loss': [], 'test acr': [], 'train acr': []}
    lr_list = {'learning_rate': []}
    model = model.to(device)
    model_best = model_best.to(device)
    f_t_a = 0  # 记录达到的最高训练准确率
    f_s_a = 0  # 记录达到的最高训练准确率
    for epoch in range(epochs):
        # adjust_learning_rate(optimizer, epoch, init_lr=learning_rate)
        if f_s_a > final_acr and f_t_a >final_acr:
            break
        with tqdm(target_loader, total=len(target_loader)) as t:  # 🌟 1. 定义进度条
         for source ,target in zip(enumerate(source_loader),enumerate(t)):
                batch_index = source[0]
                (source_x,source_y) = source[1]
                (target_x,target_y) = target[1]
                # train_loader.set_description(f"Epoch {epoch}")  # 🌟 2. 设置开头
                source_x = normalize(source_x)  # 数据归一化
                source_x = source_x.view(source_x.shape[0], source_x.shape[2], source_x.shape[1])
                target_x = normalize(target_x)  # 数据归一化
                target_x = source_x.view(target_x.shape[0], target_x.shape[2], target_x.shape[1])
                # print('train_x.shape:',train_x.shape)
                # print('train_x:', train_x)
                # print('train_y.shape:', train_y.shape)
                # print('train_y:', train_y)
                data_source, label_source = source_x .to(device), source_y .to(device)
                data_target, label_target = target_x.to(device), target_y.to(device)
                model.train()
                # 目标域和源域数据全部用以训练
                # 不同的是源域数据视为带标签数据，可以用aam损失函数进行分类损失计算，
                # 目标域视为无标签数据，只能用以和源域数据训练输出结果进行kl散度计算，用以提取共同特征
                output_source = model(data_source)
                output_target = model(data_target)
                # target = model_best(data)
                output_source = output_source.softmax(-1)
                output_target = output_target.softmax(-1)
                # print('output',output_source)
                # print('target',output_target)
                # kl散度损失函数用以漂移补偿
                loss_kr = loss_func01(output_target.log(), output_source)
                # aam损失用以分类
                loss_aam = loss_func02(output_source,label_source)
                # print('loss_kr',loss_kr)
                # print('loss_aam',loss_aam)
                loss = 10*loss_kr+loss_aam
                loss_kr_print = round(loss_kr.detach().cpu().numpy() / 1, 4)
                loss_aam_print = round(loss_aam.detach().cpu().numpy() / 1, 4)
                loss_print = round(loss.detach().cpu().numpy() / 1, 4)
                # 反向传播 更新参数 梯度归零
                loss.backward()
                optimizer.step()
                # 更新学习率
                scheduler.step()
                optimizer.zero_grad()
                model.eval()
                if batch_index % print_batch == 0:
                    # 记录各种数据到列表中，方便图像展示
                    loss_list_epoch['index'].append(str(epoch) + '_' + str(batch_index / print_batch))
                    loss_list_epoch['loss'].append(loss_print)
                    test_acr = predict(model, device, target_loader) # 测试集测试的就是目标域的预测准确率
                    # print(test_acr)
                    train_acr = predict(model, device, source_loader)# 训练集测试的就是源域的预测准确率
                    # print(train_acr)
                    loss_list_epoch['test acr'].append(test_acr)
                    loss_list_epoch['train acr'].append(train_acr)
                # 可视化学习率,将学习率记录

                for param_group in optimizer.param_groups:
                    lr_list['learning_rate'].append(param_group['lr'])
                t.set_description(f'Epoch [{epoch + 1}/{epochs}]')
                t.set_postfix(loss_all=loss_print,loss_kr= loss_kr_print,loss_aam =loss_aam_print, train_acr=train_acr, test_acr=test_acr)  # 🌟 3. 设置结尾
                time.sleep(0.00001)

                f_s_a = train_acr # 源域分类准确类
                f_t_a = test_acr # 目标域分类准确类
                if f_s_a > final_acr and f_t_a > final_acr:
                    break
    return loss_list_epoch, lr_list  # 返回两个字典，记录训练过程中的所有的数据
