from tqdm import tqdm
from Func import adjust_learning_rate
import time
import torch

final_train_acr = 0.94


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


def train(model, device, train_loader, test_loader, optimizer, loss_func, epochs, print_batch, scheduler):
    loss_list_epoch = {'index': [], 'loss': [], 'test acr': [], 'train acr': []}
    lr_list = {'learning_rate': []}
    model = model.to(device)
    f_t_a = 0  # 记录达到的最高训练准确率
    for epoch in range(epochs):
        # adjust_learning_rate(optimizer, epoch, init_lr=learning_rate)
        if f_t_a > final_train_acr:
            break
        with tqdm(train_loader, total=len(train_loader)) as t:  # 🌟 1. 定义进度条
            for batch_index, (train_x, train_y) in enumerate(t):
                # train_loader.set_description(f"Epoch {epoch}")  # 🌟 2. 设置开头
                train_x = normalize(train_x)  # 数据归一化
                train_x = train_x.view(train_x.shape[0], train_x.shape[2], train_x.shape[1])
                # print('train_x.shape:',train_x.shape)
                # print('train_x:', train_x)
                # print('train_y.shape:', train_y.shape)
                # print('train_y:', train_y)
                data, label = train_x.to(device), train_y.to(device)
                model.train()
                output = model(data)
                loss = loss_func(output, label)
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
                    test_acr = predict(model, device, test_loader)
                    train_acr = predict(model, device, train_loader)
                    loss_list_epoch['test acr'].append(test_acr)
                    loss_list_epoch['train acr'].append(train_acr)
                # 可视化学习率,将学习率记录

                for param_group in optimizer.param_groups:
                    lr_list['learning_rate'].append(param_group['lr'])
                t.set_description(f'Epoch [{epoch + 1}/{epochs}]')
                t.set_postfix(loss=loss_print, train_acr=train_acr, test_acr=test_acr)  # 🌟 3. 设置结尾
                time.sleep(0.00001)

                f_t_a = train_acr
                if f_t_a > final_train_acr:
                    break
    return loss_list_epoch, lr_list  # 返回两个字典，记录训练过程中的所有的数据
