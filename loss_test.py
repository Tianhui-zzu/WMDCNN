import math

import torch
import torch.nn.functional as F
import torch.nn as nn

x1 = torch.rand(5)
x1 = x1.view(x1.shape[0], -1)
print('x1', x1, 'x1.shape', x1.shape)
x2 = -(x1 + 1)
x3 = -(x1 + 1) * torch.log(1 - 1 / (x1 + 1))
x4 = x3.view(-1, x3.shape[0])
x5 = x1.matmul(x4)
print('x2', x2)
print('x3', x3)
print('x4', x4, 'x4.shape', x4.shape)
print('x5',x5)
# target = torch.tensor([0, 2, 3, 1,4])  # 标签 这里还有一个torch.tensor与torch.Tensor的知识点https://blog.csdn.net/weixin_40607008/article/details/107348254
# print('target',target)
# one_hot = F.one_hot(target).float()  # 对标签进行one_hot编码
# print('one_hot',one_hot)
# softmax = torch.exp(x) / torch.sum(torch.exp(x), dim=1,keepdim=True)
# print('softmax',softmax)
# logsoftmax = torch.log(softmax)
# print('logsoftmax',logsoftmax)
# print('one_hot * logsoftmax',one_hot * logsoftmax)
# nllloss = -torch.sum(one_hot * logsoftmax) / target.shape[0]
# print('nllloss',nllloss)

# Mat1 = torch.tensor([[1, 6, 7],
#                       [2, 5, 8],
#                       [3, 4, 9]])
# Mat2 = torch.tensor([[9, 6, 3],
#                       [8, 5, 2],
#                       [7, 4, 1]])
#
# Mat3 = torch.mm(Mat1, Mat2, out=None)


# print('torch.mm(Mat1, Mat2, out=None)',Mat3)
# print('Mat1*Mat2',Mat1*Mat2)
