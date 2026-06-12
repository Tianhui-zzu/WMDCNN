'测试计算模型预测准确率的功能'
import  torch
import numpy as np

output = torch.tensor([[-3.4960, -7.9463, -4.9212,  3.3531,  6.7408, -1.2303],[-3.4960, -7.9463, -4.9212,  3.3531,  6.7408, -1.2303]])
print(output)
predictions = torch.max(output, 1)[1]
print(predictions)