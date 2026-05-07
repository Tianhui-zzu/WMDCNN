import torch
import torch.nn as nn
from feature_extractor import Feature_Extractor
from classifier import PartialFC

feature_size = 8 * 112
class_num = 4
vector_dim = 112
C1_S = 64
C2_S = 32
C3_S = 16


# 多尺度transformer提取器
class MST_AAMN(nn.Module):
    def __init__(self, memory_t=None, memory_f=None):
        super(MST_AAMN, self).__init__()
        self.F_E_C1 = Feature_Extractor(256, C1_S, sensor_num=8, dropout=0.1, head=4)
        self.F_E_C2 = Feature_Extractor(256, C2_S, sensor_num=8, dropout=0.1, head=4)
        self.F_E_C3 = Feature_Extractor(256, C3_S, sensor_num=8, dropout=0.1, head=4)
        # 用来存储经由transformer提取完毕特征后的数据，方便经由pca降维展现提取特征的效果。
        self.Classifier = PartialFC(
            feature_size=feature_size,vector_dim = vector_dim, class_num = class_num, memory_f = memory_f)
        self.memory_transformer = memory_t

    def forward(self, x):
        x1 = self.F_E_C1(x)
        x2 = self.F_E_C2(x)
        x3 = self.F_E_C3(x)
        # 联合三个通道
        x = torch.cat((x1, x2, x3), 2)

        if self.memory_transformer != None:
            temp = x.view(x.size(0), -1).detach().cpu().numpy()
            # print('x.type',type(x))
            # print('temp-------',temp)
            self.memory_transformer.append(temp)
        x = self.Classifier(x)
        return x


# test
if __name__ == '__main__':
    t1 = torch.ones(1, 8, 256)
    # t1 = torch.LongTensor(20, 1, 8, 5000)
    model = MST_AAMN()
    final_out = model(t1)
    print('final_out :', final_out)
    print('final_out.shape:', final_out.shape)
