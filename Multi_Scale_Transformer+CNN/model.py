import torch
import torch.nn as nn
from classifier import Classifier, Classifier_conv_new
from feature_extractor import Feature_Extractor


class HATN_DA(nn.Module):
    def __init__(self, memory_t=None, memory_f=None):
        super(HATN_DA, self).__init__()
        self.F_E_C1 = Feature_Extractor(256, 64, sensor_num=8, dropout=0.1, head=4)
        self.F_E_C2 = Feature_Extractor(256, 32, sensor_num=8, dropout=0.1, head=4)
        self.F_E_C3 = Feature_Extractor(256, 16, sensor_num=8, dropout=0.1, head=4)
        self.C = Classifier(sensor_num=8, class_num=4, vector_dim=112,memory_f=memory_f)
        self.C_conv_new = Classifier_conv_new(sensor_num=8, class_num=4, vector_dim=112,memory_f=memory_f)
        # 用来存储经由transformer提取完毕特征后的数据，方便经由pca降维展现提取特征的效果。
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
        # print('x.shape', x.shape)
        # x = self.C(x)
        # x = self.C_conv(x)
        x = self.C_conv_new(x)
        return x


# test
if __name__ == '__main__':
    t1 = torch.ones(1, 8, 512)
    # t1 = torch.LongTensor(20, 1, 8, 5000)
    model = HATN_DA()
    final_out = model(t1)
    print('final_out :', final_out)
    print('final_out.shape:', final_out.shape)
