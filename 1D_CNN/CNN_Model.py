import torch
import torch.nn as nn
import torch.nn.functional as F


# 卷积num*8*256征向量
class cnn_model(nn.Module):
    def __init__(self, sensor_num=8, class_num=6, vector_dim=8, memory_f=None):
        super(cnn_model, self).__init__()

        # self.channel_attention = ChannelAttention(in_planes=8)
        # 自己采集的传感器一共9个,这里只取前8个传感器
        # B_S*1*64
        self.conv1 = nn.Sequential(nn.Conv1d(1, 8, 3, 1, 1), nn.MaxPool1d(2))
        # B_S*8*32
        self.conv2 = nn.Sequential(nn.Conv1d(8, 8, 3, 1, 1), nn.MaxPool1d(2))
        # B_S*8*16
        self.conv3 = nn.Sequential(nn.Conv1d(8, 8, 3, 1, 1), nn.MaxPool1d(2))
        # B_S * 8 * 8
        # self.GAP = nn.AdaptiveAvgPool1d(1)
        self.linear1 = nn.Linear(8*sensor_num * vector_dim // (8), 8*sensor_num * vector_dim // (16))
        # 8 * 4
        self.linear2 = nn.Linear(8*sensor_num * vector_dim // (16), 8*sensor_num * vector_dim // (32))
        # 8 * 2 --------------> class_num,softmax()
        self.out = nn.Linear(8*sensor_num * vector_dim // (32), class_num)
        self.memory = memory_f

    def forward(self, x):
        # 卷积层
        #  增加1维通道数 torch.Size([15, 1, 8, 256]) ->(batch_size,channels,sensor_nums,times)
        # x = torch.unsqueeze(x, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # x = x.view(x.size(0), -1, x.size(3))
        # --------------------------
        # [num, 8, 8]
        # [num, 8, 1]
        # x = x.view(x.size(0), x.size(1),-1)
        # x = self.GAP(x)
        # x(8,32) ---------->(1,256)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        # 全连接层
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.out(x)
        if self.memory != None:
            temp = x.view(x.size(0), -1).detach().cpu().numpy()
            # print('x.type',type(x))
            # print('temp-------',temp)
            self.memory.append(temp)
        # -----------------------
        return x


# test
if __name__ == '__main__':
    t1 = torch.ones(15, 1, 64)
    # t1 = torch.LongTensor(20, 1, 8, 5000)
    model = cnn_model()
    final_out = model(t1)
    print('final_out :', final_out)
    print('final_out.shape:', final_out.shape)
