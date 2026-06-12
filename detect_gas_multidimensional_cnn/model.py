import torch.nn as nn

class MDCNN(nn.Module):
    def __init__(self,memory=None):
        super(MDCNN, self).__init__()
        self.memory = memory
        self.conv1 = nn.Sequential(

            nn.Conv1d(
                in_channels=1,  #
                out_channels=64,  # 输出的通道数,等价于卷积核的个数.
                kernel_size=3,  # 卷积核的尺寸为1*3(1d)
                stride=1,  # 步长为1
                padding=1,  # 填充行数为1,由公式 padding=(kernel_size-1)/2(if padding=1),可保证经过卷积后图像特征图尺寸与原来保持一致,即仍为28*28
            ),
            nn.MaxPool1d(kernel_size=2),  # 池化操作,池化区域为(2*2),每2*2个区域内的数据取最大值代表,经过池化后的结果为:(16,14,14)
            # nn.Dropout(p=0.25)  # 25%概率将值置为0
        )
        self.conv2 = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,  # 卷积核的尺寸为3*3(2d)
                stride=1,
                padding=1,
            ),
            nn.MaxPool2d(kernel_size=2),  # 3d卷积
            # nn.Dropout(p=0.25)
        )
        self.conv3 = nn.Sequential(

            nn.Conv3d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,  # 卷积核的尺寸为3*3*3(3d)
                stride=1,
                padding=1,
            ),
            nn.MaxPool3d(kernel_size=2),  # 3d卷积
            nn.Conv3d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,  # 卷积核的尺寸为3*3*3(3d)
                stride=1,
                padding=1,
            ),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,  # 卷积核的尺寸为3*3*3(3d)
                stride=1,
                padding=1,
            ),
            nn.MaxPool3d(kernel_size=2),
            # nn.Dropout(p=0.25)
        )
        self.dense = nn.Linear(16 * 4 * 4 * 4, 10 * 6)
        self.out = nn.Linear(10 * 6, 6)

    def forward(self, x):
        x = self.conv1(x)
        # print(x)
        # print('1d卷积后x.shape', x.shape)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        # print('x.shape',x)
        x = self.conv2(x)
        # print('2d卷积后x.shape', x.shape)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3])
        x = self.conv3(x)
        # print('3d卷积后x.shape', x.shape)
        x = x.view(x.size(0), -1)  # view方法类似于reshape方法,将tensor维度重新规划两维,(batch_size,32*7*7)
        x = self.dense(x)
        # print('dense()后x.shape', x.shape)
        x = self.out(x)
        if self.memory != None:
            temp = x.view(x.size(0), -1).detach().cpu().numpy()
            # print('x.type',type(x))
            # print('temp-------',temp)
            self.memory.append(temp)
        # print('全连接层后x.shape', x.shape)
        # print('全连接层后x', x.shape)
        return x
