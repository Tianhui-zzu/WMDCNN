import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class WeightedDataset(Dataset):
    def __init__(self, weights=None):
        super(Dataset, self).__init__()
        # 样本权重,按照上述权重返回对应的样本,权重越大抽取概率越大
        # weights设置为列表
        if weights is not  None:
            self.weights = weights

    def __len__(self):
        return None

    def __getitem__(self, index):
        assert self.a <= index <= self.b
        return index, index ** 2


