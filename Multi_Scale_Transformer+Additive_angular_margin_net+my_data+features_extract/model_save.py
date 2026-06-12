import torch


def save(model, des_path):
    # 1只保存模型字典参数
    torch.save(model.state_dict(), des_path)
