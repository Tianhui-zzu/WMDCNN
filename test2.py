import  numpy as np
import  matplotlib.pyplot as plt
import  torch

def subsequent_mask(size):
    """
    用于遮住序列的一些位置
    修改了解码器中的自注意力子层，以防止位置关注后续位置。
    这种掩蔽与输出嵌入偏移一个位置的事实相结合，确保了位置i的预测只能依赖小于位置i的已知输出
    :param size: (int)向量长度
    :return: (Tensor,bool)掩码后的矩阵，尺寸为[1,size,size]
    """
    attn_shape = (1, size, size)
    # 返回函数的上三角矩阵，从k=1列开始
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


if __name__ == "__main__":
    plt.figure(figsize=(5, 5))
    print(subsequent_mask(20))
    plt.imshow(subsequent_mask(20)[0])
    plt.show()

