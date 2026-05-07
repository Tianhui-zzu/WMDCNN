'用以解决长期漂移问题'

'''
使用Incremental Learning from Stream Data论文的思想，进行长期漂移补偿，数据集Twin gas sensor arrays.
利用batch1训练完成原有模型m1，去预测batch2将batch2的数据按照分类准确赋权重，基本思想是让模型尽可能学习那些发生较大漂移的数据.
关键点在于如何给数据赋权重，因为通常的训练，每个数据的权重都是相同的，可以理解为1.
'''

import torch.utils.data as data


def getData_byWeight(samples_x,samples_y,weights,num_samples=80):
    '''

    :param samples_x: 样本集
    :param samples_y: 样本标签集
    :param weights:   样本权重集
    :return:
    '''
    samples_indexs = data.WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=False)
    indexs = [index for index in samples_indexs ]
    # print(indexs)
    # print('--------------输出对应样本的权重--------------')
    # print([weights[index] for index in indexs])
    extract_samples_x = [samples_x[index] for index in indexs]
    extract_samples_y = [samples_y[index] for index in indexs]
    return extract_samples_x,extract_samples_y,indexs
