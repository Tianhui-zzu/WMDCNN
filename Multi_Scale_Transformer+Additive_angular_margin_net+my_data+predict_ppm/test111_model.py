from additive_angular_margin_loss import *
from classifier import *

batch_size = 4
if __name__ == '__main__':
    feature_vector = torch.normal(0, 1, (batch_size, 8 * 112))
    print('feature_vector', feature_vector)
    labels = torch.LongTensor([0, 2, 1, 3])
    # print('labels.dtype', labels.dtype)
    # labels = F.one_hot(labels).float()
    print('labels', labels)
    # index = torch.where(labels != 0)[0]
    # index_y = torch.where(labels != 0)[1]
    # index = labels[index]
    # index = torch.max(labels, 1)[1]
    # print('index',index)
    # print('index_y',index_y)
    loss_func = aam_loss()
    pfc = PartialFC()
    out_vector = pfc.forward(feature_vector)
    loss = loss_func.forward(out_vector, labels)
    print('loss', loss)
    # print('---------------------------')
    # max_logits, _ = torch.max(out, dim=1, keepdim=True)  # ！！！！！！！！！！！！！！！！！！
    # print('max_logits', max_logits)
