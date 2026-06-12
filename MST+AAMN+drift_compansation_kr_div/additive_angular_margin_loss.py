import torch
import torch.nn.functional as F
import math
from torch.nn.functional import linear, normalize

mean = 0
std = 0.01


# 融合了additive_angular_margin_loss的分类器
class aam_loss(torch.nn.Module):
    def __init__(self):
        super(aam_loss, self).__init__()
        self.margin_softmax = Margin_Softmax()
        self.dict_cross_entropy = DistCrossEntropy()

    def forward(self, out_vectors, label_vectors):
        # one——hot编码
        labels = F.one_hot(label_vectors).float()
        # print('labels', labels)
        logits = out_vectors.clamp(-1, 1)  # 将特征向量和对应权重矩阵相乘结果，投影限制在(-1,1)之间
        logits = self.margin_softmax(logits, labels)
        loss = self.dict_cross_entropy(logits, labels)
        return loss

    def change_label_form(self):
        return


class Margin_Softmax(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """

    def __init__(self, s=4, margin=0.5):
        super(Margin_Softmax, self).__init__()
        self.s = s  # 用于平衡特征向量
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False



    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != 0)[0]
        index_y = torch.where(labels != 0)[1]
        # print(index)
        # print('logits', logits)
        target_logit = logits[index, index_y]
        # print('target_logit', target_logit)
        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.margin
            logits[index, index_y] = final_target_logit
            logits.cos_()
        logits = logits * self.s
        return logits


class DistCrossEntropyFunc(torch.autograd.Function):
    """
    CrossEntropy loss is calculated in parallel, allreduce denominator into single gpu and calculate softmax.
    Implemented of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, label: torch.Tensor):
        """ """
        batch_size = logits.size(0)
        # for numerical stability
        max_logits, _ = torch.max(logits, dim=1, keepdim=True)  # ！！！！！！！！！！！！！！！！！！
        # local to global
        # distributed.all_reduce(max_logits, distributed.ReduceOp.MAX)
        logits.sub_(max_logits)
        logits.exp_()
        sum_logits_exp = torch.sum(logits, dim=1, keepdim=True)
        # local to global
        # distributed.all_reduce(sum_logits_exp, distributed.ReduceOp.SUM)
        logits.div_(sum_logits_exp)
        index = torch.where(label != 0)[1]
        # index_y = torch.where(label != 0)[1]
        index = index.view(-1, 1)
        # loss
        loss = torch.zeros(batch_size, 1, device=logits.device)
        loss = logits.gather(1, index)
        # print('loss',loss)
        # print('loss.size()',loss.size())
        # print('logits.size()',logits.size())
        # distributed.all_reduce(loss, distributed.ReduceOp.SUM)
        ctx.save_for_backward(index, logits, label)
        return loss.clamp_min_(1e-30).log_().mean() * (-1)

    @staticmethod
    def backward(ctx, loss_gradient):
        """
        Args:
            loss_grad (torch.Tensor): gradient backward by last layer
        Returns:
            gradients for each input in forward function
            `None` gradients for one-hot label
        """
        (
            index,
            logits,
            label,
        ) = ctx.saved_tensors
        batch_size = logits.size(0)
        one_hot = torch.zeros(
            size=[index.size(0), logits.size(1)], device=logits.device
        )

        one_hot.scatter_(1, index, 1)
        # print('logits',logits)
        # print('logits[index]',logits[index] )
        # print(one_hot)
        logits = logits - one_hot  # 计算导数 log softmax 的导数 = 1-log softmax
        # print(logits)
        logits.div_(batch_size)

        return logits * loss_gradient.item(), None


class DistCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(DistCrossEntropy, self).__init__()

    def forward(self, logit_part, label_part):
        return DistCrossEntropyFunc.apply(logit_part, label_part)
