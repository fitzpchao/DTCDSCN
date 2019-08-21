import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    r"""Returns cosine similarity between x1 and x2, computed along dim.
    Args:
        x1 (Variable): First input.
        x2 (Variable): Second input (of size matching x1).
        dim (int, optional): Dimension of vectors. Default: 1
        eps (float, optional): Small value to avoid division by zero. Default: 1e-8
    Shape:
        - Input: :math:`(\ast_1, D, \ast_2)` where D is at position `dim`.
        - Output: :math:`(\ast_1, \ast_2)` where 1 is at position `dim`.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def cos_distance(self, a, b):
    return torch.dot(a, b)/(torch.norm(a)*torch.norm(b))

class TripletMarginLoss(nn.Module):
    def __init__(self, margin, use_ohem=False, ohem_bs=128, dist_type=0):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.dist_type = dist_type
        self.use_ohem = use_ohem
        self.ohem_bs = ohem_bs
        # print('Use_OHEM : ',self.use_ohem)

    def forward(self, anchor, positive, negative):
        # eucl distance
        # dist = torch.sum( (anchor - positive) ** 2 - (anchor - negative) ** 2, dim=1)\
        #        + self.margin

        if self.dist_type == 0:
            dist_p = F.pairwise_distance(anchor, positive)
            dist_n = F.pairwise_distance(anchor, negative)
        else:
            dist_p = cosine_similarity(anchor, positive)
            dist_n = cosine_similarity(anchor, negative)

        dist_hinge = torch.clamp(dist_p - dist_n + self.margin, min=0.0)
        if self.use_ohem:
            v, idx = torch.sort(dist_hinge, descending=True)
            loss = torch.mean(v[0:self.ohem_bs])
        else:
            loss = torch.mean(dist_hinge)

        return loss