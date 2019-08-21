import torch
import torch.nn as nn
import torch.nn.functional as F

class cdloss(nn.Module):
    def __init__(self,gamma = 1.5,size_average=True):
        super(cdloss,self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, prob, target):
        target = target.view(-1)
        #prob = F.sigmoid(logit)
        prob = prob.view(-1)
        prob_p = torch.clamp(prob, 1e-8, 1 - 1e-8)
        prob_n = torch.clamp(1.0 - prob, 1e-8, 1 - 1e-8)
        batch_loss= - torch.pow((2 - prob_p),self.gamma) * prob_p.log() * target \
                    - prob_n.log() * (1 - target) *(2 - prob_n)
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss
        return loss