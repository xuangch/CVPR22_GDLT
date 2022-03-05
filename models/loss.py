import torch
from torch import nn
from models.triplet_loss import HardTripletLoss
import torch.nn.functional as F
import numpy as np


# 各种分布特征聚合
class LossFun(nn.Module):
    def __init__(self, alpha, margin):
        super(LossFun, self).__init__()
        self.mse_loss = nn.MSELoss()
        # self.mse_loss = nn.L1Loss()
        self.triplet_loss = HardTripletLoss(margin=margin, hardest=True)
        self.alpha = alpha

    def forward(self, pred, label, feat):
        # feat (b, n, c), x (b, t, c)
        if feat is not None:
            device = feat.device
            b, n, c = feat.shape
            flat_feat = feat.view(-1, c)  # (bn, c)
            la = torch.arange(n, device=device).repeat(b)

            t_loss = self.triplet_loss(flat_feat, la)
            # t_loss = pair_diversity_loss(feat)
        else:
            self.alpha = 0
            t_loss = 0

        mse_loss = self.mse_loss(pred, label)
        # mse_loss = pearson_loss(pred, label)
        return mse_loss + self.alpha * t_loss, mse_loss, t_loss
        # return f_loss, mse_loss, t_loss, f_loss
