# ------------------------------------------------------------------------------
# Written by Miao
# ------------------------------------------------------------------------------

import torch.nn as nn


class MSELoss(nn.Module):
    """
    TODO  损失函数
    """
    def __init__(self, use_target_confidence):
        super(MSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_confidence

    def forward(self, output, target, target_confidence):

        loss = 0
        if self.use_target_weight:
            loss += self.criterion(output.mul(target_confidence), target.mul(target_confidence))
        else:
            loss += self.criterion(output, target)

        return loss
