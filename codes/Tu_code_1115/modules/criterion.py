import torch
from torch import nn

from modules.utils import cal_relative_pose

class PoseLoss(nn.Module):
    """Camera pose loss"""

    def __init__(self, s_x=0.0, s_q=-3.0, requires_grad=True):
        """
        Args:
            s_x, s_q (float): Initial value of s_x, s_q.
            requires_grad
        """
        super().__init__()
        self.s_x = nn.Parameter(torch.tensor(s_x), requires_grad=requires_grad)
        self.s_q = nn.Parameter(torch.tensor(s_q), requires_grad=requires_grad)

    def forward(self, pred, targ):
        """
        Args:
            pred (torch.Tensor): Predicted camera poses.
            targ (torch.Tensor): Target camera poses.

        Returns:
            loss (torch.Tensor): Loss value.

        Shape:
            pred: (batch_size, 6).
            targ: (batch_size, 6).
            loss: (1).
        """
        batch_size = pred.shape[0]
        pred1, pred2 = pred[:int(batch_size/2), :], pred[int(batch_size/2):, :]
        targ1, targ2 = targ[:int(batch_size/2), :], targ[int(batch_size/2):, :]
        pred_rel = cal_relative_pose(pred1, pred2)
        targ_rel = cal_relative_pose(targ1, targ2)

        l1_loss = nn.L1Loss()
        loss = (l1_loss(pred[:, :3], targ[:, :3]) * torch.exp(-self.s_x) + self.s_x +
                l1_loss(pred[:, 3:], targ[:, 3:]) * torch.exp(-self.s_q) + self.s_q +
                l1_loss(pred_rel[:, :3], targ_rel[:, :3]) * torch.exp(-self.s_x) + self.s_x +
                l1_loss(pred_rel[:, 3:], targ_rel[:, 3:]) * torch.exp(-self.s_q) + self.s_q)
        return loss
