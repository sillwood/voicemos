import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import make_non_pad_mask


class Loss(nn.Module):
    """
    Implements the clipped MSE loss and categorical loss.
    """

    def __init__(self, masked_loss):
        super(Loss, self).__init__()
        self.masked_loss = masked_loss
        self.loss_fn = nn.L1Loss()

    def forward_criterion(self, y_hat, label, masks=None):
        # might investigate how to combine masked loss with categorical output
        loss = self.loss_fn(y_hat, label)
        if masks is not None:
            loss = loss.masked_select(masks)
        loss = torch.sum(loss) / masks.sum()
        return loss

    def forward(self, pred_mean, gt_mean, lens):
        """
        Args:
            pred_mean, pred_score: [batch, time]
        """

        # return self.loss_fn(pred_mean, gt_mean)

        # make mask
        if self.masked_loss:
            masks = make_non_pad_mask(lens, xs=pred_mean, length_dim=1).to(
                pred_mean.device
            )
        else:
            masks = None

        # repeat for frame level loss
        time = pred_mean.shape[1]
        gt_mean = gt_mean.unsqueeze(1).repeat(1, time)

        main_loss = self.forward_criterion(pred_mean, gt_mean, masks)
        return main_loss
