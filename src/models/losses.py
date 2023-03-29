"Loss functions"

import torch


class GlobalMeanRemovedLoss(torch.nn.Module):

    def forward(self, _y:torch.Tensor, y:torch.Tensor) -> torch.Tensor:

        mean_dims = tuple(range(len(_y.shape)))[1:]
        reshape_shape = [-1] + [1] * (len(y.shape) - 1)

        mean_y = y.mean(dim=mean_dims).reshape(reshape_shape)
        _mean_y = _y.mean(dim=mean_dims).reshape(reshape_shape)

        norm_y = y - mean_y
        _norm_y = _y - _mean_y
        error = norm_y - _norm_y
        error = torch.abs(error)

        return error.mean()
