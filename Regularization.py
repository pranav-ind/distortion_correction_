import torch as th
import torch.nn.functional as F

import numpy as np

# Regulariser base class (standard from PyTorch)
class _Regulariser(th.nn.modules.Module):
    def __init__(self, pixel_spacing, size_average=True, reduce=True):
        super(_Regulariser, self).__init__()
        self._size_average = size_average
        self._reduce = reduce
        self._weight = 1
        self._dim = len(pixel_spacing)
        self._pixel_spacing = pixel_spacing
        self.name = "parent"
        self._mask = None

    def SetWeight(self, weight):
        print("SetWeight is deprecated. Use set_weight instead.")
        self.set_weight(weight)

    def set_weight(self, weight):
        self._weight = weight

    def set_mask(self, mask):
        self._mask = mask

    def _mask_2d(self, df):
        if not self._mask is None:
            nx, ny, d = df.shape
            return df * self._mask.image.squeeze()[:nx,:ny].unsqueeze(-1).repeat(1,1,d)
        else:
            return df

    def _mask_3d(self, df):
        if not self._mask is None:
            nx, ny, nz, d = df.shape
            return df * self._mask.image.squeeze()[:nx,:ny,:nz].unsqueeze(-1).repeat(1,1,1,d)
        else:
            return df

    # conditional return
    def return_loss(self, tensor):
        if self._size_average and self._reduce:
            return self._weight*tensor.mean()
        if not self._size_average and self._reduce:
            return self._weight*tensor.sum()
        if not self._reduce:
            return self._weight*tensor


class TVRegulariser(_Regulariser):
    def __init__(self, pixel_spacing, size_average=True, reduce=True):
        super(TVRegulariser, self).__init__(pixel_spacing, size_average, reduce)

        self.name = "TV"

        if self._dim == 2:
            self._regulariser = self._TV_regulariser_2d  # 2d regularisation
        elif self._dim == 3:
            self._regulariser = self._TV_regulariser_3d  # 3d regularisation

    def _TV_regulariser_2d(self, displacement):
        dx = th.abs(displacement[1:, 1:, :] - displacement[:-1, 1:, :])*self._pixel_spacing[0]
        dy = th.abs(displacement[1:, 1:, :] - displacement[1:, :-1, :])*self._pixel_spacing[1]

        return self._mask_2d(F.pad(dx + dy, (0, 1, 0, 1)))

    def _TV_regulariser_3d(self, displacement):
        dx = th.abs(displacement[1:, 1:, 1:, :] - displacement[:-1, 1:, 1:, :])*self._pixel_spacing[0]
        dy = th.abs(displacement[1:, 1:, 1:, :] - displacement[1:, :-1, 1:, :])*self._pixel_spacing[1]
        dz = th.abs(displacement[1:, 1:, 1:, :] - displacement[1:, 1:, :-1, :])*self._pixel_spacing[2]

        return self._mask_3d(F.pad(dx + dy + dz, (0, 1, 0, 1, 0, 1)))

    def forward(self, displacement):
        return self.return_loss(self._regulariser(displacement))