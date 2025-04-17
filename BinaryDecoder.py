from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BinaryDecoder(nn.Module):

    def __init__(self, depth: int):
        super().__init__()
        self.depth = depth

    def forward(self, src)-> Tuple[Tensor, Tensor]:
        t2 = src.repeat(1, 2).reshape(src.size(0), 2, src.size(1))
        # distances from poles
        tp = t2
        tp[:,0,:] += 1  # v - (-1)
        tp[:,1,:] -= 1  # v - (+1)
        ta = abs(tp)
        # match to minimum (mark with 0)
        tm = torch.minimum(ta[:,0,:], ta[:,1,:]).reshape(src.size(0), 1, src.size(1))
        tx = ta - tm
        # participation (1 at 0, 0 at others)
        tx -= 1
        tx = (-tx).floor()
        tz = F.threshold(tx, 0, 0) # now 0 or 1
        # loss
        tl = (-tp).multiply(tz).sum(dim=1)
        # turm to bipolar then convert to single row unipolar
        ts = -tz[:,0,:] + tz[:,1,:]
        ti = ((ts + 1) / 2).round()
        # compute index
        tb = ti * pow(2, torch.tensor(range(ti.size(1)), dtype=torch.float, device=ti.device))
        i = tb.sum(dim=-1).round().to(dtype=torch.int64)
        # done
        return (i, tl)

    def vector(self, index: Tensor)-> Tensor:
        idx = index.clone().to(dtype=torch.int)
        t2 = torch.zeros((idx.size(0),self.depth), dtype=torch.float, device=idx.device)
        for i in range(self.depth):
            t2[:, i] = idx.remainder(2)
            idx = idx.divide(2).floor().to(dtype=torch.int)
        return t2*2-1  # bipolar
