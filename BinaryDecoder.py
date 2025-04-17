from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, IntTensor


class DecoderLoss(nn.L1Loss):

    def __init__(self, vocab_size: int, padding_value: int):
        super().__init__() #(reduction="none")

    @staticmethod
    def vector(index: Tensor, sz: int)-> Tensor:
        idx = index.clone().to(dtype=torch.int)
        t2 = torch.zeros((idx.size(0),sz), dtype=torch.float, device=idx.device)
        for i in range(sz):
            t2[:, i] = idx.remainder(2)
            idx = idx.divide(2).floor().to(dtype=torch.int)
        return t2*2-1  # bipolar


class BinaryDecoder(nn.Module):

    def __init__(self, vocab_size: int, padding_value: int):
        super().__init__()

    def forward(self, src)-> Tuple[int, Tensor]:
        t = src[-1]
        t2 = t.repeat(2, 1)
        # distances from poles
        tp = t2
        tp[0] += 1  # v - (-1)
        tp[1] -= 1  # v - (+1)
        ta = abs(tp)
        # match to minimum (mark with 0)
        tm = torch.minimum(ta[0], ta[1])
        tx = ta - tm
        # participation (1 at 0, 0 at others)
        tx -= 1
        tx = (-tx).floor()
        tz = F.threshold(tx, 0, 0) # now 0 or 1
        # loss
        tl = (-tp).multiply(tz).sum(dim=0)
        # turm to bipolar then convert to single row unipolar
        ts = -tz[0] + tz[1]
        ti = ((ts + 1) / 2).round()
        # compute index
        tb = ti * pow(2, torch.tensor(range(ti.size(0)), dtype=torch.float, device=t.device))
        i = tb.sum().round().to(dtype=torch.int)
        # done
        return (i.item(), tl)
