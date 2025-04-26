import torch.nn as nn
from torch import Tensor


class MultiplyModel(nn.Module):

    def __init__(self, vocab_size, embed_size, device = None, dtype = None):
        super(MultiplyModel, self).__init__()
        self.device=device
        self.dtype=dtype
        self.embedding = nn.Embedding(vocab_size+1, embed_size, padding_idx=vocab_size, device=device, dtype=dtype)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        pass

    def target(self, mout: Tensor) -> Tensor:
        """ Returns target of training """
        pass

    def tokens(self, mout: Tensor) -> Tensor:
        """ Returns tokens of inference """
        pass
