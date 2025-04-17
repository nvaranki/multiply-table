import torch.nn as nn
import torch.nn.functional as F

from BinaryDecoder import BinaryDecoder


class MultiplyModel(nn.Module):

    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers, device = None, dtype = None):
        super(MultiplyModel, self).__init__()
        self.device=device
        self.dtype=dtype
        self.embedding = nn.Embedding(vocab_size+1, embed_size, padding_idx=vocab_size, device=device, dtype=dtype)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_size, num_heads, hidden_dim, activation=F.tanh, batch_first=True, device=device, dtype=dtype),
            num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embed_size, num_heads, hidden_dim, activation=F.tanh, batch_first=True, device=device, dtype=dtype),
            num_layers
        )
        self.bdec = BinaryDecoder(embed_size)
        self.embed_size = embed_size

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        src_embedded = self.embedding(src)
        tgt_embedded = self.embedding(tgt)
        memory = self.encoder(src_embedded, src_mask)
        output = self.decoder(tgt_embedded, memory, tgt_mask, memory_mask)
        return output
