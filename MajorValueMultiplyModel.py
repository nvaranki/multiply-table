from torch import nn, Tensor

from MultiplyModel import MultiplyModel
import torch.nn.functional as F


class MajorValueMultiplyModel(MultiplyModel):

    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers, device = None, dtype = None):
        super(MajorValueMultiplyModel, self).__init__(vocab_size, embed_size, device, dtype)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_size, num_heads, hidden_dim, batch_first=True, device=device, dtype=dtype),
            num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embed_size, num_heads, hidden_dim, batch_first=True, device=device, dtype=dtype),
            num_layers
        )
        self.fc = nn.Linear(embed_size, vocab_size, device=device, dtype=dtype)
        self.vocab_size = vocab_size

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encoder(self.embedding(src), src_mask)
        output = self.decoder(self.embedding(tgt), memory, tgt_mask, memory_mask)
        output = F.log_softmax(self.fc(output), dim=-1)
        return output.view(-1, self.vocab_size)  #TODO

    def target(self, index: Tensor) -> Tensor:
        """ Returns target of training """
        return index

    def tokens(self, mout: Tensor) -> Tensor:
        """ Returns tokens of inference """
        return mout.max(dim=-1).indices
