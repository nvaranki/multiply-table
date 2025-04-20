from typing import Optional, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Dropout, LayerNorm, Parameter
from torch.nn.init import xavier_uniform_, constant_

from BinaryDecoder import BinaryDecoder


class MultiheadAttention(nn.Module):

    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            batch_first=False,
            device=None,
            dtype=None,
        ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim
        self._qkv_same_embed_dim = True  # self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads

        self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        self.register_parameter("q_proj_weight", None)
        self.register_parameter("k_proj_weight", None)
        self.register_parameter("v_proj_weight", None)

        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        constant_(self.in_proj_bias, 0.0)

        self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        # self._reset_parameters()
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        is_batched = query.dim() == 3

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        # attn_output, attn_output_weights = F.multi_head_attention_forward(
        attn_output, attn_output_weights = self.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.bias_k,
            self.bias_v,
            self.add_zero_attn,
            self.dropout,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def multi_head_attention_forward(self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            embed_dim_to_check: int,
            num_heads: int,
            in_proj_weight: Optional[Tensor],
            in_proj_bias: Optional[Tensor],
            bias_k: Optional[Tensor],
            bias_v: Optional[Tensor],
            add_zero_attn: bool,
            dropout_p: float,
            training: bool = True,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            use_separate_proj_weight: bool = False,
            q_proj_weight: Optional[Tensor] = None,
            k_proj_weight: Optional[Tensor] = None,
            v_proj_weight: Optional[Tensor] = None,
            static_k: Optional[Tensor] = None,
            static_v: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False,
        ) -> Tuple[Tensor, Optional[Tensor]]:
        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        attn_mask = None
        head_dim = embed_dim // num_heads

        #
        # compute in-projection
        #
        # from torch.nn.functional import _in_projection_packed
        q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)

        #
        # reshape q, k, v for multihead attention and make them batch first
        #
        q = q.view(tgt_len,    bsz * num_heads, head_dim).transpose(0, 1)
        k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

        # update source sequence length after adjustments
        src_len = k.size(1)

        # adjust dropout probability
        if not training:
            dropout_p = 0.0

        #
        # (deep breath) calculate attention and out projection
        #
        q = q.view(bsz, num_heads, tgt_len, head_dim)
        k = k.view(bsz, num_heads, src_len, head_dim)
        v = v.view(bsz, num_heads, src_len, head_dim)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask, dropout_p, is_causal
        )
        attn_output = attn_output.permute(2, 0, 1, 3).view(tgt_len, bsz, embed_dim)
        return attn_output, None


class TransformerEncoderLayer(nn.Module):

    def __init__(self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
            layer_norm_eps: float = 1e-5,
            batch_first: bool = False,
            device=None,
            dtype=None
        ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        # Implementation of SA
        self.self_attn = MultiheadAttention(
            d_model,
            1, #nhead,
            dropout=dropout,
            bias=True,
            batch_first=True,
            **factory_kwargs,
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)
        self.dropout2 = Dropout(dropout)
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def forward(self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False,
        ) -> Tensor:
        x = src
        x = x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
        x = x + self._ff_block(x)

        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        return x

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        # Implementation of SA and MHA
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            **factory_kwargs,
        )
        self.multihead_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            **factory_kwargs,
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)
        self.norm_first = norm_first
        self.dropout3 = Dropout(dropout)
        # Legacy string support for activation function.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        x = tgt
        x = x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
        x = x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal)
        x = x + self._ff_block(x)
        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return x  #self.dropout1(x)

    # multi-head attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return x  #self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class MultiplyModel(nn.Module):

    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers, device = None, dtype = None):
        super(MultiplyModel, self).__init__()
        self.device=device
        self.dtype=dtype
        self.embedding = nn.Embedding(vocab_size+1, embed_size, padding_idx=vocab_size, device=device, dtype=dtype)
        self.encoder = nn.TransformerEncoder(
            TransformerEncoderLayer(embed_size, num_heads, hidden_dim, activation=F.tanh, batch_first=True, device=device, dtype=dtype),
            num_layers
        )
        self.decoder = nn.TransformerDecoder(
            # nn.TransformerDecoderLayer(embed_size, num_heads, hidden_dim, activation=F.tanh, batch_first=True, device=device, dtype=dtype),
            TransformerDecoderLayer(embed_size, num_heads, hidden_dim, activation=F.tanh, batch_first=True, device=device, dtype=dtype),
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
