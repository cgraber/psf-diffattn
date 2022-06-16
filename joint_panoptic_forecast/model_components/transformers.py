import copy
from typing import Optional, Tuple, List

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ShapeSpec, get_norm

from .transformer_layers import MultiheadAttention
from .transformer_difference_general_layers import MultiheadDifferenceGeneralAttention

# First section: re-implementation of base transformers, with the following changes:
#   1) The ability to choose between pre- and post-normalization
#   2) Attention values are returned

class TransformerEncoderLayer(nn.Module):
    """
    This is a re-implementation of the default PyTorch layer, except it also returns the attention
    weights (so that we have access to them for our own purposes)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 use_pre_norm=False, temperature=1.0, attention_type='dot_product',
                 aggregation_type='normal', attention_module='standard', add_zero_attn=False,
                use_no_match_emb=False, is_agent_aware=False, key_mode=None, value_mode=None,
                 key_dim=None):
        super().__init__()
        self.use_pre_norm = use_pre_norm
        super(TransformerEncoderLayer, self).__init__()
        if attention_module == 'difference_general':
            self.self_attn = MultiheadDifferenceGeneralAttention(d_model, nhead, dropout=dropout, temperature=temperature,
                                                attention_type=attention_type, aggregation_type=aggregation_type,
                                                add_zero_attn=add_zero_attn,
                                                use_no_match_emb=use_no_match_emb, is_agent_aware=is_agent_aware,
                                                key_mode=key_mode, value_mode=value_mode,kdim=key_dim)
        else:
            attn = MultiheadAttention
            self.self_attn = attn(d_model, nhead, dropout=dropout, temperature=temperature,
                                                attention_type=attention_type, aggregation_type=aggregation_type,
                                                add_zero_attn=add_zero_attn,
                                                use_no_match_emb=use_no_match_emb, is_agent_aware=is_agent_aware,kdim=key_dim)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                src_agent_aware_mask = None) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if self.use_pre_norm:
            src2 = self.norm1(src)
            src2, attn_wts = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                                            key_padding_mask=src_key_padding_mask,
                                            query_value=src2, agent_aware_mask=src_agent_aware_mask)
            src = src + self.dropout1(src2)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(src)))))
            src = src + self.dropout2(src2)
        else:
            src2, attn_wts = self.self_attn(src, src, src, attn_mask=src_mask,
                                            key_padding_mask=src_key_padding_mask,
                                            query_value=src, agent_aware_mask=src_agent_aware_mask)
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        return src #, attn_wts

class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None, fix_init=False):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        if fix_init:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None,
                src_agent_aware_mask = None) -> torch.Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask,
                         src_agent_aware_mask=src_agent_aware_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Module):
    """
    This is a re-implementation of the default PyTorch layer, except it also returns the attention
    weights (so that we have access to them for our own purposes)
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 use_pre_norm=False, temperature=1.0, attention_type='dot_product',
                 aggregation_type='normal', is_agent_aware=False, attention_module='standard',
                 add_zero_attn=False, use_no_match_emb=False, key_mode=None, value_mode=None,
                 key_dim=None):
        super(TransformerDecoderLayer, self).__init__()
        if attention_module == 'difference_general':
            self.self_attn = MultiheadDifferenceGeneralAttention(d_model, nhead, dropout=dropout, temperature=temperature,
                                                attention_type=attention_type, aggregation_type=aggregation_type,
                                                is_agent_aware=is_agent_aware, add_zero_attn=add_zero_attn,
                                                use_no_match_emb=use_no_match_emb, key_mode=key_mode, value_mode=value_mode,
                                                kdim=key_dim)
            self.multihead_attn = MultiheadDifferenceGeneralAttention(d_model, nhead, dropout=dropout, temperature=temperature,
                                                    attention_type=attention_type, aggregation_type=aggregation_type,
                                                    add_zero_attn=False, use_no_match_emb=use_no_match_emb,
                                                    key_mode=key_mode, value_mode=value_mode, kdim=key_dim)
        else:
            attn = MultiheadAttention
            self.self_attn = attn(d_model, nhead, dropout=dropout, temperature=temperature,
                                                attention_type=attention_type, aggregation_type=aggregation_type,
                                                is_agent_aware=is_agent_aware, add_zero_attn=add_zero_attn,
                                                use_no_match_emb=use_no_match_emb, kdim=key_dim)
            self.multihead_attn = attn(d_model, nhead, dropout=dropout, temperature=temperature,
                                                    attention_type=attention_type, aggregation_type=aggregation_type,
                                                    add_zero_attn=False, use_no_match_emb=use_no_match_emb, kdim=key_dim)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.use_pre_norm = use_pre_norm

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None, 
                memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None, 
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_agent_aware_mask: Optional[torch.Tensor] = None,
                memory_agent_aware_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if self.use_pre_norm:
            tgt2 = self.norm1(tgt)
            tgt2, self_attn = self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask, query_value=tgt2,
                                  agent_aware_mask=tgt_agent_aware_mask)
            tgt = tgt + self.dropout1(tgt2)
            tgt2 = self.norm2(tgt)
            tgt2, cross_attn = self.multihead_attn(tgt2, memory, memory, attn_mask=memory_mask,
                                                   key_padding_mask=memory_key_padding_mask,
                                                   query_value=tgt2,
                                                   agent_aware_mask=memory_agent_aware_mask)
            tgt = tgt + self.dropout2(tgt2)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm3(tgt)))))
            tgt = tgt + self.dropout3(tgt2)
        else:
            tgt2, self_attn = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask,
                                query_value=tgt, agent_aware_mask=tgt_agent_aware_mask)
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            tgt2, cross_attn = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask,
                                    query_value=tgt, agent_aware_mask=memory_agent_aware_mask)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)
        return tgt, self_attn, cross_attn


class TransformerDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers
    This is a reimplementation that allows for easy access to the computed attentions

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None, fix_init=False):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        if fix_init:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                memory_agent_aware_mask: Optional[torch.Tensor] = None,
                tgt_agent_aware_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt
        all_self_attns = []
        all_cross_attns = []
        for mod in self.layers:
            output, self_attn, cross_attn = mod(output, memory, tgt_mask=tgt_mask,
                                                memory_mask=memory_mask,
                                                tgt_key_padding_mask=tgt_key_padding_mask,
                                                memory_key_padding_mask=memory_key_padding_mask,
                                                memory_agent_aware_mask=memory_agent_aware_mask,
                                                tgt_agent_aware_mask=tgt_agent_aware_mask)
            all_self_attns.append(self_attn)
            all_cross_attns.append(cross_attn)
        if self.norm is not None:
            output = self.norm(output)

        return output, all_self_attns, all_cross_attns

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
