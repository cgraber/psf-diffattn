from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F



from joint_panoptic_forecast.model_components.conv_transformer_layers import ConvCrossAttention, ConvCrossAttention_AllSpatial, ConvSelfAttention, ConvSelfAttention_AllSpatial


class ConvTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, attn_kernel_size, feedforward_kernel_size, dim_feedforward, dropout=0.1, activation='relu',
                 use_pre_norm=False, is_agent_aware=False, attention_type='standard'):
        super().__init__()
        self.use_pre_norm = use_pre_norm
        if attention_type == 'conv':
            attn_fn = ConvSelfAttention
        elif attention_type == 'conv_all_spatial':
            attn_fn = ConvSelfAttention_AllSpatial
        else:
            raise ValueError('Attention type not recognized: ',attention_type)
        self.self_attn = attn_fn(d_model, nhead, attn_kernel_size, dropout=dropout,
                                           is_agent_aware=is_agent_aware)
        self.linear1 = nn.Conv2d(d_model, dim_feedforward, kernel_size=feedforward_kernel_size, stride=1, padding=feedforward_kernel_size//2)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Conv2d(dim_feedforward, d_model, kernel_size=feedforward_kernel_size, stride=1, padding=feedforward_kernel_size//2)

        self.norm1 = nn.GroupNorm(nhead, d_model)
        self.norm2 = nn.GroupNorm(nhead, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                src_agent_aware_mask = None, src_key_padding_mask=None) -> torch.Tensor:
        b, t,c, h, w = src.shape
        if self.use_pre_norm:
            src2 = self.norm1(src.view(b*t, c, h, w)).view(b, t, c, h, w)
            src2 = self.self_attn(src2, attn_mask=src_mask,
                                            agent_aware_mask=src_agent_aware_mask)
            src = src + self.dropout1(src2)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(src.view(b*t, c, h, w))))))
            src = src + self.dropout2(src2).view(b, t, c, h, w)
        else:
            src2 = self.self_attn(src, attn_mask=src_mask,
                                            agent_aware_mask=src_agent_aware_mask)
            src = src + self.dropout1(src2)
            src = self.norm1(src.view(b*t, c, h, w))
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src).view(b, t, c, h, w)
        return src #, attn_wts 


class ConvTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, attn_kernel_size, feedforward_kernel_size, dim_feedforward,
                 dropout=0.1, activation='relu', is_agent_aware=False,
                 use_pre_norm=False, attention_type='standard'):
        super().__init__()
        if attention_type == 'conv':
            self_attn_fn = ConvSelfAttention
            cross_attn_fn = ConvCrossAttention
        elif attention_type == 'conv_all_spatial':
            self_attn_fn = ConvSelfAttention_AllSpatial
            cross_attn_fn = ConvCrossAttention_AllSpatial
        else:
            raise ValueError('Attention type not recognized: ',attention_type)
        self.self_attn = self_attn_fn(d_model, nhead, attn_kernel_size, dropout=dropout,
                                           is_agent_aware=is_agent_aware)
        self.cross_attn = cross_attn_fn(d_model, nhead, attn_kernel_size, dropout=dropout,
                                            is_agent_aware=is_agent_aware)
        self.linear1 = nn.Conv2d(d_model, dim_feedforward, kernel_size=feedforward_kernel_size, stride=1, padding=feedforward_kernel_size//2)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Conv2d(dim_feedforward, d_model, kernel_size=feedforward_kernel_size, stride=1, padding=feedforward_kernel_size//2)

        self.norm1 = nn.GroupNorm(nhead, d_model)
        self.norm2 = nn.GroupNorm(nhead, d_model)
        self.norm3 = nn.GroupNorm(nhead, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.use_pre_norm = use_pre_norm

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

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
        b, t, c, h, w = tgt.shape
        if self.use_pre_norm:
            tgt2 = self.norm1(tgt.view(b*t, c, h, w)).view(b, t, c, h, w)
            tgt2  = self.self_attn(tgt2, attn_mask=tgt_mask,
                                  agent_aware_mask=tgt_agent_aware_mask)
            tgt = tgt + self.dropout1(tgt2)
            tgt2 = self.norm2(tgt.view(b*t, c, h, w)).view(b, t, c, h, w)
            tgt2 = self.cross_attn(tgt2, memory, attn_mask=memory_mask,
                                                   agent_aware_mask=memory_agent_aware_mask)
            tgt = tgt + self.dropout2(tgt2)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm3(tgt.view(b*t, c, h, w))))))
            tgt = tgt + self.dropout3(tgt2).view(b, t, c, h, w)
        else:
            tgt2 = self.self_attn(tgt, attn_mask=tgt_mask,
                                             agent_aware_mask=tgt_agent_aware_mask)
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            tgt2 = self.cross_attn(tgt, memory, attn_mask=memory_mask,
                                                   agent_aware_mask=memory_agent_aware_mask)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)
        return tgt, None, None


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
