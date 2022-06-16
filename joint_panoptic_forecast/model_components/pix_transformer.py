import torch
from torch import nn
import torch.nn.functional as F

import math


class PixMHA(nn.Module):
    def __init__(self, emb_size, kernel_size, num_heads, softmax_full_matrix=False):
        super().__init__()
        init_factor = math.sqrt(6/(emb_size+1))
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.softmax_full_matrix=softmax_full_matrix

        assert self.emb_size % self.num_heads == 0, 'Embedding size must be divisible by num heads!'

    def forward(self, inp_qk, inp_v):
        # inp: n_inst x c x h x w
        attn_wts = -inp_qk
        attn = torch.softmax(attn_wts, dim=0)
        out = inp_v*attn
        return out