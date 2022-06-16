import math

import torch
from torch import nn
import torch.nn.functional as F




class ConvSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, kernel_size, dropout=0., bias=True,
                 is_agent_aware=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.nheads = num_heads
        self.dropout = dropout
        self.h_emb_dim = embed_dim // num_heads
        self.is_agent_aware = is_agent_aware
        if is_agent_aware:
            self.in_emb_len = 5
        else:
            self.in_emb_len = 3
        self.conv_in = nn.Conv2d(embed_dim, self.embed_dim*self.in_emb_len, kernel_size=kernel_size,
                                 stride=1, padding=kernel_size//2)
        self.conv_out = nn.Conv2d(embed_dim, embed_dim, kernel_size=kernel_size,
                                  stride=1, padding=kernel_size//2)

    def forward(self, inp, attn_mask=None, agent_aware_mask=None):
        """
        Args:
            inp: should be [b, seq, c, h, w]
            attn_mask: if not None, is [b, seq, seq]
        """
        b, seq, c, h, w = inp.shape
        inp = inp.view(b*seq, c, h, w)
        feats = self.conv_in(inp)
        feats = feats.view(
            b, seq, self.in_emb_len*self.h_emb_dim, self.nheads*h*w
        ).permute(0,3,1,2).reshape(b*self.nheads, h*w, seq, self.in_emb_len*self.h_emb_dim)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                correct_2d_size = (seq, seq)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (b*self.nheads, seq, seq)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            attn_mask = attn_mask.unsqueeze(1)

        if self.is_agent_aware:
            agent_aware_mask = agent_aware_mask.unsqueeze(1)
            if b > 1:
                raise NotImplementedError()
            k_same, k_other, q_same, q_other, v = feats.split(self.h_emb_dim, dim=3)
            output = dot_product_agent_aware_attn(k_same, k_other, q_same, q_other, v, agent_aware_mask,
                                                  attn_mask, self.dropout, self.training)
        else:
            k, q, v = feats.split(self.h_emb_dim, dim=3)
            output = dot_product_attn(k, q, v, attn_mask, self.dropout, self.training)
        output = output.reshape(b, self.nheads*h*w, seq, self.h_emb_dim)
        output = output.permute(0, 2, 3, 1).reshape(b*seq, c, h, w)
        output = self.conv_out(output).reshape(b, seq, c, h, w)
        return output


class ConvSelfAttention_AllSpatial(nn.Module):
    def __init__(self, embed_dim, num_heads, kernel_size, dropout=0., bias=True,
                 is_agent_aware=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.nheads = num_heads
        self.dropout = dropout
        self.h_emb_dim = embed_dim // num_heads
        self.is_agent_aware = is_agent_aware
        if is_agent_aware:
            self.in_emb_len = 5
        else:
            self.in_emb_len = 3
        self.conv_in = nn.Conv2d(embed_dim, self.in_emb_len*embed_dim, kernel_size=kernel_size,
                                 stride=1, padding=kernel_size//2)
        self.conv_out = nn.Conv2d(embed_dim, embed_dim, kernel_size=kernel_size,
                                  stride=1, padding=kernel_size//2)

    def forward(self, inp, attn_mask=None, agent_aware_mask=None):
        """
        Args:
            inp: should be [b, seq, c, h, w]
            attn_mask: if not None, is [b, seq, seq]
        """
        b, seq, c, h, w = inp.shape
        inp = inp.view(b*seq, c, h, w)
        feats = self.conv_in(inp)
        feats = feats.view(
            b, seq, self.in_emb_len*self.h_emb_dim, self.nheads, h*w
        ).permute(0,3,1,2, 4).reshape(b*self.nheads, seq, self.in_emb_len*self.h_emb_dim*h*w)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                correct_2d_size = (seq, seq)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (b*self.nheads, seq, seq)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask

        if self.is_agent_aware:
            if b > 1:
                raise NotImplementedError()
            k_same, k_other, q_same, q_other, v = feats.split(self.h_emb_dim*h*w, dim=2)
            output = dot_product_agent_aware_attn(k_same, k_other, q_same, q_other, v, agent_aware_mask,
                                                  attn_mask, self.dropout, self.training)
        else:
            k, q, v = feats.split(self.h_emb_dim*h*w, dim=2)
            output = dot_product_attn(k, q, v, attn_mask, self.dropout, self.training)
        output = output.reshape(b, self.nheads, seq, self.h_emb_dim, h, w)
        output = output.transpose(1, 2).reshape(b*seq, c, h, w)
        output = self.conv_out(output).reshape(b, seq, c, h, w)
        return output

class ConvCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, kernel_size, dropout=0., bias=True,
                 is_agent_aware=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.nheads = num_heads
        self.dropout = dropout
        self.h_emb_dim = embed_dim // num_heads
        self.is_agent_aware = is_agent_aware
        if self.is_agent_aware:
            self.kv_dim = 3
            self.q_dim = 2
        else:
            self.kv_dim = 2
            self.q_dim = 1
        self.conv_kv_in = nn.Conv2d(embed_dim, self.kv_dim*embed_dim, kernel_size=kernel_size,
                                    stride=1, padding=kernel_size//2)
        self.conv_q_in = nn.Conv2d(embed_dim, self.q_dim*embed_dim, kernel_size=kernel_size,
                                    stride=1, padding=kernel_size//2)
        self.conv_out = nn.Conv2d(embed_dim, embed_dim, kernel_size=kernel_size,
                                  stride=1, padding=kernel_size//2)

        #TODO: reset params

    def forward(self, query, key_val, attn_mask=None, agent_aware_mask=None):
        """
        Args:
            inp: should be [b, seq, c, h, w]
            attn_mask: if not None, is [b, seq, seq]
        """
        b, seq, c, h, w = query.shape
        query = query.view(b*seq, c, h, w)
        q = self.conv_q_in(query)
        q = q.view(
            b, seq, self.q_dim*self.h_emb_dim, self.nheads*h*w
        ).permute(0,3,1,2).view(b*self.nheads, h*w, seq, self.q_dim*self.h_emb_dim)
        seq2 = key_val.size(1)
        key_val = key_val.view(b*seq2, c, h, w)
        kv = self.conv_kv_in(key_val)
        kv = kv.view(
            b, seq2, self.kv_dim*self.h_emb_dim, self.nheads*h*w
        ).permute(0,3,1,2).view(b*self.nheads, h*w, seq2, self.kv_dim*self.h_emb_dim)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                correct_2d_size = (seq, seq2)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (b*self.nheads, seq, seq2)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            attn_mask = attn_mask.unsqueeze(1)

        if self.is_agent_aware:
            if b > 1:
                raise NotImplementedError()
            agent_aware_mask = agent_aware_mask.unsqueeze(1).float()
            q_same, q_other = q.split(self.h_emb_dim, dim=3)
            k_same, k_other, v = kv.split(self.h_emb_dim, dim=3)
            output = dot_product_agent_aware_attn(k_same, k_other, q_same, q_other, v, agent_aware_mask,
                                                  attn_mask, self.dropout, self.training)
        else:
            k, v = kv.split(self.h_emb_dim, dim=3)
            output = dot_product_attn(k, q, v, attn_mask, self.dropout, self.training)
        output = output.reshape(b, self.nheads*h*w, seq, self.h_emb_dim)
        output = output.permute(0, 2, 3, 1).reshape(b*seq, c, h, w)
        output = self.conv_out(output).reshape(b, seq, c, h, w)
        return output


class ConvCrossAttention_AllSpatial(nn.Module):
    def __init__(self, embed_dim, num_heads, kernel_size, dropout=0., bias=True,
                 is_agent_aware=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.nheads = num_heads
        self.dropout = dropout
        self.h_emb_dim = embed_dim // num_heads
        self.is_agent_aware = is_agent_aware
        if self.is_agent_aware:
            self.kv_dim = 3
            self.q_dim = 2
        else:
            self.kv_dim = 2
            self.q_dim = 1
        self.conv_kv_in = nn.Conv2d(embed_dim, self.kv_dim*embed_dim, kernel_size=kernel_size,
                                    stride=1, padding=kernel_size//2)
        self.conv_q_in = nn.Conv2d(embed_dim, self.q_dim*embed_dim, kernel_size=kernel_size,
                                    stride=1, padding=kernel_size//2)
        self.conv_out = nn.Conv2d(embed_dim, embed_dim, kernel_size=kernel_size,
                                  stride=1, padding=kernel_size//2)


    def forward(self, query, key_val, attn_mask=None, agent_aware_mask=None):
        """
        Args:
            inp: should be [b, seq, c, h, w]
            attn_mask: if not None, is [b, seq, seq]
        """
        b, seq, c, h, w = query.shape
        query = query.view(b*seq, c, h, w)
        q = self.conv_q_in(query)
        q = q.view(
            b, seq, self.q_dim*self.h_emb_dim, self.nheads, h*w
        ).permute(0,3,1,2,4).reshape(b*self.nheads, seq, self.q_dim*self.h_emb_dim*h*w)
        seq2 = key_val.size(1)
        key_val = key_val.view(b*seq2, c, h, w)
        kv = self.conv_kv_in(key_val)
        kv = kv.view(
            b, seq2, self.kv_dim*self.h_emb_dim, self.nheads, h*w
        ).permute(0,3,1,2,4).reshape(b*self.nheads,seq2, self.kv_dim*self.h_emb_dim*h*w)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                correct_2d_size = (seq, seq2)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (b*self.nheads, seq, seq2)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask

        if self.is_agent_aware:
            if b > 1:
                raise NotImplementedError()
            agent_aware_mask = agent_aware_mask.float()
            q_same, q_other = q.split(self.h_emb_dim*h*w, dim=2)
            k_same, k_other, v = kv.split(self.h_emb_dim*h*w, dim=2)
            output = dot_product_agent_aware_attn(k_same, k_other, q_same, q_other, v, agent_aware_mask,
                                                  attn_mask, self.dropout, self.training)
        else:
            k, v = kv.split(self.h_emb_dim*h*w, dim=2)
            output = dot_product_attn(k, q, v, attn_mask, self.dropout, self.training)
        output = output.reshape(b, self.nheads, seq, self.h_emb_dim, h, w)
        output = output.transpose(1, 2).reshape(b*seq, c, h, w)
        output = self.conv_out(output).reshape(b, seq, c, h, w)
        return output




def dot_product_attn(k, q, v, attn_mask, dropout, training):
    q = q / math.sqrt(q.size(-1))
    attn = q @ k.transpose(-2, -1)
    if attn_mask is not None:
        attn += attn_mask
    attn = torch.softmax(attn, dim=-1)
    if dropout > 0.0 and training:
        attn = F.dropout(attn, p=dropout)
    output = attn @ v
    return output


def dot_product_agent_aware_attn(k_same, k_other, q_same, q_other, v,
                                 agent_aware_mask, attn_mask, dropout, training):
    q_same = q_same / math.sqrt(q_same.size(-1))
    q_other = q_other / math.sqrt(q_other.size(-1))
    attn_same = q_same @ k_same.transpose(-2, -1)
    attn_other = q_other @ k_other.transpose(-2, -1)
    agent_aware_mask = agent_aware_mask.float()
    if attn_mask is not None:
        attn_same += attn_mask
        attn_other += attn_mask
    attn_same = torch.softmax(attn_same, dim=-1)
    attn_other = torch.softmax(attn_other, dim=-1)
    attn = agent_aware_mask*attn_same + (1-agent_aware_mask)*attn_other
    if dropout > 0.0 and training:
        attn = F.dropout(attn, p=dropout)
    output = attn @ v
    return output