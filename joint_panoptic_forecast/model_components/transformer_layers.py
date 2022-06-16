import math
import torch
import warnings
from torch import Tensor
from torch.nn.functional import pad, linear, softmax, dropout
from torch.nn import Module, Parameter, Linear
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_
from typing import Optional, Tuple, List

# This whole file exists because, by default, MHA returns the average attention
# for each individual head, which...???????


# This class exists solely for Transformer; it has an annotation stating
# that bias is never None, which appeases TorchScript
class _LinearWithBias(Linear):
    bias: Tensor  # type: ignore

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=True)  # type: ignore


class MultiheadAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.
    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.
    Note that if :attr:`kdim` and :attr:`vdim` are None, they will be set
    to :attr:`embed_dim` such that query, key, and value have the same
    number of features.
    Examples::
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None,
                 temperature=1.0, attention_type='dot_product', aggregation_type='normal',
                 is_agent_aware=False, use_no_match_emb=False):
        super(MultiheadAttention, self).__init__()
        assert attention_type in ['dot_product', 'additive', 'dot_product_difference']
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.attention_type = attention_type
        self.aggregation_type = aggregation_type
        self.is_agent_aware = is_agent_aware
        if use_no_match_emb:
            raise NotImplementedError()

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.temperature = temperature
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            if self.attention_type == 'dot_product_difference':
                raise NotImplementedError()
            if self.is_agent_aware:
                raise NotImplementedError()
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            if self.is_agent_aware:
                inp_size = 5*embed_dim
                if attention_type == 'dot_product_difference':
                    inp_size += 2*embed_dim
            else:
                inp_size = 3*embed_dim
                if attention_type == 'dot_product_difference':
                    inp_size += embed_dim
            self.in_proj_weight = Parameter(torch.empty(inp_size, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(inp_size))
        else:
            self.register_parameter('in_proj_bias', None)
        if self.aggregation_type == 'difference_concat':
            out_proj_in = 2*embed_dim
        else:
            out_proj_in = embed_dim
        self.out_proj = _LinearWithBias(out_proj_in, embed_dim)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        if self.attention_type == 'additive':
            head_dim = embed_dim // num_heads
            self.additive_k_weight = Parameter(torch.empty(head_dim, head_dim))
            self.additive_q_weight = Parameter(torch.empty(head_dim, head_dim))
            self.additive_v_weight = Parameter(torch.empty(head_dim))
        else:
            self.additive_k_weight = self.additive_q_weight = self.additive_v_weight = None

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.additive_k_weight is not None:
            xavier_uniform_(self.additive_k_weight)
            xavier_uniform_(self.additive_q_weight)
            self.additive_v_weight.data.uniform_(-0.1, 0.1)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                query_value: Optional[Tensor] = None, precomputed_attention: Optional[Tensor] = None,
                agent_aware_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
    Shapes for inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: if a 2D mask: :math:`(L, S)` where L is the target sequence length, S is the
          source sequence length.
          If a 3D mask: :math:`(N\cdot\text{num\_heads}, L, S)` where N is the batch size, L is the target sequence
          length, S is the source sequence length. ``attn_mask`` ensure that position i is allowed to attend
          the unmasked positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
    Shapes for outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, temperature=self.temperature,
                attention_type=self.attention_type, additive_k=self.additive_k_weight,
                additive_q=self.additive_q_weight, additive_v=self.additive_v_weight,
                aggregation_type=self.aggregation_type, query_value=query_value,
                precomputed_attention=precomputed_attention)
        else:
            if not self.is_agent_aware:
                agent_aware_mask = None
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, temperature=self.temperature,
                attention_type=self.attention_type, additive_k=self.additive_k_weight,
                additive_q=self.additive_q_weight, additive_v=self.additive_v_weight,
                aggregation_type=self.aggregation_type, query_value=query_value,
                precomputed_attention=precomputed_attention,
                agent_aware_mask=agent_aware_mask)


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
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
    temperature: Optional[int] = 1.0,
    attention_type: Optional[str] = 'dot_product',
    additive_k: Optional[Tensor] = None,
    additive_q: Optional[Tensor] = None,
    additive_v: Optional[Tensor] = None,
    aggregation_type: Optional[str] = 'normal',
    query_value: Optional[Tensor] = None,
    precomputed_attention: Optional[Tensor] = None,
    agent_aware_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    assert attention_type in ['dot_product', 'additive', 'dot_product_difference']
    assert aggregation_type in ['normal', 'difference', 'difference_concat']
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
            # self-attention
            if agent_aware_mask is not None:
                if attention_type == 'dot_product_difference':
                    q, k, k_diff, v, q_same, k_same, k_diff_same = linear(query, in_proj_weight, in_proj_bias).chunk(7, dim=-1)
                    k = torch.cat([k, k_diff], dim=-1)
                    k_same = torch.cat([k_same, k_diff_same], dim=-1)
                else:
                    q, k, v, q_same, k_same = linear(query, in_proj_weight, in_proj_bias).chunk(5, dim=-1)
            else:
                if attention_type == 'dot_product_difference':
                    q, k, k_diff, v = linear(query, in_proj_weight, in_proj_bias).chunk(4, dim=-1)
                    k = torch.cat([k, k_diff], dim=-1)
                else:
                    q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif key is value or torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            if agent_aware_mask is not None:
                _end = 2*embed_dim
            else:
                _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)
            if agent_aware_mask is not None:
                q, q_same = q.chunk(2, dim=-1)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = _end
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                if agent_aware_mask is not None:
                    if attention_type == 'dot_product_difference':
                        k, k_diff, k_same, k_diff_same, v = linear(key, _w, _b).chunk(5, dim=-1)
                        k = torch.cat([k, k_diff], dim=-1)
                        k_same = torch.cat([k_same, k_diff_same], dim=-1)
                    else:
                        k, k_same, v = linear(key, _w, _b).chunk(3, dim=-1)
                else:
                    if attention_type == 'dot_product_difference':
                        k, k_diff, v = linear(key, _w, _b).chunk(3, dim=-1)
                        k = torch.cat([k, k_diff], dim=-1)
                    else:
                        k, v = linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            if agent_aware_mask is not None:
                _end = 2*embed_dim
            else:
                _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)
            if agent_aware_mask is not None:
                q, q_same = q.chunk(2, dim=-1)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = _end
            if agent_aware_mask is not None:
            #    #_start = 2*embed_dim
            #    #_end = embed_dim * 4
                if attention_type == 'dot_product_difference':
                    _end = _start + 4*embed_dim
                else:
                    _end = _start + 2*embed_dim
            else:
            #    _start = embed_dim
            #    _end = embed_dim * 2
                if attention_type == 'dot_product_difference':
                    _end = _start + 2*embed_dim
                else:
                    _end = _start + embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)
            if agent_aware_mask is not None:
                k, k_same = k.chunk(2, dim=-1)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = _end
            #if agent_aware_mask is not None:
            #    _start = embed_dim * 4
            #else:
            #    _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = linear(value, _w, _b)
    else:
        if agent_aware_mask is not None or attention_type == 'dot_product_difference':
            raise NotImplementedError()
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim : (embed_dim * 2)])
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2) :])
        else:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling
    if agent_aware_mask is not None:
        q_same = q_same * scaling

    if attn_mask is not None:
        assert (
            attn_mask.dtype == torch.float32
            or attn_mask.dtype == torch.float64
            or attn_mask.dtype == torch.float16
            or attn_mask.dtype == torch.uint8
            or attn_mask.dtype == torch.bool
        ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 2D attn_mask is not correct.")
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 3D attn_mask is not correct.")
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
        )
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if agent_aware_mask is not None:
            raise NotImplementedError()
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None
    if attention_type == 'dot_product_difference':
        k_size = 2*head_dim
    else:
        k_size = head_dim
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, k_size).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if agent_aware_mask is not None:
        q_same = q_same.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k_same = k_same.contiguous().view(tgt_len, bsz * num_heads, k_size).transpose(0, 1)
        


    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        if agent_aware_mask is not None:
            raise NotImplementedError()
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    if precomputed_attention is None:
        if attention_type == 'dot_product':
            attn_output_weights = torch.bmm(q, k.transpose(1, 2))
            if agent_aware_mask is not None:
                agent_aware_mask = agent_aware_mask.unsqueeze(1).expand(
                    -1, num_heads, -1, -1
                ).reshape(bsz*num_heads, *agent_aware_mask.shape[1:]).float()
                attn_output_weights = (1-agent_aware_mask)*attn_output_weights + \
                    agent_aware_mask * torch.bmm(q_same, k_same.transpose(1,2))
        elif attention_type == 'dot_product_difference':
            k, k_diff = k.chunk(2, dim=-1)
            q_inp = q.unsqueeze(2) - k_diff.unsqueeze(1)
            attn_output_weights = (q_inp * k.unsqueeze(1)).sum(-1)
            if agent_aware_mask is not None:
                raise NotImplementedError()

        elif attention_type == 'additive':
            q = linear(q, additive_q, None)
            k = linear(k, additive_k, None)
            attn_output_weights = torch.tanh(q.unsqueeze(2) + k.unsqueeze(1)) @ additive_v
        assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

        attn_output_weights = softmax(attn_output_weights/temperature, dim=-1)
        attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)
    else:
        attn_output_weights = precomputed_attention.reshape(bsz*num_heads, tgt_len, src_len)

    #if aggregation_type == 'normal':
    attn_output = torch.bmm(attn_output_weights, v)
    #elif aggregation_type == 'difference':
    #    tmp_v = v.unsqueeze(1) - v.unsqueeze(2)
    #    attn_output = (attn_output_weights.unsqueeze(-1) * tmp_v).sum(2)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    if aggregation_type == 'difference':
        attn_output = query_value - attn_output
    elif aggregation_type == 'difference_concat':
        attn_output = torch.cat([attn_output, query_value], dim=-1)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        #return attn_output, attn_output_weights.sum(dim=1) / num_heads #WHY WOULD THEY DO THIS?????
        return attn_output, attn_output_weights
    else:
        return attn_output, None