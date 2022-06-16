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


class MultiheadDifferenceGeneralAttention(Module):
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
                 is_agent_aware=False, use_no_match_emb=False, key_mode=None,
                 value_mode=None):
        super(MultiheadDifferenceGeneralAttention, self).__init__()
        assert attention_type in ['dot_product', 'additive', 'dot_product_difference']
        assert key_mode in ['standard', 'diff_only', 'diff_concat', 'diff_concat_full']
        assert value_mode in ['standard', 'diff_only', 'diff_concat', 'diff_concat_full']
        self.embed_dim = embed_dim
        self.key_mode = key_mode
        self.value_mode = value_mode
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.attention_type = attention_type
        self.aggregation_type = aggregation_type
        self.is_agent_aware = is_agent_aware
        self.use_no_match_emb = use_no_match_emb
        use_key_diff = key_mode in ['diff_only', 'diff_concat', 'diff_concat_full']
        use_value_diff = value_mode in ['diff_only', 'diff_concat', 'diff_concat_full']
        if value_mode == 'diff_concat':
            q_val_size = embed_dim // 2
        else:
            q_val_size = embed_dim
        if value_mode == 'diff_concat_full':
            v_size = 2*embed_dim
        else:
            v_size = embed_dim
        if key_mode == 'diff_concat_full':
            qk_size = 2*embed_dim
        else:
            qk_size = embed_dim
        
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.temperature = temperature
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            if self.is_agent_aware:
                raise NotImplementedError()
            self.q_proj_weight = Parameter(torch.Tensor(2*embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(2*embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(2*embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            if self.is_agent_aware:
                inp_size = 4*qk_size + v_size
                if use_key_diff:
                    inp_size += 2*qk_size
                if use_value_diff:
                    inp_size += q_val_size
            else:
                inp_size = 2*qk_size + v_size
                if use_key_diff:
                    inp_size += qk_size
                if use_value_diff:
                    inp_size += q_val_size
            self.in_proj_weight = Parameter(torch.empty(inp_size, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(inp_size))
        else:
            self.register_parameter('in_proj_bias', None)
        if self.aggregation_type == 'difference_concat' or value_mode == 'diff_concat_full':
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
        if self.use_no_match_emb:
            self.no_match_emb = Parameter(torch.empty(1, embed_dim))
        else:
            self.no_match_emb = None

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
        if self.no_match_emb is not None:
            xavier_uniform_(self.no_match_emb)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadDifferenceGeneralAttention, self).__setstate__(state)

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
            return multi_head_difference_attention_forward(
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
                precomputed_attention=precomputed_attention,
                no_match_emb=self.no_match_emb, key_mode=self.key_mode,
                value_mode=self.value_mode)
        else:
            if not self.is_agent_aware:
                agent_aware_mask = None
            return multi_head_difference_attention_forward(
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
                agent_aware_mask=agent_aware_mask,
                no_match_emb=self.no_match_emb, key_mode=self.key_mode,
                value_mode=self.value_mode)


def multi_head_difference_attention_forward(
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
    no_match_emb: Optional[Tensor] = None,
    use_key_diff = False,
    key_mode = None,
    value_mode = None,
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
    assert attention_type in ['dot_product', 'additive']
    assert aggregation_type in ['normal', 'difference', 'difference_concat']
    assert key_mode in ['standard', 'diff_only', 'diff_concat', 'diff_concat_full']
    assert value_mode in ['standard', 'diff_only', 'diff_concat', 'diff_concat_full']

    use_key_diff = key_mode in ['diff_only', 'diff_concat', 'diff_concat_full']
    use_value_diff = value_mode in ['diff_only', 'diff_concat', 'diff_concat_full']
    tgt_len, bsz, embed_dim = query.size()
    src_len, _, _ = key.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5
    orig_query_size = query.size(0)
    orig_key_size = key.size(0)

    if value_mode == 'diff_concat':
        q_val_size = embed_dim // 2
        v_size = embed_dim
    elif value_mode == 'diff_concat_full':
        q_val_size = embed_dim
        v_size = 2*embed_dim
    else:
        q_val_size = v_size = embed_dim
    q_val_head_dim = q_val_size // num_heads
    v_head_dim = v_size // num_heads

    if key_mode == 'diff_concat_full':
        qk_size = 2*embed_dim
        qk_head_dim = qk_size // num_heads
    else:
        qk_size = embed_dim
        qk_head_dim = head_dim

    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
            # self-attention
            if add_zero_attn:
                if no_match_emb is not None:
                    no_match_emb = no_match_emb.reshape(1, 1, embed_dim).expand(-1, bsz, -1)
                    query = torch.cat([query, no_match_emb], dim=0)
                else:
                    query = torch.cat([query, torch.zeros(1, bsz, embed_dim, device=query.device, dtype=query.dtype)], dim=0)
            if agent_aware_mask is not None:
                if use_key_diff and use_value_diff:
                    q_diff, q_diff_same, q_val, k_diff, k_diff_same, k_mult, k_mult_same, v = linear(query, in_proj_weight, in_proj_bias).split(
                        [qk_size, qk_size, q_val_size, qk_size, qk_size, qk_size, qk_size, v_size], dim=-1
                    )
                elif use_key_diff:
                    #q_diff, q_diff_same, k_diff, k_diff_same, k_mult, k_mult_same, v = linear(query, in_proj_weight, in_proj_bias).chunk(7, dim=-1)
                    q_diff, q_diff_same, k_diff, k_diff_same, k_mult, k_mult_same, v = linear(query, in_proj_weight, in_proj_bias).split(
                        [qk_size, qk_size, qk_size, qk_size, qk_size, qk_size, v_size], dim=-1
                    )

                elif use_value_diff:
                    q_diff, q_diff_same, q_val, k_mult, k_mult_same, v = linear(query, in_proj_weight, in_proj_bias).split(
                        [qk_size, qk_size, qk_size, qk_size, qk_size, v_size], dim=-1
                    )
                else:
                    #q_diff, q_diff_same, k_mult, k_mult_same, v = linear(query, in_proj_weight, in_proj_bias).chunk(5, dim=-1)
                    q_diff, q_diff_same, k_mult, k_mult_same, v = linear(query, in_proj_weight, in_proj_bias).split(
                        [embed_dim, embed_dim, embed_dim, embed_dim, v_size], dim=-1
                    )
            else:
                if use_key_diff and use_value_diff:
                    q_diff, q_val, k_diff, k_mult, v = linear(query, in_proj_weight, in_proj_bias).split(
                        [qk_size, q_val_size, qk_size, qk_size, v_size], dim=-1
                    )
                elif use_key_diff:
                    #q_diff, k_diff, k_mult, v = linear(query, in_proj_weight, in_proj_bias).chunk(4, dim=-1)
                    q_diff, k_diff, k_mult, v = linear(query, in_proj_weight, in_proj_bias).split(
                        [qk_size, qk_size, qk_size, v_size], dim=-1)
                elif use_value_diff:
                    q_diff, q_val, k_mult, v = linear(query, in_proj_weight, in_proj_bias).split(
                        [qk_size, q_val_size, qk_size, v_size], dim=-1
                    )
                else:
                    q_diff, k_mult, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

            if add_zero_attn:
                q = q[:-1]
                k = k[:-1]
                v = v[:-1]
                if no_match_emb is None:
                    v = torch.cat([v[:, :-1], torch.zeros(tgt_len, 1, bsz, embed_dim, device=query.device, dtype=query.dtype)], dim=1)
                src_len += 1
                

        elif key is value or torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            if add_zero_attn:
                if no_match_emb is not None:
                    no_match_emb = no_match_emb.reshape(1, 1, embed_dim).expand(-1, bsz, -1)
                    key = torch.cat([key, no_match_emb], dim=0)
                else:
                    key = torch.cat([key, torch.zeros(1, bsz, embed_dim, device=key.device, dtype=key.dtype)], dim=0)
                src_len += 1
            _b = in_proj_bias
            _start = 0
            incr = 1
            if agent_aware_mask is not None:
                incr += 1
            if use_value_diff:
                if value_mode in ['diff_concat_full', 'diff_concat']:
                    incr += 0.5
                else:
                    incr += 1
            _end = int(incr*qk_size)

            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)
            if agent_aware_mask is not None:
                if use_value_diff:
                    q_diff, q_diff_same, q_val = q.split([qk_size, qk_size, q_val_size], dim=-1)
                else:
                    q_diff, q_diff_same = q.chunk(2, dim=-1)
            elif use_value_diff:
                q_diff, q_val = q.split([qk_size, q_val_size], dim=-1)
            else:
                q_diff = q 

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
                    if use_key_diff:
                        #k_diff, k_diff_same, k_mult, k_mult_same, v = linear(key, _w, _b).chunk(5, dim=-1)
                        k_diff, k_diff_same, k_mult, k_mult_same, v = linear(key, _w, _b).split(
                            [qk_size, qk_size, qk_size, qk_size, v_size], dim=-1)
                    else:
                        k_mult, k_mult_same, v = linear(key, _w, _b).chunk(3, dim=-1)
                elif use_key_diff:
                    #k_diff, k_mult, v = linear(key, _w, _b).chunk(3, dim=-1)
                    k_diff, k_mult, v = linear(key, _w, _b).split([qk_size, qk_size, v_size], dim=-1)
                else:
                    k_mult, v = linear(key, _w, _b).chunk(2, dim=-1)
                if add_zero_attn and no_match_emb is None:
                    v = torch.cat([v[:, :-1], torch.zeros(tgt_len, 1, bsz, embed_dim, device=query.device, dtype=query.dtype)], dim=1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            incr = 1
            if agent_aware_mask is not None:
                incr += 1
            if use_value_diff:
                incr += 1
            _end = incr*qk_size
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)
            if agent_aware_mask is not None:
                if use_value_diff:
                    q_diff, q_diff_same, q_val = q.split([head_dim, head_dim, q_val_size], dim=-1)
                else:
                    q_diff, q_diff_same = q.chunk(2, dim=-1)
            elif use_value_diff:
                q_diff, q_val = q.split([head_dim, q_val_size], dim=-1)
            else:
                q_diff = q

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = _end
            incr = 1
            if agent_aware_mask is not None and use_key_diff:
                incr += 3
            else:
                if agent_aware_mask is not None:
                    incr += 1
                if use_key_diff:
                    incr += 1
            _end = _start + incr*embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)
            if agent_aware_mask is not None:
                if use_key_diff:
                    k_diff, k_diff_same, k_mult, k_mult_same = k.chunk(4, dim=-1)
                else:
                    k_mult, k_mult_same = k.chunk(2, dim=-1)
            elif use_key_diff:
                k_diff, k_mult = k.chunk(2, dim=-1)
            else:
                k_mult = k

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
            if add_zero_attn and no_match_emb is None:
                v = torch.cat([v[:, :-1], torch.zeros(tgt_len, 1, bsz, embed_dim, device=query.device, dtype=query.dtype)], dim=1)
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
    #q = q * scaling
    k_mult = k_mult * scaling
    if agent_aware_mask is not None:
        k_mult = k_mult * scaling


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
            if list(attn_mask.size()) != [1, orig_query_size, orig_key_size]:
                raise RuntimeError("The size of the 2D attn_mask is not correct.")
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, orig_query_size, orig_key_size]:
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
    q_diff = q_diff.contiguous().view(tgt_len, bsz * num_heads, qk_head_dim).transpose(0, 1)
    k_mult = k_mult.contiguous().view(-1, bsz * num_heads, qk_head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, v_head_dim).transpose(0, 1)
    if agent_aware_mask is not None:
        q_diff_same = q_diff_same.contiguous().view(tgt_len, bsz * num_heads, qk_head_dim).transpose(0, 1)
        k_mult_same = k_mult_same.contiguous().view(-1, bsz * num_heads, qk_head_dim).transpose(0, 1)
    if use_value_diff:
        q_val = q_val.contiguous().view(tgt_len, bsz * num_heads, q_val_head_dim).transpose(0, 1)
    if use_key_diff:
        k_diff = k_diff.contiguous().view(-1, bsz * num_heads, qk_head_dim).transpose(0, 1)
        if agent_aware_mask is not None:
            k_diff_same = k_diff_same.contiguous().view(-1, bsz * num_heads, qk_head_dim).transpose(0, 1)
        
    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k_mult.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        if agent_aware_mask is not None:
            raise NotImplementedError()
        #src_len += 1
        #if no_match_emb is not None:
        #    k = torch.cat([k, no_match_emb.reshape(-1)])
        #else:
        #    k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        #v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    if precomputed_attention is None:
        if attention_type == 'dot_product':
            #attn_output_weights = torch.bmm(q, k.transpose(1, 2))
            #attn_output_weights =  (q * k).sum(-1)
            if key_mode == 'standard':
                attn_output_weights = torch.bmm(q_diff, k_mult.transpose(1, 2))
            elif key_mode == 'diff_only':
                #    attn_output_weights = ((q_diff.unsqueeze(2)-k_diff.unsqueeze(1))*k_mult.unsqueeze(1)).sum(-1)
                attn_output_weights = q_diff @ k_mult.transpose(-1, -2) - ((k_diff*k_mult).sum(-1)).unsqueeze(1)
            else:
                l = q_diff.size(-1) // 2
                attn_output_weights = torch.bmm(q_diff[:, :, :l], k_mult[:, :, :l].transpose(1,2))
                #attn_output_weights = attn_output_weights + \
                #    ((q_diff[:, :, l:].unsqueeze(2)-k_diff[:, :, l:].unsqueeze(1))*k_mult[:, :, l:].unsqueeze(1)).sum(-1)
                attn_output_weights = attn_output_weights + \
                     (q_diff[:, :, l:] @ k_mult[:, :, l:].transpose(-1, -2)) - ((k_diff[:, :, l:]*k_mult[:, :, l:]).sum(-1)).unsqueeze(1)
                    
            if agent_aware_mask is not None:
                agent_aware_mask = agent_aware_mask.unsqueeze(1).expand(
                    -1, num_heads, -1, -1
                ).reshape(bsz*num_heads, *agent_aware_mask.shape[1:]).float()
                if key_mode == 'standard':
                    aa_attn = torch.bmm(q_diff_same, k_mult_same.transpose(1, 2))
                elif key_mode == 'diff_only':
                    #aa_attn = ((q_diff_same.unsqueeze(2)-k_diff_same.unsqueeze(1))*k_mult_same.unsqueeze(1)).sum(-1)
                    aa_attn = q_diff_same @ k_mult_same.transpose(-1, -2) - ((k_diff_same*k_mult_same).sum(-1)).unsqueeze(1)
                else:
                    l = q_diff.size(-1) // 2
                    aa_attn = torch.bmm(q_diff_same[:, :, :l], k_mult_same[:, :, :l].transpose(1,2))
                    #aa_attn = aa_attn + \
                    #    ((q_diff_same[:, :, l:].unsqueeze(2)-k_diff_same[:, :, l:].unsqueeze(1))*k_mult_same[:, :, l:].unsqueeze(1)).sum(-1)
                    aa_attn = aa_attn + \
                        (q_diff_same[:, :, l:] @ k_mult_same[:, :, l:].transpose(-1, -2)) - ((k_diff_same[:, :, l:]*k_mult_same[:, :, l:]).sum(-1)).unsqueeze(1)
                    
                attn_output_weights = (1-agent_aware_mask)*attn_output_weights + \
                    agent_aware_mask * aa_attn
        else:
            raise NotImplementedError()
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

    if value_mode == 'standard':
        attn_output = torch.bmm(attn_output_weights, v)
    elif value_mode == 'diff_only':
        #attn_output = (attn_output_weights.unsqueeze(-1) * (q_val.unsqueeze(2)-v.unsqueeze(1))).sum(-2)
        attn_output = attn_output_weights.sum(-1,keepdim=True)*q_val - attn_output_weights @ v
    else:
        l = v.size(-1) // 2
        #attn_output = torch.cat([
        #    torch.bmm(attn_output_weights, v[:, :, :l]),
        #    (attn_output_weights.unsqueeze(-1) * (q_val.unsqueeze(2)-v[:, :, l:].unsqueeze(1))).sum(-2)
        #], dim=-1)

        attn_output = torch.cat([
            torch.bmm(attn_output_weights, v[:, :, :l]),
            attn_output_weights.sum(-1,keepdim=True) * q_val - attn_output_weights @ v[:, :, l:]
        ], dim=-1)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, v_head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, v_size)
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