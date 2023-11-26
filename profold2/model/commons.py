"""A lot of modules in Alphafold2
  """
import functools

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from profold2.model import functional, profiler
from profold2.utils import default, exists


# helpers
def init_zero_(layer):
  nn.init.constant_(layer.weight, 0.)
  if exists(layer.bias):
    nn.init.constant_(layer.bias, 0.)

def default_list_get(value, n):
  if isinstance(value, (tuple, list)):
    assert len(value) <= n
    value = list(value)
    while len(value) < n:
      value = value + [value[-1]]
    return value
  return [value] * n

def embedd_dim_get(dim):
  if isinstance(dim, (tuple, list)):
    assert len(dim) == 2  # (dim_single, dim_pairwise)
    return dim
  return (dim, dim)

def embedd_dropout_get(p):
  if isinstance(p, (tuple, list)):
    assert len(p) == 2  # (p_single, p_pairwise)
    return p
  return (p, p)

def shared_dropout(x, p, broadcast_dim=None, training=True):
  if exists(broadcast_dim) and 0 < p < 1.0:
    if training:
      shape = list(x.shape)
      assert len(shape) == 4  # (b m i d)
      assert broadcast_dim in (0, 1)  # (# shared across rows and columns)
      shape[broadcast_dim + 1] = 1
      m = torch.bernoulli(torch.full(shape, 1 - p, device=x.device))
      return m * x / (1 - p)
    return x
  return F.dropout(x, p=p, training=training)

# helper classes
class Always(nn.Module):

  def __init__(self, val):
    super().__init__()
    self.val = val

  def forward(self, x):
    del x
    return self.val


# feed forward
class GEGLU(nn.Module):
  """Gated GELU"""
  def forward(self, x):
    x, gates = x.chunk(2, dim=-1)
    return x * F.gelu(gates)


class FeedForward(nn.Module):
  """FeedForward layer in transformer
    """
  def __init__(self, dim, mult=4, dropout=0.):
    super().__init__()
    self.norm = nn.LayerNorm(dim)

    self.net = nn.Sequential(nn.Linear(dim, dim * mult * 2), GEGLU(),
                             nn.Dropout(dropout), nn.Linear(dim * mult, dim))
    init_zero_(self.net[-1])

  def forward(self, x):
    x = self.norm(x)
    return self.net(x)


class Attention(nn.Module):
  """Multi-head Attention
    """
  def __init__(self,
               dim_q,
               dim_kv,
               heads=8,
               dim_head=64,
               dropout=0.,
               gating=True,
               global_query_attn=False):
    super().__init__()
    self.heads = heads
    self.dim_head = dim_head
    self.scale = dim_head**-0.5

    dim_inner = dim_head * heads
    self.to_q = nn.Linear(dim_q, dim_inner, bias=False)
    if global_query_attn:
      self.to_kv = nn.Linear(dim_kv, dim_head * 2, bias=False)
    else:
      self.to_kv = nn.Linear(dim_kv, dim_inner * 2, bias=False)
    self.to_out = nn.Linear(dim_inner, dim_q)

    self.gating = nn.Linear(dim_kv, dim_inner) if gating else None
    if exists(self.gating):
      nn.init.constant_(self.gating.weight, 0.)
      nn.init.constant_(self.gating.bias, 1.)

    self.dropout = nn.Dropout(dropout)
    self.global_query_attn = global_query_attn
    init_zero_(self.to_out)

  def forward(self,
              x,
              mask=None,
              attn_bias=None,
              context=None,
              context_mask=None):
    device, h, d = x.device, self.heads, self.dim_head

    m = default(context, x)

    q, k, v = (self.to_q(x), *self.to_kv(m).chunk(2, dim=-1))

    b, i, n = q.shape[0], q.shape[-2], k.shape[-2]

    q, k, v = map(lambda t: rearrange(t, '... i (h d) -> ... h i d', d=d),
                  (q, k, v))


    # query / key similarities
    if self.global_query_attn:
      # as in the paper, for the extra MSAs
      # they average the queries along the rows of the MSAs
      # they named this particular module MSAColumnGlobalAttention

      k, v = map(lambda t: repeat(t, '... r i d-> ... (r h) i d', h=h), (k, v))
      if exists(mask):
        q = torch.sum(q * mask[..., None, :, None],
                      dim=-2) / (torch.sum(mask[..., None, :, None], dim=-2) + 1e-10)
      else:
        q = q.mean(dim=-2)
      q = repeat(q, '... h d -> ... h i d', i=n)

    # masking
    attn_mask, mask_value = None, -torch.finfo(q.dtype).max
    if exists(mask):
      mask = default(mask, lambda: torch.ones(b, i, device=device))
      context_mask = mask if not exists(context) else default(
          context_mask,
          lambda: torch.ones(b, k.shape[-2], device=device))
      attn_mask = rearrange(mask.bool(), '... i -> ... () i ()') * rearrange(
          context_mask.bool(), '... j -> ... () () j')
      assert attn_mask.dtype == torch.bool
      
    # pytorch 2.0+
    if hasattr(F, 'scaled_dot_product_attention') and (
          not exists(attn_bias) or not self.training):
      if exists(attn_mask) and exists(attn_bias):
        attn_mask = attn_bias.masked_fill(~attn_mask, mask_value)
      elif exists(attn_bias):
        attn_mask = attn_bias
      # See: https://github.com/pytorch/pytorch/issues/96099
      dropout_p = self.dropout.p if self.training else 0.0
      out = F.scaled_dot_product_attention(
          q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
    else:
      # scale
      q = q * self.scale

      dots = torch.einsum('... h i d,... h j d -> ... h i j', q, k)

      # add attention bias,
      # if supplied (for pairwise to msa attention communication)

      if exists(attn_bias):
        dots = dots + attn_bias

      # masking
      if exists(attn_mask):
        dots = dots.masked_fill(~attn_mask, mask_value)

      # attention

      # dots = dots - dots.max(dim=-1, keepdim=True).values
      attn = F.softmax(dots, dim=-1)
      attn = self.dropout(attn)

      # aggregate
      out = torch.einsum('... h i j, ... h j d -> ... h i d', attn, v)

    # merge heads
    out = rearrange(out, '... h i d -> ... i (h d)')

    # gating
    if exists(self.gating):
      gates = self.gating(m)
      out = out * gates.sigmoid()

    # combine to out
    out = self.to_out(out)
    return out


class AxialAttention(nn.Module):
  """AxialAttention
    """
  def __init__(self,
               dim_node,
               dim_edge,
               heads,
               row_attn=True,
               col_attn=True,
               accept_edges=False,
               **kwargs):
    super().__init__()
    assert not (not row_attn and
                not col_attn), 'row or column attention must be turned on'

    self.row_attn = row_attn
    self.col_attn = col_attn

    self.norm = nn.LayerNorm(dim_node)
    self.attn = Attention(dim_q=dim_node,
                          dim_kv=dim_node,
                          heads=heads,
                          **kwargs)
    self.edges_to_attn_bias = nn.Sequential(
        nn.Linear(dim_edge, heads, bias=False),
        Rearrange('b i j h -> b h i j')) if accept_edges else None

  def forward(self, x, edges=None, mask=None, shard_size=None):
    assert self.row_attn ^ self.col_attn, 'has to be either row or column attention, but not both'  # pylint: disable=line-too-long

    x = self.norm(x)

    # axial attention
    if self.col_attn:
      axial_dim = 2
      mask_fold_axial_eq = 'b h w -> (b w) h'
      input_fold_eq = 'b h w d -> (b w) h d'
      output_fold_eq = '(b w) h d -> b h w d'

    elif self.row_attn:
      axial_dim = 1
      mask_fold_axial_eq = 'b h w -> (b h) w'
      input_fold_eq = 'b h w d -> (b h) w d'
      output_fold_eq = '(b h) w d -> b h w d'

    def run_attn(x, mask, attn_bias):
      _, h, w, _ = x.shape

      if exists(attn_bias):
        attn_bias = repeat(attn_bias,
                           'b h i j -> (b x) h i j',
                           x=x.shape[axial_dim])

      x = rearrange(x, input_fold_eq)
      if exists(mask):
        mask = rearrange(mask, mask_fold_axial_eq)
      out = self.attn(x, mask=mask, attn_bias=attn_bias)
      out = rearrange(out, output_fold_eq, h=h, w=w)
      return out

    attn_bias = None
    if exists(self.edges_to_attn_bias) and exists(edges):
      attn_bias = self.edges_to_attn_bias(edges)  # pylint: disable=not-callable
    if exists(attn_bias) and self.col_attn:
      attn_bias = rearrange(attn_bias, '... i j -> ... j i')

    return functional.sharded_apply(
        run_attn, [x, mask],
        attn_bias,
        shard_size=None if self.training else shard_size,
        shard_dim=axial_dim,
        cat_dim=axial_dim)


class TriangleMultiplicativeModule(nn.Module):
  """TriangleMultiplicative
    """
  def __init__(self, *, dim, hidden_dim=None, mix='ingoing'):
    super().__init__()
    assert mix in {'ingoing',
                   'outgoing'}, 'mix must be either ingoing or outgoing'

    hidden_dim = default(hidden_dim, dim)
    self.norm = nn.LayerNorm(dim)

    self.left_proj = nn.Linear(dim, hidden_dim)
    self.right_proj = nn.Linear(dim, hidden_dim)

    self.left_gate = nn.Linear(dim, hidden_dim)
    self.right_gate = nn.Linear(dim, hidden_dim)
    self.out_gate = nn.Linear(dim, dim)

    # initialize all gating to be identity

    for gate in (self.left_gate, self.right_gate, self.out_gate):
      nn.init.constant_(gate.weight, 0.)
      nn.init.constant_(gate.bias, 1.)

    if mix == 'outgoing':
      self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
    elif mix == 'ingoing':
      self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'

    self.to_out_norm = nn.LayerNorm(hidden_dim)
    self.to_out = nn.Linear(hidden_dim, dim)

  def forward(self, x, mask=None):
    assert x.shape[1] == x.shape[2], 'feature map must be symmetrical'
    if exists(mask):
      mask = rearrange(mask, 'b i j -> b i j ()')

    x = self.norm(x)

    left = self.left_proj(x)
    right = self.right_proj(x)

    if exists(mask):
      left = left * mask
      right = right * mask

    left_gate = self.left_gate(x).sigmoid()
    right_gate = self.right_gate(x).sigmoid()
    out_gate = self.out_gate(x).sigmoid()

    left = left * left_gate
    right = right * right_gate

    out = torch.einsum(self.mix_einsum_eq, left, right)

    out = self.to_out_norm(out)
    out = out * out_gate
    out = self.to_out(out)
    return out


# evoformer blocks


class OuterProductMean(nn.Module):
  """OuterProductMean
    """
  def __init__(self, dim_msa, dim_pairwise, dim_hidden=32, eps=1e-5):
    super().__init__()

    self.eps = eps
    self.norm = nn.LayerNorm(dim_msa)
    dim_hidden = default(dim_hidden, dim_pairwise)

    self.left_proj = nn.Linear(dim_msa, dim_hidden)
    self.right_proj = nn.Linear(dim_msa, dim_hidden)
    self.proj_out = nn.Linear(dim_hidden**2, dim_pairwise)

  def forward(self, x, mask=None, shard_size=None):
    x = self.norm(x)
    left = self.left_proj(x)
    right = self.right_proj(x)

    def run_outer_sum(left, right, mask):
      outer = rearrange(left, 'b m i c -> b m i () c ()') * rearrange(
          right, 'b m j d -> b m () j () d')
      if exists(mask):
        # masked mean, if there are padding in the rows of the MSA
        mask = rearrange(mask, 'b m i -> b m i () () ()') * rearrange(
            mask, 'b m j -> b m () j () ()') > 0
        outer = outer.masked_fill(~mask, 0.)
      outer = outer.sum(dim=1, keepdim=True)
      outer = self.proj_out(rearrange(outer, '... c d -> ... (c d)'))
      return outer

    def run_mask_sum(mask):
      # masked mean, if there are padding in the rows of the MSA
      mask = rearrange(mask, 'b m i -> b m i () ()') * rearrange(
          mask, 'b m j -> b m () j ()') > 0
      return mask.sum(dim=1, keepdim=True)

    def run_iter_sum(chunk_iter):
      return sum(chunk_iter)

    outer = functional.sharded_apply(
        run_outer_sum, [left, right, mask],
        shard_size=None if self.training else shard_size,
        shard_dim=1,
        cat_dim=run_iter_sum)
    if exists(mask):
      mask = functional.sharded_apply(
          run_mask_sum, [mask],
          shard_size=None if self.training else shard_size,
          shard_dim=1,
          cat_dim=run_iter_sum)
      outer = outer.sum(dim=1) / (mask.sum(dim=1) + self.eps)
    else:
      outer = outer.mean(dim=1)

    return outer


class PairwiseAttentionBlock(nn.Module):
  """PairwiseAttentionBlock
    """
  def __init__(
      self,
      dim_msa,
      dim_pairwise,
      heads,
      dim_head,
      dropout=0.,
      disabled_outer_mean=False,
      multiplication_first=True,
  ):
    super().__init__()
    self.multiplication_first = multiplication_first

    self.outer_mean = OuterProductMean(
        dim_msa, dim_pairwise) if not disabled_outer_mean else None
    self.triangle_attention_outgoing = AxialAttention(dim_node=dim_pairwise,
                                                      dim_edge=dim_pairwise,
                                                      heads=heads,
                                                      dim_head=dim_head,
                                                      row_attn=True,
                                                      col_attn=False,
                                                      accept_edges=True)
    self.triangle_attention_ingoing = AxialAttention(dim_node=dim_pairwise,
                                                     dim_edge=dim_pairwise,
                                                     heads=heads,
                                                     dim_head=dim_head,
                                                     row_attn=False,
                                                     col_attn=True,
                                                     accept_edges=True)
    self.triangle_multiply_outgoing = TriangleMultiplicativeModule(
        dim=dim_pairwise, mix='outgoing')
    self.triangle_multiply_ingoing = TriangleMultiplicativeModule(
        dim=dim_pairwise, mix='ingoing')

    _, dropout = embedd_dropout_get(dropout)
    self.dropout_rowwise_fn = functools.partial(shared_dropout,
                                                broadcast_dim=0,
                                                p=dropout)
    self.dropout_column_fn = functools.partial(shared_dropout,
                                               broadcast_dim=1,
                                               p=dropout)

  def forward(self,
              x,
              mask=None,
              msa_repr=None,
              msa_mask=None,
              shard_size=None):
    if exists(msa_repr):
      assert exists(self.outer_mean)
      with profiler.record_function("OuterProductMean"):
        x = x + self.outer_mean(msa_repr, mask=msa_mask, shard_size=shard_size)

    if self.multiplication_first:
      with profiler.record_function("TriangleMultiplicative"):
        x = x + self.dropout_rowwise_fn(
            self.triangle_multiply_outgoing(x, mask=mask),
            training=self.training)
        x = x + self.dropout_rowwise_fn(
            self.triangle_multiply_ingoing(x, mask=mask),
            training=self.training)
      with profiler.record_function("TriangleAttention"):
        x = x + self.dropout_rowwise_fn(
            self.triangle_attention_outgoing(x, edges=x, mask=mask,
                                             shard_size=shard_size),
            training=self.training)
        x = x + self.dropout_column_fn(
            self.triangle_attention_ingoing(x, edges=x, mask=mask,
                                            shard_size=shard_size),
            training=self.training)
    else:
      with profiler.record_function("TriangleAttention"):
        x = x + self.dropout_rowwise_fn(
            self.triangle_attention_outgoing(x, edges=x, mask=mask,
                                             shard_size=shard_size),
            training=self.training)
        x = x + self.dropout_column_fn(
            self.triangle_attention_ingoing(x, edges=x, mask=mask,
                                            shard_size=shard_size),
            training=self.training)
      with profiler.record_function("TriangleMultiplicative"):
        x = x + self.dropout_rowwise_fn(
            self.triangle_multiply_outgoing(x, mask=mask),
            training=self.training)
        x = x + self.dropout_rowwise_fn(
            self.triangle_multiply_ingoing(x, mask=mask),
            training=self.training)
    return x


class MsaAttentionBlock(nn.Module):
  """MsaAttentionBlock
    """
  def __init__(
      self,
      dim_msa,
      dim_pairwise,
      heads,
      dim_head,
      dropout=0.,
      global_column_attn=False,
  ):
    super().__init__()

    self.row_attn = AxialAttention(dim_node=dim_msa,
                                   dim_edge=dim_pairwise,
                                   heads=heads,
                                   dim_head=dim_head,
                                   row_attn=True,
                                   col_attn=False,
                                   accept_edges=True)
    self.col_attn = AxialAttention(dim_node=dim_msa,
                                   dim_edge=dim_pairwise,
                                   heads=heads,
                                   dim_head=dim_head,
                                   row_attn=False,
                                   col_attn=True,
                                   global_query_attn=global_column_attn)

    dropout, _ = embedd_dropout_get(dropout)
    self.dropout_fn = functools.partial(shared_dropout,
                                        broadcast_dim=0,
                                        p=dropout)

  def forward(self, x, mask=None, pairwise_repr=None, shard_size=None):
    with profiler.record_function("MSARowAttention"):
      x = x + self.dropout_fn(self.row_attn(x, mask=mask, edges=pairwise_repr,
                                            shard_size=shard_size),
                              training=self.training)
    with profiler.record_function("MSAColumnAttention"):
      x = x + self.col_attn(x, mask=mask, shard_size=shard_size)
    return x


class RelativePositionEmbedding(nn.Module):
  """RelativePositionEmbedding
    """
  def __init__(self, dim, max_rel_dist):
    super().__init__()

    _, dim_pairwise = embedd_dim_get(dim)
    self.max_rel_dist = max_rel_dist
    self.embedding = nn.Embedding(max_rel_dist * 2 + 1, dim_pairwise)

  def forward(self, seq_index):
    seq_rel_dist = rearrange(seq_index, '... i -> ... i ()') - rearrange(
        seq_index, '... j -> ... () j')
    seq_rel_dist = seq_rel_dist.clamp(-self.max_rel_dist,
                                      self.max_rel_dist) + self.max_rel_dist
    return self.embedding(seq_rel_dist)


class PairwiseEmbedding(nn.Module):
  """PairwiseEmbedding
    """
  def __init__(self, dim, max_rel_dist=0):
    super().__init__()

    dim_single, dim_pairwise = embedd_dim_get(dim)
    self.to_pairwise_repr = nn.Linear(dim_single, dim_pairwise * 2)
    self.relative_pos_emb = RelativePositionEmbedding(
        dim, max_rel_dist) if max_rel_dist > 0 else None

  def embeddings(self):
    return dict(position=self.relative_pos_emb.embedding.weight)

  def forward(self, x, x_mask, seq_index=None):
    (_, n), device = x.shape[:2], x.device

    x_left, x_right = self.to_pairwise_repr(x).chunk(2, dim=-1)
    x = rearrange(x_left, 'b i d -> b i () d') + rearrange(
        x_right, 'b j d-> b () j d')  # create pair-wise residue embeds
    x_mask = rearrange(x_mask, 'b i -> b i ()') * rearrange(
        x_mask, 'b j -> b () j') if exists(x_mask) else None
    if exists(self.relative_pos_emb):
      seq_index = default(seq_index, lambda: torch.arange(n, device=device))
      x = x + self.relative_pos_emb(seq_index)
    return x, x_mask


def checkpoint_sequential_nargs(functions, segments, inputs, **kwargs):
  # Hack for keyword-only parameter in a python 2.7-compliant way
  preserve = kwargs.pop('preserve_rng_state', True)

  def run_function(start, end, functions):

    def forward(*inputs):
      for j in range(start, end + 1):
        inputs = functions[j](inputs, **kwargs)
      return inputs

    return forward

  if isinstance(functions, torch.nn.Sequential):
    functions = list(functions.children())

  segment_size = len(functions) // segments
  # the last chunk has to be non-volatile
  end = -1
  for start in range(0, segment_size * (segments - 1), segment_size):
    end = start + segment_size - 1
    if torch.is_grad_enabled():
      inputs = checkpoint(run_function(start, end, functions),
                          *inputs,
                          preserve_rng_state=preserve)
    else:
      inputs = run_function(start, end, functions)(*inputs)
  return run_function(end + 1, len(functions) - 1, functions)(*inputs)
