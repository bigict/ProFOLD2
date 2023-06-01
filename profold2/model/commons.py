"""A lot of modules in Alphafold2
  """
import functools

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from profold2.model import functional
from profold2.utils import default, exists


# helpers
def init_zero_(layer):
  nn.init.constant_(layer.weight, 0.)
  if exists(layer.bias):
    nn.init.constant_(layer.bias, 0.)


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
  def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
    super().__init__()
    inner_dim = dim_head * heads
    self.heads = heads
    self.scale = dim_head**-0.5

    self.to_q = nn.Linear(dim, inner_dim, bias=False)
    self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
    self.to_out = nn.Linear(inner_dim, dim)

    self.gating = nn.Linear(dim, inner_dim)
    nn.init.constant_(self.gating.weight, 0.)
    nn.init.constant_(self.gating.bias, 1.)

    self.dropout = nn.Dropout(dropout)
    init_zero_(self.to_out)

  def forward(self,
              x,
              mask=None,
              attn_bias=None,
              context=None,
              context_mask=None,
              tie_dim=None):
    device, h, has_context = x.device, self.heads, exists(context)

    context = default(context, x)

    q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))

    i, _ = q.shape[-2], k.shape[-2]

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                  (q, k, v))

    # scale

    q = q * self.scale

    # query / key similarities
    if exists(tie_dim):
      # as in the paper, for the extra MSAs
      # they average the queries along the rows of the MSAs
      # they named this particular module MSAColumnGlobalAttention

      q, k = map(lambda t: rearrange(t, '(b r) ... -> b r ...', r=tie_dim),
                 (q, k))
      q = q.mean(dim=1)

      dots = torch.einsum('b h i d, b r h j d -> b r h i j', q, k)
      dots = rearrange(dots, 'b r ... -> (b r) ...')
    else:
      dots = torch.einsum('b h i d, b h j d -> b h i j', q, k)

    # add attention bias,
    # if supplied (for pairwise to msa attention communication)

    if exists(attn_bias):
      dots = dots + attn_bias

    # masking

    if exists(mask):
      mask = default(mask, lambda: torch.ones(1, i, device=device).bool())
      context_mask = mask if not has_context else default(
          context_mask,
          lambda: torch.ones(1, k.shape[-2], device=device).bool())
      mask_value = -torch.finfo(dots.dtype).max
      mask = mask[:, None, :, None] * context_mask[:, None, None, :]
      dots = dots.masked_fill(~mask.bool(), mask_value)

    # attention

    # dots = dots - dots.max(dim=-1, keepdim=True).values
    attn = dots.softmax(dim=-1)
    attn = self.dropout(attn)

    # aggregate

    out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

    # merge heads

    out = rearrange(out, 'b h n d -> b n (h d)')

    # gating

    gates = self.gating(x)
    out = out * gates.sigmoid()

    # combine to out

    out = self.to_out(out)
    return out


class AxialAttention(nn.Module):
  """AxialAttention
    """
  def __init__(self,
               dim,
               heads,
               row_attn=True,
               col_attn=True,
               accept_edges=False,
               global_query_attn=False,
               **kwargs):
    super().__init__()
    assert not (not row_attn and
                not col_attn), 'row or column attention must be turned on'

    self.row_attn = row_attn
    self.col_attn = col_attn
    self.global_query_attn = global_query_attn

    dim_node, dim_edge = embedd_dim_get(dim)
    self.norm = nn.LayerNorm(dim_node)
    self.attn = Attention(dim=dim_node, heads=heads, **kwargs)
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
      tie_dim = x.shape[axial_dim] if self.global_query_attn else None

      x = rearrange(x, input_fold_eq)
      if exists(mask):
        mask = rearrange(mask, mask_fold_axial_eq)
      out = self.attn(x, mask=mask, attn_bias=attn_bias, tie_dim=tie_dim)
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
    self.out_gate = nn.Linear(dim, hidden_dim)

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
    return self.to_out(out)


# evoformer blocks


class OuterMean(nn.Module):
  """OuterProductMean
    """
  def __init__(self, dim, dim_hidden=None, eps=1e-5):
    super().__init__()

    self.eps = eps
    dim_single, dim_pairwise = embedd_dim_get(dim)
    self.norm = nn.LayerNorm(dim_single)
    dim_hidden = default(dim_hidden, dim_pairwise)

    self.left_proj = nn.Linear(dim_single, dim_hidden)
    self.right_proj = nn.Linear(dim_single, dim_hidden)
    self.proj_out = nn.Linear(dim_hidden, dim_pairwise)

  def forward(self, x, mask=None, shard_size=None):
    x = self.norm(x)
    left = self.left_proj(x)
    right = self.right_proj(x)

    def run_outer_sum(left, right, mask):
      outer = rearrange(left, 'b m i d -> b m i () d') * rearrange(
          right, 'b m j d -> b m () j d')
      if exists(mask):
        # masked mean, if there are padding in the rows of the MSA
        mask = rearrange(mask, 'b m i -> b m i () ()') * rearrange(
            mask, 'b m j -> b m () j ()') > 0
        outer = outer.masked_fill(~mask, 0.)
      return outer.sum(dim=1, keepdim=True)

    def run_mask_sum(mask):
      # masked mean, if there are padding in the rows of the MSA
      mask = rearrange(mask, 'b m i -> b m i () ()') * rearrange(
          mask, 'b m j -> b m () j ()') > 0
      return mask.sum(dim=1, keepdim=True)

    outer = functional.sharded_apply(
        run_outer_sum, [left, right, mask],
        shard_size=None if self.training else shard_size,
        shard_dim=1,
        cat_dim=1)
    if exists(mask):
      mask = functional.sharded_apply(
          run_mask_sum, [mask],
          shard_size=None if self.training else shard_size,
          shard_dim=1,
          cat_dim=1)
      outer = outer.sum(dim=1) / torch.clamp(mask.sum(dim=1) + self.eps, min=1)
    else:
      outer = outer.mean(dim=1)

    return self.proj_out(outer)


class PairwiseAttentionBlock(nn.Module):
  """PairwiseAttentionBlock
    """
  def __init__(
      self,
      dim,
      heads,
      dim_head,
      dropout=0.,
      global_column_attn=False,
  ):
    super().__init__()

    _, dim_pairwise = embedd_dim_get(dim)

    self.outer_mean = OuterMean(dim)
    self.triangle_attention_outgoing = AxialAttention(dim=dim_pairwise,
                                                      heads=heads,
                                                      dim_head=dim_head,
                                                      row_attn=True,
                                                      col_attn=False,
                                                      accept_edges=True)
    self.triangle_attention_ingoing = AxialAttention(
        dim=dim_pairwise,
        heads=heads,
        dim_head=dim_head,
        row_attn=False,
        col_attn=True,
        accept_edges=True,
        global_query_attn=global_column_attn)
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
      x = x + self.outer_mean(msa_repr, mask=msa_mask, shard_size=shard_size)

    x = x + self.dropout_rowwise_fn(
        self.triangle_multiply_outgoing(x, mask=mask),
        training=self.training)
    x = x + self.dropout_rowwise_fn(
        self.triangle_multiply_ingoing(x, mask=mask),
        training=self.training)
    x = x + self.dropout_rowwise_fn(
        self.triangle_attention_outgoing(x, edges=x, mask=mask,
                                         shard_size=shard_size),
        training=self.training)
    x = x + self.dropout_column_fn(
        self.triangle_attention_ingoing(x, edges=x, mask=mask,
                                        shard_size=shard_size),
        training=self.training)
    return x


class MsaAttentionBlock(nn.Module):
  """MsaAttentionBlock
    """
  def __init__(self, dim, heads, dim_head, dropout=0.):
    super().__init__()

    dim_single, dim_pairwise = embedd_dim_get(dim)
    self.row_attn = AxialAttention(dim=(dim_single, dim_pairwise),
                                   heads=heads,
                                   dim_head=dim_head,
                                   row_attn=True,
                                   col_attn=False,
                                   accept_edges=True)
    self.col_attn = AxialAttention(dim=(dim_single, dim_pairwise),
                                   heads=heads,
                                   dim_head=dim_head,
                                   row_attn=False,
                                   col_attn=True)

    dropout, _ = embedd_dropout_get(dropout)
    self.dropout_fn = functools.partial(shared_dropout,
                                        broadcast_dim=0,
                                        p=dropout)

  def forward(self, x, mask=None, pairwise_repr=None):
    x = x + self.dropout_fn(self.row_attn(x, mask=mask, edges=pairwise_repr),
                            training=self.training)
    x = x + self.col_attn(x, mask=mask)
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
