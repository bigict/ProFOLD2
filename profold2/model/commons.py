"""A lot of modules in AlphaFold2
  """
import contextlib
import dataclasses
import functools
import logging
import typing

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from profold2.model import functional, kernel, profiler
from profold2.utils import default, env, exists, version_cmp

logger = logging.getLogger(__name__)

_tensor_inplace_op = env('profold2_tensor_inplace_op', defval=0, dtype=int)


@contextlib.contextmanager
def torch_default_dtype(dtype):
  prev_dtype = torch.get_default_dtype()
  torch.set_default_dtype(dtype)
  yield
  torch.set_default_dtype(prev_dtype)


@contextlib.contextmanager
def torch_allow_tf32(allow=True):
  if hasattr(torch.backends,
             'cuda') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
    matmul_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow
  if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn, 'allow_tf32'):
    cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = allow
  yield
  if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn, 'allow_tf32'):
    torch.backends.cudnn.allow_tf32 = cudnn_allow_tf32
  if hasattr(torch.backends,
             'cuda') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
    torch.backends.cuda.matmul.allow_tf32 = matmul_allow_tf32


# helpers
def init_linear_(layer, w=0., b=0.):
  nn.init.constant_(layer.weight, w)
  if exists(layer.bias):
    nn.init.constant_(layer.bias, b)


def default_list_get(value, n):
  if isinstance(value, (tuple, list)):
    assert len(value) <= n
    value = list(value)
    while len(value) < n:
      value = value + [value[-1]]
    return value
  return [value] * n


def embedd_dim_get(dim):
  return default_list_get(dim, 2)  # (dim_node, dim_edge)


def embedd_dropout_get(p):
  return default_list_get(p, 2)  # (p_single, p_pairwise)


def tensor_add(x, y):
  if _tensor_inplace_op:
    x += y
  else:
    x = x + y
  return x


def tensor_sub(x, y):
  if _tensor_inplace_op:
    x -= y
  else:
    x = x - y
  return x


def tensor_mul(x, y):
  if _tensor_inplace_op:
    x *= y
  else:
    x = x * y
  return x


def tensor_div(x, y):
  if _tensor_inplace_op:
    x /= y
  else:
    x = x / y
  return x


def max_neg_value(t):
  return -torch.finfo(t.dtype).max


def shared_dropout(x, p, broadcast_dim=None, training=True):
  if exists(broadcast_dim) and 0 < p < 1.0:
    if training:
      shape = list(x.shape)
      assert len(shape) >= 3  # (... m i d)
      # shared across rows(-3) and columns(-2)
      assert broadcast_dim % len(shape) - len(shape) in (-3, -2)
      shape[broadcast_dim] = 1
      m = torch.bernoulli(torch.full(shape, 1 - p, device=x.device))
      return m * x / (1 - p)
    return x
  return F.dropout(x, p=p, training=training)


def evoformer_attn(q, k, v, attn_mask, dropout_p=0.0, scale=None, dtype=None):
  del dropout_p, scale
  assert kernel.is_available()
  dtype_from, dtype_to = q.dtype, default(dtype, torch.float16)
  if not exists(dtype) and hasattr(torch.cuda, 'is_bf16_supported'):
    if torch.cuda.is_bf16_supported():
      dtype_to = torch.bfloat16
  q, k, v = map(
      lambda t: rearrange(t.to(dtype=dtype_to), '... h i d -> ... i h d'), (q, k, v)
  )
  mask, _, attn_bias = attn_mask
  # HACK: experience value
  mask_value = 1e4 if dtype == torch.float16 else 1e6  # max_neg_value(q)
  if exists(mask):
    mask = rearrange(mask.to(dtype=dtype_to), '... m i -> ... m () () i')
    mask = mask_value * (mask - 1.0)
  if exists(attn_bias):
    attn_bias = torch.clamp(attn_bias, min=-mask_value).to(dtype=dtype_to)
  o = kernel.evoformer_attn(q, k, v, [mask, attn_bias])
  return rearrange(o.to(dtype=dtype_from), '... i h d -> ... h i d')


def pytorch_attn(q, k, v, attn_mask, dropout_p=0.0, scale=None, dtype=None):
  assert hasattr(F, 'scaled_dot_product_attention')
  dtype_from, dtype_to = q.dtype, default(dtype, torch.float16)
  if not exists(dtype) and hasattr(torch.cuda, 'is_bf16_supported'):
    if torch.cuda.is_bf16_supported():
      dtype_to = torch.bfloat16
  q, k, v = map(lambda t: t.to(dtype=dtype_to), (q, k, v))
  _, attn_mask, attn_bias = attn_mask
  # HACK: experience value
  mask_value = 1e4 if dtype == torch.float16 else 1e6  # max_neg_value(q)
  if exists(attn_mask) and exists(attn_bias):
    attn_mask = torch.clamp(
        attn_bias.masked_fill(~attn_mask, mask_value), min=-mask_value
    ).to(dtype=dtype_to)
  elif exists(attn_bias):
    attn_mask = attn_bias
  # See https://github.com/pytorch/pytorch/issues/96099
  o = F.scaled_dot_product_attention(
      q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, scale=scale
  )
  return o.to(dtype=dtype_from)


# helper classes
class ConfigMeta(type):
  """Metaclass that synthesizes a __post_init__ that coerces dicts to Config subclass
  instances.
  """
  def __new__(mcs, clsname, bases, attrs):
    cls = super().__new__(mcs, clsname, bases, attrs)

    old_post_init = getattr(cls, '__post_init__', None)

    def _post_init(self) -> None:
      # Use get_type_hints instead of Field.type to ensure that forward references are
      # resolved.
      type_hints = typing.get_type_hints(self.__class__)
      fields = dataclasses.fields(self.__class__)
      for field in fields:
        type_hint = type_hints[field.name]
        if issubclass(type(type_hint), ConfigMeta):
          field_value = getattr(self, field.name)
          if exists(field_value) and isinstance(field_value, dict):
            setattr(self, field.name, type_hint(**field_value))
      if exists(old_post_init):
        old_post_init(self)

    cls.__post_init__ = _post_init

    return dataclasses.dataclass()(cls)


class BasicConfig(metaclass=ConfigMeta):
  def asdict(self):
    return dataclasses.asdict(self)


class Always(nn.Module):
  def __init__(self, val):
    super().__init__()
    self.val = val

  def forward(self, x):
    del x
    return self.val


class APC(nn.Module):
  """Average Product Correction
     https://doi.org/10.1093/bioinformatics/btm604
    """
  def __init__(self, dim=None):
    super().__init__()
    self.dim = dim

  def forward(self, x):
    return functional.apc(x, dim=self.dim)


# feed forward
class GEGLU(nn.Module):
  """Gated GELU"""
  def forward(self, x):
    x, gates = x.chunk(2, dim=-1)
    return x * F.gelu(gates)


class FeedForward(nn.Module):
  """FeedForward layer in transformer
    """
  def __init__(self, dim, mult=4, dropout=0., activation=None):
    super().__init__()
    self.norm = nn.LayerNorm(dim)

    # NOTE: activation: ReLU (AlphaFold2), GatedGELU (ProFOLD)
    activation = default(activation, env('FeedForward_activation', defval='GatedGELU'))
    if activation == 'GatedGELU':
      activation = GEGLU
      amplifier = 2
    else:
      assert activation == 'ReLU'
      activation = nn.ReLU
      amplifier = 1

    self.net = nn.Sequential(
        nn.Linear(dim, dim * mult * amplifier), activation(), nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )
    init_linear_(self.net[-1])

  def forward(self, x, shard_dim=-2, shard_size=None):
    x = self.norm(x)
    # return self.net(x)
    return functional.sharded_apply(
        self.net, [x],
        shard_size=None if self.training else shard_size,
        shard_dim=shard_dim,
        cat_dim=shard_dim
    )


class Attention(nn.Module):
  """Multi-head Attention
    """
  def __init__(
      self,
      dim_q,
      dim_kv,
      heads=8,
      dim_head=64,
      dropout=0.,
      gating=True,
      q_use_bias=False,
      kv_use_bias=False,
      global_query_attn=False
  ):
    super().__init__()
    self.heads = heads
    self.dim_head = dim_head
    self.scale = dim_head**-0.5

    dim_inner = dim_head * heads
    self.to_q = nn.Linear(dim_q, dim_inner, bias=q_use_bias)
    if global_query_attn:
      self.to_kv = nn.Linear(dim_kv, dim_head * 2, bias=kv_use_bias)
    else:
      self.to_kv = nn.Linear(dim_kv, dim_inner * 2, bias=kv_use_bias)
    self.to_out = nn.Linear(dim_inner, dim_q)

    self.gating = nn.Linear(dim_kv, dim_inner) if gating else None
    if exists(self.gating):
      init_linear_(self.gating, b=1.)

    self.dropout = nn.Dropout(dropout)
    self.global_query_attn = global_query_attn
    init_linear_(self.to_out)

  def forward(
      self,
      x,
      mask=None,
      attn_bias=None,
      context=None,
      context_mask=None,
      attn_fn=None
  ):
    device, h, d = x.device, self.heads, self.dim_head

    m = default(context, x)

    q, k, v = (self.to_q(x), *self.to_kv(m).chunk(2, dim=-1))

    q, k, v = map(lambda t: rearrange(t, '... i (h d) -> ... h i d', d=d), (q, k, v))

    # query / key similarities
    if self.global_query_attn:
      # as in the paper, for the extra MSAs
      # they average the queries along the rows of the MSAs
      # they named this particular module MSAColumnGlobalAttention

      k, v = map(lambda t: repeat(t, '... r i d-> ... (r h) i d', h=h), (k, v))
      n = q.shape[-2]
      if exists(mask):
        q = functional.masked_mean(value=q, mask=mask[..., None, :, None], dim=-2)
      else:
        q = q.mean(dim=-2)
      q = repeat(q, '... h d -> ... h i d', i=n)

    # masking
    attn_mask, mask_value = None, max_neg_value(q)
    if exists(mask):
      context_mask = mask if not exists(context) else default(
          context_mask, torch.ones(*m.shape[:-1], dtype=torch.bool, device=device)
      )
      attn_mask = rearrange(mask.bool(), '... i -> ... () i ()'
                           ) * rearrange(context_mask.bool(), '... j -> ... () () j')
      assert attn_mask.dtype == torch.bool

    # pytorch 2.0+
    if exists(attn_fn):
      dropout_p = self.dropout.p if self.training else 0.0
      out = attn_fn(
          q,
          k,
          v,
          attn_mask=[mask, attn_mask, attn_bias],
          dropout_p=dropout_p,
          scale=self.scale
      )
    else:
      # scale
      q = tensor_mul(q, self.scale)

      dots = torch.einsum('... h i d,... h j d -> ... h i j', q, k)

      # add attention bias,
      # if supplied (for pairwise to msa attention communication)

      if exists(attn_bias):
        dots = tensor_add(dots, attn_bias)

      # masking
      if exists(attn_mask):
        dots = dots.masked_fill(~attn_mask, mask_value)  # pylint: disable=invalid-unary-operand-type

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
      gates = self.gating(x)
      out = tensor_mul(out, gates.sigmoid())

    # combine to out
    out = self.to_out(out)
    return out


class AxialAttention(nn.Module):
  """AxialAttention
    """
  def __init__(
      self,
      dim_node,
      dim_edge,
      heads,
      row_attn=True,
      col_attn=True,
      accept_edges=False,
      **kwargs
  ):
    super().__init__()
    assert not (
        not row_attn and not col_attn
    ), 'row or column attention must be turned on'

    self.row_attn = row_attn
    self.col_attn = col_attn

    self.norm = nn.LayerNorm(dim_node)
    self.attn = Attention(dim_q=dim_node, dim_kv=dim_node, heads=heads, **kwargs)
    # FIX: to be backward compatible
    accept_edge_norm = env('AxialAttention_accept_edge_norm', defval=1, dtype=int)
    self.edges_to_attn_bias = nn.Sequential(
        nn.LayerNorm(dim_edge) if accept_edge_norm else nn.Identity(dim_edge),
        nn.Linear(dim_edge, heads, bias=not accept_edge_norm),
        Rearrange('... i j h -> ... h i j')
    ) if accept_edges else None
    accept_kernel_fn = env('AxialAttention_accept_kernel_fn', defval=0, dtype=int)
    if not kernel.is_available() and accept_kernel_fn:
      logger.warning('kernel is not available! disabled it.')
      accept_kernel_fn = 0
    accept_kernel_dtype = env('AxialAttention_accept_kernel_dtype')
    if accept_kernel_dtype in ('float16', 'f16'):
      accept_kernel_dtype = torch.float16
    elif accept_kernel_dtype in ('bfloat16', 'bf16'):
      accept_kernel_dtype = torch.bfloat16
    self.attn_fn = functools.partial(
        evoformer_attn, dtype=accept_kernel_dtype
    ) if accept_kernel_fn else None

  def forward(self, x, edges=None, mask=None, edge_mask=None, shard_size=None):
    assert self.row_attn ^ self.col_attn, 'has to be either row or column attention, but not both'  # pylint: disable=line-too-long

    x = self.norm(x)

    # axial attention
    if self.col_attn:
      axial_dim = 2
      mask_fold_axial_eq = '... h w -> ... w h'
      input_fold_eq = '... h w d -> ... w h d'
      output_fold_eq = '... w h d -> ... h w d'

    elif self.row_attn:
      axial_dim = 1
      mask_fold_axial_eq = '... h w -> ... h w'
      input_fold_eq = '... h w d -> ... h w d'
      output_fold_eq = '... h w d -> ... h w d'

    def run_attn(x, mask, attn_bias):
      _, h, w, _ = x.shape

      attn_fn = None
      if exists(attn_bias):
        attn_bias = rearrange(attn_bias, '... h i j -> ... () h i j')
        attn_fn = self.attn_fn

      x = rearrange(x, input_fold_eq)
      if exists(mask):
        mask = rearrange(mask, mask_fold_axial_eq)
      out = self.attn(x, mask=mask, attn_bias=attn_bias, attn_fn=attn_fn)
      out = rearrange(out, output_fold_eq, h=h, w=w)
      return out

    attn_bias = None
    if exists(self.edges_to_attn_bias) and exists(edges):
      attn_bias = self.edges_to_attn_bias(edges)  # pylint: disable=not-callable
      if exists(edge_mask):
        attn_mask = rearrange(edge_mask.bool(), '... i j -> ... () i j')
        attn_bias = attn_bias.masked_fill(~attn_mask, max_neg_value(attn_bias))
    if exists(attn_bias) and self.col_attn:
      attn_bias = rearrange(attn_bias, '... i j -> ... j i')

    return functional.sharded_apply(
        run_attn, [x, mask],
        attn_bias,
        shard_size=None if self.training else shard_size,
        shard_dim=axial_dim,
        cat_dim=axial_dim
    )


class TriangleMultiplicativeModule(nn.Module):
  """TriangleMultiplicative
    """
  def __init__(self, *, dim, hidden_dim=None, mix='ingoing'):
    super().__init__()
    assert mix in {'ingoing', 'outgoing'}, 'mix must be either ingoing or outgoing'

    hidden_dim = default(hidden_dim, dim)
    self.norm = nn.LayerNorm(dim)

    self.left_proj = nn.Linear(dim, hidden_dim)
    self.right_proj = nn.Linear(dim, hidden_dim)

    self.left_gate = nn.Linear(dim, hidden_dim)
    self.right_gate = nn.Linear(dim, hidden_dim)
    self.out_gate = nn.Linear(dim, dim)

    # initialize all gating to be identity

    for gate in (self.left_gate, self.right_gate, self.out_gate):
      init_linear_(gate, b=1.)

    if mix == 'outgoing':
      self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
      self.shard_dim = -3
    elif mix == 'ingoing':
      self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'
      self.shard_dim = -2

    self.to_out_norm = nn.LayerNorm(hidden_dim)
    self.to_out = nn.Linear(hidden_dim, dim)

  def forward(self, x, mask=None, shard_size=None):
    assert x.shape[-2] == x.shape[-3], 'feature map must be symmetrical'
    if exists(mask):
      mask = rearrange(mask, '... i j -> ... i j ()')

    x = self.norm(x)

    def proj_with_gating(proj_op, gating_op, x, mask):
      t = proj_op(x)
      if exists(mask):
        t = tensor_mul(t, mask)
      return tensor_mul(t, gating_op(x).sigmoid())

    def out_with_gating(x, mask, right):
      left = proj_with_gating(self.left_proj, self.left_gate, x, mask)

      out = torch.einsum(self.mix_einsum_eq, left, right)

      out = self.to_out_norm(out)
      out = tensor_mul(out, self.out_gate(x).sigmoid())
      out = self.to_out(out)
      return out

    right = proj_with_gating(self.right_proj, self.right_gate, x, mask)
    if exists(mask):
      return functional.sharded_apply(
          out_with_gating, [x, mask],
          right,
          shard_size=None if self.training else shard_size,
          shard_dim=self.shard_dim,
          cat_dim=self.shard_dim
      )
    return functional.sharded_apply(
        out_with_gating, [x],
        mask,
        right,
        shard_size=None if self.training else shard_size,
        shard_dim=self.shard_dim,
        cat_dim=self.shard_dim
    )


# evoformer blocks


class OuterProductMean(nn.Module):
  """OuterProductMean
    """
  def __init__(self, dim_msa, dim_pairwise, dim_hidden=32, eps=1e-5):
    super().__init__()

    # NOTE: eps: 1e-3 (AlphaFold), 1e-5 (ProFOLD)
    self.eps = env('OuterProductMean_eps', defval=eps, dtype=float)
    self.norm = nn.LayerNorm(dim_msa)
    dim_hidden = default(dim_hidden, dim_pairwise)

    self.left_proj = nn.Linear(dim_msa, dim_hidden)
    self.right_proj = nn.Linear(dim_msa, dim_hidden)
    self.proj_out = nn.Linear(dim_hidden**2, dim_pairwise)

  def forward(self, x, mask=None, shard_size=None):
    x = self.norm(x)

    def run_proj(op, x, mask):
      x = op(x)
      if exists(mask):
        x = tensor_mul(x, mask[..., None])
      return x

    def run_outer_sum(left, right):
      outer = torch.einsum('... m i c,... m j d -> ... i j c d', left, right)
      outer = self.proj_out(rearrange(outer, '... c d -> ... (c d)'))
      return outer

    def run_mask_sum(left, right):
      # masked mean, if there are padding in the rows of the MSA
      return torch.einsum('... m i,... m j -> ... i j', left, right)

    left = run_proj(self.left_proj, x, mask)
    right = run_proj(self.right_proj, x, mask)

    outer = functional.sharded_apply(
        run_outer_sum, [left],
        right,
        shard_size=None if self.training else shard_size,
        shard_dim=-2,
        cat_dim=-3
    )
    if exists(mask):
      mask = functional.sharded_apply(
          run_mask_sum, [mask],
          mask,
          shard_size=None if self.training else shard_size,
          shard_dim=-1,
          cat_dim=-2
      )
      outer = tensor_div(outer, mask[..., None] + self.eps)
    else:
      outer = tensor_div(outer, x.shape[-3] + self.eps)

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
        dim_msa, dim_pairwise
    ) if not disabled_outer_mean else None
    self.triangle_attention_outgoing = AxialAttention(
        dim_node=dim_pairwise,
        dim_edge=dim_pairwise,
        heads=heads,
        dim_head=dim_head,
        row_attn=True,
        col_attn=False,
        accept_edges=True
    )
    self.triangle_attention_ingoing = AxialAttention(
        dim_node=dim_pairwise,
        dim_edge=dim_pairwise,
        heads=heads,
        dim_head=dim_head,
        row_attn=False,
        col_attn=True,
        accept_edges=True
    )
    self.triangle_multiply_outgoing = TriangleMultiplicativeModule(
        dim=dim_pairwise, mix='outgoing'
    )
    self.triangle_multiply_ingoing = TriangleMultiplicativeModule(
        dim=dim_pairwise, mix='ingoing'
    )

    _, dropout = embedd_dropout_get(dropout)
    self.dropout_rowwise_fn = functools.partial(
        shared_dropout, broadcast_dim=-3, p=dropout
    )
    self.dropout_column_fn = functools.partial(
        shared_dropout, broadcast_dim=-2, p=dropout
    )

  def forward(self, x, mask=None, msa_repr=None, msa_mask=None, shard_size=None):
    if exists(msa_repr):
      assert exists(self.outer_mean)
      with profiler.record_function('OuterProductMean'):
        x = tensor_add(
            x, self.outer_mean(msa_repr, mask=msa_mask, shard_size=shard_size)
        )

    if self.multiplication_first:
      with profiler.record_function('TriangleMultiplicative'):
        x = tensor_add(
            x,
            self.dropout_rowwise_fn(
                self.triangle_multiply_outgoing(x, mask=mask, shard_size=shard_size),
                training=self.training
            )
        )
        x = tensor_add(
            x,
            self.dropout_rowwise_fn(
                self.triangle_multiply_ingoing(x, mask=mask, shard_size=shard_size),
                training=self.training
            )
        )
      with profiler.record_function('TriangleAttention'):
        x = tensor_add(
            x,
            self.dropout_rowwise_fn(
                self.triangle_attention_outgoing(
                    x, edges=x, mask=mask, edge_mask=mask, shard_size=shard_size
                ),
                training=self.training
            )
        )
        x = tensor_add(
            x,
            self.dropout_column_fn(
                self.triangle_attention_ingoing(
                    x, edges=x, mask=mask, edge_mask=mask, shard_size=shard_size
                ),
                training=self.training
            )
        )
    else:
      with profiler.record_function('TriangleAttention'):
        x = tensor_add(
            x,
            self.dropout_rowwise_fn(
                self.triangle_attention_outgoing(
                    x, edges=x, mask=mask, edge_mask=mask, shard_size=shard_size
                ),
                training=self.training
            )
        )
        x = tensor_add(
            x,
            self.dropout_column_fn(
                self.triangle_attention_ingoing(
                    x, edges=x, mask=mask, edge_mask=mask, shard_size=shard_size
                ),
                training=self.training
            )
        )
      with profiler.record_function('TriangleMultiplicative'):
        x = tensor_add(
            x,
            self.dropout_rowwise_fn(
                self.triangle_multiply_outgoing(x, mask=mask, shard_size=shard_size),
                training=self.training
            )
        )
        x = tensor_add(
            x,
            self.dropout_rowwise_fn(
                self.triangle_multiply_ingoing(x, mask=mask, shard_size=shard_size),
                training=self.training
            )
        )
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

    self.row_attn = AxialAttention(
        dim_node=dim_msa,
        dim_edge=dim_pairwise,
        heads=heads,
        dim_head=dim_head,
        row_attn=True,
        col_attn=False,
        accept_edges=True
    )
    self.col_attn = AxialAttention(
        dim_node=dim_msa,
        dim_edge=dim_pairwise,
        heads=heads,
        dim_head=dim_head,
        row_attn=False,
        col_attn=True,
        global_query_attn=global_column_attn
    )

    dropout, _ = embedd_dropout_get(dropout)
    self.dropout_fn = functools.partial(shared_dropout, broadcast_dim=-3, p=dropout)

  def forward(
      self, x, mask=None, pairwise_repr=None, pairwise_mask=None, shard_size=None
  ):
    with profiler.record_function('MSARowAttention'):
      x = tensor_add(
          x,
          self.dropout_fn(
              self.row_attn(
                  x,
                  mask=mask,
                  edges=pairwise_repr,
                  edge_mask=pairwise_mask,
                  shard_size=shard_size
              ),
              training=self.training
          )
      )
    with profiler.record_function('MSAColumnAttention'):
      x = tensor_add(x, self.col_attn(x, mask=mask, shard_size=shard_size))
    return x


# classes
class InvariantPointAttention(nn.Module):
  """Invariant Point Attention
    """
  def __init__(
      self,
      *,
      dim,
      heads=12,
      scalar_key_dim=16,
      scalar_value_dim=16,
      point_key_dim=4,
      point_value_dim=8,
      pairwise_repr_dim=None,
      require_pairwise_repr=True,
      qkv_use_bias=False,
      gating=False,
      point_weight_init=1.,
      eps=1e-8
  ):
    super().__init__()
    self.eps = eps
    self.heads = heads
    self.require_pairwise_repr = require_pairwise_repr

    # num attention contributions
    num_attn_logits = 3 if require_pairwise_repr else 2

    # qkv projection for scalar attention (normal)
    self.scalar_attn_logits_scale = (num_attn_logits * scalar_key_dim)**-0.5

    self.to_scalar_q = nn.Linear(dim, scalar_key_dim * heads, bias=qkv_use_bias)
    self.to_scalar_k = nn.Linear(dim, scalar_key_dim * heads, bias=qkv_use_bias)
    self.to_scalar_v = nn.Linear(dim, scalar_value_dim * heads, bias=qkv_use_bias)

    self.gating = nn.Linear(dim, scalar_value_dim * heads) if gating else None
    if exists(self.gating):
      init_linear_(self.gating, b=1.)

    # qkv projection for point attention (coordinate and orientation aware)
    point_weight_init_value = torch.log(
        torch.exp(torch.full((heads, ), point_weight_init)) - 1.
    )
    self.point_weights = nn.Parameter(point_weight_init_value)

    self.point_attn_logits_scale = ((num_attn_logits * point_key_dim) * (9 / 2))**-0.5

    self.to_point_q = nn.Linear(dim, point_key_dim * heads * 3, bias=qkv_use_bias)
    self.to_point_k = nn.Linear(dim, point_key_dim * heads * 3, bias=qkv_use_bias)
    self.to_point_v = nn.Linear(dim, point_value_dim * heads * 3, bias=qkv_use_bias)

    # pairwise representation projection to attention bias
    if require_pairwise_repr:
      pairwise_repr_dim = default(pairwise_repr_dim, dim)

    if require_pairwise_repr:
      self.pairwise_attn_logits_scale = num_attn_logits**-0.5
      self.to_pairwise_attn_bias = nn.Sequential(
          nn.Linear(pairwise_repr_dim, heads), Rearrange('... i j h -> ... h i j')
      )
    else:
      self.to_pairwise_attn_bias = nn.Sequential(
          nn.LayerNorm(pairwise_repr_dim),
          nn.Linear(pairwise_repr_dim, heads, bias=False),
          Rearrange('... i j h -> ... h i j')
      )

    # combine out - scalar dim +
    #               pairwise dim +
    #               point dim * (3 for coordinates in R3 and then 1 for norm)
    self.to_out = nn.Linear(
        heads * (
            scalar_value_dim +
            (pairwise_repr_dim if require_pairwise_repr else 0) + point_value_dim *
            (3 + 1)
        ), dim
    )
    init_linear_(self.to_out)

  def forward(
      self, single_repr, pairwise_repr=None, *, rotations, translations, mask=None
  ):
    x, h, eps = single_repr, self.heads, self.eps
    assert not (
        self.require_pairwise_repr and not exists(pairwise_repr)
    ), 'pairwise representation must be given as second argument'

    # get queries, keys, values for scalar and point (coordinate-aware)
    # attention pathways
    q_scalar, k_scalar, v_scalar = self.to_scalar_q(x), self.to_scalar_k(
        x
    ), self.to_scalar_v(x)
    q_point, k_point, v_point = self.to_point_q(x), self.to_point_k(x), self.to_point_v(
        x
    )

    # split out heads
    q_scalar, k_scalar, v_scalar = map(
        lambda t: rearrange(t, '... i (h d) -> ... h i d', h=h),
        (q_scalar, k_scalar, v_scalar)
    )
    q_point, k_point, v_point = map(
        lambda t: rearrange(t, '... i (h d c) -> ... h i d c', h=h, c=3),
        (q_point, k_point, v_point)
    )

    translations = rearrange(translations, '... i c -> ... () i () c')

    # rotate qkv points into global frame
    q_point = torch.einsum(
        '... h i d c,... i r c -> ... h i d r', q_point, rotations
    ) + translations
    k_point = torch.einsum(
        '... h i d c,... i r c -> ... h i d r', k_point, rotations
    ) + translations
    v_point = torch.einsum(
        '... h i d c,... i r c -> ... h i d r', v_point, rotations
    ) + translations

    # derive attn logits for scalar and pairwise
    attn_logits = tensor_mul(
        torch.einsum('... h i d,... h j d -> ... h i j', q_scalar, k_scalar),
        self.scalar_attn_logits_scale
    )

    # combine attn logits
    if exists(pairwise_repr):
      attn_logits_pairwise = self.to_pairwise_attn_bias(pairwise_repr)
      if self.require_pairwise_repr:
        attn_logits_pairwise = tensor_mul(
            attn_logits_pairwise, self.pairwise_attn_logits_scale
        )
      attn_logits = tensor_add(attn_logits, attn_logits_pairwise)

    # derive attn logits for point attention
    def calc_point_dist(q_point, k_point):
      point_dist = functional.squared_cdist(
          rearrange(q_point, '... h i d c -> ... h d i c'),
          rearrange(k_point, '... h j d c -> ... h d j c')
      )
      return torch.sum(point_dist, dim=-3)

    point_dist = functional.sharded_apply(
        calc_point_dist, [q_point, k_point],
        shard_size=None if self.training else 1,
        shard_dim=-2,
        cat_dim=functools.partial(functools.reduce, tensor_add)
    )

    point_weights = F.softplus(self.point_weights)
    point_weights = rearrange(point_weights, 'h -> h () ()')

    point_dist = tensor_mul(
        point_dist, -0.5 * point_weights * self.point_attn_logits_scale
    )

    attn_logits = tensor_add(attn_logits, point_dist)

    # mask
    if exists(mask):
      mask = rearrange(mask, '... i -> ... () i ()'
                      ) * rearrange(mask, '... j -> ... () () j')
      mask_value = max_neg_value(attn_logits)
      attn_logits = attn_logits.masked_fill(~mask, mask_value)

    # attention
    attn = F.softmax(attn_logits, dim=-1)

    # disable TF32 for precision
    with torch_allow_tf32(allow=False):

      # aggregate values
      results_scalar = torch.einsum('... h i j,... h j d -> ... h i d', attn, v_scalar)

      if self.require_pairwise_repr:
        results_pairwise = torch.einsum(
            '... h i j,... i j d -> ... h i d', attn, pairwise_repr
        )

      # aggregate point values
      results_points = torch.einsum(
          '... h i j,... h j d c -> ... h i d c', attn, v_point
      )

      # rotate aggregated point values back into local frame
      results_points = torch.einsum(
          '... h i d c,... i r c -> ... h i d r', results_points - translations,
          rotations.transpose(-1, -2)
      )
      results_points_norm = torch.sqrt(torch.square(results_points).sum(dim=-1) + eps)

    # merge back heads
    results_scalar = rearrange(results_scalar, '... h i d -> ... i (h d)')
    # gating
    if exists(self.gating):
      results_scalar = tensor_mul(results_scalar, self.gating(x).sigmoid())  # pylint: disable=not-callable

    results_points = rearrange(results_points, '... h i d c -> ... i (h d c)')
    results_points_norm = rearrange(results_points_norm, '... h i d -> ... i (h d)')

    results = (results_scalar, results_points, results_points_norm)

    if self.require_pairwise_repr:
      results_pairwise = rearrange(results_pairwise, '... h i d -> ... i (h d)', h=h)
      results = (*results, results_pairwise)

    # concat results and project out
    results = torch.cat(results, dim=-1)
    return self.to_out(results)


class FrameAttentionBlock(nn.Module):
  """FrameAttention"""
  def __init__(self, *, dim_msa, dim_pairwise, dropout=.0, **kwargs):
    super().__init__()

    self.norm = nn.LayerNorm(dim_msa)
    self.attn = InvariantPointAttention(
        dim=dim_msa, pairwise_repr_dim=dim_pairwise, **kwargs
    )

    dropout, _ = embedd_dropout_get(dropout)
    self.dropout_fn = nn.Dropout(dropout)

  def forward(self, x, mask, pairwise_repr, frames):
    quaternions, translations = frames
    # No rotation gradients between iterations to stabilize training.
    rotations = functional.quaternion_to_matrix(quaternions).detach()
    x = tensor_add(
        x,
        self.dropout_fn(
            self.attn(
                self.norm(x.float()),
                mask=mask.bool(),
                pairwise_repr=pairwise_repr.float(),
                rotations=rotations.float(),
                translations=translations.float()
            )
        )
    )
    return x


class FrameUpdater(nn.Module):
  """FrameUpdate"""
  def __init__(self, dim, dropout):
    super().__init__()

    self.to_affine_norm = nn.LayerNorm(dim)
    self.to_affine_update = nn.Linear(dim, 6)

    dropout, _ = embedd_dropout_get(dropout)
    self.dropout_fn = nn.Dropout(dropout)

  def forward(self, x, frames):
    # update quaternion and translation
    x = self.to_affine_norm(self.dropout_fn(x))
    quaternion_update, translation_update = self.to_affine_update(x).chunk(2, dim=-1)
    quaternion_update = F.pad(quaternion_update, (1, 0), value=1.)
    # FIX: make sure quaternion_update is standardized
    quaternion_update = functional.l2_norm(quaternion_update)

    quaternions, translations = frames
    # No rotation gradients between iterations to stabilize training.
    rotations = functional.quaternion_to_matrix(quaternions).detach()
    quaternions = functional.quaternion_multiply(quaternions, quaternion_update)
    translations = torch.einsum(
        '... i c,... i r c -> ... i r', translation_update, rotations
    ) + translations
    return quaternions, translations


class RelativePositionEmbedding(nn.Module):
  """RelativePositionEmbedding
    """
  def __init__(self, dim, max_rel_dist):
    super().__init__()

    self.max_rel_dist = max_rel_dist
    self.embedding = nn.Embedding(max_rel_dist * 2 + 1, dim)

  def forward(self, seq_index):
    seq_rel_dist = rearrange(seq_index, '... i -> ... i ()'
                            ) - rearrange(seq_index, '... j -> ... () j')
    seq_rel_dist = seq_rel_dist.clamp(
        -self.max_rel_dist, self.max_rel_dist
    ) + self.max_rel_dist
    return self.embedding(seq_rel_dist)


class PairwiseEmbedding(nn.Module):
  """PairwiseEmbedding
    """
  def __init__(self, dim_msa, dim_pairwise, max_rel_dist=0):
    super().__init__()

    self.to_pairwise_repr = nn.Linear(dim_msa, dim_pairwise * 2)
    self.relative_pos_emb = RelativePositionEmbedding(
        dim_pairwise, max_rel_dist
    ) if max_rel_dist > 0 else None

  def embeddings(self):
    return {'position': self.relative_pos_emb.embedding.weight}

  def forward(self, x, x_mask=None, seq_index=None):
    (_, n), device = x.shape[:2], x.device

    x_left, x_right = self.to_pairwise_repr(x).chunk(2, dim=-1)
    x = rearrange(x_left, '... i d -> ... i () d') + rearrange(
        x_right, '... j d-> ... () j d'
    )  # create pair-wise residue embeds
    if exists(self.relative_pos_emb):
      seq_index = default(seq_index, lambda: torch.arange(n, device=device))
      x = tensor_add(x, self.relative_pos_emb(seq_index))
    if exists(x_mask):
      x_mask = rearrange(x_mask, '... i -> ... i ()') * rearrange(
          x_mask, '... j -> ... () j'
      )
      return x, x_mask
    return x


def checkpoint_sequential_nargs(functions, segments, *inputs, **kwargs):
  # Hack for keyword-only parameter in a python 2.7-compliant way
  preserve = kwargs.pop('preserve_rng_state', True)

  def run_function(start, end, functions):
    def forward(*inputs):
      for j in range(start, end + 1):
        inputs = functions[j](*inputs, **kwargs)
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
      if version_cmp(torch.__version__, '1.11.0') >= 0:
        inputs = checkpoint(
            run_function(start, end, functions),
            *inputs,
            preserve_rng_state=preserve,
            use_reentrant=True
        )  # compatible with torch 2.4+
      else:
        inputs = checkpoint(
            run_function(start, end, functions), *inputs, preserve_rng_state=preserve
        )
    else:
      inputs = run_function(start, end, functions)(*inputs)
  return run_function(end + 1, len(functions) - 1, functions)(*inputs)


def layer_stack(moduleclass, depth, *args, **kwargs):
  class _LayerStack(nn.Module):
    def __init__(self, *args, **kwargs):
      super().__init__()
      self.layers = nn.ModuleList(moduleclass(*args, **kwargs) for _ in range(depth))  # pylint: disable=missing-kwargs

    def forward(self, *args, **kwargs):
      with profiler.record_function(moduleclass.__name__):
        return checkpoint_sequential_nargs(
            self.layers, len(self.layers), *args, **kwargs
        )

  return _LayerStack(*args, **kwargs)
