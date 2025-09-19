"""Diffusion model for generating 3D-structure"""
import functools
import logging
import math
from typing import Optional, Union

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from profold2.common import chemical_components, residue_constants
from profold2.model import commons, functional
from profold2.utils import compose, default, env, exists

logger = logging.getLogger(__name__)


class ScatterUtil:
  """ScatterUtil"""
  @staticmethod
  def add(
      index: torch.Tensor,
      src: torch.Tensor,
      dim: Optional[int] = -1,
      out: Optional[torch.Tensor] = None,
      out_dim: Optional[int] = None
  ) -> torch.Tensor:
    if not exists(out):
      size = list(src.shape)
      if exists(out_dim):
        size[dim] = out_dim
      else:
        size[dim] = int(torch.max(index)) + 1
      out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return torch.scatter_add(out, dim, index.long(), src)

  @staticmethod
  def sum(
      index: torch.Tensor,
      src: torch.Tensor,
      dim: Optional[int] = -1,
      out: Optional[torch.Tensor] = None,
      out_dim: Optional[int] = None
  ) -> torch.Tensor:
    return ScatterUtil.add(index, src, dim=dim, out=out, out_dim=out_dim)

  @staticmethod
  def mean(
      index: torch.Tensor,
      src: torch.Tensor,
      dim: Optional[int] = -1,
      out: Optional[torch.Tensor] = None,
      out_dim: Optional[int] = None
  ) -> torch.Tensor:
    out = ScatterUtil.sum(index, src, dim=dim, out=out, out_dim=out_dim)
    out_dim = out.shape[dim]

    one = torch.ones_like(index, dtype=src.dtype, device=src.device)
    count = torch.clamp(ScatterUtil.sum(index, one, dim=dim, out_dim=out_dim), min=1)

    return commons.tensor_div(out, count)


class AtomUtil:
  """AtomUtil"""
  @staticmethod
  def pad(
      t: torch.Tensor,
      dim: int,
      pad: Union[tuple[int], list[int]],
  ) -> torch.Tensor:
    dim = dim % t.dim()
    if pad != (0, 0):
      pad = (0, 0) * (t.dim() - dim - 1) + pad
      t = F.pad(t, pad=pad)
    return t

  @staticmethod
  def reshape(
      t: torch.Tensor, dim: int, shape: Union[tuple[int], list[int]]
  ) -> torch.Tensor:
    dim = dim % t.dim()
    assert t.shape[dim] == math.prod(shape)
    return torch.reshape(t, t.shape[:dim] + tuple(shape) + t.shape[dim + 1:])

  @staticmethod
  def permute(t: torch.Tensor, dim: int) -> torch.Tensor:
    dim = dim % t.dim()
    if dim + 1 < t.dim():
      dims = tuple(range(dim)) + (-1, ) + tuple(range(dim, t.dim() - 1))
      t = torch.permute(t, dims)
    return t

  @staticmethod
  def unfold(
      q_window_size: int,
      k_window_size: int,
      dim: int,
      q: torch.Tensor,
      k: Optional[torch.Tensor] = None,
  ) -> tuple[torch.Tensor, Optional[torch.Tensor], int]:
    assert k_window_size % 2 == 0 and q_window_size % 2 == 0
    assert k_window_size >= q_window_size
    assert not exists(k) or q.shape[dim] == k.shape[dim]

    n = q.shape[dim] // q_window_size + (1 if q.shape[dim] % q_window_size > 0 else 0)
    q_padding = 0, n * q_window_size - q.shape[dim]
    # q
    q = AtomUtil.reshape(
        AtomUtil.pad(q, dim=dim, pad=q_padding), dim=dim, shape=(n, q_window_size)
    )
    # k
    if exists(k):
      k_padding = (
          (k_window_size - q_window_size) // 2,
          n * q_window_size + (k_window_size - q_window_size) // 2 - k.shape[dim]
      )
      k = AtomUtil.pad(k, dim=dim, pad=k_padding)
      k = AtomUtil.permute(
          k.unfold(dim, size=k_window_size, step=q_window_size), dim=dim
      )
    return q, k, q_padding[1]

  @staticmethod
  def gather(
      t: torch.Tensor,
      atom_to_token_idx: torch.Tensor,
      q_window_size: Optional[int] = None,
      k_window_size: Optional[int] = None
  ) -> torch.Tensor:
    if exists(q_window_size) and exists(k_window_size):  # pair
      n = t.shape[-2]
      t = rearrange(t, '... i j d -> ... (i j) d')

      q_atom_to_token_idx, k_atom_to_token_idx, *_ = AtomUtil.unfold(
          q_window_size,
          k_window_size,
          dim=-1,
          q=atom_to_token_idx,
          k=atom_to_token_idx
      )
      atom_to_token_idx = rearrange(
          q_atom_to_token_idx[..., :, None] * n + k_atom_to_token_idx[..., None, :],
          '... c i j -> ... (c i j)'
      )

    t = functional.batched_gather(t, atom_to_token_idx, dim=-2, has_batch_dim=True)

    if exists(q_window_size) and exists(k_window_size):  # pair
      t = rearrange(t, '... (c i j) d -> ... c i j d', i=q_window_size, j=k_window_size)
    return t

  @staticmethod
  def flatten(
      atom_to_token_idx: torch.Tensor,
      atom_within_token_idx: torch.Tensor,
      coord: torch.Tensor,
      coord_mask: Optional[torch.Tensor] = None
  ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    coord = functional.batched_gather(
        rearrange(coord, '... i c d -> ... (i c) d'),
        atom_to_token_idx * coord.shape[-2] + atom_within_token_idx,
        has_batch_dim=True
    )
    if exists(coord_mask):
      coord_mask = functional.batched_gather(
          rearrange(coord_mask, '... i c -> ... (i c)'),
          atom_to_token_idx * coord_mask.shape[-1] + atom_within_token_idx,
          has_batch_dim=True
      )
      assert coord.shape[:-1] == coord_mask.shape
    return coord, coord_mask

  @staticmethod
  def unflatten(
      atom_to_token_idx: torch.Tensor,
      atom_within_token_idx: torch.Tensor,
      coord: torch.Tensor,
      coord_mask: Optional[torch.Tensor] = None,
      num_tokens: int = None
  ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    num_tokens = default(num_tokens, int(torch.max(atom_to_token_idx) + 1))
    c = residue_constants.atom14_type_num
    index = atom_to_token_idx * c + atom_within_token_idx

    coord = torch.index_copy(
        torch.zeros(
            coord.shape[:-2] + (num_tokens * c, coord.shape[-1]),
            device=coord.device,
            dtype=coord.dtype
        ), -2, index, coord
    )
    coord = rearrange(coord, '... (i c) d -> ... i c d', c=c)
    if exists(coord_mask):
      coord_mask = torch.index_copy(
          torch.zeros(
              coord_mask.shape[:-1] + (num_tokens * c, ),
              device=coord_mask.device,
              dtype=coord_mask.dtype
          ), -1, index, coord_mask
      )
      coord_mask = rearrange(coord_mask, '... (i c) -> ... i c', c=c)
    return coord, coord_mask


class InputFeatureEmbedder(nn.Module):
  """InputFeatureEmbedder
    """
  def __init__(
      self,
      dim=(128, 16),
      dim_token=384,
      num_tokens=len(residue_constants.restypes_with_x),
      depth=3,
      heads=4,
      atom_query_window_size=32,
      atom_key_window_size=128,
      atom_feats=None
  ):
    super().__init__()

    self.atom_encoder = AtomAttentionEncoder(
        dim=dim,
        dim_token=dim_token,
        depth=depth,
        heads=heads,
        atom_query_window_size=atom_query_window_size,
        atom_key_window_size=atom_key_window_size,
        atom_feats=atom_feats
    )
    self.num_tokens = num_tokens

  def forward(self, batch, shard_size=None):
    single_repr = self.atom_encoder(batch, shard_size=shard_size)
    return torch.cat(
        (F.one_hot(batch['seq'], self.num_tokens + 1), single_repr), dim=-1  # pylint: disable=not-callable
    )


class RelativePositionEncoding(nn.Module):
  """RelativePositionEncoding
    """
  def __init__(self, dim, r_max=32, s_max=2):
    super().__init__()

    _, dim_pairwise = commons.embedd_dim_get(dim)
    self.r_max = r_max
    self.s_max = s_max

    # (d_{ij}^{seq_index}, d_{ij}^{token_index}, b_{ij}^{seq_entity}, d{ij}^{seq_sym}
    # 2*(r_{max} + 1) + 2*(r_{max} + 1) + 1 + 2*(s_{max} + 1)
    self.proj = nn.Linear(
        2 * 2 * self.r_max + 2 * self.s_max + 7, dim_pairwise, bias=False
    )

  def forward(
      self, seq_index, seq_color, seq_sym, seq_entity, token_index, shard_size=None
  ):
    def run_proj(seq_index_j, seq_color_j, seq_sym_j, seq_entity_j, token_index_j):
      bij_seq_index = (seq_index[..., :, None] == seq_index_j[..., None, :])
      bij_seq_color = (seq_color[..., :, None] == seq_color_j[..., None, :])
      bij_seq_entity = (seq_entity[..., :, None] == seq_entity_j[..., None, :])

      dij_seq_index = F.one_hot(  # pylint: disable=not-callable
          torch.where(
              bij_seq_color,
              torch.clamp(
                  seq_index[..., :, None] - seq_index_j[..., None, :],
                  min=-self.r_max, max=self.r_max
              ) + self.r_max,
              2 * self.r_max + 1
          ).long(),
          2 * (self.r_max + 1)
      )
      dij_token_index = F.one_hot(  # pylint: disable=not-callable
          torch.where(
              bij_seq_color * bij_seq_index,
              torch.clamp(
                  token_index[..., :, None] - token_index_j[..., None, :],
                  min=-self.r_max, max=self.r_max
              ) + self.r_max,
              2 * self.r_max + 1
          ).long(),
          2 * (self.r_max + 1)
      )
      dij_seq_sym = F.one_hot(  # pylint: disable=not-callable
          torch.where(
              seq_entity[..., :, None] == seq_entity_j[..., None, :],
              torch.clamp(
                  seq_sym[..., :, None] - seq_sym_j[..., None, :],
                  min=-self.s_max, max=self.s_max
              ) + self.s_max,
              2 * self.s_max + 1
          ).long(),
          2 * (self.s_max + 1)
      )
      return self.proj(
          torch.cat(
              (dij_seq_index, dij_token_index, bij_seq_entity[..., None], dij_seq_sym),
              dim=-1
          ).float()
      )

    return functional.sharded_apply(
        run_proj, [seq_index, seq_color, seq_sym, seq_entity, token_index],
        shard_size=None if self.training else shard_size,
        shard_dim=-1,
        cat_dim=-2
    )


class AtomPairwiseEmbedding(nn.Module):
  """AtomPairwiseEmbedding
    """
  def __init__(self, dim):
    super().__init__()

    dim_single, dim_pairwise = commons.embedd_dim_get(dim)
    self.to_pairwise_repr = nn.Linear(dim_single, dim_pairwise * 2, bias=False)

  def forward(self, x, query_window_size=32, key_window_size=128):
    x_i, x_j = torch.chunk(self.to_pairwise_repr(x), 2, dim=-1)
    x_i, x_j, *_ = AtomUtil.unfold(
        query_window_size, key_window_size, dim=-2, q=x_i, k=x_j
    )
    return x_i[..., None, :] + x_j[..., None, :, :]


class DiffusionTransformerBlock(nn.Module):
  """DiffusionTransformerBlock"""
  def __init__(self, *, dim, dim_cond, heads, dropout=0., **kwargs):
    super().__init__()

    dim_single, dim_pairwise = commons.embedd_dim_get(dim)

    self.attn = commons.AttentionPairBias(
        dim_node=dim_single,
        dim_edge=dim_pairwise,
        dim_cond=dim_cond,
        heads=heads,
        q_use_bias=True,
        o_use_bias=False,
        g_use_bias=False,
        **kwargs
    )
    self.ff = commons.ConditionedFeedForward(dim_single, dim_cond)
    self.dropout_fn = functools.partial(commons.shaped_dropout, p=dropout)

  def forward(
      self,
      x,
      *,
      single_cond,
      pair_cond,
      mask=None,
      context=None,
      context_cond=None,
      context_mask=None,
      pair_bias=None,
      pair_mask=None,
      shard_size=None
  ):
    def dropout_wrap(f, *args, **kwargs):
      shape = x.shape[:-2] + (1, 1)
      return self.dropout_fn(f(*args, **kwargs), shape=shape, training=self.training)

    if exists(context_cond):
      context = default(context, x)
      if not torch.is_tensor(context):
        x, context = context(x)

    # run attn and ff parallel: x += attn(x) + ff(x)
    x = commons.tensor_add(
        x,
        dropout_wrap(
            self.attn,
            x,
            pair_cond,
            cond=single_cond,
            mask=mask,
            edge_bias=pair_bias,
            edge_mask=pair_mask,
            context=context,
            context_cond=context_cond,
            context_mask=context_mask
        ) + dropout_wrap(self.ff, x, single_cond, shard_size=shard_size)
    )

    if exists(context_cond):
      x = rearrange(x, '... c i d -> ... (c i) d')

    return x


class AtomTransformer(nn.Module):
  """AtomTransformer"""
  def __init__(
      self,
      dim,
      depth=3,
      heads=4,
      dim_head=32,
      query_window_size=32,
      key_window_size=128
  ):
    super().__init__()

    dim_single, _ = commons.embedd_dim_get(dim)

    self.query_window_size = query_window_size
    self.key_window_size = key_window_size

    self.difformer = commons.layer_stack(
        DiffusionTransformerBlock,
        depth,
        checkpoint_segment_size=env(
            'profold2_atomtransformer_checkpoint_segment_size', defval=1, dtype=int
        ),
        dim=dim,
        dim_cond=dim_single,
        has_context=True,
        heads=heads,
        dim_head=dim_head
    )

  def _context(self, query):
    # HACK: split query to (query, context)
    query, context, *_ = AtomUtil.unfold(
        self.query_window_size, self.key_window_size, dim=-2, q=query, k=query
    )
    return query, context

  def forward(
      self,
      single_repr,
      single_cond,
      pair_cond,
      mask=None,
      pair_mask=None,
      shard_size=None
  ):
    query_cond, context_cond, padding = AtomUtil.unfold(
        self.query_window_size,
        self.key_window_size,
        dim=-2,
        q=single_cond,
        k=single_cond
    )
    if exists(mask):
      query_mask, context_mask, *_ = AtomUtil.unfold(
          self.query_window_size, self.key_window_size, dim=-1, q=mask, k=mask
      )
    else:
      query_mask, context_mask = None, None
    query = self.difformer(
        single_repr,
        single_cond=query_cond,
        pair_cond=pair_cond,
        context=self._context,
        context_cond=context_cond,
        mask=query_mask,
        context_mask=context_mask,
        pair_mask=pair_mask,
        shard_size=shard_size
    )
    if exists(mask):
      return commons.tensor_mul(query[..., :-padding, :], mask[..., None])
    return query[..., :-padding, :]


class AtomAttentionEncoder(nn.Module):
  """AtomAttentionEncoder"""
  def __init__(
      self,
      dim,
      dim_trunk=(384, 128),
      dim_token=768,
      depth=3,
      heads=4,
      atom_query_window_size=32,
      atom_key_window_size=128,
      atom_feats=None,
      has_coords=False
  ):
    super().__init__()

    dim_single, dim_pairwise = commons.embedd_dim_get(dim)
    dim_trunk_single, dim_trunk_pairwise = commons.embedd_dim_get(dim_trunk)

    self.atom_feats = default(
        atom_feats, (
            ('ref_pos', 3, nn.Identity()),
            ('ref_charge', 1, compose(torch.arcsinh, Rearrange('... -> ... ()'))),
            ('ref_mask', 1, Rearrange('... -> ... ()')),
            (
                'ref_element',
                chemical_components.elem_type_num,
                functools.partial(
                    F.one_hot, num_classes=chemical_components.elem_type_num
                )
            ),
            (
                'ref_atom_name_chars',
                chemical_components.name_char_channel * chemical_components.name_char_num,
                compose(
                    functools.partial(
                        F.one_hot, num_classes=chemical_components.name_char_num
                    ),
                    Rearrange('... c d -> ... (c d)')
                )
            )
        )
    )
    dim_atom_feats = sum(d for _, d, _ in self.atom_feats)
    self.to_single_cond = nn.Linear(dim_atom_feats, dim_single, bias=False)
    self.to_pair_cond = nn.Linear(3 + 1 + 1, dim_pairwise, bias=False)

    if has_coords:
      self.from_trunk_single_cond = nn.Sequential(
          nn.LayerNorm(dim_trunk_single, bias=False),
          nn.Linear(dim_trunk_single, dim_single, bias=False)
      )
      self.from_trunk_pair_cond = nn.Sequential(
          nn.LayerNorm(dim_trunk_pairwise, bias=False),
          nn.Linear(dim_trunk_pairwise, dim_pairwise, bias=False)
      )
      self.from_coord = nn.Linear(3, dim_single, bias=False)

    self.outer_add = AtomPairwiseEmbedding(dim)
    # self.outer_ff = commons.FeedForward(dim_pairwise)
    self.outer_ff = commons.layer_stack(
        nn.Sequential, 3, nn.ReLU(), nn.Linear(dim_pairwise, dim_pairwise, bias=False)
    )

    self.transformer = AtomTransformer(
        dim,
        depth=depth,
        heads=heads,
        query_window_size=atom_query_window_size,
        key_window_size=atom_key_window_size
    )

    self.to_out = nn.Linear(dim_single, dim_token, bias=False)

  def forward(
      self,
      batch,
      r_noisy=None,
      trunk_single_cond=None,
      trunk_pair_cond=None,
      shard_size=None
  ):
    atom_to_token_idx = batch['atom_to_token_idx']

    # create the atom single conditioning: Embed per-atom meta data
    atom_single_cond = self.to_single_cond(
        torch.cat([f(batch[k]) for k, _, f in self.atom_feats], dim=-1)
    ) * batch['ref_mask'][..., None]

    # TODO: add ref_mask
    # embed offsets between atom reference position, pairwise inverse squared
    # distances, and the valid mask.
    ref_pos_i, ref_pos_j, *_ = AtomUtil.unfold(
        self.transformer.query_window_size,
        self.transformer.key_window_size,
        dim=-2,
        q=batch['ref_pos'],
        k=batch['ref_pos']
    )
    ref_space_uid_i, ref_space_uid_j, *_ = AtomUtil.unfold(
        self.transformer.query_window_size,
        self.transformer.key_window_size,
        dim=-1,
        q=batch['ref_space_uid'],
        k=batch['ref_space_uid']
    )
    dij_ref = ref_pos_i[..., :, None, :] - ref_pos_j[..., None, :, :]
    bij_ref = (ref_space_uid_i[..., :, None] == ref_space_uid_j[..., None, :])
    pair_cond = self.to_pair_cond(
        torch.cat(
            (
                dij_ref * bij_ref[..., None],
                1 / (1 + torch.sum(dij_ref**2, dim=-1, keepdim=True)) * bij_ref[..., None],
                bij_ref[..., None]
            ),
            dim=-1
        )
    )

    # initialise the atom single representation as the single conditioning.
    query, query_cond, mask = atom_single_cond, atom_single_cond, batch['ref_mask']

    # if provided, add trunk embeddings and noisy positions.
    assert not hasattr(self, 'from_trunk_single_cond') ^ exists(trunk_single_cond)
    if exists(trunk_single_cond):
      # broadcast the single embedding from the trunk
      query_cond = commons.tensor_add(
          AtomUtil.gather(
              self.from_trunk_single_cond(trunk_single_cond), atom_to_token_idx
          ), query_cond
      )
    assert not hasattr(self, 'from_trunk_pair_cond') ^ exists(trunk_pair_cond)
    if exists(trunk_pair_cond):
      # broadcast the pair embedding from the trunk
      pair_cond = commons.tensor_add(
          pair_cond,
          AtomUtil.gather(
              self.from_trunk_pair_cond(trunk_pair_cond),
              atom_to_token_idx,
              self.transformer.query_window_size,
              self.transformer.key_window_size
          )
      )
    assert not hasattr(self, 'from_coord') ^ exists(r_noisy)
    if exists(r_noisy):
      # add the noisy positions.
      query = commons.tensor_add(self.from_coord(r_noisy), query_cond)
      query_cond = rearrange(query_cond, '... i d -> ... () i d')
      pair_cond = rearrange(pair_cond, '... c i j d -> ... () c i j d')
      mask = rearrange(mask, '... i -> ... () i')

    # add the combined single conditioning to the pair representation.
    pair_cond = commons.tensor_add(
        pair_cond,
        self.outer_add(
            F.relu(query_cond),
            self.transformer.query_window_size,
            self.transformer.key_window_size
        )
    )
    # run a small MLP on the pair activations.
    pair_cond = commons.tensor_add(pair_cond, self.outer_ff(pair_cond))
    # cross attention transformer.
    query = self.transformer(
        query, query_cond, pair_cond, mask=mask, shard_size=shard_size
    )
    # aggregate per-atom representation to per-token representation.
    token_single_cond = F.relu(self.to_out(query))
    atom_to_token_idx = repeat(
        atom_to_token_idx, '... i -> ... i d', d=token_single_cond.shape[-1]
    )
    if exists(r_noisy):
      atom_to_token_idx = rearrange(atom_to_token_idx, '... i d -> ... () i d')
    token_single_cond = ScatterUtil.mean(atom_to_token_idx, token_single_cond, dim=-2)

    query_skip, query_cond_skip, pair_skip = query, query_cond, pair_cond
    return token_single_cond, query_skip, query_cond_skip, pair_skip


class AtomAttentionDecoder(nn.Module):
  """AtomAttentionDecoder"""
  def __init__(
      self,
      dim,
      dim_token=768,
      depth=3,
      heads=4,
      atom_query_window_size=32,
      atom_key_window_size=128,
  ):
    super().__init__()

    dim_single, _ = commons.embedd_dim_get(dim)

    self.from_token = nn.Linear(dim_token, dim_single, bias=False)
    self.transformer = AtomTransformer(
        dim,
        depth=depth,
        heads=heads,
        query_window_size=atom_query_window_size,
        key_window_size=atom_key_window_size
    )
    self.to_out = nn.Sequential(
        nn.LayerNorm(dim_single, bias=False), nn.Linear(dim_single, 3, bias=False)
    )

  def forward(
      self,
      batch,
      token_single_cond,
      query_skip,
      context_skip,
      pair_skip,
      shard_size=None
  ):
    # broadcast per-token activations to per-atom activations and add the skip
    # connection
    atom_single_cond = commons.tensor_add(
        AtomUtil.gather(
            self.from_token(token_single_cond),
            rearrange(batch['atom_to_token_idx'], '... i -> ... () i')
        ), query_skip
    )
    # cross attention transformer
    atom_single_cond = self.transformer(
        atom_single_cond, context_skip, pair_skip, shard_size=shard_size
    )
    # map to positions update
    return self.to_out(atom_single_cond)


class FourierEmbedding(nn.Module):
  """FourierEmbedding
    """
  def __init__(self, dim, seed=2147483647):
    super().__init__()

    generator = torch.Generator()
    generator.manual_seed(
        env('profold2_fourier_embedding_seed', defval=seed, dtype=int)
    )
    # randomly generate weight/bias once before training
    self.w = nn.Parameter(torch.randn(dim, generator=generator), requires_grad=False)
    self.b = nn.Parameter(torch.randn(dim, generator=generator), requires_grad=False)

  def forward(self, t):
    # compute embeddings. scale w by t
    v = t[..., None] * self.w + self.b
    return torch.cos(2 * torch.pi * v)


class DiffusionConditioning(nn.Module):
  """DiffusionConditioning"""
  def __init__(self, dim, dim_noise=256, dim_inputs=449, sigma_data=16.0):
    super().__init__()

    dim_single, dim_pairwise = commons.embedd_dim_get(dim)

    self.from_single = nn.Sequential(
        nn.LayerNorm(dim_single + dim_inputs, bias=False),
        nn.Linear(dim_single + dim_inputs, dim_single, bias=False)
    )
    self.from_pairwise = nn.Sequential(
        nn.LayerNorm(dim_pairwise * 2, bias=False),
        nn.Linear(dim_pairwise * 2, dim_pairwise, bias=False)
    )
    self.from_pos_emb = RelativePositionEncoding(dim=dim)
    self.from_noise = nn.Sequential(
        FourierEmbedding(dim_noise),
        nn.LayerNorm(dim_noise, bias=False),
        nn.Linear(dim_noise, dim_single, bias=False)
    )
    self.to_single = commons.residue_stack(
        commons.FeedForward, 2, dim_single, mult=2, activation='SwiGLU', use_bias=False
    )
    self.to_pairwise = commons.residue_stack(
        commons.FeedForward,
        2,
        dim_pairwise,
        mult=2,
        activation='SwiGLU',
        use_bias=False
    )

    self.sigma_data = sigma_data

  def forward(
      self,
      batch,
      *,
      noise_level,
      inputs,
      trunk_single_cond,
      trunk_pair_cond,
      shard_size=None
  ):
    # single conditioning
    s = self.from_single(torch.cat((trunk_single_cond, inputs), dim=-1))
    s = commons.tensor_add(
        rearrange(s, '... i d -> ... () i d'),
        rearrange(
            self.from_noise(torch.log(noise_level / self.sigma_data) / 4.),
            '... m d -> ... m () d'
        )
    )
    s = self.to_single(s)

    # pair conditioning
    # TODO: token_index
    x = self.from_pairwise(
        torch.cat(
            (
                trunk_pair_cond,
                self.from_pos_emb(
                    batch['seq_index'],
                    batch['seq_color'],
                    batch['seq_sym'],
                    batch['seq_entity'],
                    batch['seq_index'],
                    shard_size=shard_size
                )
            ),
            dim=-1
        )
    )
    x = self.to_pairwise(x)

    return s, x


class DiffusionModule(nn.Module):
  """DiffusionModule"""
  def __init__(
      self,
      dim,
      dim_atom=(128, 16),
      dim_token=768,
      dim_noise=256,
      sigma_data=16.0,
      atom_encoder_depth=3,
      atom_encoder_head_num=4,
      atom_encoder_feats=None,
      transformer_depth=24,
      transformer_group_size=4,
      transformer_head_num=16,
      transformer_dim_head=48,
      transformer_dropout_min=0,
      transformer_dropout_max=0,
      atom_decoder_depth=3,
      atom_decoder_head_num=4,
      atom_query_window_size=32,
      atom_key_window_size=128,
  ):
    super().__init__()

    dim_single, dim_pairwise = commons.embedd_dim_get(dim)

    self.conditioning = DiffusionConditioning(
        dim, dim_noise=dim_noise, sigma_data=sigma_data
    )

    self.atom_encoder = AtomAttentionEncoder(
        dim=dim_atom,
        dim_trunk=dim,
        dim_token=dim_token,
        depth=atom_encoder_depth,
        heads=atom_encoder_head_num,
        atom_query_window_size=atom_query_window_size,
        atom_key_window_size=atom_key_window_size,
        atom_feats=atom_encoder_feats,
        has_coords=True
    )
    self.transformer_in = nn.Sequential(
        nn.LayerNorm(dim_single, bias=False),
        nn.Linear(dim_single, dim_token, bias=False)
    )
    assert transformer_depth % transformer_group_size == 0
    self.transformer = commons.layer_stack(
        DiffusionTransformerBlock,
        depth=transformer_depth // transformer_group_size,
        checkpoint_segment_size=env(
            'profold2_diffuser_checkpoint_segment_size', defval=1, dtype=int
        ),
        dim=(dim_token, dim_pairwise),
        dim_cond=dim_single,
        group_size=transformer_group_size,
        heads=transformer_head_num,
        dim_head=transformer_dim_head
    )
    self.transformer_out = nn.LayerNorm(dim_token, bias=False)
    self.atom_decoder = AtomAttentionDecoder(
        dim=dim_atom,
        dim_token=dim_token,
        depth=atom_decoder_depth,
        heads=atom_decoder_head_num,
        atom_query_window_size=atom_query_window_size,
        atom_key_window_size=atom_key_window_size
    )

  def forward(
      self,
      batch,
      *,
      x_noisy,
      x_mask,
      noise_level,
      inputs,
      trunk_single_cond,
      trunk_pair_cond,
      shard_size=None
  ):
    # conditioning
    single_cond, pair_cond = self.conditioning(
        batch,
        noise_level=noise_level,
        inputs=inputs,
        trunk_single_cond=trunk_single_cond,
        trunk_pair_cond=trunk_pair_cond,
        shard_size=shard_size
    )

    ##################################################
    # EDM: r_noisy = c_in * x_noisy
    #      where c_in = 1 / sqrt(sigma_data^2 + sigma^2)
    ##################################################

    # scale positions to dimensionless
    r_noisy = x_noisy / torch.sqrt(
        self.conditioning.sigma_data**2 + noise_level**2
    )[..., None, None]

    ##################################################
    # EDM: r_update = F_theta(r_noisy, c_noise(sigma))
    ##################################################

    # sequence-local Atom Attention and aggregation to coasrse-grained tokens
    token_cond, query_skip, context_skip, pair_skip = self.atom_encoder(
        batch,
        r_noisy,
        trunk_single_cond=trunk_single_cond,
        trunk_pair_cond=trunk_pair_cond
    )

    # full self-attention on token level
    token_cond = commons.tensor_add(token_cond, self.transformer_in(single_cond))
    token_cond = self.transformer(
        token_cond, single_cond=single_cond, pair_cond=pair_cond
    )
    token_cond = self.transformer_out(token_cond)

    # broadcast token activations to atoms and run Sequence-local Atom Attention
    r_update = self.atom_decoder(batch, token_cond, query_skip, context_skip, pair_skip)

    ##################################################
    # EDM: D = c_skip * x_noisy + c_out * r_update
    #      c_skip = sigma_data^2 / (sigma_data^2 + sigma^2)
    #      c_out = (sigma_data * sigma) / sqrt(sigma_data^2 + sigma^2)
    #      s_ratio = 1 + (sigma / sigma_data)^2
    #      c_skip = 1 / s_ratio
    #      c_out = sigma / sqrt(s_ratio)
    ##################################################

    # rescale updates to positions and combine with input positions
    noise_level = noise_level[..., None, None]
    s_ratio = 1 + (noise_level / self.conditioning.sigma_data)**2
    x_denoised = x_noisy / s_ratio + r_update * noise_level / torch.sqrt(s_ratio)

    return x_denoised


class DiffusionNoiseSampler(nn.Module):
  """DiffusionNoiseSampler: sample the noise-level."""
  def __init__(self, mean=-1.2, std=1.5, sigma_data=16.):
    super().__init__()

    self.mean = mean
    self.std = std
    self.sigma_data = sigma_data

  def forward(self, *size, device=None):
    x = torch.randn(*size, device=device)
    return torch.exp(self.mean + self.std * x) * self.sigma_data


class DiffusionNoiseScheduler(nn.Module):
  """DiffusionNoiseScheduler: schedule the noise-level (time steps)."""
  def __init__(self, sigma_min=4e-4, sigma_max=160., rho=7., sigma_data=16.):
    super().__init__()

    self.s_min = sigma_min
    self.s_max = sigma_max
    self.rho = rho
    self.s_data = sigma_data

  def forward(self, steps=200, device=None, dtype=torch.float32):
    # t = torch.linspace(0, 1, steps + 1, device=device, dtype=dtype)
    s_min, s_max = self.s_min**(1. / self.rho), self.s_max**(1. / self.rho)
    return [
        self.s_data * (s_max + (s_min - s_min) * t / steps)**self.rho
        for t in range(steps + 1)
    ]
    # return self.s_data * (s_max + (s_min - s_max) * t)**self.rho


class CentreRandomAugmentation(nn.Module):
  def __init__(self, trans_scale_factor=1.0):
    super().__init__()

    self.trans_scale_factor = trans_scale_factor

  def forward(self, x, mask=None, batch_size=None):
    if exists(mask):
      c = functional.masked_mean(value=x, mask=mask, dim=-2, keepdim=True)
    else:
      c = torch.mean(x, dim=-2, keepdim=True)

    if exists(batch_size):
      x = repeat(x, '... i d -> n i d', n=batch_size)
    x = commons.tensor_sub(x, c)
    R, t = functional.rigids_from_randn(x.shape[:-1], device=x.device)  # pylint: disable=invalid-name
    x = functional.rigids_apply((R, t * self.trans_scale_factor), x)

    return x


class DiffusionSampler(nn.Module):
  """DiffusionSampler"""
  def __init__(
      self,
      dim,
      dim_atom=(128, 16),
      dim_noise=256,
      sigma_mean=-1.2,
      sigma_std=1.5,
      sigma_min=4e-4,
      sigma_max=160,
      rho=7.,
      sigma_data=16.0,
      diffuser_atom_encoder_depth=3,
      diffuser_atom_encoder_head_num=4,
      diffuser_atom_encoder_feats=None,
      diffuser_transformer_depth=24,
      diffuser_transformer_group_size=4,
      diffuser_transformer_head_num=16,
      diffuser_atom_decoder_depth=3,
      diffuser_atom_decoder_head_num=4,
      diffuser_atom_query_window_size=32,
      diffuser_atom_key_window_size=128,
  ):
    super().__init__()

    dim_single, dim_pairwise = commons.embedd_dim_get(dim)

    self.rand_augmenter = CentreRandomAugmentation()
    self.noise_sampler = DiffusionNoiseSampler(
        sigma_mean=sigma_mean, sigma_std=sigma_std, sigma_data=sigma_data
    )
    self.noise_scheduler = DiffusionNoiseScheduler(
        sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, sigma_data=sigma_data
    )
    self.diffuser = DiffusionModule(
        self,
        dim,
        dim_atom=dim_atom,
        dim_noise=dim_noise,
        sigma_data=sigma_data,
        atom_encoder_depth=diffuser_atom_encoder_depth,
        atom_encoder_head_num=diffuser_atom_encoder_head_num,
        atom_encoder_feats=diffuser_atom_encoder_feats,
        transformer_depth=diffuser_transformer_depth,
        transformer_group_size=diffuser_transformer_group_size,
        transformer_head_num=diffuser_transformer_head_num,
        atom_decoder_depth=diffuser_atom_decoder_depth,
        atom_decoder_head_num=diffuser_atom_decoder_head_num,
        atom_query_window_size=diffuser_atom_query_window_size,
        atom_key_window_size=diffuser_atom_key_window_size
    )

  def forward(
      self,
      batch,
      trunk_single_cond=None,
      trunk_pair_cond=None,
      diffuer_batch_size=None,
      shard_size=None
  ):
    # apply random rotation and translation
    coord, coord_mask = AtomUtil.flatten(
        batch['atom_to_token_idx'],
        batch['atom_within_token_idx'],
        batch['coord'],
        batch['coord_mask']
    )
    x_noisy = self.rand_augmenter(
        coord, mask=coord_mask, batch_size=diffuer_batch_size
    )
    # sigma: independent noise-level [..., N_sample]
    noise_level = self.noise_sampler(
        x.shape[:-1], batch_size=diffuer_batch_size, device=x.device
    )
    # noise
    noise_level = torch.randn_like(x) * noise_level[..., None, None]
    # denoised_x
    x_denoised = self.diffuser(
        batch,
        x_noisy=x_noisy,
        x_mask=x_mask,
        noise_level=noise_level,
        trunk_single_cond=trunk_single_cond,
        trunk_pair_cond=trunk_pair_cond,
        shard_size=shard_size
    )
    return x_noisy, x_denoised, sigma

  def sample(
      self,
      batch,
      trunk_single_cond=None,
      trunk_pair_cond=None,
      steps=200,
      gamma0: float = 0.8,
      gamma_min: float = 1.0,
      noise_scale_lambda: float = 1.003,
      step_scale_eta: float = 1.5,
      shard_size=None
  ):
    assert not self.training
    noise_level = self.noise_scheduler(steps=steps)

    x = noise_level[0] * torch.randn()
    for tau in range(1, len(noise_level)):
      x = self.rand_augmenter(x)
      gamma = gamma0 if noise_level[tau] > gamma_min else 0
      t_hat = noise_level[tau - 1] * (gamma + 1)
      delta = noise_scale_lambda * math.sqrt(t_hat**2 - noise_level[tau - 1]**2)
      x_noisy = x + delta

      # denoise
      x_denoised = self.diffuser(batch, x_noisy=x_noisy)

