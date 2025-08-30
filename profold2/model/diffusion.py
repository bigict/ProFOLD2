"""Diffusion model for generating 3d-structure"""
import functools
import logging
import math
from typing import Optional, Union

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from profold2.common import residue_constants
from profold2.model import commons, functional
from profold2.utils import default, exists

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
      m: Optional[torch.Tensor] = None
  ) -> tuple[
      torch.Tensor,
      Optional[torch.Tensor],
      tuple[int, int],
      tuple[Optional[torch.Tensor], Optional[torch.Tensor]]
  ]:
    assert k_window_size % 2 == 0 and q_window_size % 2 == 0
    assert k_window_size >= q_window_size
    assert q.shape[dim] == k.shape[dim]

    n = q.shape[dim] // q_window_size + (1 if q.shape[dim] % q_window_size > 0 else 0)

    q_padding = 0, n * q_window_size - q.shape[dim]
    # q & m
    q = AtomUtil.reshape(
        AtomUtil.pad(q, dim=dim, pad=q_padding), dim=dim, shape=(n, q_window_size)
    )
    if exists(m):
      qm = AtomUtil.reshape(
          AtomUtil.pad(m, dim=dim, pad=q_padding), dim=dim, shape=(n, q_window_size)
      )
    else:
      qm = None

    # k & m
    k_padding = (
        (k_window_size - q_window_size) // 2,
        n * q_window_size + (k_window_size - q_window_size) // 2 - k.shape[dim]
    )
    if exists(k):
      k = AtomUtil.pad(k, dim=dim, pad=k_padding)
      k = AtomUtil.permute(
          k.unfold(dim, size=k_window_size, step=q_window_size), dim=dim
      )
    if exists(m):
      km = AtomUtil.pad(m, dim=dim, pad=k_padding)
      km = AtomUtil.permute(
          km.unfold(dim, size=k_window_size, step=q_window_size), dim=dim
      )
    else:
      km = None

    return q, k, q_padding, (qm, km)


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
      atom_query_window_size=64,
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
    return functional.sharded_apply(
        self._forward, [seq_index, seq_color, seq_sym, seq_entity, token_index],
        shard_size=None if self.training else shard_size,
        shard_dim=-1,
        cat_dim=-2
    )

  def _forward(self, seq_index, seq_color, seq_sym, seq_entity, token_index):
    bij_seq_index = (seq_index[..., :, None] == seq_index[..., None, :])
    bij_seq_color = (seq_color[..., :, None] == seq_color[..., None, :])
    bij_seq_entity = (seq_entity[..., :, None] == seq_entity[..., None, :])

    dij_seq_index = F.one_hot(  # pylint: disable=not-callable
        torch.where(
            bij_seq_color,
            torch.clamp(
                seq_index[..., :, None] - seq_index[..., None, :],
                min=-self.r_max, max=self.r_max
            ) + self.r_max,
            2 * self.r_max + 1
        ),
        2 * self.r_max + 1
    )
    dij_token_index = F.one_hot(  # pylint: disable=not-callable
        torch.where(
            bij_seq_color * bij_seq_index,
            torch.clamp(
                token_index[..., :, None] - token_index[..., None, :],
                min=-self.r_max, max=self.r_max
            ) + self.r_max,
            2 * self.r_max + 1
        ),
        2 * self.r_max + 1
    )
    dij_seq_sym = F.one_hot(  # pylint: disable=not-callable
        torch.where(
            seq_entity[..., :, None] == seq_entity[..., None, :],
            torch.clamp(
                seq_sym[..., :, None] - seq_sym[..., None, :],
                min=-self.s_max, max=self.s_max
            ) + self.s_max,
            2 * self.s_max + 1
        ),
        2 * self.s_max + 1
    )
    return self.proj(
        torch.cat(
            (dij_seq_index, dij_token_index, bij_seq_entity, dij_seq_sym), dim=-1
        )
    )


class AtomPairwiseEmbedding(nn.Module):
  """AtomPairwiseEmbedding
    """
  def __init__(self, dim):
    super().__init__()

    dim_single, dim_pairwise = commons.embedd_dim_get(dim)
    self.to_pairwise_repr = nn.Linear(dim_single, dim_pairwise * 2)

  def forward(self, x, atom_query_window_size=64, atom_key_window_size=128):
    x_i, x_j = torch.chunk(self.to_pairwise_repr(x), 2, dim=-1)
    x_i, x_j, *_ = AtomUtil.unfold(
        atom_query_window_size, atom_key_window_size, dim=-2, q=x_i, k=x_j
    )
    return rearrange(x_i,
                     '... i d -> ... i () d') + rearrange(x_j, '... j d -> ... () j d')


class DiffusionTransformerBlock(nn.Module):
  """DiffusionTransformerBlock"""
  def __init__(self, dim, dim_cond, heads, dropout=0., **kwargs):
    super().__init__()

    dim_single, _ = commons.embedd_dim_get(dim)

    self.attn = commons.AttentionPairBias(dim, dim_cond=dim_cond, heads=heads, **kwargs)
    self.ff = commons.SwiGLUForward(dim_single, dim_cond)
    self.dropout_fn = functools.partial(commons.shaped_dropout, p=dropout)

  def forward(
      self,
      x,
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

    return x, single_cond, pair_cond


class AtomTransformer(nn.Module):
  """AtomTransformer"""
  def __init__(
      self,
      dim,
      depth=3,
      heads=4,
      dim_head=32,
      query_window_size=64,
      key_window_size=128
  ):
    super().__init__()

    dim_single, _ = commons.embedd_dim_get(dim)

    self.query_window_size = query_window_size
    self.key_window_size = key_window_size

    self.difformer = commons.layer_stack(
        DiffusionTransformerBlock,
        depth,
        dim,
        dim_cond=dim_single,
        has_context=True,
        heads=heads,
        dim_head=dim_head,
        q_use_bias=True
    )

  def _context(self, query):
    query, context, *_ = AtomUtil.unfold(
        self.query_window_size, self.key_window_size, dim=-2, q=query, k=query
    )
    return query, context

  def forward(self, single_repr, single_cond, pair_cond, shard_size=None):
    query_cond, context_cond, (_, padding), *_ = AtomUtil.unfold(
        self.query_window_size,
        self.key_window_size,
        dim=-2,
        q=single_cond,
        k=single_cond
    )
    query, query_cond, pair_cond = self.difformer(
        single_repr,
        query_cond,
        pair_cond,
        context=self._context,
        context_cond=context_cond,
        shard_size=shard_size
    )
    return query[..., :-padding, :]


class AtomAttentionEncoder(nn.Module):
  """AtomAttentionEncoder"""
  def __init__(
      self,
      dim,
      dim_token=(384, 128),
      depth=3,
      heads=4,
      atom_query_window_size=64,
      atom_key_window_size=128,
      atom_feats=None,
      has_noisy=False
  ):
    super().__init__()

    dim_single, dim_pairwise = commons.embedd_dim_get(dim)
    dim_token_single, dim_token_pairwise = commons.embedd_dim_get(dim_token)

    self.atom_feats = default(
        atom_feats, (
            ('ref_pos', 3, nn.Identity()),
            ('ref_charge', 1, Rearrange('... -> ... ()')),
            ('ref_mask', 1, Rearrange('... -> ... ()')),
            ('ref_element', 128, nn.Identity()),
            ('ref_atom_name_chars', 4 * 64, nn.Identity())
        )
    )
    dim_atom_feats = sum(d for _, d, _ in self.atom_feats)
    self.to_single_cond = nn.Linear(dim_atom_feats, dim_single, bias=False)
    self.to_pair_cond = nn.Linear(3 + 1 + 1, dim_pairwise, bias=False)

    if has_noisy:
      self.from_trunk_single_cond = nn.Sequential(
          nn.LayerNorm(dim_token_single),
          nn.Linear(dim_token_single, dim_single, bias=False)
      )
      self.from_trunk_pair_cond = nn.Sequential(
          nn.LayerNorm(dim_token_pairwise),
          nn.Linear(dim_token_pairwise, dim_pairwise, bias=False)
      )
      self.from_noisy_emb = nn.Sequential(nn.Linear(3, dim_single, bias=False))

    self.outer_add = AtomPairwiseEmbedding(dim)
    self.outer_ff = commons.FeedForward(dim_pairwise)

    self.transformer = AtomTransformer(
        dim,
        depth=depth,
        heads=heads,
        query_window_size=atom_query_window_size,
        key_window_size=atom_key_window_size
    )

    self.to_out = nn.Linear(dim_single, dim_token_single, bias=False)

  def forward(
      self,
      batch,
      atom_noisy_emb=None,
      trunk_single_cond=None,
      trunk_pair_cond=None,
      shard_size=None
  ):
    atom_to_token_idx = batch['atom_to_token_idx']

    # create the atom single conditioning: Embed per-atom meta data
    atom_single_cond = self.to_single_cond(
        torch.cat([f(batch[k]) for k, _, f in self.atom_feats], dim=-1)
    )

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
    pair_cond = functional.sharded_apply(
        self.to_pair_cond,
        torch.cat(
            (
                dij_ref,
                1 / ( 1 + torch.sum(dij_ref**2, dim=-1, keepdim=True)),
                bij_ref[..., None]
            ),
            dim=-1
        ),
        shard_size=None if self.training else shard_size,
        shard_dim=-1,
        cat_dim=-2
    ) * bij_ref[..., None]

    # initialise the atom single representation as the single conditioning.
    query, context = atom_single_cond, atom_single_cond

    # if provided, add trunk embeddings and noisy positions.
    assert not hasattr(self, 'from_trunk_single_cond') ^ exists(trunk_single_cond)
    if exists(trunk_single_cond):
      # broadcast the single embedding from the trunk
      context = commons.tensor_add(
          functional.batched_gather(
              self.from_trunk_single_cond(trunk_single_cond),
              atom_to_token_idx,
              has_batch_dim=True
          ),
          context
      )
    assert not hasattr(self, 'from_trunk_pair_cond') ^ exists(trunk_pair_cond)
    if exists(trunk_pair_cond):
      # broadcast the pair embedding from the trunk
      q_atom_to_token_idx, k_atom_to_token_idx, *_ = AtomUtil.unfold(
          self.transformer.query_window_size,
          self.transformer.key_window_size,
          dim=-1,
          q=atom_to_token_idx,
          k=atom_to_token_idx
      )
      l, m = q_atom_to_token_idx.shape[-1], k_atom_to_token_idx.shape[-1]
      q_atom_to_token_idx = repeat(q_atom_to_token_idx, '... l -> ... l m', m=m)
      k_atom_to_token_idx = repeat(k_atom_to_token_idx, '... m -> ... l m', l=l)
      pair_cond = commons.tensor_add(
          pair_cond,
          self.from_trunk_pair_cond(trunk_pair_cond)[
              ..., q_atom_to_token_idx, k_atom_to_token_idx, :
          ]
      )
    assert not hasattr(self, 'from_noisy_emb') ^ exists(atom_noisy_emb)
    if exists(atom_noisy_emb):
      # add the noisy positions.
      atom_noisy_emb, *_ = AtomUtil.unfold(
          self.transformer.query_window_size,
          self.transformer.key_window_size,
          dim=-2,
          q=atom_noisy_emb
      )
      query = commons.tensor_add(atom_noisy_emb, self.from_noisy_emb(query))

    # add the combined single conditioning to the pair representation.
    pair_cond = commons.tensor_add(pair_cond, self.outer_add(F.relu(context)))
    # run a small MLP on the pair activations.
    pair_cond = commons.tensor_add(pair_cond, self.outer_ff(pair_cond))
    # cross attention transformer.
    query = self.transformer(query, context, pair_cond, shard_size=shard_size)
    # aggregate per-atom representation to per-token representation.
    token_single_cond = F.relu(self.to_out(query))
    token_single_cond = ScatterUtil.mean(
        repeat(atom_to_token_idx, '... i -> ... i d', d=token_single_cond.shape[-1]),
        token_single_cond,
        dim=-2
    )
    query_skip, context_skip, pair_skip = query, context, pair_cond
    return token_single_cond, query_skip, context_skip, pair_skip


class AtomAttentionDecoder(nn.Module):
  """AtomAttentionDecoder"""
  def __init__(
      self,
      dim,
      dim_token=(384, 128),
      depth=3,
      heads=4,
      atom_query_window_size=64,
      atom_key_window_size=128,
  ):
    super().__init__()

    dim_single, _ = commons.embedd_dim_get(dim)
    dim_token_single, _ = commons.embedd_dim_get(dim_token)

    self.from_token = nn.Sequential(
        nn.LayerNorm(dim_token_single),
        nn.Linear(dim_token_single, dim_single, bias=False)
    )
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
        functional.batched_gather(
            self.from_token(token_single_cond),
            batch['atom_to_token_idx'],
            has_batch_dim=True
        ),
        query_skip
    )
    # cross attention transformer
    atom_single_cond = self.transformer(
        atom_single_cond, context_skip, pair_skip, shard_size=shard_size
    )
    # map to positions update
    return self.to_out(atom_single_cond)


class DiffusionConditioning(nn.Module):
  """DiffusionConditioning"""
  def __init__(self, dim, dim_noise=256, sigma_data=16.0):
    super().__init__()

    dim_single, dim_pairwise = commons.embedd_dim_get(dim)

    self.from_single = nn.Sequential(
        nn.LayerNorm(dim_single), nn.Linear(dim_single, dim_single, bias=False)
    )
    self.from_pairwise = nn.Sequential(
        nn.LayerNorm(dim_pairwise), nn.Linear(dim_pairwise, dim_pairwise, bias=False)
    )
    self.from_noise = nn.Sequential(
        commons.FourierEmbedding(dim_noise),
        nn.LayerNorm(dim_noise),
        nn.Linear(dim_noise, dim_single, bias=False)
    )
    self.to_single = commons.residue_stack(commons.FeedForward, 2, dim_single, mult=2)
    self.to_pairwise = commons.residue_stack(
        commons.FeedForward, 2, dim_pairwise, mult=2
    )

    self.sigma_data = sigma_data

  def forward(self, batch, *, noise_level, trunk_single_cond, trunk_pair_cond):
    # single conditioning
    s = self.from_single(trunk_single_cond)
    s = commons.tensor_add(
        s, self.from_noise(torch.log(noise_level / self.sigma_data) / 4.)
    )
    s, *_ = self.to_single(s)

    # pair conditioning
    x = self.from_pairwise(trunk_pair_cond)
    x, *_ = self.to_pairwise(x)

    return s, x


class DiffusionModule(nn.Module):
  """DiffusionModule"""
  def __init__(
      self,
      dim,
      dim_atom=(128, 16),
      dim_noise=256,
      sigma_data=16.0,
      atom_encoder_depth=3,
      atom_encoder_head_num=4,
      atom_encoder_feats=None,
      transformer_depth=24,
      transformer_group_size=4,
      transformer_head_num=16,
      transformer_dropout_min=0,
      transformer_dropout_max=0,
      atom_decoder_depth=3,
      atom_decoder_head_num=4,
      atom_query_window_size=64,
      atom_key_window_size=128,
  ):
    super().__init__()

    dim_single, dim_pairwise = commons.embedd_dim_get(dim)

    self.conditioning = DiffusionConditioning(
        dim, dim_noise=dim_noise, sigma_data=sigma_data
    )

    self.atom_encoder = AtomAttentionEncoder(
        dim=dim_atom,
        dim_token=dim,
        depth=atom_encoder_depth,
        heads=atom_encoder_head_num,
        atom_query_window_size=atom_query_window_size,
        atom_key_window_size=atom_key_window_size,
        atom_feats=atom_encoder_feats
    )
    self.transformer_in = nn.Sequential(
        nn.LayerNorm(dim_single), nn.Linear(dim_single, dim_single, bias=False)
    )
    assert transformer_depth % transformer_group_size == 0
    self.transformer = commons.layer_stack(
        DiffusionTransformerBlock,
        depth=transformer_depth // transformer_group_size,
        dim=dim,
        dim_cond=dim_single,
        group_size=transformer_group_size,
        heads=transformer_head_num
    )
    self.transformer_out = nn.Sequential(nn.LayerNorm(dim_single))
    self.atom_decoder = AtomAttentionDecoder(
        dim=dim_atom,
        dim_token=dim,
        depth=atom_decoder_depth,
        heads=atom_decoder_head_num,
        atom_query_window_size=atom_query_window_size,
        atom_key_window_size=atom_key_window_size
    )

  def forward(
      self,
      batch,
      *,
      positions_noisy,
      noise_level,
      trunk_single_cond,
      trunk_pair_cond,
      shard_size=None
  ):
    # conditioning
    single_cond, pair_cond = self.conditioning(
        batch,
        noise_level=noise_level,
        trunk_single_cond=trunk_single_cond,
        trunk_pair_cond=trunk_pair_cond
    )
    # scale positions to dimensionless

    token_cond, query_skip, context_skip, pair_skip = self.atom_encoder(batch)

    # full self-attention on token level
    token_cond = commons.tensor_add(token_cond, self.transformer_in(single_cond))
    token_cond, *_ = self.transformer(token_cond, single_cond, pair_cond)
    token_cond = self.transformer_out(token_cond)

    position_update = self.atom_decoder(
        batch, token_cond, query_skip, context_skip, pair_skip
    )

    return position_update


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
    t = torch.linspace(0, 1, steps + 1, device=device, dtype=dtype)
    s_min, s_max = self.s_min ** (1. / self.rho), self.s_max ** (1. / self.rho)
    return self.s_data * (s_max + (s_min - s_max) * t) ** self.rho


class CentreRandomAugmentation(nn.Module):
  def __init__(self, trans_scale_factor=1.0):
    self.trans_scale_factor = trans_scale_factor

  def forward(self, x, mask=None, batch_size=None):
    if exists(mask):
      c = functional.masked_mean(value=x, mask=mask, dim=-2, keepdim=True)
    else:
      c = torch.mean(x, dim=-2, keepdim=True)

    if exists(batch_size):
      x = repeat(x, '... i d -> n i d', n=batch_size)
    x = commons.tensor_sub(x, c)
    R, t = functional.rigids_from_randn(x.shape[:-1], device=x.device)
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
      diffuser_atom_query_window_size=64,
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
      batch_size=None,
      shard_size=None
  ):
    # apply random rotation and translation
    positions_noisy = self.rand_augmenter(
        batch['coord'], mask=batch['coord_mask'], batch_size=batch_size
    )
    # sigma: independent noise-level [..., N_sample]
    sigma = self.noise_scheduler(x.shape[:-1], batch_size=batch_size, device=x.device)
    # noise
    noise_level = torch.randn_like(x) * sigma[..., None, None]
    # denoised_x
    position_denoised = self.diffuser(
        batch,
        positions_noisy=positions_noisy,
        noise_level=noise_level,
        trunk_single_cond=trunk_single_cond,
        trunk_pair_cond=trunk_pair_cond,
        shard_size=shard_size
    )
    return positions_noisy, position_denoised, sigma
