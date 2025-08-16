"""Diffusion model for generating 3d-structure"""
from typing import Optional, Union

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from profold2.model import commons, functional
from profold2.utils import default, exists


class ScatterUtil:
  @staticmethod
  def add(
      index: torch.Tensor,
      src: torch.Tensor,
      dim: Optional[int] = -1,
      out: Optional[torch.Tensor] = None,
      out_dim: Optional[int] = None
  ):
    if not exists(out):
      size = list(src.shape)
      if exists(out_dim):
        size[dim] = out_dim
      else:
        size[dim] = int(torch.max(index)) + 1
      out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return torch.scatter_add(out, dim, index, src)


  @staticmethod
  def sum(
      index: torch.Tensor,
      src: torch.Tensor,
      dim: Optional[int] = -1,
      out: Optional[torch.Tensor] = None,
      out_dim: Optional[int] = None
  ):
    return ScatterUtil.add(index, src, dim=dim, out=out, out_dim=out_dim)


  @staticmethod
  def mean(
      index: torch.Tensor,
      src: torch.Tensor,
      dim: Optional[int] = -1,
      out: Optional[torch.Tensor] = None,
      out_dim: Optional[int] = None
  ):
    out = ScatterUtil.sum(index, src, dim=dim, out=out, out_dim=out_dim)
    out_dim = out.shape[dim]

    one = torch.ones_like(index, dtype=src.dtype, device=src.device)
    count = torch.clamp(ScatterUtil.sum(index, one, dim=dim, out_dim=out_dim), min=1)

    return commons.tensor_div(out, count)


class AtomUtil:
  @staticmethod
  def pad(t: torch.Tensor, dim: int, pad: Union[tuple[int], list[int]]):
    dim = len(t.shape) - dim % len(t.shape)
    pad = (0, 0) * (len(t.shape) - dim - 1) + pad
    return F.pad(t, pad=pad)


  @staticmethod
  def reshape(t: torch.Tensor, dim: int, shape: Union[tuple[int], list[int]]):
    dim = len(t.shape) - dim % len(t.shape)
    return torch.reshape(t, t.shape[:dim] + tuple(shape) + t.shape[dim + 1:])


  @staticmethod
  def unfold(
      q_window_size: int,
      k_window_size: int,
      dim: int,
      q: torch.Tensor,
      k: torch.Tensor,
      v: Optional[torch.Tensor] = None
  ):
    assert k_window_size % 2 == 0 and q_window_size % 2 == 0
    assert k_window_size >= q_window_size
    assert q.shape[dim] == k.shape[dim]
    if exists(v):
      assert q.shape[dim] == v.shape[dim]

    n = q.shape[dim] // q_window_size + (1 if q.shape[dim] % q_window_size > 0 else 0)

    # padding q
    q_padding = 0, n - q.shape[dim]
    if q_padding != (0, 0):
      q = AtomUtil.pad(q, dim=dim, pad=q_padding)
    # reshape q
    q = AtomUtil.reshape(q, dim=dim, shape=(n, q_window_size))

    # padding k
    k_padding = (
        (k_window_size - q_window_size) // 2,
        n * q_window_size + (k_window_size - q_window_size) // 2 - k.shape[dim]
    )
    if k_padding != (0, 0):
      k = AtomUtil.pad(k, dim=dim, pad=k_padding)
    assert k.shape[dim] == n * k_window_size
    # reshape k
    k = AtomUtil.reshape(k, dim=dim, shape=(n, k_window_size))

    # unfold k
    k = k.unfold(dim, size=k_window_size, steps=q_window_size)
    
    if exists(v):
      if k_padding != (0, 0):
        v = AtomUtil.pad(v, dim=dim, pad=k_padding)
      # reshape v
      v = AtomUtil.reshape(v, dim=dim, shape=(n, k_window_size))

      # unfold v
      v = v.unfold(dim, size=k_window_size, steps=q_window_size)

    return q, k, v


  @staticmethod
  def fold(tensor, dim=None):
    pass


class InputFeatureEmbedder(nn.Module):
  """InputFeatureEmbedder
    """
  def __init__(self, dim, dim_atom=(128, 16), dim_token=None):
    super().__init__()

    dim_single, _ = commons.embedd_dim_get(dim)
    dim_token = default(dim_token, dim_single)

    self.atom_encoder = AtomAttentionEncoder(dim_atom)


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

  def _forward(
      self, seq_index, seq_color, seq_sym, seq_entity, token_index, shard_size=None
  ):
    bij_seq_index = (seq_index[..., :, None] == seq_index[..., None, :])
    bij_seq_color = (seq_color[..., :, None] == seq_color[..., None, :])
    bij_seq_entity = (seq_entity[..., :, None] == seq_entity[..., None, :])

    dij_seq_index = F.one_hot(
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
    dij_token_index = F.one_hot(
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
    dij_seq_sym = F.one_hot(
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

    dim_single, dim_pairwise = embedd_dim_get(dim)
    self.to_pairwise_repr = nn.Linear(dim_single, dim_pairwise * 2)

  def forward(self, x, atom_query_window_size=64, atom_key_window_size=128):
    x_i, x_j, _ = AtomUtil.unfold(
        atom_key_window_size,
        atom_key_window_size,
        dim=-2,
        q=x,
        k=x
    )
    return rearrange(x_i, '... i d -> ... i () d') + rearrange(
        x_j, '... j d -> ... () j d'
    )


class DiffusionConditioning(nn.Module):
  def __init__(self, dim, dim_noise=256, sigma_data=16.0):
    super().__init__()

    dim_single, dim_pairwise = commons.embedd_dim_get(dim)

    self.from_single = nn.Sequential(
        nn.LayerNorm(dim=dim_single), nn.Linear(dim_single, dim_single, bias=False)
    )
    self.from_pairwise = nn.Sequential(
        nn.LayerNorm(dim=dim_pairwise),
        nn.Linear(dim_pairwise, dim_pairwise, bias=False)
    )
    self.from_noise = nn.Sequential(
        commons.FourierEmbedding(dim_noise),
        nn.LayerNorm(dim=dim_noise),
        nn.Linear(dim_noise, dim_single, bias=False)
    )
    self.to_single = commons.residue_stack(commons.FeedForward, 2, dim_single, mult=2)
    self.to_pairwise = commons.residue_stack(
        commons.FeedForward, 2, dim_pairwise, mult=2
    )

    self.sigma_data = sigma_data

  def forward(self, single_repr, pairwise_repr, t_hat):
    # single conditioning
    s = self.from_single(single_repr)
    s = commons.tensor_add(s, self.from_noise(torch.log(t_hat / self.sigma_data) / 4.))
    s = self.to_single(s)

    # pair conditioning
    x = self.from_pairwise(pairwise_repr)
    x = self.to_pairwise(x)

    return s, x


class AtomTransformer(nn.Module):
  def __init__(self, dim, depth=3, heads=4, dim_atom=(128, 16)):
    super().__init__()

    self.difformer = commons.layer_stack(
        commons.DiffusionTransformerBlock, depth=depth, heads=heads
    )

  def forward(self, x):
    return self.difformer(x)


class AtomAttentionEncoder(nn.Module):
  def __init__(
      self, dim,
      dim_token=384,
      depth=3,
      heads=4,
      atom_query_window_size=64,
      atom_key_window_size=128,
      atom_feats=None
  ):
    super().__init__()

    dim_single, dim_pairwise = commons.embedd_dim_get(dim)

    self.atom_query_window_size = atom_query_window_size
    self.atom_key_window_size = atom_key_window_size
    self.atom_feats = default(
        atom_feats, (
            ('ref_pos', 3),
            ('ref_charge', 1),
            ('ref_mask', 1),
            ('ref_element', 128),
            ('ref_atom_name_chars', 4 * 64)
        )
    )
    dim_atom_feats = sum(map(lambda kv: kv[1], self.atom_feats))
    self.to_single_repr = nn.Linear(dim_atom_feats, dim_single, bias=False)
    self.to_pairwise_repr = nn.Linear(3 + 1 + 1, dim_pairwise, bias=False)

    self.outer_add = AtomPairwiseEmbedding(dim)
    self.outer_ff = commons.FeedForward(dim)

    self.transformer = AtomTransformer(dim, depth=depth, heads=heads)

    self.to_out = nn.Linear(dim_single, dim_token, bias=False)

  def forward(self, batch, single_repr=None, pairwise_repr=None):
    atom_to_token_idx = batch['atom_to_token_idx']

    # create the atom single conditioning: Embed per-atom meta data
    atom_single_repr = self.to_single_repr(
        torch.cat([batch[k] for k in self.atom_feats], dim=-1)
    )

    # embed offsets between atom reference position, pairwise inverse squared
    # distances, and the valid mask.
    ref_pos_i, ref_pos_j, _ = AtomUtil.unfold(
        self.atom_query_window_size,
        self.atom_key_window_size,
        dim=-2,
        q=batch['ref_pos'],
        k=batch['ref_pos']
    )
    ref_space_uid_i, ref_space_uid_j, _ = AtomUtil.unfold(
        self.atom_query_window_size,
        self.atom_key_window_size,
        dim=-2,
        q=batch['ref_space_uid'],
        k=batch['ref_space_uid']
    )
    dij_ref = ref_pos_i[..., :, None] - ref_pos_j[..., None, :]
    bij_ref = (
        ref_space_uid_i[..., :, None] == ref_space_uid_j[..., None, :]
    )
    atom_pairwise_repr = functional.sharded_apply(
        self.to_pairwise_repr,
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

    # add the combined single conditioning to the pair representation.
    atom_pairwise_repr = commons.tensor_add(
        atom_pairwise_repr, self.outer_add(F.relu(atom_single_repr))
    )
    # run a small MLP on the pair activations.
    atom_pairwise_repr = commons.tensor_add(
        atom_pairwise_repr, self.outer_ff(atom_pairwise_repr)
    )
    # cross attention transformer.
    atom_single_repr = self.transformer(x)
    # aggregate per-atom representation to per-token representation
    single_repr = ScatterUtil.mean(
        F.relu(self.to_out(atom_single_repr)), atom_to_token_idx, dim=-2
    )
    return single_repr


class AtomAttentionDecoder(nn.Module):
  def __init__(self, dim, depth, heads):
    super().__init__()

    self.transformer = AtomTransformer(dim, depth, heads)

  def forward(self, x):
    # broadcase per-token activations to per-atom activations and add the skip connection
    # cross attention transformer
    # map to positions update
    return self.transformer(x)


class DiffusionSchedule:
  pass


class DiffusionModule(nn.Module):
  def __init__(
      self,
      dim,
      dim_noise=256,
      sigma_data=16.0,
      difformer_depth=24,
      difformer_head_num=16
  ):
    super().__init__()

    dim_single, dim_pairwise = commons.embedd_dim_get(dim)

    self.conditioning = DiffusionSchedule(
        dim, dim_noise=dim_noise, sigma_data=sigma_data
    )

    self.atom_encoder = None
    self.difformer = commons.layer_stack(
        commons.DiffusionTransformerBlock,
        depth=difformer_depth,
        heads=difformer_head_num
    )
    self.atom_decoder = None
