"""Helper functions for different atom layouts and conversion between them."""
import math
from typing import Optional, Union

import torch
from torch.nn import functional as F
from einops import rearrange, repeat

from profold2.common import residue_constants
from profold2.model import functional
from profold2.utils import default, exists


def padding(
    t: torch.Tensor,
    dim: int,
    pad: Union[tuple[int], list[int]],
) -> torch.Tensor:
  dim = dim % t.dim()
  if pad != (0, 0):
    pad = (0, 0) * (t.dim() - dim - 1) + pad
    t = F.pad(t, pad=pad)
  return t


def reshape(
    t: torch.Tensor, dim: int, shape: Union[tuple[int], list[int]]
) -> torch.Tensor:
  dim = dim % t.dim()
  assert t.shape[dim] == math.prod(shape)
  return torch.reshape(t, t.shape[:dim] + tuple(shape) + t.shape[dim + 1:])


def permute(t: torch.Tensor, dim: int) -> torch.Tensor:
  dim = dim % t.dim()
  if dim + 1 < t.dim():
    dims = tuple(range(dim)) + (-1, ) + tuple(range(dim, t.dim() - 1))
    t = torch.permute(t, dims)
  return t


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
  q = reshape(padding(q, dim=dim, pad=q_padding), dim=dim, shape=(n, q_window_size))
  # k
  if exists(k):
    k_padding = (
        (k_window_size - q_window_size) // 2,
        n * q_window_size + (k_window_size - q_window_size) // 2 - k.shape[dim]
    )
    k = padding(k, dim=dim, pad=k_padding)
    k = permute(k.unfold(dim, size=k_window_size, step=q_window_size), dim=dim)
  return q, k, q_padding[1]


def gather(
    t: torch.Tensor,
    atom_to_token_idx: torch.Tensor,
    q_window_size: Optional[int] = None,
    k_window_size: Optional[int] = None
) -> torch.Tensor:
  if exists(q_window_size) and exists(k_window_size):  # pair
    n = t.shape[-2]
    t = rearrange(t, '... i j d -> ... (i j) d')

    q_atom_to_token_idx, k_atom_to_token_idx, *_ = unfold(
        q_window_size, k_window_size, dim=-1, q=atom_to_token_idx, k=atom_to_token_idx
    )
    atom_to_token_idx = rearrange(
        q_atom_to_token_idx[..., :, None] * n + k_atom_to_token_idx[..., None, :],
        '... c i j -> ... (c i j)'
    )

  t = functional.batched_gather(t, atom_to_token_idx, dim=-2, has_batch_dim=True)

  if exists(q_window_size) and exists(k_window_size):  # pair
    t = rearrange(t, '... (c i j) d -> ... c i j d', i=q_window_size, j=k_window_size)
  return t


def flatten(
    atom_to_token_idx: torch.Tensor,
    atom_within_token_idx: torch.Tensor,
    coord: Optional[torch.Tensor] = None,
    coord_mask: Optional[torch.Tensor] = None
) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[torch.Tensor]]]:
  assert exists(coord) or exists(coord_mask)

  if exists(coord):
    coord = functional.batched_gather(
        rearrange(coord, '... i c d -> ... (i c) d'),
        atom_to_token_idx * coord.shape[-2] + atom_within_token_idx,
        dim=-2,
        has_batch_dim=True
    )
  if exists(coord_mask):
    coord_mask = functional.batched_gather(
        rearrange(coord_mask, '... i c -> ... (i c)'),
        atom_to_token_idx * coord_mask.shape[-1] + atom_within_token_idx,
        dim=-1,
        has_batch_dim=True
    )
    assert not exists(coord) or coord.shape[:-1] == coord_mask.shape
  if exists(coord) and exists(coord_mask):
    return coord, coord_mask
  elif exists(coord_mask):
    return coord_mask
  return coord


def unflatten(
    atom_to_token_idx: torch.Tensor,
    atom_within_token_idx: torch.Tensor,
    coord: Optional[torch.Tensor] = None,
    coord_mask: Optional[torch.Tensor] = None,
    num_tokens: int = None
) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[torch.Tensor]]]:
  assert exists(coord) or exists(coord_mask)
  num_tokens = default(num_tokens, int(torch.max(atom_to_token_idx) + 1))
  c = residue_constants.atom14_type_num
  index = (atom_to_token_idx * c + atom_within_token_idx).long()

  if exists(coord):
    coord = torch.scatter(
        torch.zeros(
            coord.shape[:-2] + (num_tokens * c, coord.shape[-1]),
            device=coord.device,
            dtype=coord.dtype
        ), -2, repeat(index, '... -> ... d', d=coord.shape[-1]), coord
    )
    coord = rearrange(coord, '... (i c) d -> ... i c d', c=c)
  if exists(coord_mask):
    coord_mask = torch.scatter(
        torch.zeros(
            coord_mask.shape[:-1] + (num_tokens * c, ),
            device=coord_mask.device,
            dtype=coord_mask.dtype
        ), -1, index, coord_mask
    )
    coord_mask = rearrange(coord_mask, '... (i c) -> ... i c', c=c)
  if exists(coord) and exists(coord_mask):
    return coord, coord_mask
  elif exists(coord_mask):
    return coord_mask
  return coord
