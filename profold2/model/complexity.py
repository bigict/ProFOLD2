"""Functions for computing sequence complexities
"""

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from profold2.model import functional


def gather_edge_idx(
    params: torch.Tensor, edge_idx: torch.Tensor
) -> torch.Tensor:
  """Gather values from edge indices

  Args:
      params (torch.Tensor): value tensor with shape
          `(num_batch, num_residues, ...)`
      edge_idx (torch.Tensor): Edge indices with shape
          `(num_batch, num_residues, num_residues)`
  Returns:
      values (torch.Tensor): gathered values with shape
          `(num_batch, num_residues, num_residues, ...)`,
  """
  num_residues = edge_idx.shape[-2]
  edge_idx = rearrange(edge_idx, 'b i j -> b (i j)')
  feats = functional.batched_gather(params, edge_idx, has_batch_dim=True)
  feats = rearrange(feats, 'b (i j) ... -> b i j ...', i=num_residues)
  return feats


def compositions(
    S: torch.Tensor,
    C: torch.Tensor,
    w: int = 30,
):
  """Compute local compositions per residue

  Args:
      S (torch.Tensor): Sequence tensor with shape
          `(num_batch, num_seqs, num_residues, alphabet)`
      C (torch.Tensor): Chain map with shape `(num_batch, num_residues)`
      w (int, optional): Window size
  Returns:
      P (torch.Tensor): Local compositions with shape
          `(num_batch, num_residues - w + 1, alphabet)`,
      N (torch.Tensor): Local counts with shape
          `(num_batch, num_residues - w + 1, alphabet)`
  """
  b, m, n, _ = S.shape

  # Build neighborhoods and masks
  kx = torch.arange(w, device=S.device) - w // 2
  edge_idx = repeat(
      torch.arange(n, device=S.device)[:, None] + kx[None, :],
      '... -> b ...', b=b
  )
  mask_ij = (edge_idx > 0) & (edge_idx < n)
  edge_idx = torch.clamp(edge_idx, min=0, max=n - 1)
  C_i = C[..., None]
  C_j = torch.squeeze(gather_edge_idx(C_i, edge_idx), dim=-1)
  mask_ij = (mask_ij & C_i.eq(C_j) & (C_i > 0) & (C_j > 0))


  # Sum neighborhood composition
  S_j = repeat(mask_ij, '... i j -> ... () i j ()') * rearrange(
      gather_edge_idx(rearrange(S, 'b m i c -> b i m c'), edge_idx),
      'b i j m c -> b m i j c'
  )
  N = torch.sum(S_j, dim=-2)

  num_N = torch.sum(N, dim=-1, keepdim=True)
  P = N / (num_N + 1e-5)
  return P, N, edge_idx, mask_ij


def complexity_lcp(
    S: torch.Tensor,
    C: torch.Tensor,
    mask: torch.Tensor,
    w: int = 30,
    entropy_min: float = 2.32,
    method: str = 'naive',
    differentiable: bool = True,
    eps: float = 1e-5,
    min_coverage: float = 0.9,
) -> torch.Tensor:
  """Compute the Local Composition Perplexity metric.

  Args:
      S (torch.Tensor): Sequence tensor with shape `(num_batch, num_seqs, num_residues)`
          (index tensor) or `(num_batch, num_seqs, num_residues, alphabet)`.
      C (torch.LongTensor): Chain map with shape `(num_batch, num_residues)`.
      w (int): Window size.
      eps (float): Small number for numerical stability in division and logarithms.
  Returns:
      U (torch.Tensor): Complexities with shape `(num_batch, num_seqs)`.
  """

  # adjust window size based on sequence length
  if S.shape[-2] < w:
    w = S.shape[-2]

  S = S * mask[..., None]

  P, N, edge_idx, mask_ij = compositions(S, C, w=w)

  # Only count windows with `min_coverage`
  min_N = int(min_coverage * w)
  mask_coverage = torch.sum(N, dim=-1) >= min_N

  H = estimate_entropy(N, method=method)
  U = torch.square(
      torch.clamp(mask_coverage * (torch.exp(H) - np.exp(entropy_min)), max=0)
  )

  # Compute entropy as a function of perturbed counts
  if differentiable:
    # Compute how a mutation changes entropy for each neighbor
    N_neighbors = rearrange(
        gather_edge_idx(rearrange(N, 'b m i c -> b i m c'), edge_idx),
        'b i j m c -> b m i j c'
    )
    mask_coverage_j = rearrange(
        gather_edge_idx(rearrange(mask_coverage, 'b m i -> b i m'), edge_idx),
        'b i j m -> b m i j'
    )
    N_ij = (N_neighbors - S[..., None, :])[..., None, :] + torch.eye(
        N.shape[-1], device=N.device)
    N_ij = torch.clamp(N_ij, min=0)
    H_ij = estimate_entropy(N_ij, method=method)
    U_ij = torch.square(
        torch.clamp(torch.exp(H_ij) - np.exp(entropy_min), max=0)
    )
    U_ij = mask_ij[..., None, :, :, None] * mask_coverage_j[..., None] * U_ij
    U_differentiable = torch.sum(U_ij.detach() * S[..., None, :], dim=(-1, -2))
    U = U.detach() + U_differentiable - U_differentiable.detach()

  U = torch.sum(U * mask, dim=-1)
  return U

def estimate_entropy(
    N: torch.Tensor, method: str = 'chao-shen', eps: float = 1e-11
):
  """Estimate entropy from counts.

      See Chao, A., & Shen, T. J. (2003) for more details.

  Args:
      N (torch.Tensor): Tensor of counts with shape `(..., num_bins)`.

  Returns:
      H (torch.Tensor): Estimated entropy with shape `(...)`.
  """
  N_total = torch.sum(N, dim=-1, keepdim=True)
  P = N.float() / (N_total + eps)

  if method == 'chao-shen':
    # Estimate coverage and adjusted frequencies
    singletons = N.long().eq(1).sum(-1, keepdim=True).float()
    C = 1.0 - singletons / (N_total + eps)
    P_adjust = C * P
    P_inclusion = torch.clamp(1.0 - (1.0 - P_adjust) ** N_total, min=eps)
    H = -torch.sum(P_adjust * torch.log(P_adjust.clamp(min=eps)) / P_inclusion, dim=-1)
  elif method == 'miller-maddow':
    bins = (N > 0).float().sum(-1)
    bias = (bins - 1) / (2 * N_total[..., 0] + eps)
    H = -(P * torch.log(P + eps)).sum(-1) + bias
  elif method == 'laplace':
    N = N.float() + 1 / N.shape[-1]
    N_total = torch.sum(N, dim=-1, keepdim=True)
    P = N / (N_total + eps)
    H = -torch.sum(P * torch.log(P), dim=-1)
  else:
    H = -torch.sum(P * torch.log(P + eps), dim=-1)
  return H

if __name__ == '__main__':
  from profold2.common import residue_constants
  
  b, m, n = 2, 5, 50
  S = torch.randint(0, len(residue_constants.restypes_with_x_and_gap), size=(b, m, n))
  S = F.one_hot(
      S.long(), num_classes=len(residue_constants.restypes_with_x_and_gap)
  )
  print(S.shape)
  C = torch.ones(b, n)
  mask = torch.ones(b, 1, n)

  # compositions(S, C, mask)
  U = complexity_lcp(S, C, mask)
  print(U)
