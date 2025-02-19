"""Samplers
"""

from typing import Callable, List, Literal, Optional, Tuple

from tqdm.auto import tqdm

import numpy as np
import torch
from torch.distributions import categorical
import torch.nn.functional as F
from einops import repeat

from profold2.model import functional, potts
from profold2.utils import exists


def init_masks(
    logits_init: torch.Tensor,
    mask_sample: Optional[torch.Tensor] = None,
    S: Optional[torch.Tensor] = None,
    ban_S: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Parse sampling masks and an initial sequence.

  Args:
      logits_init (torch.Tensor): Logits for sequence initialization with shape
          `(num_batch, num_nodes, alphabet)`.
      mask_sample (torch.Tensor, optional): Binary sampling mask indicating which
          positions are free to change with shape `(num_batch, num_nodes)` or which
          tokens are valid at each position with shape
          `(num_batch, num_nodes, alphabet)`. In the latter case, `mask_sample` will
          take priority over `S` except for positions in which `mask_sample` is
          all zero.
      S (torch.Tensor optional): Initial sequence with shape
          `(num_batch, num_seqs, num_nodes)`.
      ban_S (list of int, optional): Optional list of alphabet indices to ban from
          all positions during sampling.

  Returns:
      mask_sample (torch.Tensor): Finalized position specific mask with shape
          `(num_batch, num_nodes, alphabet)`.
      S (torch.Tensor): Self-consistent initial `S` with shape
          `(num_batch, num_nodes)`.
  """
  if not exists(S) and exists(mask_sample):
    raise Exception("To use masked sampling, please provide an initial S")

  m = S.shape[-2] if exists(S) else 1
  logits_init = repeat(logits_init, 'b i d -> b m i d', m=m)

  if exists(mask_sample):
    O_init = F.one_hot(S.long(), num_class=logits_init.shape[-1]).float()
    if mask_sample.dim() == logits_init.dim():
      # Mutation-restricted sampling
      mask_zero = (torch.sum(mask_sample, dim=-1, keepdim=True) == 0)
      mask_S = ((mask_zero * O_init + mask_sample) > 0)
    elif mask_sample.dim() == logits_init.dim() - 1:
      # Position-restricted sampling
      mask_S = mask_sample[..., None] + (1 - mask_sample[..., None]) * O_init
    else:
      raise NotImplementedError
  else:
    mask_S = torch.ones_like(logits_init)

  if exists(ban_S):
    mask_S[..., ban_S] = 0

  logits_init_masked = 1000 * mask_S + logits_init
  S = categorical.Categorical(logits=logits_init_masked).sample()

  return mask_S, S


@torch.no_grad
def from_potts(
    h: torch.Tensor,
    J: torch.Tensor,
    mask: torch.Tensor,
    S: Optional[torch.Tensor] = None,
    mask_sample: Optional[torch.Tensor] = None,
    mask_ban: Optional[List[int]] = None,
    num_sweeps: int = 100,
    temperature: float = 1.0,
    temperature_init: float = 1.0,
    annealing_fraction: float = 0.8,
    penalty_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    differentiable_penalty: bool = True,
    rejection_step: bool = False,
    proposal: Literal['dlmc', 'gibbs'] = 'dlmc',
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Sample from Potts model with MCMC.

  Args:
      h (torch.Tensor): Potts model fields :math:`h_i(s_i)` with shape
          `(num_batch, num_nodes, num_states)`.
      J (Tensor): Potts model couplings :math:`J_{ij}(s_i, s_j)` with shape
          `(num_batch, num_nodes, num_nodes, num_states, num_states)`.
      mask (torch.Tensor): Node mask with shape
          `(num_batch, num_seqs, num_nodes)`.
      S (torch.LongTensor, optional): Sequence for initialization with
          shape `(num_batch, num_seqs, num_nodes)`.
      mask_sample (torch.Tensor, optional): Binary sampling mask indicating
          positions which are free to change with shape
          `(num_batch, num_nodes)` or which tokens are acceptable at each position
          with shape `(num_batch, num_nodes, alphabet)`.
      mask_ban (list of int, optional): Optional list of alphabet indices to ban
          from all positions during sampling.
      num_sweeps (int): Number of sweeps of Chromatic Gibbs to perform,
          i.e. the depth of sampling as measured by the number of times
          every position has had an opportunity to update.
      temperature (float): Final sampling temperature.
      temperature_init (float): Initial sampling temperature, which will
          be linearly interpolated to `temperature` over the course of
          the burn in phase.
      annealing_fraction (float): Fraction of the total sampling run during
          which temperature annealing occurs.
      penalty_func (Callable, optional): An optional penalty function which
          takes a sequence `S` and outputes a `(num_batch, num_seqs)` shaped tensor
          of energy adjustments, for example as regularization.
      differentiable_penalty (bool): If True, gradients of penalty function
          will be used to adjust the proposals.
      rejection_step (bool): If True, perform a Metropolis-Hastings
          rejection step.
      proposal (str): MCMC proposal for Potts sampling. Currently implemented
              proposals are `dlmc` for Discrete Langevin Monte Carlo [1] or `gibbs`
              for Gibbs sampling with graph coloring.
              [1] Sun et al. Discrete Langevin Sampler via Wasserstein Gradient Flow (2023).

  Returns:
      S (torch.LongTensor): Sampled sequences with
          shape `(num_batch, num_nodes)`.
      U (torch.Tensor): Sampled energies with shape `(num_batch)`. Lower is more
          favorable.
  """
  # Initialize masked proposals and mask h
  mask_S, S = init_masks(-h, mask_sample, S, ban_S=mask_ban)
  mask_mutatable = (torch.sum(mask_S, dim=-1) > 1)

  # Block update schedule
  num_iterations = num_sweeps

  num_iterations_annealing = int(annealing_fraction * num_iterations)
  temperatures = np.linspace(
      temperature_init, temperature, num_iterations_annealing
  ).tolist() + [temperature] * (num_iterations - num_iterations_annealing)

  _energy_proposal = lambda _S, _T: potts.mcmc_proposal(
      _S,
      h,
      J,
      mask,
      T=_T,
      penalty_func=penalty_func,
      differentiable_penalty=differentiable_penalty,
      proposal=proposal
  )

  for i, T_i in enumerate(tqdm(temperatures, desc="Potts Sampling")):
    # Cycle through Gibbs updates random sites to the update with fixed prob
    mask_update = torch.ones_like(S, dtype=torch.bool)
    if exists(mask_mutatable):
      mask_update = mask_update * mask_mutatable

    # Compute current energy and local conditionals
    U, logp = _energy_proposal(S, T_i)

    # Propose
    S_new = categorical.Categorical(logits=logp).sample()
    S_new = torch.where(mask_update, S_new, S)

    # Metropolis-Hastings adjusment
    if rejection_step:
      raise NotImplementedError
    else:
      S = S_new

    U, _ = potts.energy(S, h, J, mask)

  return S, U


if __name__ == '__main__':
  from profold2.common import residue_constants
  from profold2.model import complexity

  b, m, n = 2, 5, 50
  S = torch.randint(0, len(residue_constants.restypes_with_x_and_gap), size=(b, m, n))
  F.one_hot(
      S.long(), num_classes=len(residue_constants.restypes_with_x_and_gap)
  )
  h = torch.rand(b, n, len(residue_constants.restypes_with_x_and_gap))
  J = torch.rand(b, n, n, len(residue_constants.restypes_with_x_and_gap), len(residue_constants.restypes_with_x_and_gap))
  C = torch.ones(b, n)
  mask = torch.ones(b, 1, n)

  penalty_func = lambda _S: complexity.complexity_lcp(_S, C, mask)
  # compositions(S, C, mask)
  S, U = from_potts(h, J, mask, S=S, penalty_func=penalty_func)
  print(S)
