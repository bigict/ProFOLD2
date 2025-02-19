"""Module for building Potts models.
"""

from typing import Literal, Tuple

import torch
import torch.nn.functional as F

from profold2.utils import exists

def predict(
    S: torch.Tensor,
):
  pass

def energy(
    S: torch.Tensor, h: torch.Tensor, J: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Compute Potts model energy from sequences.

  Inputs:
      S (torch.Tensor): Sequence with shape `(num_batch, num_seqs, num_nodes)`.
      h (torch.Tensor): Potts model fields :math:`h_i(s_i)` with shape
          `(num_batch, num_nodes, num_states)`.
      J (Tensor): Potts model couplings :math:`J_{ij}(s_i, s_j)` with shape
          `(num_batch, num_nodes, num_nodes, num_states, num_states)`.

  Outputs:
      U (torch.Tensor): Potts total energies with shape `(num_batch)`.
          Lower energies are more favorable.
  """
  S = F.one_hot(
      S.long(), num_classes=h.shape[-1]
  ).float() * mask[..., None]

  J_i = torch.einsum('b m j d,b i j c d -> b m i c', S, J)
  U_i = h[..., None, :, :] + J_i

  # Correct for double counting in total energy
  U = torch.sum((U_i - 0.5 * J_i) * S, dim=(-1, -2))

  return U, U_i


def _proposal_gibbs(
    S: torch.Tensor,
    h: torch.Tensor,
    J: torch.Tensor,
    mask: torch.Tensor,
    T: float = 1.0,
    penalty_func=None,
    differentiable_penalty: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
  # Compute energy gap
  U, U_i = energy(S, h, J, mask)
  if exists(penalty_func):
    if differentiable_penalty:
      with torch.enable_grad():
        O = F.one_hot(S.long(), num_classes=h.shape[-1]).float()
        O.requires_grad = True
        U_penalty = penalty_func(O)
        U_i_adjustment = torch.autograd.grad(U_penalty.sum(), [O])[0].detach()
        U_penalty = U_penalty.detach()
        O.requires_grad = False
      U_i = U_i + 0.5 * U_i_adjustment
    else:
      U_penalty = penalty_func(O)
    U = U + U_penalty

  # Compute local equilibrium distribution
  logP_i = F.log_softmax(-U_i / T, dim=-1)
  return U, logP_i


def _proposal_dlmc(
    S: torch.Tensor,
    h: torch.Tensor,
    J: torch.Tensor,
    mask: torch.Tensor,
    T: float = 1.0,
    penalty_func=None,
    differentiable_penalty: bool = True,
    dt: float = 0.1,
    balancing_func:str = 'sigmoid',
) -> Tuple[torch.Tensor, torch.Tensor]:
  # Compute energy gap
  U, U_i = energy(S, h, J, mask)
  if exists(penalty_func):
    O = F.one_hot(S.long(), num_classes=h.shape[-1]).float()
    if differentiable_penalty:
      with torch.enable_grad():
        O.requires_grad = True
        U_penalty = penalty_func(O)
        U_i_adjustment = torch.autograd.grad(U_penalty.sum(), [O])[0].detach()
        U_penalty = U_penalty.detach()
        O.requires_grad = False
        U_i_adjustment = U_i_adjustment - torch.sum(
            U_i_adjustment * O, dim=-1, keepdim=True
        )
      U_i = U_i + U_i_adjustment
    else:
      U_penalty = penalty_func(O)
    U = U + U_penalty

  # Compute local equilibrium distribution
  logP_j = F.log_softmax(-U_i / T, dim=-1)

  # Compute transition log probabilities
  O = F.one_hot(S.long(), num_classes=h.shape[-1])
  logP_i = torch.sum(logP_j * O, dim=-1, keepdim=True)
  if balancing_func == 'sqrt':
    logQ_ij = (logP_j - logP_i) * 0.5
  elif balancing_func == 'sigmoid':
    logQ_ij = F.logsigmoid(logP_j - logP_i)
  else:
    raise NotImplementedError

  rate = torch.exp(logQ_ij - logP_j)

  # Compute transition probability
  logP_ij = logP_j + torch.log(-(-dt * rate).expm1())
  p_flip = torch.sum((1.0 - O) * logP_ij.exp(), dim=-1, keepdim=True)


  # DEBUG:
  # flux = ((1. - O) * torch.exp(log_Q_ij)).mean([1,2], keepdim=True)
  # print(f" ->Flux is {flux.item():0.2f}, FlipProb is {p_flip.mean():0.2f}")

  logP_ii = (1.0 - p_flip).clamp(1e-5).log()
  logP_ij = (1.0 - O) * logP_ij + O * logP_ii
  return U, logP_ij


def mcmc_proposal(
    S: torch.Tensor,
    h: torch.Tensor,
    J: torch.Tensor,
    mask: torch.Tensor,
    T: float = 1.0,
    penalty_func=None,
    differentiable_penalty: bool = True,
    proposal: Literal['dlmc', 'gibbs'] = 'dlmc',
):
  if proposal == 'dlmc':
    return _proposal_dlmc(
        S,
        h,
        J,
        mask,
        T=T,
        penalty_func=penalty_func,
        differentiable_penalty=differentiable_penalty
    )
  elif proposal == 'gibbs':
    return _proposal_gibbs(
        S,
        h,
        J,
        mask,
        T=T,
        penalty_func=penalty_func,
        differentiable_penalty=differentiable_penalty
    )
  else:
    raise NotImplementedError
