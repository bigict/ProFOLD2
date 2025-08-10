"""Folding the protein based on s_{i}
  """
import functools
import logging

import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F
from einops import rearrange, repeat

from profold2.common import residue_constants
from profold2.model import commons, functional
from profold2.utils import exists, torch_default_dtype

logger = logging.getLogger(__name__)


def Transition(dim, mult=1., num_layers=2, act=nn.ReLU):  # pylint: disable=invalid-name
  layers = []
  dim_hidden = dim * mult

  for ind in range(num_layers):
    is_first = ind == 0
    is_last = ind == (num_layers - 1)
    dim_in = dim if is_first else dim_hidden
    dim_out = dim if is_last else dim_hidden

    layers.append(nn.Linear(dim_in, dim_out))

    if is_last:
      continue

    layers.append(act())

  return nn.Sequential(*layers)


class IPABlock(nn.Module):
  """One transformer block based on IPA
     In the paper, they used 3 layer transition (feedforward) block
    """
  def __init__(self, *, dim, ff_mult=1, ff_num_layers=3, dropout=.0, **kwargs):
    super().__init__()
    dim_single, _ = commons.embedd_dim_get(dim)

    self.attn_norm = nn.LayerNorm(dim_single)
    self.attn = commons.InvariantPointAttention(dim=dim, **kwargs)

    self.ff_norm = nn.LayerNorm(dim_single)
    self.ff = Transition(dim_single, mult=ff_mult, num_layers=ff_num_layers)

    self.dropout_fn = functools.partial(F.dropout, p=dropout)

  def forward(self, x, **kwargs):
    x = commons.tensor_add(x, self.attn(x, **kwargs))
    x = self.attn_norm(self.dropout_fn(x, training=self.training))

    x = commons.tensor_add(x, self.ff(x))
    x = self.ff_norm(self.dropout_fn(x, training=self.training))
    return x


class AngleNetBlock(nn.Module):
  def __init__(self, dim, channel=128):
    super().__init__()

    self.net = nn.Sequential(
        nn.ReLU(), nn.Linear(dim, channel), nn.ReLU(), nn.Linear(channel, dim)
    )

  def forward(self, x):
    return commons.tensor_add(x, self.net(x))


class AngleNet(nn.Module):
  """Predict the torsion angles
    """
  def __init__(self, dim, channel=128, num_blocks=2):
    super().__init__()

    self.projection = nn.Linear(dim, channel)
    self.projection_init = nn.Linear(dim, channel)

    self.blocks = nn.Sequential(
        *[AngleNetBlock(channel, channel) for _ in range(num_blocks)]
    )

    self.to_groups = nn.Linear(
        channel, (residue_constants.restype_rigid_group_num - 1) * 2
    )

  def forward(self, single_repr, single_repr_init=None):
    act = self.projection(F.relu(single_repr))
    if exists(single_repr_init):
      act += self.projection_init(F.relu(single_repr_init))

    # Mapping with some angle residual blocks
    act = self.blocks(act)

    # Map activations to torsion angles. (b l n 7 2)
    angles = rearrange(self.to_groups(F.relu(act)), '... (n d)->... n d', d=2)

    return angles


class StructureModule(nn.Module):
  """Iteratively updating rotations and translations
    """
  def __init__(
      self,
      dim,
      structure_module_depth,
      structure_module_heads,
      dropout=.0,
      position_scale=1.0,
      **kwargs
  ):
    super().__init__()
    dim_single, dim_pairwise = commons.embedd_dim_get(dim)

    assert structure_module_depth >= 1
    self.structure_module_depth = structure_module_depth
    self.position_scale = position_scale
    with torch_default_dtype(torch.float32):
      self.ipa_block = IPABlock(
          dim=dim_single,
          pairwise_repr_dim=dim_pairwise,
          heads=structure_module_heads,
          dropout=dropout,
          **kwargs
      )

      self.to_affine_update = nn.Linear(dim_single, 6)

    self.single_repr_norm = nn.LayerNorm(dim_single)
    self.pairwise_repr_norm = nn.LayerNorm(dim_pairwise)
    self.single_repr_dim = nn.Sequential(nn.Linear(dim_single, dim_single))

    self.to_angles = AngleNet(dim_single)

  def forward(self, representations, batch):
    b, n, device = *batch['seq'].shape[:2], batch['seq'].device

    single_repr, pairwise_repr = representations['single'], representations['pair']

    single_repr = self.single_repr_norm(single_repr)
    pairwise_repr = self.pairwise_repr_norm(pairwise_repr)

    single_repr_init = single_repr
    single_repr = self.single_repr_dim(single_repr)

    # prepare float32 precision for equivariance
    original_dtype = single_repr.dtype
    single_repr, pairwise_repr = map(lambda t: t.float(), (single_repr, pairwise_repr))

    outputs = []

    # iterative refinement with equivariant transformer in high precision
    with torch_default_dtype(torch.float32):
      # initial frames
      if 'frames' in representations and exists(representations['frames']):
        quaternions, translations = representations['frames']
      else:
        quaternions = torch.tensor([1., 0., 0., 0.], device=device)
        quaternions = repeat(quaternions, 'd -> b n d', b=b, n=n)
        translations = torch.zeros((b, n, 3), device=device)
      rotations = functional.quaternion_to_matrix(quaternions).detach()

      # go through the layers and apply invariant point attention and
      # feedforward
      for i in range(self.structure_module_depth):
        is_last = i == (self.structure_module_depth - 1)

        with autocast(enabled=False):
          single_repr = self.ipa_block(
              single_repr.float(),
              mask=batch['mask'].bool(),
              pairwise_repr=pairwise_repr.float(),
              rotations=rotations.float(),
              translations=translations.float()
          )

        # update quaternion and translation
        quaternion_update, translation_update = self.to_affine_update(single_repr
                                                                     ).chunk(2, dim=-1)
        quaternion_update = F.pad(quaternion_update, (1, 0), value=1.)
        # FIX: make sure quaternion_update is standardized
        quaternion_update = functional.l2_norm(quaternion_update)

        quaternions = functional.quaternion_multiply(quaternions, quaternion_update)
        translations = torch.einsum(
            'b n c, b n r c -> b n r', translation_update, rotations
        ) + translations
        rotations = functional.quaternion_to_matrix(quaternions)
        # No rotation gradients between iterations to stabilize training.
        if not is_last:
          rotations = rotations.detach()

        if self.training or is_last:
          angles = self.to_angles(single_repr, single_repr_init=single_repr_init)
          frames = functional.rigids_from_angles(
              batch['seq'],
              functional.rigids_scale((rotations, translations), self.position_scale),
              functional.l2_norm(angles)
          )
          coords = functional.rigids_to_positions(frames, batch['seq'])
          coords.type(original_dtype)
          outputs.append(
              dict(
                  frames=functional.rigids_scale(
                      (rotations, translations), self.position_scale
                  ),
                  act=single_repr,
                  atoms=dict(frames=frames, coords=coords, angles=angles)
              )
          )

    return outputs
