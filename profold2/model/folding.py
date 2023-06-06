"""Folding the protein based on s_{i}
  """
import functools
import logging

import torch
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from einops import rearrange, repeat

from profold2.model.commons import init_zero_, embedd_dim_get
from profold2.model.functional import (l2_norm, quaternion_multiply,
                                       quaternion_to_matrix, rigids_from_angles,
                                       rigids_scale, rigids_to_positions)
from profold2.utils import (default, exists,
                            torch_allow_tf32, torch_default_dtype)

logger = logging.getLogger(__name__)


def max_neg_value(t):
  return -torch.finfo(t.dtype).max


# classes
class InvariantPointAttention(nn.Module):
  """Invariant Point Attention
    """
  def __init__(self,
               *,
               dim,
               heads=8,
               scalar_key_dim=16,
               scalar_value_dim=16,
               point_key_dim=4,
               point_value_dim=4,
               pairwise_repr_dim=None,
               require_pairwise_repr=True,
               qkv_use_bias=False,
               eps=1e-8):
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

    # qkv projection for point attention (coordinate and orientation aware)
    point_weight_init_value = torch.log(
        torch.exp(torch.full((heads,), 1.)) - 1.)
    self.point_weights = nn.Parameter(point_weight_init_value)

    self.point_attn_logits_scale = ((num_attn_logits * point_key_dim) *
                                    (9 / 2))**-0.5

    self.to_point_q = nn.Linear(dim, point_key_dim * heads * 3, bias=qkv_use_bias)
    self.to_point_k = nn.Linear(dim, point_key_dim * heads * 3, bias=qkv_use_bias)
    self.to_point_v = nn.Linear(dim, point_value_dim * heads * 3, bias=qkv_use_bias)

    # pairwise representation projection to attention bias
    pairwise_repr_dim = default(pairwise_repr_dim,
                                dim) if require_pairwise_repr else 0

    if require_pairwise_repr:
      self.pairwise_attn_logits_scale = num_attn_logits**-0.5

      self.to_pairwise_attn_bias = nn.Sequential(
          nn.Linear(pairwise_repr_dim, heads),
          Rearrange('b ... h -> (b h) ...'))

    # combine out - scalar dim +
    #               pairwise dim +
    #               point dim * (3 for coordinates in R3 and then 1 for norm)
    self.to_out = nn.Linear(
        heads * (scalar_value_dim + pairwise_repr_dim + point_value_dim *
                 (3 + 1)), dim)

  def forward(self,
              single_repr,
              pairwise_repr=None,
              *,
              rotations,
              translations,
              mask=None):
    x, b, h, eps = single_repr, single_repr.shape[0], self.heads, self.eps
    assert not (self.require_pairwise_repr and not exists(pairwise_repr)
               ), 'pairwise representation must be given as second argument'

    # get queries, keys, values for scalar and point (coordinate-aware)
    # attention pathways
    q_scalar, k_scalar, v_scalar = self.to_scalar_q(x), self.to_scalar_k(
        x), self.to_scalar_v(x)
    q_point, k_point, v_point = self.to_point_q(x), self.to_point_k(
        x), self.to_point_v(x)

    # split out heads
    q_scalar, k_scalar, v_scalar = map(
        lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
        (q_scalar, k_scalar, v_scalar))
    q_point, k_point, v_point = map(
        lambda t: rearrange(t, 'b n (h d c) -> (b h) n d c', h=h, c=3),
        (q_point, k_point, v_point))

    rotations = repeat(rotations, 'b n d r -> (b h) n d r', h=h)
    translations = repeat(translations, 'b n c -> (b h) n () c', h=h)

    # rotate qkv points into global frame
    q_point = torch.einsum('b n d c, b n r c -> b n d r', q_point,
                           rotations) + translations
    k_point = torch.einsum('b n d c, b n r c -> b n d r', k_point,
                           rotations) + translations
    v_point = torch.einsum('b n d c, b n r c -> b n d r', v_point,
                           rotations) + translations

    # derive attn logits for scalar and pairwise
    attn_logits_scalar = torch.einsum('b i d, b j d -> b i j', q_scalar,
                                      k_scalar) * self.scalar_attn_logits_scale

    if self.require_pairwise_repr:
      attn_logits_pairwise = self.to_pairwise_attn_bias(
          pairwise_repr) * self.pairwise_attn_logits_scale

    # derive attn logits for point attention
    point_qk_diff = rearrange(q_point, 'b i d c -> b i () d c') - rearrange(
        k_point, 'b j d c -> b () j d c')
    point_dist = (point_qk_diff**2).sum(dim=-2)

    point_weights = F.softplus(self.point_weights)
    point_weights = repeat(point_weights, 'h -> (b h) () () ()', b=b)

    attn_logits_points = -0.5 * (point_dist * point_weights *
                                 self.point_attn_logits_scale).sum(dim=-1)

    # combine attn logits
    attn_logits = attn_logits_scalar + attn_logits_points

    if self.require_pairwise_repr:
      attn_logits = attn_logits + attn_logits_pairwise

    # mask
    if exists(mask):
      mask = rearrange(mask, 'b i -> b i ()') * rearrange(mask, 'b j -> b () j')
      mask = repeat(mask, 'b i j -> (b h) i j', h=h)
      mask_value = max_neg_value(attn_logits)
      attn_logits = attn_logits.masked_fill(~mask, mask_value)

    # attention
    attn = F.softmax(attn_logits, dim=-1)

    # disable TF32 for precision
    # with torch_allow_tf32(allow=False), autocast(enabled=False):
    with torch_allow_tf32(allow=False):

      # aggregate values
      results_scalar = torch.einsum('b i j, b j d -> b i d', attn, v_scalar)

      attn_with_heads = rearrange(attn, '(b h) i j -> b h i j', h=h)

      if self.require_pairwise_repr:
        results_pairwise = torch.einsum('b h i j, b i j d -> b h i d',
                                        attn_with_heads, pairwise_repr)

      # aggregate point values
      results_points = torch.einsum('b i j, b j d c -> b i d c', attn, v_point)

      # rotate aggregated point values back into local frame
      results_points = torch.einsum('b n d c, b n r c -> b n d r',
                                    results_points - translations,
                                    rotations.transpose(-1, -2))
      results_points_norm = torch.sqrt(
          torch.square(results_points).sum(dim=-1) + eps)

    # merge back heads
    results_scalar = rearrange(results_scalar, '(b h) n d -> b n (h d)', h=h)
    results_points = rearrange(results_points,
                               '(b h) n d c -> b n (h d c)',
                               h=h)
    results_points_norm = rearrange(results_points_norm,
                                    '(b h) n d -> b n (h d)',
                                    h=h)

    results = (results_scalar, results_points, results_points_norm)

    if self.require_pairwise_repr:
      results_pairwise = rearrange(results_pairwise,
                                   'b h n d -> b n (h d)',
                                   h=h)
      results = (*results, results_pairwise)

    # concat results and project out
    results = torch.cat(results, dim=-1)
    return self.to_out(results)


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
  def __init__(
      self,
      *,
      dim,
      ff_mult=1,
      ff_num_layers=3,
      dropout=.0,
      **kwargs):
    super().__init__()
    dim_single, _ = embedd_dim_get(dim)

    self.attn_norm = nn.LayerNorm(dim_single)
    self.attn = InvariantPointAttention(dim=dim, **kwargs)

    self.ff_norm = nn.LayerNorm(dim_single)
    self.ff = Transition(dim_single, mult=ff_mult, num_layers=ff_num_layers)

    self.dropout_fn = functools.partial(F.dropout, p=dropout)

  def forward(self, x, **kwargs):
    x = self.attn(x, **kwargs) + x
    x = self.attn_norm(self.dropout_fn(x, training=self.training))

    x = self.ff(x) + x
    x = self.ff_norm(self.dropout_fn(x, training=self.training))
    return x


class AngleNetBlock(nn.Module):

  def __init__(self, dim, channel=128):
    super().__init__()

    self.net = nn.Sequential(nn.ReLU(), nn.Linear(dim, channel), nn.ReLU(),
                             nn.Linear(channel, dim))

  def forward(self, x):
    return x + self.net(x)


class AngleNet(nn.Module):
  """Predict the torsion angles
    """
  def __init__(self, dim, channel=128, num_blocks=2):
    super().__init__()

    self.projection = nn.Linear(dim, channel)
    self.projection_init = nn.Linear(dim, channel)

    self.blocks = nn.Sequential(
        *[AngleNetBlock(channel, channel) for _ in range(num_blocks)])

    self.to_groups = nn.Linear(channel, 14)

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
  def __init__(self,
               dim,
               structure_module_depth,
               structure_module_heads,
               dropout=.0,
               position_scale=1.0,
               **kwargs):
    super().__init__()
    dim_single, dim_pairwise = embedd_dim_get(dim)

    assert structure_module_depth >= 1
    self.structure_module_depth = structure_module_depth
    self.position_scale = position_scale
    with torch_default_dtype(torch.float32):
      self.ipa_block = IPABlock(dim=dim_single,
                                pairwise_repr_dim=dim_pairwise,
                                heads=structure_module_heads,
                                dropout=dropout,
                                **kwargs)

      self.to_affine_update = nn.Linear(dim_single, 6)

    init_zero_(self.ipa_block.attn.to_out)

    self.single_repr_norm = nn.LayerNorm(dim_single)
    self.pairwise_repr_norm = nn.LayerNorm(dim_pairwise)
    self.single_repr_dim = nn.Sequential(nn.Linear(dim_single, dim_single))

    self.to_angles = AngleNet(dim_single)

  def forward(self, representations, batch):
    b, n, device = *batch['seq'].shape[:2], batch['seq'].device

    single_repr, pairwise_repr = representations['single'], representations[
        'pair']

    single_repr = self.single_repr_norm(single_repr)
    pairwise_repr = self.pairwise_repr_norm(pairwise_repr)

    single_repr_init = single_repr
    single_repr = self.single_repr_dim(single_repr)

    # prepare float32 precision for equivariance
    original_dtype = single_repr.dtype
    single_repr, pairwise_repr = map(lambda t: t.float(),
                                     (single_repr, pairwise_repr))

    outputs = []

    # iterative refinement with equivariant transformer in high precision
    with torch_default_dtype(torch.float32):
      quaternions = torch.tensor([1., 0., 0., 0.],
                                 device=device)  # initial rotations
      quaternions = repeat(quaternions, 'd -> b n d', b=b, n=n)
      translations = torch.zeros((b, n, 3), device=device)

      # go through the layers and apply invariant point attention and
      # feedforward
      for i in range(self.structure_module_depth):
        is_last = i == (self.structure_module_depth - 1)

        rotations = quaternion_to_matrix(quaternions)
        # No rotation gradients between iterations to stabilize training.
        if not is_last:
          rotations = rotations.detach()

        single_repr = self.ipa_block(single_repr,
                                     mask=batch['mask'].bool(),
                                     pairwise_repr=pairwise_repr,
                                     rotations=rotations,
                                     translations=translations)

        # update quaternion and translation
        quaternion_update, translation_update = self.to_affine_update(
            single_repr).chunk(2, dim=-1)
        quaternion_update = F.pad(quaternion_update, (1, 0), value=1.)
        # FIX: make sure quaternion_update is standardized
        quaternion_update = l2_norm(quaternion_update)

        quaternions = quaternion_multiply(quaternions, quaternion_update)
        translations = torch.einsum('b n c, b n r c -> b n r',
                                    translation_update,
                                    rotations) + translations

        if self.training or is_last:
          angles = self.to_angles(single_repr,
                                  single_repr_init=single_repr_init)
          frames = rigids_from_angles(
              batch['seq'],
              rigids_scale((rotations, translations), self.position_scale),
              l2_norm(angles))
          frames = rigids_scale(frames, 1.0 / self.position_scale)
          coords = rigids_to_positions(frames, batch['seq'])
          coords.type(original_dtype)
          outputs.append(
              dict(frames=(rotations, translations),
                   act=single_repr,
                   atoms=dict(frames=frames, coords=coords, angles=angles)))

    return outputs
