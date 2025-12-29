import collections
import functools
from typing import Optional, Union

import numpy as np
import torch
from torch.nn import functional as F
from einops import rearrange, repeat

from profold2.common import residue_constants
from profold2.model import accelerator
from profold2.utils import default, exists


def apc(x, dim=None, epsilon=1e-8):
  """Perform average product correct, used for contact prediction."""
  i, j = default(dim, (-1, -2))
  a1 = torch.sum(x, dim=i, keepdim=True)
  a2 = torch.sum(x, dim=j, keepdim=True)
  a12 = torch.sum(x, dim=(i, j), keepdim=True)

  avg = a1 * a2 / (a12 + epsilon)
  return x - avg


def l2_norm(v, dim=-1, epsilon=1e-12):
  return v / torch.clamp(
      torch.linalg.norm(v, dim=dim, keepdim=True),  # pylint: disable=not-callable
      min=epsilon
  )


def squared_cdist(x, y, keepdim=False):
  return torch.sum(
      (x[..., :, None, :] - y[..., None, :, :])**2, dim=-1, keepdim=keepdim
  )


def masked_mean(mask, value, dim=None, keepdim=False, epsilon=1e-10):
  if exists(dim):
    return torch.sum(mask * value, dim=dim, keepdim=keepdim
                    ) / (epsilon + torch.sum(mask, dim=dim, keepdim=keepdim))
  return torch.sum(mask * value) / (epsilon + torch.sum(mask))


def scatter_add(
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


def scatter_sum(
    index: torch.Tensor,
    src: torch.Tensor,
    dim: Optional[int] = -1,
    out: Optional[torch.Tensor] = None,
    out_dim: Optional[int] = None
) -> torch.Tensor:
  return scatter_add(index, src, dim=dim, out=out, out_dim=out_dim)


def scatter_mean(
    index: torch.Tensor,
    src: torch.Tensor,
    dim: Optional[int] = -1,
    out: Optional[torch.Tensor] = None,
    out_dim: Optional[int] = None
):
  out = scatter_sum(index, src, dim=dim, out=out, out_dim=out_dim)
  out_dim = out.shape[dim]

  one = torch.ones_like(index, dtype=src.dtype, device=src.device)
  count = torch.clamp(scatter_sum(index, one, dim=dim, out_dim=out_dim), min=1)

  return out / count


@functools.lru_cache(maxsize=8)
def make_mask(restypes, mask, device=None):
  num_class = len(restypes)
  if exists(mask) and mask:
    m = [restypes.index(i) for i in mask]
    # Shape (k, c)
    m = F.one_hot(  # pylint: disable=not-callable
        torch.as_tensor(m, dtype=torch.int, device=device).long(), num_class)
    # Shape (c)
    m = ~(torch.sum(m, dim=0) > 0)
    return m.float()
  return torch.as_tensor([1.0] * num_class, device=device)


def batched_gather(params, indices, dim=1, has_batch_dim=False):
  b, device = indices.shape[0], indices.device
  if isinstance(params, np.ndarray):
    params = torch.from_numpy(params).to(device)
  if not has_batch_dim:
    params = repeat(params, 'n ... -> b n ...', b=b)
  c = len(params.shape) - len(indices.shape)
  assert c >= 0
  ext = list(map(chr, range(ord('o'), ord('o') + c)))
  kwargs = dict(zip(ext, params.shape[-c:]))
  ext = ' '.join(ext)
  return torch.gather(
      params, dim, repeat(indices.long(), f'b n ... -> b n ... {ext}', **kwargs)
  )


def sharded_apply(
    fn, sharded_args, *args, shard_size=1, shard_dim=0, cat_dim=0, **kwargs
):
  """Sharded apply.

  Applies `fn` over shards to sharded_args
  """
  def run_fn(*sharded_args):
    return fn(*(sharded_args + args), **kwargs)

  def run_chunk(*sharded_args):
    assert len(sharded_args) > 0 and exists(sharded_args[0])
    batched_dim = sharded_args[0].shape[shard_dim]
    assert all(
        map(lambda x: not exists(x) or x.shape[shard_dim] == batched_dim, sharded_args)
    )
    for slice_args in zip(
        *map(lambda x: torch.split(x, shard_size, dim=shard_dim), sharded_args)
    ):
      yield run_fn(*slice_args)

  # shard size None denotes no sharding
  if not exists(shard_size):
    return run_fn(*sharded_args)

  if isinstance(cat_dim, int):
    return torch.cat(list(run_chunk(*sharded_args)), dim=cat_dim)
  return cat_dim(run_chunk(*sharded_args))


"""
The transformation matrices returned from the functions in this file assume
the points on which the transformation will be applied are column vectors.
i.e. the R matrix is structured as

    R = [
            [Rxx, Rxy, Rxz],
            [Ryx, Ryy, Ryz],
            [Rzx, Rzy, Rzz],
        ]  # (3, 3)

This matrix can be applied to column vectors by post multiplication
by the points e.g.

    points = [[0], [1], [2]]  # (3 x 1) xyz coordinates of a point
    transformed_points = R * points

To apply the same matrix to points which are row vectors, the R matrix
can be transposed and pre multiplied by the points:

e.g.
    points = [[0, 1, 2]]  # (1 x 3) xyz coordinates of a point
    transformed_points = points * R.transpose(1, 0)
"""


def quaternion_to_matrix(quaternions):
  """
  Convert rotations given as quaternions to rotation matrices.

  Args:
      quaternions: quaternions with real part first,
          as tensor of shape (..., 4).

  Returns:
      Rotation matrices as tensor of shape (..., 3, 3).
  """
  r, i, j, k = torch.unbind(quaternions, -1)
  two_s = 2.0 / (quaternions * quaternions).sum(-1)

  o = torch.stack(
      (
          1 - two_s * (j * j + k * k),
          two_s * (i * j - k * r),
          two_s * (i * k + j * r),
          two_s * (i * j + k * r),
          1 - two_s * (i * i + k * k),
          two_s * (j * k - i * r),
          two_s * (i * k - j * r),
          two_s * (j * k + i * r),
          1 - two_s * (i * i + j * j),
      ),
      -1,
  )
  return o.reshape(quaternions.shape[:-1] + (3, 3))


def _copysign(a, b):
  """
  Return a tensor where each element has the absolute value taken from the,
  corresponding element of a, with sign taken from the corresponding
  element of b. This is like the standard copysign floating-point operation,
  but is not careful about negative 0 and NaN.

  Args:
      a: source tensor.
      b: tensor whose signs will be used, of the same shape as a.

  Returns:
      Tensor of the same shape as a with the signs of b.
  """
  signs_differ = (a < 0) != (b < 0)
  return torch.where(signs_differ, -a, a)


def _sqrt_positive_part(x):
  """
  Returns torch.sqrt(torch.max(0, x))
  but with a zero subgradient where x is 0.
  """
  ret = torch.zeros_like(x)
  positive_mask = x > 0
  ret[positive_mask] = torch.sqrt(x[positive_mask])
  return ret


def matrix_to_quaternion(matrix):
  """
  Convert rotations given as rotation matrices to quaternions.

  Args:
      matrix: Rotation matrices as tensor of shape (..., 3, 3).

  Returns:
      quaternions with real part first, as tensor of shape (..., 4).
  """
  if matrix.size(-1) != 3 or matrix.size(-2) != 3:
    raise ValueError(f'Invalid rotation matrix  shape f{matrix.shape}.')
  m00 = matrix[..., 0, 0]
  m11 = matrix[..., 1, 1]
  m22 = matrix[..., 2, 2]
  o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
  x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
  y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
  z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
  o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
  o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
  o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
  return torch.stack((o0, o1, o2, o3), -1)


def standardize_quaternion(quaternions):
  """
  Convert a unit quaternion to a standard form: one in which the real
  part is non negative.

  Args:
      quaternions: Quaternions with real part first,
          as tensor of shape (..., 4).

  Returns:
      Standardized quaternions as tensor of shape (..., 4).
  """
  return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_raw_multiply(a, b):
  """
  Multiply two quaternions.
  Usual torch rules for broadcasting apply.

  Args:
      a: Quaternions as tensor of shape (..., 4), real part first.
      b: Quaternions as tensor of shape (..., 4), real part first.

  Returns:
      The product of a and b, a tensor of quaternions shape (..., 4).
  """
  aw, ax, ay, az = torch.unbind(a, -1)
  bw, bx, by, bz = torch.unbind(b, -1)
  ow = aw * bw - ax * bx - ay * by - az * bz
  ox = aw * bx + ax * bw + ay * bz - az * by
  oy = aw * by - ax * bz + ay * bw + az * bx
  oz = aw * bz + ax * by - ay * bx + az * bw
  return torch.stack((ow, ox, oy, oz), -1)


def quaternion_multiply(a, b):
  """
  Multiply two quaternions representing rotations, returning the quaternion
  representing their composition, i.e. the versor with nonnegative real part.
  Usual torch rules for broadcasting apply.

  Args:
      a: Quaternions as tensor of shape (..., 4), real part first.
      b: Quaternions as tensor of shape (..., 4), real part first.

  Returns:
      The product of a and b, a tensor of quaternions of shape (..., 4).
  """
  ab = quaternion_raw_multiply(a, b)
  return standardize_quaternion(ab)


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks=None):
  """Create pseudo beta features."""

  is_gly = torch.eq(
      aatype, residue_constants.restype_order[('G', residue_constants.PROT)]
  )
  ca_idx = residue_constants.atom_order['CA']
  cb_idx = residue_constants.atom_order['CB']
  pseudo_beta = torch.where(
      torch.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
      all_atom_positions[..., ca_idx, :], all_atom_positions[..., cb_idx, :]
  )

  if exists(all_atom_masks):
    pseudo_beta_mask = torch.where(
        is_gly, all_atom_masks[..., ca_idx], all_atom_masks[..., cb_idx]
    )
    pseudo_beta_mask = pseudo_beta_mask.float()
    return pseudo_beta, pseudo_beta_mask

  return pseudo_beta


def distogram_from_positions(breaks, x, y=None):
  lo_breaks = torch.square(breaks)
  hi_breaks = torch.cat(
      (lo_breaks[1:], torch.full((1, ), 1e8, device=breaks.device)), dim=-1
  )
  dist2 = squared_cdist(x, default(y, x), keepdim=True)
  dgram = (dist2 > lo_breaks) * (dist2 < hi_breaks)
  return dgram.float()


def lddt(
    pred_cdist, true_cdist, cdist_mask, cutoff=None, per_residue=True, smooth=False
):
  """Computes the lddt score for a batch of coordinates.
      https://academic.oup.com/bioinformatics/article/29/21/2722/195896
      Inputs:
      * pred_cdist: (..., i, j, d) array of predicted cdist of 3D points.
      * true_cdist: (..., i, j, d) array of true cdist of 3D points.
      * cdist_mask : (..., i, j) binary-valued array. 1 for cdist that exist in
          the true cdist
      * cutoff: maximum inclusion radius in reference struct.
      Outputs:
      * (..., i) lddt scores ranging between 0 and 1
  """
  assert len(pred_cdist.shape) >= 2 and len(true_cdist.shape) >= 2
  cutoff = default(cutoff, 15.0)
  eps = 1e-10

  # NOTE: cdist_mask should exclude self-interaction
  cdist_to_score = (true_cdist < cutoff) * cdist_mask

  # Shift unscored distances to be far away
  dist_l1 = torch.abs(true_cdist - pred_cdist)

  # Normalize over the appropriate axes.
  reduce_dim = -1 if per_residue else (-2, -1)

  # True lDDT uses a number of fixed bins.
  # We ignore the physical plausibility correction to lDDT, though.
  if smooth:
    score = 0.25 * sum(
        torch.sum(cdist_to_score * F.sigmoid(t - dist_l1), dim=reduce_dim)
        for t in (0.5, 1.0, 2.0, 4.0)
    )
  else:
    score = 0.25 * sum(
        torch.sum(cdist_to_score * (dist_l1 < t), dim=reduce_dim)
        for t in (0.5, 1.0, 2.0, 4.0)
    )

  return (score + eps) / (torch.sum(cdist_to_score, dim=reduce_dim) + eps)


def plddt(logits):
  """Compute per-residue pLDDT from logits
  """
  device = logits.device if hasattr(logits, 'device') else None
  # Shape (b, l, c)
  c = logits.shape[-1]
  width = 1.0 / c
  centers = torch.arange(start=0.5 * width, end=1.0, step=width, device=device)
  probs = F.softmax(logits, dim=-1)
  return torch.einsum('c,... c -> ...', centers, probs)


def bin_centers_from_breaks(breaks):
  # Add half-step to get the center
  step = (breaks[..., 1:] - breaks[..., :-1])
  step = torch.cat((step, torch.mean(step, dim=-1, keepdim=True)), dim=-1)
  bin_centers = breaks + step
  bin_centers = torch.cat((bin_centers, bin_centers[..., -1:] + step[..., -1:]), dim=-1)
  return bin_centers


def pae(logits, breaks, mask=None, return_mae=False):
  """Computes aligned confidence metrics from logits
  """
  probs = F.softmax(logits, dim=-1)
  bin_centers = bin_centers_from_breaks(breaks)

  expected_align_error = torch.sum(probs * bin_centers, dim=-1)
  if exists(mask):
    pair_mask = mask[..., :, None] * mask[..., None, :]
    expected_align_error = expected_align_error * pair_mask

  if return_mae:  # return max aligned error
    return expected_align_error, bin_centers[..., -1]
  return expected_align_error


def ptm(logits, breaks, mask=None, seq_color=None):
  """Compute predicted TM alignment
  """
  assert logits.shape[-2] == logits.shape[-3]
  assert len(breaks.shape) == 1
  probs = F.softmax(logits, dim=-1)
  bin_centers = bin_centers_from_breaks(breaks)

  # Clip num_res to avoid negative/undefined d0.
  if exists(mask):
    n = torch.clamp(torch.sum(mask, dim=-1), min=19)
    n = repeat(n, '... -> ... d', d=bin_centers.shape[0])
  else:
    n = max(logits.shape[-2], 19)

  # Compute d_0(num_res) as defined by TM-score, eqn. (5) in Yang & Skolnick
  # "Scoring function for automated assessment of protein structure template
  # quality", 2004: http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
  d0 = 1.24 * (n - 15)**(1. / 3) - 1.8

  # TM-Score term for every bin.
  tm_per_bin = 1. / (1 + (bin_centers / d0)**2)
  # E_distances tm(distance).
  # predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)
  predicted_tm_term = torch.einsum('... i j d, ... d->... i j', probs, tm_per_bin)

  if exists(mask):
    pair_mask = mask[..., :, None] * mask[..., None, :]
    if exists(seq_color):
      pair_color = (seq_color[..., :, None] != seq_color[..., None, :])
      if torch.any(pair_color):  # iptm
        pair_mask *= pair_color

  if exists(mask):
    per_alignment = masked_mean(value=predicted_tm_term, mask=pair_mask, dim=-1)
  else:
    per_alignment = torch.mean(per_alignment, dim=-1)

  return torch.amax(per_alignment, dim=-1)


Rigids = collections.namedtuple('Rigids', ['rotations', 'translations'])


def rotations_from_vecs(v1, v2, epsilon=1e-8):
  e1 = l2_norm(v1, epsilon=epsilon)
  c = torch.sum(e1 * v2, dim=-1, keepdim=True)
  u2 = v2 - e1 * c
  e2 = l2_norm(u2, epsilon=epsilon)
  e3 = torch.cross(e1, e2, dim=-1)
  R = torch.stack((e1, e2, e3), dim=-1)  # pylint: disable=invalid-name

  return R


def rotations_from_randn(*shape, device=None, epsilon=1e-8):
  # create a random rotation (Gram-Schmidt orthogonalization of two random normal
  # vectors)
  v1 = torch.randn(*shape, 3, device=device)
  v2 = torch.randn(*shape, 3, device=device)
  return rotations_from_vecs(v1, v2, epsilon=epsilon)


def rigids_from_randn(*shape, device=None, epsilon=1e-8):
  R = rotations_from_randn(*shape, device=device, epsilon=epsilon)
  t = torch.randn(*shape, 3, device=device)
  return R, t


def rigids_from_3x3(points, indices=None, epsilon=1e-6):
  """Create rigids from 3 points.
  This creates a set of rigid transformations from 3 points by Gram Schmidt
  orthogonalization.
  """
  indices = default(indices, (0, 1, 2))

  # Shape (b, l, 3, 3)
  assert points.shape[-1] == 3
  assert points.shape[-2] >= 3
  v1 = points[..., indices[0], :] - points[..., indices[1], :]
  v2 = points[..., indices[2], :] - points[..., indices[1], :]

  e1 = l2_norm(v1, epsilon=epsilon)
  c = torch.sum(e1 * v2, dim=-1, keepdim=True)
  u2 = v2 - e1 * c
  e2 = l2_norm(u2, epsilon=epsilon)
  e3 = torch.cross(e1, e2, dim=-1)
  R = torch.stack((e1, e2, e3), dim=-1)  # pylint: disable=invalid-name
  t = points[..., indices[1], :]

  return R, t


def rigids_from_4x4(m):
  """Create rigids from 4x4 array
  """
  # Shape (..., 4, 4)
  assert m.shape[-2:] == (4, 4)
  return m[..., :3, :3], m[..., :3, 3]


def angles_from_positions(aatypes, coords, coord_mask, placeholder_for_undefined=False):
  # prev_coords, prev_mask = coords[..., :-1, :, :], coord_mask[..., :-1, :]
  # this_coords, this_mask = coords[..., 1:, :, :], coord_mask[..., 1:, :]

  #omega_points = torch.stack((prev_coords[...,]))
  # (N, 7, 4, 14)
  torsion_atom14_idx = F.one_hot(  # pylint: disable=not-callable
      batched_gather(residue_constants.chi_angles_atom14_indices,
                     aatypes).long(), residue_constants.atom14_type_num)
  torsion_atom14_exists = batched_gather(
      residue_constants.chi_angles_atom14_exists, aatypes
  )

  # (N, 7, 4, 3)
  torsion_points = torch.einsum(
      '... n d,... g m n -> ... g m d', coords, torsion_atom14_idx.float()
  )
  # (N, 7, 4)
  torsion_point_mask = torch.einsum(
      '... n,... g m n -> ... g m', coord_mask.float(), torsion_atom14_idx.float()
  )

  # fix omega, phi angles
  for i in range(torsion_points.shape[-4] - 1, 0, -1):
    torsion_points[..., i, 0, :2, :] = torsion_points[..., i - 1, 0, :2, :]  # omega
    torsion_point_mask[..., i, 0, :2] = torsion_point_mask[..., i - 1, 0, :2]

    torsion_points[..., i, 1, :1, :] = torsion_points[..., i - 1, 1, :1, :]  # phi
    torsion_point_mask[..., i, 1, :1] = torsion_point_mask[..., i - 1, 1, :1]

  torsion_points[..., 0, 0, :2, :] = 0  # omega
  torsion_point_mask[..., 0, 0, :2] = 0
  torsion_points[..., 0, 1, :1, :] = 0  # phi
  torsion_point_mask[..., 0, 1, :1] = 0

  # Create a frame from the first three atoms:
  # First atom: point on x-y-plane
  # Second atom: point on negative x-axis
  # Third atom: origin
  # torsion_frames = rigids_from_3x3(torsion_points, indices=(1, 2, 0))
  torsion_frames = rotations_from_vecs(
      torsion_points[..., 2, :] - torsion_points[..., 1, :],
      torsion_points[..., 0, :] - torsion_points[..., 2, :]
  ), torsion_points[..., 2, :]

  # Compute the position of the forth atom in this frame (y and z coordinate
  # define the chi angle)
  def to_angles(rotations, translations, torsion_points_4):
    local_points = torch.einsum(
        '... w,... w h -> ... h', torsion_points_4 - translations, rotations
    )
    angles_sin_cos = torch.stack((local_points[..., 2], local_points[..., 1]), dim=-1)
    return angles_sin_cos / torch.sqrt(
        1e-8 + torch.sum(angles_sin_cos**2, dim=-1, keepdim=True)
    )

  torsion_angles = to_angles(*torsion_frames, torsion_points[..., 3, :])
  torsion_angles_mask = torch.prod(torsion_point_mask, dim=-1) * torsion_atom14_exists

  # Mirror psi, because we computed it from the Oxygen-atom.
  mirror_vec = [1., 1., -1.] + [1.] * (residue_constants.chi_angles_num - 3)
  torsion_angles *= rearrange(
      torch.as_tensor(mirror_vec, device=torsion_angles.device), 'g -> g ()'
  )

  # Create alternative angles for ambiguous atom names.
  chi_is_ambiguous = batched_gather(
      np.asarray(residue_constants.chi_pi_periodic, dtype=np.float32)[
          ..., :residue_constants.chi_angles_num - 3
      ], aatypes
  )
  mirror_ambiguous = torch.cat(
      (
          torch.ones(aatypes.shape + (3, ), dtype=torch.float32, device=aatypes.device),
          1 - 2 * chi_is_ambiguous,
      ),
      dim=-1
  )
  torsion_angles_alt = torsion_angles * mirror_ambiguous[..., None]
  if placeholder_for_undefined:
    # Add placeholder torsions in place of undefined torsion angles
    # (e.g. N-terminus pre-omega)
    placeholder_torsions = torch.stack(
        [
            torch.ones(torsion_angles.shape[:-1], device=torsion_angles.device),
            torch.zeros(torsion_angles.shape[:-1], device=torsion_angles.device)
        ],
        dim=-1
    )
    torsion_angles = torsion_angles * torsion_angles_mask[
        ..., None] + placeholder_torsions * (1 - torsion_angles_mask[..., None])
    torsion_angles_alt = torsion_angles_alt * torsion_angles_mask[
        ..., None] + placeholder_torsions * (1 - torsion_angles_mask[..., None])

  return dict(
      torsion_angles=torsion_angles,  # pylint: disable=use-dict-literal
      torsion_angles_mask=torsion_angles_mask,
      torsion_angles_alt=torsion_angles_alt
  )


def rigids_from_positions(aatypes, coords, coord_mask):
  # (N, 8, 3, 14)
  group_atom14_idx = F.one_hot(  # pylint: disable=not-callable
      batched_gather(residue_constants.restype_rigid_group_atom14_idx,
                     aatypes).long(), residue_constants.atom14_type_num)
  # Compute a mask whether the group exists.
  # (N, 8)
  group_exists = batched_gather(residue_constants.restype_rigid_group_mask, aatypes)
  if not exists(coords):
    return dict(atom_affine_exists=group_exists)  # pylint: disable=use-dict-literal

  group_points = torch.einsum(
      '... n d,... g m n -> ... g m d', coords, group_atom14_idx.float()
  )
  group_point_mask = torch.einsum(
      '... n,... g m n', coord_mask.float(), group_atom14_idx.float()
  )

  # Compute a mask whether ground truth exists for the group
  group_mask = torch.all(group_point_mask > 0, dim=-1) * group_exists

  # Compute the Rigids.
  # group_affine = rigids_from_3x3(group_points)
  group_affine = (
      rotations_from_vecs(
          group_points[..., 1, :] - group_points[..., 0, :],
          group_points[..., 2, :] - group_points[..., 1, :]
      ), group_points[..., 1, :]
  )
  # Adapt backbone frame to old convention (mirror x-axis and z-axis).
  rots = torch.tile(
      torch.eye(3, dtype=torch.float32, device=group_points.device),
      [residue_constants.restype_rigid_group_num, 1, 1]
  )
  rots[0, 0, 0] = -1
  rots[0, 2, 2] = -1
  group_affine = rigids_rotate(group_affine, rots)

  # The frames for ambiguous rigid groups are just rotated by 180 degree around
  # the x-axis. The ambiguous group is always the last chi-group.
  restype_rigid_group_is_ambiguous = np.zeros(
      [residue_constants.restype_num + 1, residue_constants.restype_rigid_group_num],
      dtype=np.float32
  )
  restype_rigid_group_rotations = np.tile(
      np.eye(3, dtype=np.float32), [
          residue_constants.restype_num + 1, residue_constants.restype_rigid_group_num,
          1, 1
      ]
  )

  for resname, _ in residue_constants.residue_atom_renaming_swaps.items():
    restype = residue_constants.restype_order[residue_constants.restype_3to1[resname]]
    chi_idx = int(sum(residue_constants.torsion_angles_mask[restype]) - 1)
    restype_rigid_group_is_ambiguous[restype, chi_idx + 1] = 1
    restype_rigid_group_rotations[restype, chi_idx + 1, 1, 1] = -1
    restype_rigid_group_rotations[restype, chi_idx + 1, 2, 2] = -1

  group_is_ambiguous = batched_gather(restype_rigid_group_is_ambiguous, aatypes)
  group_ambiguous_rotations = batched_gather(restype_rigid_group_rotations, aatypes)

  # Create the alternative ground truth frames.
  group_affine_alt = rigids_rotate(group_affine, group_ambiguous_rotations)

  return dict(
      atom_affine=group_affine,  # pylint: disable=use-dict-literal
      atom_affine_exists=group_exists,
      atom_affine_mask=group_mask,
      atom_affine_is_ambiguous=group_is_ambiguous,
      atom_affine_alt=group_affine_alt
  )


def rigids_to_positions(frames, aatypes):
  # Shape ((b, l, 8, 3, 3), (b, l, 8, 3))
  rotations, translations = frames

  # Shape (b, l, 14)
  group_idx = batched_gather(residue_constants.restype_atom14_to_rigid_group, aatypes)
  # Shape (b, l, 14, 8)
  group_mask = F.one_hot(
      group_idx.long(), num_classes=residue_constants.restype_rigid_group_num
  ).float()  # pylint: disable=not-callable

  rotations = torch.einsum('... m n,... n h w->... m h w', group_mask, rotations)
  translations = torch.einsum('... m n,... n h->... m h', group_mask, translations)

  # Gather the literature atom positions for each residue.
  # Shape (b, l, 14, 3)
  group_pos = batched_gather(
      residue_constants.restype_atom14_rigid_group_positions, aatypes
  )

  # Transform each atom from it's local frame to the global frame.
  positions = rigids_apply((rotations, translations), group_pos)

  # Mask out non-existing atoms.
  mask = batched_gather(residue_constants.restype_atom14_mask, aatypes)
  return positions * mask[..., None]


def rigids_slice(frames, start=0, end=None):
  rotations, translations = frames
  return rotations[..., start:end, :, :], translations[..., start:end, :]


def rigids_rearrange(frames, ops1, ops2=None):
  rotations, translations = frames
  return rearrange(rotations, ops1), rearrange(translations, default(ops2, ops1))


def rigids_multiply(a, b):
  rotations, translations = b
  rotations, _ = rigids_rotate(a, rotations)
  return rotations, rigids_apply(a, translations)


def rigids_apply(frames, points):
  rotations, translations = frames
  return torch.einsum('... h w,... w -> ... h', rotations, points) + translations


def rigids_rotate(frames, mat3x3):
  rotations, translations = frames
  rotations = torch.einsum('... h d, ... d w -> ... h w', rotations, mat3x3)
  return rotations, translations


def rigids_scale(frames, position_scale):
  rotations, translations = frames
  return rotations, translations * position_scale


def rigids_from_angles(aatypes, backb_frames, angles):
  """Create rigids from torsion angles
  """
  # Shape (b, l, 3, 3), (b, l, 3)
  backb_rotations, backb_trans = backb_frames
  assert backb_rotations.shape[-2:] == (3, 3) and backb_trans.shape[-1] == 3
  # Shape (b, l)
  assert aatypes.shape == backb_rotations.shape[:-2]
  # Shape (b, l, n, 2) s.t. n <= 7
  assert angles.shape[-1] == 2
  assert angles.shape[-2] < residue_constants.restype_rigid_group_num
  assert angles.shape[:-2] == aatypes.shape

  _, _, n = angles.shape[:3]

  # Gather the default frames for all rigids (b, l, 8, 3, 3), (b, l, 8, 3)
  m = batched_gather(residue_constants.restype_rigid_group_default_frame, aatypes)
  default_frames = rigids_slice(rigids_from_4x4(m), 0, n + 1)

  #
  # Create the rotation matrices according to the given angles
  #

  # Insert zero rotation for backbone group.
  # Shape (b, l, n+1, 2)
  angles = torch.cat(
      (
          rearrange(
              torch.stack(
                  (torch.zeros_like(aatypes), torch.ones_like(aatypes)), dim=-1
              ), '... i r -> ... i () r'
          ), angles
      ),
      dim=-2
  )
  sin_angles, cos_angles = torch.unbind(angles, dim=-1)
  zeros, ones = torch.zeros_like(sin_angles), torch.ones_like(cos_angles)
  # Shape (b, l, n+1, 3, 3)
  rotations = torch.stack(
      (
          ones,  zeros,       zeros,
          zeros, cos_angles, -sin_angles,
          zeros, sin_angles,  cos_angles
      ),
      dim=-1
  )
  rotations = rearrange(rotations, '... (h w) -> ... h w', h=3, w=3)

  # Apply rotations to the frames.
  atom_frames = rigids_rotate(default_frames, rotations)

  # \chi_2, \chi_3, and \chi_4 frames do not transform to the backbone frame
  # but to the previous frame. So chain them up accordingly.
  depend = batched_gather(residue_constants.restype_rigid_group_depend, aatypes)

  def to_prev_frames(frames, idx):
    rotations, translations = frames

    assert rotations.device == translations.device
    assert rotations.shape[:-2] == translations.shape[:-1]

    ri, ti = rigids_multiply(
        (
            torch.gather(
                rotations, -3, repeat(depend[..., idx], '... i -> ... i () c d', c=3, d=3)
            ),
            torch.gather(
                translations, -2, repeat(depend[..., idx], '... i -> ... i () d', d=3)
            ),
        ),
        (rotations[..., idx:idx + 1, :, :], translations[..., idx:idx + 1, :])
    )
    return torch.cat(
        (rotations[..., :idx, :, :], ri, rotations[..., idx + 1:, :, :]), dim=-3
    ), torch.cat(
        (translations[..., :idx, :], ti, translations[..., idx + 1:, :]), dim=-2
    )

  for i in range(5, residue_constants.chi_angles_num + 1):
    atom_frames = to_prev_frames(atom_frames, i)

  return rigids_multiply(
      rigids_rearrange(
          backb_frames, '... i c d -> ... i () c d', '... i d -> ... i () d'
      ), atom_frames
  )


def fape(
    pred_frames,
    true_frames,
    frames_mask,
    pred_points,
    true_points,
    points_mask,
    clamp_distance=None,
    clamp_ratio=None,
    dij_weight=None,
    use_weighted_mask=True,
    epsilon=1e-8
):
  """ FAPE(Frame Aligined Point Error) - Measure point error under different
  alignments
  """
  # Shape (b, l, 3, 3), (b, l, 3)
  pred_rotations, pred_trans = pred_frames
  assert pred_rotations.shape[-2:] == (3, 3) and pred_trans.shape[-1] == 3
  # Shape (b, l, 3, 3), (b, l, 3)
  true_rotations, true_trans = true_frames
  assert true_rotations.shape[-2:] == (3, 3) and true_trans.shape[-1] == 3
  ## Shape (b, l)
  #assert frames_mask.shape[:2] == points_mask.shape[:2]
  # Shape (b, n, 3)
  assert pred_points.shape[-1] == 3 and true_points.shape[-1] == 3
  # Shape (b, n)
  assert pred_points.shape[:-1] == points_mask.shape
  clamp_ratio = default(clamp_ratio, 1.0)
  if isinstance(clamp_ratio, torch.Tensor):
    assert torch.all(torch.logical_and(0.0 <= clamp_ratio, clamp_ratio <= 1.0))
  else:
    assert 0.0 <= clamp_ratio <= 1.0

  def to_local(R, t, points):
    # invert apply frames: R^t (x - t)
    return torch.einsum(
        '... j w,... w h -> ... j h', points[..., None, :, :] - t[..., :, None, :], R
    )

  pred_xij = to_local(pred_rotations, pred_trans, pred_points)
  true_xij = to_local(true_rotations, true_trans, true_points)

  # Shape (b, l, l, n)
  dij = torch.sqrt(torch.sum((pred_xij - true_xij)**2, dim=-1) + epsilon)
  if exists(clamp_distance):
    if isinstance(clamp_ratio, torch.Tensor) or (0 <= clamp_ratio < 1):
      dij = (1 - clamp_ratio) * dij + clamp_ratio * torch.clamp(dij, 0, clamp_distance)
    else:
      dij = torch.clamp(dij, 0, clamp_distance)
  dij_mask = rearrange(frames_mask, '... i -> ... i ()'
                      ) * rearrange(points_mask, '... j -> ... () j')

  dij_weight = default(dij_weight, 1.0)
  if use_weighted_mask:
    dij_mask = dij_mask * dij_weight
  else:
    dij *= dij_weight

  return masked_mean(value=dij, mask=dij_mask, epsilon=epsilon)


def between_ca_ca_distance_loss(
    pred_points, points_mask, residue_index, tau=1.5, epsilon=1e-6
):
  assert pred_points.shape[-1] == 3
  assert pred_points.shape[:-1] == points_mask.shape
  assert points_mask.shape[:-1] == residue_index.shape

  ca_idx = residue_constants.atom_order['CA']

  this_ca_point = pred_points[..., :-1, ca_idx, :]
  this_ca_mask = points_mask[..., :-1, ca_idx]
  next_ca_point = pred_points[..., 1:, ca_idx, :]
  next_ca_mask = points_mask[..., 1:, ca_idx]
  no_gap_mask = (residue_index[..., 1:] - residue_index[..., :-1]) == 1

  ca_ca_distance = torch.sqrt(
      epsilon + torch.sum((this_ca_point - next_ca_point)**2, dim=-1)
  )
  violations = torch.gt(ca_ca_distance - residue_constants.ca_ca, tau)
  mask = this_ca_mask * next_ca_mask * no_gap_mask
  return masked_mean(mask=mask, value=violations, epsilon=epsilon)


def between_residue_bond_loss(
    pred_points, points_mask, residue_index, aatypes, tau=12.0, epsilon=1e-6
):
  assert pred_points.shape[-1] == 3
  assert pred_points.shape[:-1] == points_mask.shape
  assert points_mask.shape[:-1] == residue_index.shape
  assert aatypes.shape == residue_index.shape

  n_idx = residue_constants.atom_order['N']
  ca_idx = residue_constants.atom_order['CA']
  c_idx = residue_constants.atom_order['C']

  # Get the positions of the relevant backbone atoms.
  this_ca_point = pred_points[..., :-1, ca_idx, :]
  this_ca_mask = points_mask[..., :-1, ca_idx]
  this_c_point = pred_points[..., :-1, c_idx, :]
  this_c_mask = points_mask[..., :-1, c_idx]
  next_n_point = pred_points[..., 1:, n_idx, :]
  next_n_mask = points_mask[..., 1:, n_idx]
  next_ca_point = pred_points[..., 1:, ca_idx, :]
  next_ca_mask = points_mask[..., 1:, ca_idx]
  no_gap_mask = (residue_index[..., 1:] - residue_index[..., :-1]) == 1

  # Compute bond length
  c_n_bond_length = torch.sqrt(
      epsilon + torch.sum((this_c_point - next_n_point)**2, dim=-1)
  )
  ca_c_bond_length = torch.sqrt(
      epsilon + torch.sum((this_ca_point - this_c_point)**2, dim=-1)
  )
  n_ca_bond_length = torch.sqrt(
      epsilon + torch.sum((next_n_point - next_ca_point)**2, dim=-1)
  )

  # Compute loss for the C--N bond.
  # The C-N bond to proline has slightly different length because of the ring.
  def bond_length_loss(pred_length, gt, mask):
    gt_length, gt_stddev = gt
    length_errors = torch.sqrt(epsilon + (pred_length - gt_length)**2)
    length_loss = F.relu(length_errors - tau * gt_stddev)
    return (
        length_loss, masked_mean(mask=mask, value=length_loss, epsilon=epsilon),
        mask * (length_errors > tau * gt_stddev)
    )

  next_is_proline = (aatypes[..., 1:] == residue_constants.resname_to_idx['PRO'])
  c_n_bond_labels = (
      (~next_is_proline) * residue_constants.between_res_bond_length_c_n[0] +
      next_is_proline * residue_constants.between_res_bond_length_c_n[1]
  )
  c_n_bond_stddev = (
      (~next_is_proline) * residue_constants.between_res_bond_length_stddev_c_n[0] +
      next_is_proline * residue_constants.between_res_bond_length_stddev_c_n[1]
  )
  c_n_bond_errors, c_n_loss, c_n_violation_mask = bond_length_loss(
      c_n_bond_length, (c_n_bond_labels, c_n_bond_stddev),
      this_c_mask * next_n_mask * no_gap_mask
  )

  # Compute loss for the angles.
  c_ca_unit_vec = (this_ca_point - this_c_point) / ca_c_bond_length[..., None]
  c_n_unit_vec = (next_n_point - this_c_point) / c_n_bond_length[..., None]
  n_ca_unit_vec = (next_ca_point - next_n_point) / n_ca_bond_length[..., None]

  def bond_angle_loss(x, y, gt, mask):
    gt_angle, gt_stddev = gt
    pred_angle = torch.sum(x * y, dim=-1)
    angle_errors = torch.sqrt(epsilon + (pred_angle - gt_angle)**2)
    angle_loss = F.relu(angle_errors - tau * gt_stddev)
    return (
        angle_loss, masked_mean(mask=mask, value=angle_loss, epsilon=epsilon),
        mask * (angle_errors > tau * gt_stddev)
    )

  ca_c_n_erros, ca_c_n_loss, ca_c_n_violation_mask = bond_angle_loss(
      c_ca_unit_vec, c_n_unit_vec, residue_constants.between_res_cos_angles_ca_c_n,
      this_ca_mask * this_c_mask * next_n_mask * no_gap_mask
  )
  c_n_ca_errors, c_n_ca_loss, c_n_ca_violation_mask = bond_angle_loss(
      -c_n_unit_vec, n_ca_unit_vec, residue_constants.between_res_cos_angles_c_n_ca,
      this_c_mask * next_n_mask * next_ca_mask * no_gap_mask
  )

  # Compute a per residue loss (equally distribute the loss to both
  # neighbouring residues).
  per_residue_violation = c_n_bond_errors + ca_c_n_erros + c_n_ca_errors
  per_residue_violation = 0.5 * (
      F.pad(per_residue_violation, (0, 1)) + F.pad(per_residue_violation, (1, 0))
  )

  # Compute hard violations.
  per_residue_violation_mask = torch.amax(
      torch.stack(
          (c_n_violation_mask, ca_c_n_violation_mask, c_n_ca_violation_mask), dim=0
      ),
      dim=0
  )
  per_residue_violation_mask = torch.maximum(
      F.pad(per_residue_violation_mask, (0, 1)),
      F.pad(per_residue_violation_mask, (1, 0))
  )

  return dict(
      c_n_bond_loss=c_n_loss,  # pylint: disable=use-dict-literal
      ca_c_n_angle_loss=ca_c_n_loss,
      c_n_ca_angle_loss=c_n_ca_loss,
      per_residue_violation=per_residue_violation,
      per_residue_violation_mask=per_residue_violation_mask
  )


def between_residue_clash_loss(
    pred_points, points_mask, residue_index, aatypes, tau=1.5, epsilon=1e-6
):
  """Loss to penalize steric clashes between residues"""
  assert pred_points.shape[-1] == 3
  assert pred_points.shape[:-1] == points_mask.shape
  assert points_mask.shape[:-1] == residue_index.shape
  assert aatypes.shape == residue_index.shape

  atom_radius = batched_gather(residue_constants.atom14_van_der_waals_radius, aatypes)
  atom_radius = atom_radius[..., :points_mask.shape[-1]]
  assert atom_radius.shape == points_mask.shape

  # Create the distance matrix
  dists = torch.sqrt(
      epsilon + torch.sum(
          (
              pred_points[..., :, None, :, None, :] -
              pred_points[..., None, :, None, :, :]
          )**2,
          dim=-1
      )
  )

  # Create the mask for valid distances.
  dist_mask = points_mask[..., :, None, :, None] * points_mask[..., None, :, None, :]
  # Mask out all the duplicate entries in the lower triangular matrix.
  # Also mask out the diagonal (atom-pairs from the same residue) -- these atoms
  # are handled separately.
  dist_mask = dist_mask * (
      residue_index[..., :, None, None, None] < residue_index[..., None, :, None, None]
  )

  # Backbone C--N bond between subsequent residues is no clash.
  n_idx = residue_constants.atom_order['N']
  c_idx = residue_constants.atom_order['C']
  atom_type_num = points_mask.shape[-1]
  c_atom = F.one_hot(  # pylint: disable=not-callable
      torch.as_tensor(c_idx, dtype=torch.long, device=pred_points.device), atom_type_num
  )
  n_atom = F.one_hot(  # pylint: disable=not-callable
      torch.as_tensor(n_idx, dtype=torch.long, device=pred_points.device), atom_type_num
  )
  neighbour_mask = (residue_index[..., :, None] + 1 == residue_index[..., None, :])
  c_n_bonds = neighbour_mask[..., None, None] * torch.reshape(
      c_atom[:, None] * n_atom[None, :],
      (1, ) * len(neighbour_mask.shape) + (atom_type_num, atom_type_num)
  )
  dist_mask *= (~c_n_bonds.bool())

  # Disulfide bridge between two cysteines is no clash.
  cys_sg_idx = residue_constants.restype_name_to_atom14_names['CYS'].index('SG')
  if cys_sg_idx < points_mask.shape[-1]:
    cys_sg_atom = F.one_hot(  # pylint: disable=not-callable
        torch.as_tensor(cys_sg_idx, dtype=torch.long, device=pred_points.device),
        points_mask.shape[-1]
    )
    disulfide_bonds = cys_sg_atom[:, None] * cys_sg_atom[None, :]
    dist_mask *= (~disulfide_bonds.bool())

  # Compute the lower bound for the allowed distances.
  dist_lower_bound = dist_mask * (
      atom_radius[..., :, None, :, None] + atom_radius[..., None, :, None, :]
  )

  # Compute the error.
  dist_errors = dist_mask * F.relu(dist_lower_bound - tau - dists)

  # Compute the mean loss.
  #clash_loss = masked_mean(mask=dist_mask, value=dist_errors, epsilon=epsilon)
  clash_loss = torch.sum(dist_errors) / (epsilon + torch.sum(dist_mask))

  # Compute the per atom loss sum.
  per_atom_clash = (
      torch.sum(dist_errors, dim=(-4, -2)) + torch.sum(dist_errors, dim=(-3, -1))
  )

  num_atoms = torch.sum(points_mask, dim=(-2, -1), keepdim=True)

  # Compute the hard clash mask.
  clash_mask = dist_mask * (dists < (dist_lower_bound - tau))
  per_atom_clash_mask = torch.maximum(
      torch.amax(clash_mask, dim=(-4, -2)), torch.amax(clash_mask, dim=(-3, -1))
  )

  return {
      'between_residue_clash_loss': torch.sum(per_atom_clash / (1e-6 + num_atoms)),
      'between_residue_per_atom_clash': per_atom_clash,
      'between_residue_per_atom_clash_mask': per_atom_clash_mask
  }


def within_residue_clash_loss(
    pred_points, points_mask, residue_index, aatypes, tau1=1.5, tau2=15, epsilon=1e-12
):
  """Loss to penalize steric clashes within residues"""
  assert pred_points.shape[-1] == 3
  assert pred_points.shape[:-1] == points_mask.shape
  assert points_mask.shape[:-1] == residue_index.shape
  assert aatypes.shape == residue_index.shape

  # Compute the mask for each residue.
  dist_mask = points_mask[..., :, None] * points_mask[..., None, :] * (
      ~torch.eye(points_mask.shape[-1], dtype=torch.bool, device=points_mask.device)
  )

  # Distance matrix
  dists = torch.sqrt(
      epsilon + torch.sum(
          (pred_points[..., :, None, :] - pred_points[..., None, :, :])**2, dim=-1
      )
  )

  # Compute the loss.
  restype_atom14_bounds = residue_constants.make_atom14_dists_bounds(
      overlap_tolerance=tau1, bond_length_tolerance_factor=tau2
  )
  num_atoms = points_mask.shape[-1]
  atom_lower_bound = batched_gather(
      restype_atom14_bounds['lower_bound'][..., :num_atoms, :num_atoms], aatypes
  )
  atom_upper_bound = batched_gather(
      restype_atom14_bounds['upper_bound'][..., :num_atoms, :num_atoms], aatypes
  )

  lower_errors = F.relu(atom_lower_bound - dists)
  upper_errors = F.relu(dists - atom_upper_bound)

  #clash_loss = masked_mean(mask=dist_mask,
  #        value=lower_errors+upper_errors,
  #        epsilon=epsilon)
  dist_errors = dist_mask * (lower_errors + upper_errors)
  clash_loss = torch.sum(dist_errors) / (epsilon + torch.sum(dist_mask))

  # Compute the per atom loss sum.
  per_atom_clash = torch.sum(dist_errors, dim=-2) + torch.sum(dist_errors, dim=-1)

  num_atoms = torch.sum(points_mask, dim=(-2, -1), keepdim=True)

  # Compute the violations mask.
  per_atom_clash_mask = dist_mask * (
      (dists < atom_lower_bound) | (dists > atom_upper_bound)
  )
  per_atom_clash_mask = torch.maximum(
      torch.amax(per_atom_clash_mask, dim=-2), torch.amax(per_atom_clash_mask, dim=-1)
  )

  return {
      'within_residue_clash_loss': torch.sum(per_atom_clash / (1e-6 + num_atoms)),
      'within_residue_per_atom_clash': per_atom_clash,
      'within_residue_per_atom_clash_mask': per_atom_clash_mask
  }


def symmetric_ground_truth_create_alt(seq, coord, coord_mask):
  coord_exists = batched_gather(residue_constants.restype_atom14_mask, seq)
  if not exists(coord):
    return dict(coord_exists=coord_exists)  # pylint: disable=use-dict-literal

  # pick the transformation matrices for the given residue sequence
  # shape (num_res, 14, 14)
  renaming_transform = batched_gather(residue_constants.RENAMING_MATRICES, seq)

  coord_alt = torch.einsum('... m d,... m n->... n d', coord, renaming_transform)
  coord_alt_mask = torch.einsum(
      '... m,... m n->... n', coord_mask.float(), renaming_transform
  )

  is_symmetric_mask = 1.0 - np.eye(residue_constants.RENAMING_MATRICES.shape[-1])
  coord_is_symmetric = batched_gather(
      np.sum(is_symmetric_mask * residue_constants.RENAMING_MATRICES, axis=-1) > 0, seq
  )

  return dict(
      coord_alt=coord_alt,  # pylint: disable=use-dict-literal
      coord_alt_mask=coord_alt_mask.bool(),
      coord_is_symmetric=coord_is_symmetric,
      coord_exists=coord_exists
  )


def symmetric_ground_truth_find_optimal(
    coord_pred,
    coord_exists,
    coord,
    coord_mask,
    coord_alt,
    coord_alt_mask,
    coord_is_symmetric,
    epsilon=1e-10
):
  """Find optimal renaming for ground truth that maximizes LDDT. """
  assert coord_pred.shape == coord.shape
  assert coord_pred.shape == coord_alt.shape
  assert coord_exists.shape == coord_mask.shape
  assert coord_exists.shape == coord_alt_mask.shape
  assert coord_exists.shape == coord_is_symmetric.shape

  def to_distance(points):
    return torch.sqrt(
        epsilon + torch.sum(
            (points[..., :, None, :, None, :] - points[..., None, :, None, :, :])**2,
            dim=-1
        )
    )

  # Create the pred distance matrix.
  # shape (N, N, 14, 14)
  pred_dist = to_distance(coord_pred)
  # Compute distances for ground truth with original and alternative names.
  # shape (N, N, 14, 14)
  gt_dist = to_distance(coord)
  gt_alt_dist = to_distance(coord_alt)

  def to_lddt(x, y):
    return torch.sqrt(epsilon + (x - y)**2)

  # Compute LDDT's.
  # shape (N, N, 14, 14)
  lddt_prim = to_lddt(pred_dist, gt_dist)
  lddt_alt = to_lddt(pred_dist, gt_alt_dist)

  # Create a mask for ambiguous atoms in rows vs. non-ambiguous atoms
  # in cols.
  # shape (N ,N, 14, 14)
  mask = (
      rearrange(coord_mask * coord_is_symmetric, '... i c -> ... i () c ()') *   # rows
      rearrange(coord_mask * (~coord_is_symmetric), '... j d -> ... () j () d')  # cols
  )

  # Aggregate distances for each residue to the non-amibuguous atoms.
  # shape (N)
  per_res_lddt = torch.sum(lddt_prim * mask, dim=(-3, -2, -1))
  per_res_lddt_alt = torch.sum(lddt_alt * mask, dim=(-3, -2, -1))

  # Decide for each residue, whether alternative naming is better.
  # shape (N)
  return per_res_lddt_alt < per_res_lddt  # alt_naming_is_better


def symmetric_ground_truth_renaming(
    coord_pred,
    coord_exists,
    coord,
    coord_mask,
    coord_alt,
    coord_alt_mask,
    coord_is_symmetric,
    epsilon=1e-10
):
  """Find optimal renaming of ground truth based on the predicted positions. """
  alt_naming_is_better = symmetric_ground_truth_find_optimal(
      coord_pred,
      coord_exists,
      coord,
      coord_mask,
      coord_alt,
      coord_alt_mask,
      coord_is_symmetric,
      epsilon=epsilon
  )

  def renaming(m, x, y):
    return (~m) * x + m * y

  coord_renamed = renaming(alt_naming_is_better[..., None, None], coord, coord_alt)
  coord_renamed_mask = renaming(
      alt_naming_is_better[..., None], coord_mask, coord_alt_mask
  )

  return {
      'alt_naming_is_better': alt_naming_is_better,
      'coord_renamed': coord_renamed,
      'coord_renamed_mask': coord_renamed_mask
  }


def contact_precision(
    pred: torch.Tensor,
    truth: torch.Tensor,
    ratios: Optional[list[float]] = None,
    ranges: Optional[list[tuple[int, int | None]]] = None,
    mask: Optional[torch.Tensor] = None,
    cutoff: Optional[float] = 8.
):
  if not exists(ratios):
    ratios = [1, .5, .2, .1]
  if not exists(ranges):
    ranges = [(6, 12), (12, 24), (24, None)]

  # (..., l, l)
  assert truth.shape[-1] == truth.shape[-2]
  assert pred.shape == truth.shape

  seq_len = truth.shape[-1]
  mask1s = torch.ones_like(truth, dtype=torch.int8)
  if exists(mask):
    mask1s = mask1s * (mask[..., :, None] * mask[..., None, :])
  mask_ranges = map(
      lambda r: torch.triu(mask1s, default(r[0], 0)) - torch.
      triu(mask1s, default(r[1], seq_len)), ranges
  )

  pred_truth = torch.stack((pred, truth), dim=-1)
  for (i, j), m in zip(ranges, mask_ranges):
    masked_pred_truth = pred_truth[m.bool()]
    sorter_idx = (-masked_pred_truth[:, 0]).argsort()
    sorted_pred_truth = masked_pred_truth[sorter_idx]

    num_corrects = (
        (0 < sorted_pred_truth[:, 1]) & (sorted_pred_truth[:, 1] <= cutoff)
    ).sum()
    for ratio in ratios:
      num_tops = max(1, min(num_corrects, int(seq_len * ratio)))
      assert 0 < num_tops <= seq_len
      top_labels = sorted_pred_truth[:num_tops, 1]
      pred_corrects = ((0 < top_labels) & (top_labels <= cutoff))
      pred_corrects = torch.sum(pred_corrects, dim=-1, keepdim=True)
      yield (i, j), ratio, pred_corrects / float(num_tops)


def rmsd(
    x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None, eps=1e-8
):
  """Calculate the RMSD between x and y.
  Args:
      x (torch.Tensor): source atom coordinates with shape `(num_res, 3)`.
      y (torch.Tensor): target atom coordinates with shape `(num_res, 3)`.
      mask (torch.Tensor optional): with shape `(num_res, )`.
  Returns:
      RMSD value.
    """
  d = torch.sum(torch.square(x - y), dim=-1)
  if exists(mask):
    d = masked_mean(value=d, mask=mask)
  else:
    d = torch.mean(d)
  return torch.sqrt(d + eps)


def kabsch_rotation(
    x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None
):
  """Calculate the best rotation that minimises the RMSD between x and y.
  Args:
      x (torch.Tensor): source atom coordinates with shape `(num_res, 3)`.
      y (torch.Tensor): target atom coordinates with shape `(num_res, 3)`.
      mask (torch.Tensor optional): with shape `(num_res, )`.
  Returns:
      r (torch.Tensor): rotation matrix with shape `(3, 3)`
    """
  assert x.shape[-2:] == y.shape[-2:] and x.shape[-1] == 3

  if exists(mask):
    y = y * mask[..., None]

  with accelerator.autocast(enabled=False):
    x, y = x.float(), y.float()

    # optimal rotation matrix via SVD of the convariance matrix {x.T * y}
    # v, _, w = torch.linalg.svd(x.T @ y)
    v, _, w = torch.linalg.svd(torch.einsum('... i c,... i d -> ... c d', x, y))

    # determinant sign for direction correction
    d = torch.sign(torch.det(v) * torch.det(w))
    v[..., -1, -1] = v[..., -1, -1] * d
    # Create Rotation matrix U
    r = v @ w
  return r


def kabsch_transform(
    x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None
):
  """Calculate the best rotation that minimises the RMSD between x and y.
  Args:
      x (torch.Tensor): source atom coordinates with shape `(num_res, 3)`.
      y (torch.Tensor): target atom coordinates with shape `(num_res, 3)`.
      mask (torch.Tensor optional): with shape `(num_res, )`.
  Returns:
      r (torch.Tensor): rotation matrix with shape `(3, 3)`
    """
  assert x.shape[-2:] == y.shape[-2:]

  if exists(mask):
    x_center = masked_mean(value=x, mask=mask[..., None], dim=-2)
    y_center = masked_mean(value=y, mask=mask[..., None], dim=-2)
  else:
    x_center = torch.mean(x, dim=-2)
    y_center = torch.mean(y, dim=-2)

  R = kabsch_rotation(x - x_center[..., None, :], y - y_center[..., None, :], mask=mask)  # pylint: disable=invalid-name
  t = x_center - torch.einsum('... h w, ... w -> ... h', R, y_center)

  return R, t


def kabsch_align(x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None):
  """ Kabsch alignment of x into y. Assumes x, y are both (num_res, 3).
    """
  # center x and y to the origin
  if exists(mask):
    x_ = x - masked_mean(value=x, mask=mask[..., None], dim=-2, keepdim=True)
    y_ = y - masked_mean(value=y, mask=mask[..., None], dim=-2, keepdim=True)
  else:
    x_ = x - x.mean(dim=-2, keepdim=True)
    y_ = y - y.mean(dim=-2, keepdim=True)

  # calculate rotations
  r = kabsch_rotation(x_, y_, mask=mask)

  # apply rotations
  x_ = x_ @ r

  # return centered and aligned
  return x_, y_


def tmscore(
    x: torch.Tensor, y: torch.Tensor, n: Optional[Union[int, torch.Tensor]] = None
):
  """ Assumes x, y are both (..., num_res, 3). """
  n = default(n, x.shape[-2])

  if isinstance(n, torch.Tensor):
    n = torch.clamp(n, min=21)
  else:
    n = max(21, n)
  d0 = 1.24 * (n - 15)**(1.0 / 3.0) - 1.8
  # get distance
  dist = torch.sqrt(torch.sum((x - y)**2, dim=-1))
  # formula (see wrapper for source):
  return torch.nanmean(1.0 / (1.0 + (dist / d0)**2), dim=-1)


def optimal_transform_create(pred_points, true_points, points_mask):
  """Transform ground truth onto prediction such that anchors are aligned"""
  ca_idx = residue_constants.atom_order['CA']

  pred_ca = pred_points[..., ca_idx, :]
  true_ca = true_points[..., ca_idx, :]
  mask_ca = points_mask[..., ca_idx]

  # mask_ca = mask_ca * (seq_color == seq_anchor)

  if torch.any(torch.isnan(pred_points)) or torch.any(torch.isinf(pred_points)):
    pred_points = torch.nan_to_num(pred_points, nan=0.0, posinf=1.0, neginf=1.0)

  if torch.any(mask_ca):
    pred_ca = pred_ca[..., mask_ca, :]
    true_ca = true_ca[mask_ca, :]
  else:
    with torch.no_grad():
      pred_ca = true_ca

  return kabsch_transform(pred_ca, true_ca)

  return R, t


def seq_crop_mask(fgt_seq_index, fgt_seq_color, seq_index, seq_color, seq_anchor):
  fgt_seq_index = fgt_seq_index[fgt_seq_color == seq_anchor]
  seq_index = seq_index[seq_color == seq_anchor]

  return torch.any(
      (rearrange(fgt_seq_index, 'i -> i ()') - rearrange(seq_index, 'j -> () j')) == 0,
      dim=-1
  )


def seq_crop_apply(fgt_coord, fgt_coord_mask, color_mask, crop_mask):
  return fgt_coord[color_mask][crop_mask], fgt_coord_mask[color_mask][crop_mask]


def seq_crop_candidate(fgt_seq_color, fgt_seq_entity, seq_anchor):
  """Find out all seq_colors that having the same seq_entity with `seq_anchor`"""
  seq_entity = torch.unique(fgt_seq_entity[fgt_seq_color == seq_anchor])
  seq_color = torch.unique(fgt_seq_color[fgt_seq_entity == seq_entity])
  return seq_color


def optimal_permutation_find(
    fgt_coord,
    fgt_coord_mask,
    fgt_seq_index,
    fgt_seq_color,
    fgt_seq_entity,
    seq_index,
    seq_color,
    pred_points
):
  used = set()
  ca_idx = residue_constants.atom_order['CA']

  for seq_color_i in filter(lambda x: x > 0, torch.unique(seq_color)):
    seq_color_opt, rmsd_opt = None, None

    crop_mask = seq_crop_mask(
        fgt_seq_index, fgt_seq_color, seq_index, seq_color, seq_color_i
    )
    pred_points_i = pred_points[..., seq_color == seq_color_i, :, :]
    for seq_color_j in seq_crop_candidate(fgt_seq_color, fgt_seq_entity, seq_color_i):
      if int(seq_color_j) not in used:
        true_points_j, points_mask_j = seq_crop_apply(
            fgt_coord, fgt_coord_mask, fgt_seq_color == seq_color_j, crop_mask
        )
        r = rmsd(
            pred_points_i[..., ca_idx, :],
            true_points_j[..., ca_idx, :],
            points_mask_j[..., ca_idx]
        )
        if not exists(rmsd_opt) or r < rmsd_opt:
          seq_color_opt, rmsd_opt = seq_color_j, r

    assert exists(seq_color_opt)
    used.add(int(seq_color_opt))

    if seq_color_i != seq_color_opt:
      yield seq_color_i, seq_color_opt


def multi_chain_permutation_alignment(value, batch):
  """Permute chains with identical sequences such that they are best-effort aligned
     with those of the prediction."""
  if 'coord_fgt' in batch:
    for bdx in range(batch['seq'].shape[0]):
      if batch['seq_anchor'][bdx] > 0:
        coord_opt, coord_mask_opt, rmsd_opt = None, None, None

        crop_mask = seq_crop_mask(
            batch['seq_index_fgt'][bdx],
            batch['seq_color_fgt'][bdx],
            batch['seq_index'][bdx],
            batch['seq_color'][bdx],
            batch['seq_anchor'][bdx]
        )
        for c in seq_crop_candidate(
            batch['seq_color_fgt'][bdx],
            batch['seq_entity_fgt'][bdx],
            batch['seq_anchor'][bdx]
        ):
          true_points, points_mask = seq_crop_apply(
              batch['coord_fgt'][bdx],
              batch['coord_mask_fgt'][bdx],
              batch['seq_color_fgt'][bdx] == c,
              crop_mask
          )
          pred_points = value['coords'][bdx][
              ..., batch['seq_color'][bdx] == batch['seq_anchor'][bdx], :, :
          ]
          T = optimal_transform_create(pred_points, true_points, points_mask)  # pylint: disable=invalid-name

          coord, coord_mask = map(
              torch.clone, (batch['coord'][bdx], batch['coord_mask'][bdx])
          )
          for seq_color_i, seq_color_j in optimal_permutation_find(
              rigids_apply(T, batch['coord_fgt'][bdx]),
              batch['coord_mask_fgt'][bdx],
              batch['seq_index_fgt'][bdx],
              batch['seq_color_fgt'][bdx],
              batch['seq_entity_fgt'][bdx],
              batch['seq_index'][bdx],
              batch['seq_color'][bdx],
              value['coords'][bdx]
          ):
            crop_mask_i = seq_crop_mask(
                batch['seq_index_fgt'][bdx],
                batch['seq_color_fgt'][bdx],
                batch['seq_index'][bdx],
                batch['seq_color'][bdx],
                seq_color_i
            )
            true_points, points_mask = seq_crop_apply(
                batch['coord_fgt'][bdx],
                batch['coord_mask_fgt'][bdx],
                batch['seq_color_fgt'][bdx] == seq_color_j,
                crop_mask_i
            )
            coord[batch['seq_color'][bdx] == seq_color_i, ...] = true_points
            coord_mask[batch['seq_color'][bdx] == seq_color_i, ...] = points_mask

          ca_idx = residue_constants.atom_order['CA']
          r = rmsd(
              value['coords'][bdx, ..., ca_idx, :],
              rigids_apply(T, coord[..., ca_idx, :]),
              coord_mask[..., ca_idx]
          )
          if not exists(rmsd_opt) or r < rmsd_opt:
            coord_opt, coord_mask_opt = coord, coord_mask
            rmsd_opt = r
        assert exists(coord_opt) and exists(coord_mask_opt)
        assert coord_opt.shape == batch['coord'][bdx].shape
        assert coord_mask_opt.shape == batch['coord_mask'][bdx].shape

        # Apply the optimal coordinates
        batch['coord'][bdx], batch['coord_mask'][bdx] = coord_opt, coord_mask_opt
    if torch.any(batch['seq_anchor'] > 0):
      if 'coord_alt' in batch:
        batch.update(
            symmetric_ground_truth_create_alt(
                batch['seq'], batch['coord'], batch['coord_mask']
            )
        )
      if 'backbone_affine' in batch:
        n_idx = residue_constants.atom_order['N']
        ca_idx = residue_constants.atom_order['CA']
        c_idx = residue_constants.atom_order['C']

        batch['backbone_affine'] = rigids_from_3x3(
            batch['coord'], indices=(c_idx, ca_idx, n_idx)
        )

        coord_mask = batch['coord_mask']
        coord_mask = torch.stack(
            (coord_mask[..., c_idx], coord_mask[..., ca_idx], coord_mask[..., n_idx]),
            dim=-1
        )
        batch['backbone_affine_mask'] = torch.all(coord_mask != 0, dim=-1)
      if 'atom_affine' in batch:
        batch.update(
            rigids_from_positions(batch['seq'], batch['coord'], batch['coord_mask'])
        )
      if 'torsion_angles' in batch:
        batch.update(
            angles_from_positions(batch['seq'], batch['coord'], batch['coord_mask'])
        )
  return batch
