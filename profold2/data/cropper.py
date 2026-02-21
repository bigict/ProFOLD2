"""Crop algorithms"""
import logging

import numpy as np
import torch
from torch.nn import functional as F

from profold2.common import residue_constants
from profold2.data.utils import str_seq_index
from profold2.utils import default, env, exists

logger = logging.getLogger(__file__)


def crop(
    protein,
    min_crop_len=None,
    max_crop_len=None,
    min_crop_pae=False,
    max_crop_plddt=False,
    crop_probability=0.0,
    crop_algorithm='random',
    **kwargs
):
  def _crop_length(n, do_crop):
    assert exists(min_crop_len) or exists(max_crop_len)

    if not exists(max_crop_len):
      assert min_crop_len < n
      return np.random.randint(min_crop_len, n + 1) if do_crop else n
    elif not exists(min_crop_len):
      assert max_crop_len < n
      return max_crop_len
    assert min_crop_len <= max_crop_len and (min_crop_len < n or max_crop_len < n)
    return np.random.randint(
        min_crop_len, min(n, max_crop_len) + 1
    ) if do_crop else min(max_crop_len, n)

  def _random_sampler(protein, n):
    l = _crop_length(n, np.random.random() < crop_probability)
    logger.debug(
        'min_crop_len=%s, max_crop_len=%s, n=%s, l=%s', min_crop_len, max_crop_len, n, l
    )
    i, j, w = 0, l, None
    if not 'coord_mask' in protein or torch.any(protein['coord_mask']):
      if (
          min_crop_pae and 'coord_pae' in protein and
          protein['coord_pae'].shape[-1] == n
      ):
        assert protein['coord_pae'].shape[-1] == protein['coord_pae'].shape[-2]
        w = torch.cumsum(torch.cumsum(protein['coord_pae'], dim=-1), dim=-2)
        w = torch.cat(
            (
                w[l - 1:l, l - 1],
                torch.diagonal(
                    w[l:, l:] - w[:n - l, l:] - w[l:, :n - l] + w[:n - l, :n - l],
                    dim1=-2,
                    dim2=-1
                )
            ),
            dim=-1
        ) / (l**2)
        w = 1 / (w + 1e-8)
        w = torch.pow(w, 1.3)
      elif max_crop_plddt and 'coord_plddt' in protein:
        ca_idx = residue_constants.atom_order['CA']
        plddt = protein['coord_plddt'][..., ca_idx]
        w = torch.cumsum(plddt, dim=-1)
        assert len(w.shape) == 1
        w = torch.cat((w[l - 1:l], w[l:] - w[:-l]), dim=-1)  # pylint: disable=invalid-unary-operand-type
        assert w.shape[0] == plddt.shape[-1] - l + 1
        w = torch.pow(w / l, 2.0)
      while True:
        if exists(w):
          i = int(torch.multinomial(w, 1))
        else:
          i = np.random.randint(n - l + 1)
        j = i + l
        if not 'coord_mask' in protein or torch.any(protein['coord_mask'][i:j]):
          break
    return dict(i=i, j=j, d=list(range(i, j)), l=n)

  def _knn_sampler(protein, n):

    assert exists(min_crop_len) or exists(max_crop_len)
    assert 'coord' in protein and 'coord_mask' in protein

    if exists(max_crop_len
             ) and n <= max_crop_len and crop_probability < np.random.random():
      assert not exists(min_crop_len) or min_crop_len < n
      return None

    ca_idx = residue_constants.atom_order['CA']
    ca_coord = protein['coord'][..., ca_idx, :]
    ca_coord_mask = protein['coord_mask'][..., ca_idx]
    logger.debug('knn_sampler: seq_len=%d', n)

    min_len = 32  # default(min_crop_len, 32)
    # max_len = default(max_crop_len, 256)
    max_len = _crop_length(n, np.random.random() < crop_probability)
    gamma = 0.004

    eps = 1e-1
    dist2 = torch.sum(torch.square(ca_coord[:, None, :] - ca_coord[None, :, :]), dim=-1)
    mask = ca_coord_mask[:, None] * ca_coord_mask[None, :]
    dist2 = dist2.masked_fill(~mask, torch.max(dist2))

    spatial_interface_ratio = kwargs.get('crop_spatial_interface_ratio', 0.0)
    if np.random.random() < spatial_interface_ratio:
      cutoff = kwargs.get('crop_spatial_interface_cutoff', 15)
      seq_color = protein['seq_color']
      p = torch.sum(
          (seq_color[:, None] != seq_color[None, :]) * (dist2 < cutoff**2), dim=-1
      ) + 1e-3
      p /= torch.sum(p, dim=-1)
      ridx = np.random.choice(n, p=p.numpy())
    else:
      ridx = np.random.randint(n)

    dist2 = dist2[ridx]
    opt_h = torch.zeros(n + 1, max_len + 1, dtype=torch.float)

    for i in range(1, n + 1):
      for j in range(1, min(i, max_len) + 1):
        opt_h[i, j] = opt_h[i - 1, j - 1] + 1.0 / (dist2[i - 1] + eps)
        if min_len <= j < i:
          opt_v = opt_h[i - min_len - 1, j -
                        min_len] + torch.sum(1 / (dist2[i - min_len:i] + eps)) - gamma
          opt_h[i, j] = max(opt_h[i, j], opt_v)
    # Traceback
    new_order = []
    i, j = n + 1, max_len
    while j > 0:
      _, i = torch.max(opt_h[:i, j], dim=-1)

      # To s.t. len(Ci) >= min_len
      if new_order and i + 1 == new_order[0]:
        window = 1
      else:
        window = min(j, min_len)

      new_order = list(range(max(0, i - window), i)) + new_order
      i, j = i - window + 1, j - window
    cidx = protein['seq_index'][ridx].item()
    logger.debug(
        '_knn_sampler: ridx=%s, cidx=%s, %s', ridx, cidx,
        str_seq_index(torch.as_tensor(new_order))
    )
    return dict(d=new_order, c=cidx, l=n)

  def _auto_sampler(protein, n):
    if (min_crop_pae and 'coord_pae' in protein) or (
        max_crop_plddt and 'coord_plddt' in protein and
        torch.any(protein['coord_plddt'] < 1.0)
    ) or n > env('profold2_data_knn_sampler_max_length', defval=65536, dtype=int):
      return _random_sampler(protein, n)
    return _knn_sampler(protein, n)

  logger.debug('protein_clips_fn: crop_algorithm=%s', crop_algorithm)
  sampler_list = dict(auto=_auto_sampler, knn=_knn_sampler, random=_random_sampler)

  assert crop_algorithm in sampler_list

  n = len(protein['str_seq'])
  if (exists(max_crop_len) and max_crop_len < n
     ) or (exists(min_crop_len) and min_crop_len < n and crop_probability > 0):
    sampler_fn = sampler_list[crop_algorithm]
    if crop_algorithm != 'random' and (
        'coord' not in protein or 'coord_mask' not in protein
    ):
      sampler_fn = sampler_list['random']
      logger.debug(
          'protein_clips_fn: crop_algorithm=%s downgrad to: random', crop_algorithm
      )
    return sampler_fn(protein, n)

  return None


def apply(protein, new_order, seq_feats=None, msa_feats=None, var_feats=None):
  # Update seq related feats
  protein['str_seq'] = ''.join(protein['str_seq'][k] for k in new_order)

  for field in ('str_msa', 'str_var'):
    if field in protein:
      for j in range(len(protein[field])):
        protein[field][j] = ''.join(protein[field][j][k] for k in new_order)

  # Update tensors
  new_order = torch.as_tensor(new_order)

  for field in default(
      seq_feats, (
          'seq', 'seq_index', 'seq_color', 'seq_entity', 'seq_sym', 'mask',
          'coord', 'coord_mask', 'coord_plddt', 'sta_type_mask'
      )
  ):
    if field in protein:
      protein[field] = torch.index_select(protein[field], 0, new_order)
  for field in ('coord_pae', ):
    if field in protein:
      protein[field] = torch.index_select(protein[field], 0, new_order)
      protein[field] = torch.index_select(protein[field], 1, new_order)
  for field in ('sta', ):
    if field in protein:
      l = protein[field].shape[0]
      protein[field] = F.one_hot(protein[field].long(), l + 1)  # shape: i c j
      protein[field] = torch.index_select(protein[field], 0, new_order)
      protein[field] = torch.index_select(
          protein[field], 2, torch.cat((torch.as_tensor([0]), new_order + 1), dim=0)
      )
      protein[field] = torch.argmax(protein[field], dim=-1)  # shape: i c
  for field in default(msa_feats, ('msa', 'msa_mask', 'del_msa')):
    if field in protein:
      protein[field] = torch.index_select(protein[field], 1, new_order)
  for field in default(
      var_feats, ('variant', 'del_var', 'variant_mask', 'variant_task_mask')
  ):
    if field in protein:
      protein[field] = torch.index_select(protein[field], 1, new_order)

  return protein
