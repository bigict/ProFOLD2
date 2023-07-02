"""Utils from data module
 """
import logging
import os

import numpy as np
import torch
from einops import rearrange

from profold2.common import protein, residue_constants
from profold2.utils import exists


def domain_parser(ca_coord,
                  ca_mask,
                  max_len=255,
                  alpha=0.43,
                  cutoff=8.0,
                  epsilon=1e-5):
  n = ca_mask.shape[0]
  ca_dist = torch.cdist(ca_coord, ca_coord) < cutoff
  ca_dist_mask = rearrange(ca_mask, 'n -> () n') * rearrange(
      ca_mask, 'n -> n ()')

  positions, weights = [], []
  for i in range(1, n):
    p, q = max(0, i - max_len), min(n, i + max_len)
    ll, lr = torch.sum(ca_mask[p:i, ...]), torch.sum(ca_mask[i:q, ...])
    if ll > 0 and lr > 0:
      positions.append(i)
      weights.append(
          torch.sum(ca_dist[i:q, p:i, ...] * ca_dist_mask[i:q, p:i, ...]) /
          ((ll * lr)**alpha))
  assert len(positions) == len(weights)

  p = torch.full((n,), (min(weights) + epsilon) if weights else 1.0)
  for i, j in enumerate(positions):
    p[j] = weights[i] + epsilon

  p = 1.0 / (p * torch.sum(1.0 / p) + epsilon)
  return p


def batch_data_crop(batch, max_seq_len=None):
  # preprocess
  str_seqs = list(batch['str_seq'])
  int_seqs = batch['seq']

  b, n = int_seqs.shape
  assert b == len(str_seqs)
  if exists(max_seq_len) and n > max_seq_len:
    assert max_seq_len > 0

    clips = {}

    mask = batch['mask']
    coords = batch['coord']
    coord_mask = batch['coord_mask']

    for k in range(b):
      if len(str_seqs[k]) <= max_seq_len:
        continue
      i, j = 0, max_seq_len
      if torch.any(coord_mask[k, ...]):
        while True:
          i = np.random.randint(n - max_seq_len)
          j = i + max_seq_len
          if torch.any(coord_mask[k, i:j, ...]):
            break
      clips[k] = dict(i=i, j=j, l=len(str_seqs[k]))
      str_seqs[k] = str_seqs[k][i:j]
      int_seqs[k, :j - i] = torch.clone(int_seqs[k, i:j])
      mask[k, :j - i] = torch.clone(mask[k, i:j])
      coords[k, :j - i, ...] = torch.clone(coords[k, i:j, ...])
      coord_mask[k, :j - i, ...] = torch.clone(coord_mask[k, i:j, ...])

    int_seqs = int_seqs[:, :max_seq_len]
    mask = mask[:, :max_seq_len]
    coords = coords[:, :max_seq_len, ...]
    coord_mask = coord_mask[:, :max_seq_len, ...]

    batch.update(seq=int_seqs,
                 mask=mask,
                 str_seq=str_seqs,
                 coord=coords,
                 coord_mask=coord_mask,
                 clips=clips)
  return batch


def cycling(loader, cond=lambda x: True):
  epoch = 0
  while True:
    logging.info('epoch: %d', epoch)

    data_iter = iter(loader)
    for data in data_iter:
      if cond(data):
        yield epoch, data

    epoch += 1


def weights_from_file(filename_list):
  if filename_list:
    for filename in filename_list.split(','):
      with open(filename, 'r', encoding='utf-8') as f:
        for line in filter(lambda x: len(x) > 0 and not x.startswith('#'),
                           map(lambda x: x.strip(), f)):
          items = line.split()
          yield float(items[0])


def embedding_get_labels(name, mat):
  if name == 'token':
    return [
        residue_constants.
        restypes_with_x[i if i < len(residue_constants.restypes_with_x) else -1]
        for i in range(mat.shape[0])
    ]
  return None


def filter_from_file(filename):
  if filename:
    with open(filename, 'r', encoding='utf-8') as f:
      for line in filter(lambda x: len(x) > 0, map(lambda x: x.strip(), f)):
        yield line


def pdb_from_prediction(batch, headers, idx=None):

  def to_pdb_str(b):
    str_seq = batch['str_seq'][b]
    seq_len = len(str_seq)
    #aatype = batch['seq'][b,...].numpy()
    aatype = np.array([
        residue_constants.restype_order_with_x.get(
            aa, residue_constants.unk_restype_index) for aa in str_seq
    ])
    features = dict(aatype=aatype, residue_index=np.arange(seq_len))

    coords = headers['folding']['coords'].detach().cpu()  # (b l c d)
    restype_atom14_mask = np.copy(residue_constants.restype_atom14_mask)
    # includes = ['N', 'CA', 'C', 'CB']
    includes = []
    if includes:
      includes = set(includes)
      for i in range(residue_constants.restype_num):
        resname = residue_constants.restype_1to3[residue_constants.restypes[i]]
        atom_list = residue_constants.restype_name_to_atom14_names[resname]
        for j in range(restype_atom14_mask.shape[1]):
          if restype_atom14_mask[i, j] > 0 and atom_list[j] not in includes:
            restype_atom14_mask[i, j] = 0
    coord_mask = np.asarray(
        [restype_atom14_mask[restype] for restype in aatype])
    b_factors = None
    if 'confidence' in headers and 'plddt' in headers['confidence']:
      plddt = headers['confidence']['plddt'].detach().cpu().numpy()  # (b l)
      b_factors = coord_mask * plddt[b][..., None]

    result = dict(structure_module=dict(
        final_atom_mask=coord_mask, final_atom_positions=coords[b,
                                                                ...].numpy()))
    prot = protein.from_prediction(features=features,
                                   result=result,
                                   b_factors=b_factors)
    return protein.to_pdb(prot)

  if exists(idx):
    return to_pdb_str(idx)
  return [to_pdb_str(i) for i in range(len(batch['pid']))]


def pdb_save(batch, headers, prefix='/tmp', step=None):
  for idx, pid in enumerate(batch['pid']):
    pdb_str = pdb_from_prediction(batch, headers, idx=idx)

    if exists(step):
      p = os.path.join(prefix, f'{pid}_{step}_{idx}.pdb')
    else:
      p = os.path.join(prefix, f'{pid}.pdb')
    with open(p, 'w') as f:
      f.write(pdb_str)

      str_seq = batch['str_seq'][idx]
      if 'mask' in batch:
        masked_seq_len = torch.sum(batch['mask'][idx, ...], dim=-1)
      else:
        masked_seq_len = len(str_seq)
      logging.debug('step: %s/%s, length: %s/%s, PDB save: %s',
          step, idx, masked_seq_len, len(str_seq), pid)
