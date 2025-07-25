"""Utils from data module
 """
import re

import numpy as np
import torch

from profold2.common import protein, residue_constants
from profold2.utils import exists


def decompose_pid(pid, return_domain=False):
  k = pid.find('/')
  if k != -1:
    pid, domains = pid[:k], pid[k + 1:]
  else:
    domains = None

  pid, *_ = pid.split('|')  # HACK: pid|attr1|attr2/2-256

  k = pid.rfind('_')
  if k != -1:
    pid, chain = pid[:k], pid[k + 1:]
  else:
    chain = None

  if return_domain:
    return pid, chain, domains
  return pid, chain


def compose_pid(pid, chain, domains=None):
  if exists(chain):
    pid = f'{pid}_{chain}'
  if exists(domains):
    pid = f'{pid}/{domains}'
  return pid


_seq_index_pattern = '(\\d+)-(\\d+)'


def seq_index_split(text):
  for s in text.split(','):
    r = re.match(_seq_index_pattern, s)
    assert r
    yield tuple(map(int, r.group(1, 2)))


def seq_index_join(seq_index):
  return ','.join(f'{i}-{j}' for i, j in seq_index)


def str_seq_index(seq_index):
  assert exists(seq_index) and seq_index.shape[0] > 0
  domains = []
  s, e = None, None
  for i in range(seq_index.shape[0]):
    if seq_index[i] - 1 != e:
      if exists(s) and exists(e):
        domains += [(s, e)]
      s = seq_index[i]
    e = seq_index[i]
  if exists(s) and exists(e):
    domains += [(s, e)]
  return seq_index_join(domains)


def yield_seq_index(description):
  fields = description.split()
  for f in fields[1:]:
    r = re.match(f'.*:({_seq_index_pattern}(,{_seq_index_pattern})*)', f)
    if r:
      for p, q in seq_index_split(r.group(1)):
        yield p, q
      break


def parse_seq_index(description, input_sequence, seq_index):
  # description: pid field1 field2 ...
  def seq_index_check(positions):
    for i in range(len(positions) - 1):
      p, q = positions[i]
      m, n = positions[i + 1]
      assert p <= q and m <= n, (p, q, m, n, description)
      assert q <= m
    m, n = positions[-1]
    assert m <= n
    assert sum(map(lambda p: p[1] - p[0] + 1, positions)) == len(input_sequence)

  positions = list(yield_seq_index(description))
  if positions:
    seq_index_check(positions)
    p, q = positions[0]
    start, gap = p, 0
    for m, n in positions[1:]:
      gap += m - q - 1
      seq_index[m - start - gap:n - start - gap +
                1] = torch.arange(m - start, n - start + 1, dtype=seq_index.dtype)
      p, q = m, n

  return seq_index


def weights_from_file(filename_list):
  if filename_list:
    for filename in filename_list.split(','):
      with open(filename, 'r', encoding='utf-8') as f:
        for line in filter(
            lambda x: len(x) > 0 and not x.startswith('#'), map(lambda x: x.strip(), f)
        ):
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


def tensor_to_numpy(t):
  if t.dtype in (torch.float16, torch.bfloat16):
    t = t.float()
  return t.detach().cpu().numpy()


def pdb_from_prediction(batch, headers, idx=None):
  def to_pdb_str(b):
    str_seq = batch['str_seq'][b]
    seq_len = len(str_seq)
    #aatype = batch['seq'][b,...].numpy()
    aatype = np.array(
        [
            residue_constants.restype_order_with_x.get(
                aa, residue_constants.unk_restype_index
            ) for aa in str_seq
        ]
    )
    if 'seq_index' in batch and exists(batch['seq_index'][b]):
      seq_index = tensor_to_numpy(batch['seq_index'][b])
    else:
      seq_index = np.arange(seq_len)
    features = {
        'aatype': aatype,
        'residue_index': seq_index,
    }
    if 'seq_color' in batch:
      features['seq_color'] = tensor_to_numpy(batch['seq_color'][b] - 1)

    coords = tensor_to_numpy(headers['folding']['coords'][b, ...])  # (b l c d)
    restype_atom14_mask = np.copy(residue_constants.restype_atom14_mask)
    if 'coord_exists' in batch:
      coord_mask = tensor_to_numpy(batch['coord_exists'][b, ...])
    else:
      coord_mask = np.asarray([restype_atom14_mask[restype] for restype in aatype])
    b_factors = None
    if 'confidence' in headers and 'plddt' in headers['confidence']:
      plddt = tensor_to_numpy(headers['confidence']['plddt'][b, ...])  # (b l)
      b_factors = coord_mask * plddt[..., None]

    result = dict(
        structure_module=dict(final_atom_mask=coord_mask, final_atom_positions=coords)
    )
    prot = protein.from_prediction(
        features=features, result=result, b_factors=b_factors
    )
    return protein.to_pdb(prot)

  if exists(idx):
    return to_pdb_str(idx)
  return [to_pdb_str(i) for i in range(len(batch['pid']))]
