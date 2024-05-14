"""Dataset for structure
 """
import os
from collections import defaultdict
import contextlib
import functools
import json
import logging
from io import BytesIO
import pathlib
import string
import zipfile

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from einops import rearrange, repeat

from profold2.common import residue_constants
from profold2.data.parsers import parse_fasta
from profold2.data.utils import (compose_pid, decompose_pid, domain_parser,
                                 parse_seq_index, seq_index_join,
                                 seq_index_split, str_seq_index)
from profold2.utils import default, exists, timing

logger = logging.getLogger(__file__)

FEAT_PDB = 0x01
FEAT_MSA = 0x02
FEAT_ALL = 0xff


def _make_msa_features(sequences, msa_idx=0, max_msa_depth=None):
  """Constructs a feature dict of MSA features."""

  def parse_a4m(sequences):
    deletion_matrix = []
    for msa_sequence in sequences:
      deletion_vec = []
      deletion_count = 0
      for j in msa_sequence:
        if j.islower():
          deletion_count += 1
        else:
          deletion_vec.append(deletion_count)
          deletion_count = 0
      deletion_matrix.append(deletion_vec)
    # Make the MSA matrix out of aligned (deletion-free) sequences
    deletion_table = str.maketrans('', '', string.ascii_lowercase)
    aligned_sequences = [s.translate(deletion_table) for s in sequences]
    return aligned_sequences, deletion_matrix

  msa_depth = len(sequences)
  if 0 < msa_idx < msa_depth:
    t = sequences[msa_idx]
    sequences[msa_idx] = sequences[1]
    sequences[1] = t
  if exists(max_msa_depth) and len(sequences) > max_msa_depth:
    n = 2 if 0 < msa_idx < msa_depth else 1
    sequences = sequences[:n] + list(
        np.random.choice(sequences[n:], size=max_msa_depth -
                         n, replace=False) if max_msa_depth > n else [])
  msa, del_matirx = parse_a4m(sequences)

  int_msa = []
  for sequence in msa:
    int_msa.append([
        residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE[
            residue_constants.HHBLITS_AA_TO_ID[res]] for res in sequence
    ])

  return dict(msa=torch.as_tensor(int_msa, dtype=torch.int),
              str_msa=msa,
              msa_row_mask=torch.ones(len(int_msa), dtype=torch.bool),
              del_msa=torch.as_tensor(del_matirx, dtype=torch.int),
              num_msa=msa_depth)


def _make_seq_features(sequence, description, seq_color=1, max_seq_len=None):
  residue_index = torch.arange(len(sequence), dtype=torch.int)
  residue_index = parse_seq_index(description, sequence, residue_index)

  sequence = sequence[:max_seq_len]
  residue_index = residue_index[:max_seq_len]
  seq_color = torch.full((len(sequence),), seq_color, dtype=torch.int)

  seq = torch.as_tensor(residue_constants.sequence_to_onehot(
      sequence=sequence,
      mapping=residue_constants.restype_order_with_x,
      map_unknown_to_x=True),
                        dtype=torch.int).argmax(-1).to(torch.int)
  #residue_index = torch.arange(len(sequence), dtype=torch.int)
  str_seq = ''.join(
      map(
          lambda a: a if a in residue_constants.restype_order_with_x else
          residue_constants.restypes_with_x[-1], sequence))
  mask = torch.ones(len(sequence), dtype=torch.bool)

  return dict(seq=seq,
              seq_index=residue_index,
              seq_color=seq_color,
              str_seq=str_seq,
              mask=mask)


def _make_feats_shrinked(item, new_order, seq_feats=None, msa_feats=None):
  # Update seq related feats
  item['str_seq'] = ''.join(item['str_seq'][k] for k in new_order)

  for field in ('str_msa',):
    if field in item:
      for j in range(len(item['str_msa'])):
        item['str_msa'][j] = ''.join(item['str_msa'][j][k] for k in new_order)

  # Update tensors
  new_order = torch.as_tensor(new_order)

  for field in default(seq_feats, ('coord', 'coord_mask', 'coord_plddt')):
    if field in item:
      item[field] = torch.index_select(item[field], 0, new_order)
  for field in ('coord_pae',):
    if field in item:
      item[field] = torch.index_select(item[field], 0, new_order)
      item[field] = torch.index_select(item[field], 1, new_order)
  for field in default(msa_feats, ('msa', 'del_msa')):
    if field in item:
      item[field] = torch.index_select(item[field], 1, new_order)

  return item


def _protein_clips_fn(protein,
                      min_crop_len=None,
                      max_crop_len=None,
                      min_crop_pae=False,
                      max_crop_plddt=False,
                      crop_probability=0.0,
                      crop_algorithm='random',
                      **kwargs):

  def _crop_length(n, crop):
    assert exists(min_crop_len) or exists(max_crop_len)

    if not exists(max_crop_len):
      assert min_crop_len < n
      return np.random.randint(min_crop_len, n + 1) if crop else n
    elif not exists(min_crop_len):
      assert max_crop_len < n
      return max_crop_len
    assert min_crop_len <= max_crop_len and (min_crop_len < n or
                                             max_crop_len < n)
    return np.random.randint(min_crop_len,
                             min(n, max_crop_len) +
                             1) if crop else min(max_crop_len, n)

  def _random_sampler(protein, n):
    l = _crop_length(n, np.random.random() < crop_probability)
    logger.debug('min_crop_len=%s, max_crop_len=%s, n=%s, l=%s', min_crop_len,
                 max_crop_len, n, l)
    i, j, w = 0, l, None
    if not 'coord_mask' in protein or torch.any(protein['coord_mask']):
      if (min_crop_pae and 'coord_pae' in protein and
          protein['coord_pae'].shape[-1] == n):
        assert protein['coord_pae'].shape[-1] == protein['coord_pae'].shape[-2]
        w = torch.cumsum(torch.cumsum(protein['coord_pae'], dim=-1), dim=-2)
        w = torch.cat(
            (w[l - 1:l, l - 1],
             torch.diagonal(
                 w[l:, l:] - w[:n - l, l:] - w[l:, :n - l] + w[:n - l, :n - l],
                 dim1=-2,
                 dim2=-1)),
            dim=-1) / (l**2)
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

  def _domain_sampler(protein, n):

    def _cascade_sampler(weights, width=4):
      if len(weights) <= width:
        i = torch.multinomial(weights, 1)
        return i.item()

      p = torch.zeros((width,))
      l, k = len(weights) // width, len(weights) % width
      for i in range(width):
        v, w = l * i + min(i, k), l * (i + 1) + min(i + 1, k)
        p[i] = torch.amax(weights[v:w])
      i = _cascade_sampler(p, width=width)
      v, w = l * i + min(i, k), l * (i + 1) + min(i + 1, k)
      return v + _cascade_sampler(weights[v:w], width=width)

    def _domain_next(weights, i, min_len=None, max_len=None):
      min_len, max_len = default(min_len, 80), default(max_len, 255)

      direction = np.random.randint(2)
      for _ in range(2):
        #if direction % 2 == 0 and torch.sum(ca_mask[i:]) >= min_len:
        if direction % 2 == 0 and i + min_len < n:
          j = min(len(weights), i + max_len)
          return j if i + min_len >= j else i + min_len + _cascade_sampler(
              weights[i + min_len:j])
        #if direction % 2 == 1 and torch.sum(ca_mask[:i]) >= min_len:
        if direction % 2 == 1 and i > min_len:
          j = max(0, i - max_len)
          return j + _cascade_sampler(
              weights[j:i - min_len]) if i - min_len > j else j
        direction += 1
      return None

    assert exists(min_crop_len) or exists(max_crop_len)
    assert 'coord' in protein and 'coord_mask' in protein

    if exists(max_crop_len
             ) and n <= max_crop_len and crop_probability < np.random.random():
      assert not exists(min_crop_len) or min_crop_len < n
      return None

    ca_idx = residue_constants.atom_order['CA']
    ca_coord, ca_coord_mask = protein['coord'][
        ..., ca_idx, :], protein['coord_mask'][..., ca_idx]
    logger.debug('domain_sampler: seq_len=%d', n)
    weights = domain_parser(ca_coord, ca_coord_mask, max_len=max_crop_len)
    intra_domain_probability = kwargs.get('intra_domain_probability', 0)
    while True:
      i = _cascade_sampler(weights)
      if np.random.random() < intra_domain_probability:
        logger.debug('domain_intra: seq_len=%d, i=%d', n, i)
        half = max_crop_len // 2 + max_crop_len % 2
        if i + 1 < n - i:  # i <= n // 2
          i = max(0, i - half)
          j = min(i + max_crop_len, n)
        else:
          i = min(n, i + half)
          j = max(0, i - max_crop_len)
      else:
        j = _domain_next(weights, i, min_len=min_crop_len, max_len=max_crop_len)
        logger.debug('domain_next: seq_len=%d, i=%d, j=%s', n, i, str(j))
      if j is not None and torch.any(ca_coord_mask[min(i, j):max(i, j)]):
        break
    return dict(i=min(i, j), j=max(i, j), d=list(range(i, j)), l=n)

  def _knn_sampler(protein, n):

    assert exists(min_crop_len) or exists(max_crop_len)
    assert 'coord' in protein and 'coord_mask' in protein

    if exists(max_crop_len
             ) and n <= max_crop_len and crop_probability < np.random.random():
      assert not exists(min_crop_len) or min_crop_len < n
      return None

    ca_idx = residue_constants.atom_order['CA']
    ca_coord, ca_coord_mask = protein['coord'][
        ..., ca_idx, :], protein['coord_mask'][..., ca_idx]
    logger.debug('knn_sampler: seq_len=%d', n)

    min_len = 32  # default(min_crop_len, 32)
    # max_len = default(max_crop_len, 256)
    max_len = _crop_length(n, np.random.random() < crop_probability)
    gamma = 0.004

    ridx = np.random.randint(n)
    eps = 1e-1
    dist2 = torch.sum(torch.square(
        rearrange(ca_coord, 'i d -> i () d') -
        rearrange(ca_coord, 'j d -> () j d')),
                      dim=-1)
    mask = rearrange(ca_coord_mask, 'i -> i ()') * rearrange(
        ca_coord_mask, 'j -> () j')
    dist2 = dist2.masked_fill(~mask, torch.max(dist2))
    dist2 = dist2[ridx]
    opt_h = torch.zeros(n + 1, max_len + 1, dtype=torch.float)

    for i in range(1, n + 1):
      for j in range(1, min(i, max_len) + 1):
        opt_h[i, j] = opt_h[i - 1, j - 1] + 1.0 / (dist2[i - 1] + eps)
        if min_len <= j < i:
          opt_v = opt_h[i - min_len - 1, j - min_len] + torch.sum(
              1 / (dist2[i - min_len:i] + eps)) - gamma
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
        window = min_len

      new_order = list(range(max(0, i - window), i)) + new_order
      i, j = i - window + 1, j - window
    cidx = protein['seq_index'][ridx].item()
    logger.debug('_knn_sampler: ridx=%s, cidx=%s, %s', ridx, cidx,
                 str_seq_index(torch.as_tensor(new_order)))
    return dict(d=new_order, c=cidx, l=n)


  def _auto_sampler(protein, n):
    if ((min_crop_pae and 'coord_pae' in protein) or
        (max_crop_plddt and 'coord_plddt' in protein and
         torch.any(protein['coord_plddt'] < 1.0))):
      return _random_sampler(protein, n)
    return _knn_sampler(protein, n)


  logger.debug('protein_clips_fn: crop_algorithm=%s', crop_algorithm)
  sampler_list = dict(random=_random_sampler,
                      auto=_auto_sampler,
                      domain=_domain_sampler,
                      knn=_knn_sampler)

  assert crop_algorithm in sampler_list

  n = len(protein['str_seq'])
  if (exists(max_crop_len) and
      max_crop_len < n) or (exists(min_crop_len) and min_crop_len < n and
                            crop_probability > 0):
    sampler_fn = sampler_list[crop_algorithm]
    if crop_algorithm != 'random' and ('coord' not in protein or
                                       'coord_mask' not in protein):
      sampler_fn = sampler_list['random']
      logger.debug('protein_clips_fn: crop_algorithm=%s downgrad to: random',
                   crop_algorithm)
    return sampler_fn(protein, n)

  return None


def _protein_crop_fn(protein, clip):
  assert clip

  if 'd' in clip:
    return _make_feats_shrinked(protein,
                                clip['d'],
                                seq_feats=('seq', 'seq_index', 'mask',
                                           'coord', 'coord_mask',
                                           'coord_plddt', 'seq_color'))

  i, j = clip['i'], clip['j']
  protein['str_seq'] = protein['str_seq'][i:j]
  for field in ('seq', 'seq_index', 'seq_color', 'mask', 'coord', 'coord_mask',
                'coord_plddt'):
    if field in protein:
      protein[field] = protein[field][i:j, ...]
  for field in ('coord_pae',):
    if field in protein:
      protein[field] = protein[field][i:j, i:j]
  for field in ('str_msa',):
    if field in protein:
      protein[field] = [v[i:j] for v in protein[field]]
  for field in ('msa', 'del_msa'):
    if field in protein:
      protein[field] = protein[field][:, i:j, ...]

  return protein


def _protein_crop_fmt(clip):
  assert exists(clip), clip
  if 'd' in clip:
    clip['d'] = str_seq_index(torch.as_tensor(clip['d']))
  return clip

class ProteinSequenceDataset(torch.utils.data.Dataset):
  """Construct a `Dataset` from sequences
   """

  def __init__(self,
               sequences,
               descriptions=None,
               domain_as_seq=False,
               msa=None,
               msa_as_seq=False):
    self.sequences = sequences
    self.domain_as_seq = domain_as_seq
    self.descriptions = descriptions
    self.msa = msa
    assert not exists(self.descriptions) or len(self.sequences) == len(
        self.descriptions)
    assert not exists(self.msa) or len(self.sequences) == len(self.msa)
    assert (not msa_as_seq) or exists(msa)
    self.msa_depth = np.cumsum(np.asarray([len(m) for m in self.msa
                                          ])) if msa_as_seq else None

  def __getitem__(self, idx):
    seq_idx, msa_idx = idx, 0
    if exists(self.msa_depth):
      seq_idx = np.sum(self.msa_depth < idx)
      msa_idx = idx - (self.msa_depth[seq_idx - 1] if seq_idx > 0 else 0)  # pylint: disable=unsubscriptable-object
    input_sequence = self.sequences[seq_idx]
    seq = torch.as_tensor(residue_constants.sequence_to_onehot(
        sequence=input_sequence,
        mapping=residue_constants.restype_order_with_x,
        map_unknown_to_x=True),
                          dtype=torch.int).argmax(-1).to(torch.int)
    residue_index = torch.arange(len(input_sequence), dtype=torch.int)
    str_seq = ''.join(
        map(
            lambda a: a if a in residue_constants.restype_order_with_x else
            residue_constants.restypes_with_x[-1], input_sequence))
    mask = torch.ones(len(input_sequence), dtype=torch.bool)
    if exists(self.descriptions) and exists(self.descriptions[seq_idx]):
      desc = self.descriptions[seq_idx]
      residue_index = parse_seq_index(desc, input_sequence, residue_index)
      desc = desc.split()[0]
    else:
      desc = str(seq_idx)
    seq_color = torch.ones(len(input_sequence), dtype=torch.int)
    if self.domain_as_seq:
      seq_color += torch.cat(
          (torch.zeros(1, dtype=torch.int),
           torch.cumsum(residue_index[:-1] + 1 != residue_index[1:], dim=-1)))
    ret = dict(pid=desc,
               seq=seq,
               seq_index=residue_index,
               seq_color=seq_color,
               str_seq=str_seq,
               mask=mask)
    if exists(self.msa) and exists(self.msa[seq_idx]):
      ret.update(_make_msa_features(self.msa[seq_idx], msa_idx=msa_idx))
    if msa_idx > 0:
      ret = self.msa_as_seq(ret, msa_idx)
    return ret

  def __len__(self):
    if exists(self.msa_depth):
      return int(self.msa_depth[-1])  # pylint: disable=unsubscriptable-object
    return len(self.sequences)

  def msa_as_seq(self, item, idx):
    assert idx > 0
    assert 'str_msa' in item and 'del_msa' in item
    assert len(item['str_seq']) == len(item['str_msa'][0]), (idx, item['pid'],
                                                             item['str_seq'],
                                                             item['str_msa'][0])

    # swap(msa[1], msa[idx])
    assert len(item['str_msa'][0]) == len(item['str_msa'][1])
    if item['str_seq'] != item['str_msa'][1]:
      assert item['str_seq'] != item['str_msa'][1], (item['pid'], idx,
                                                     item['str_msa'][0],
                                                     item['str_msa'][1])
      item['str_msa'][0] = item['str_msa'][1]
      item['str_msa'][1] = item['str_seq']
      assert item['str_seq'] != item['str_msa'][0], (item['pid'], idx,
                                                     item['str_msa'][0],
                                                     item['str_msa'][1])
      item['str_seq'] = item['str_msa'][0]
      assert item['str_seq'] == item['str_msa'][0]

      i, new_order = 0, []

      while i < len(item['str_seq']):
        if item['str_seq'][i] != residue_constants.restypes_with_x_and_gap[-1]:
          new_order.append(i)
        i += 1
      logger.debug('msa_as_seq: %s@%s k=%d, i=%d', item['pid'], idx,
                   len(new_order), i)
      assert 0 <= len(new_order) <= i, (len(new_order), i)

      if len(new_order) < i:
        item = _make_feats_shrinked(item, new_order)

      # Renew seq related feats
      pid = item['pid']
      item['pid'] = f'{pid}@{idx}'
      item.update(
          _make_seq_features(item['str_seq'],
                             item['pid'],
                             seq_color=item['seq_color'][0],
                             max_seq_len=None))

      # Fix seq_index
      del_seq = torch.cumsum(item['del_msa'][0],
                             dim=-1,
                             dtype=item['del_msa'].dtype)
      item['seq_index'] = item['seq_index'] + del_seq

    return item

  @staticmethod
  def collate_fn(batch):
    fields = ('pid', 'seq', 'seq_index', 'seq_color', 'mask', 'str_seq')
    pids, seqs, seqs_idx, seqs_clr, masks, str_seqs = list(
        zip(*[[b[k] for k in fields] for b in batch]))
    lengths = tuple(len(s) for s in str_seqs)
    max_batch_len = max(lengths)

    padded_seqs = pad_for_batch(seqs, max_batch_len, 'seq')
    padded_seqs_idx = pad_for_batch(seqs_idx, max_batch_len, 'seq_index')
    padded_seqs_clr = pad_for_batch(seqs_clr, max_batch_len, 'seq_index')
    padded_masks = pad_for_batch(masks, max_batch_len, 'msk')

    ret = dict(pid=pids,
               seq=padded_seqs,
               seq_index=padded_seqs_idx,
               seq_color=padded_seqs_clr,
               mask=padded_masks,
               str_seq=str_seqs)

    fields = ('msa', 'msa_row_mask', 'str_msa', 'del_msa', 'num_msa')
    if all(all(field in b for field in fields) for b in batch):
      msas, msa_row_msks, str_msas, del_msas, num_msa = list(
          zip(*[[b[k] for k in fields] for b in batch]))

      msas = pad_for_batch(msas, max_batch_len, 'msa')
      msa_row_msks = pad_for_batch(msa_row_msks, max_batch_len, 'msa_row_mask')
      del_msas = pad_for_batch(del_msas, max_batch_len, 'del_msa')
      ret.update(msa=msas,
                 msa_row_mask=msa_row_msks,
                 str_msa=str_msas,
                 del_msa=del_msas,
                 num_msa=torch.as_tensor(num_msa))

    return ret


class ProteinStructureDataset(torch.utils.data.Dataset):
  """Construct a `Dataset` from a zip or filesystem
   """

  def __init__(self,
               data_dir,
               data_idx=None,
               data_crop_fn=None,
               max_msa_depth=128,
               pseudo_linker_prob=0.0,
               pseudo_linker_shuffle=True,
               data_rm_mask_prob=0.0,
               msa_as_seq_prob=0.0,
               msa_as_seq_topn=None,
               msa_as_seq_clustering=False,
               msa_as_seq_min_alr=0.75,
               msa_as_seq_min_ident=0.0,
               feat_flags=FEAT_ALL & (~FEAT_MSA)):
    super().__init__()

    self.data_dir = pathlib.Path(data_dir)
    data_idx = default(data_idx, 'name.idx')
    if zipfile.is_zipfile(self.data_dir):
      self.data_dir = zipfile.ZipFile(self.data_dir)  # pylint: disable=consider-using-with
    self.data_crop_fn = data_crop_fn
    self.max_msa_depth = max_msa_depth
    self.pseudo_linker_prob = pseudo_linker_prob
    self.pseudo_linker_shuffle = pseudo_linker_shuffle
    self.data_rm_mask_prob = data_rm_mask_prob
    self.msa_as_seq_prob = msa_as_seq_prob
    self.msa_as_seq_topn = msa_as_seq_topn
    self.msa_as_seq_clustering = msa_as_seq_clustering
    self.msa_as_seq_min_alr = msa_as_seq_min_alr
    self.msa_as_seq_min_ident = msa_as_seq_min_ident
    self.feat_flags = feat_flags
    logger.info('load idx data from: %s', data_idx)
    with self._fileobj(data_idx) as f:
      self.pids = list(
          map(
              lambda x: x.split(),
              filter(lambda x: len(x) > 0 and not x.startswith('#'),
                     map(lambda x: self._ftext(x).strip(), f))))

    self.mapping = {}
    if self._fstat('mapping.idx'):
      with self._fileobj('mapping.idx') as f:
        for line in filter(lambda x: len(x) > 0,
                           map(lambda x: self._ftext(x).strip(), f)):
          v, k = line.split()
          self.mapping[k] = v

    self.resolu = {}
    if self._fstat('resolu.idx'):
      with self._fileobj('resolu.idx') as f:
        for line in filter(lambda x: len(x) > 0,
                           map(lambda x: self._ftext(x).strip(), f)):
          k, v = line.split()
          self.resolu[k] = float(v)

    self.chain_list = {}
    if self._fstat('chain.idx'):
      with self._fileobj('chain.idx') as f:
        for line in filter(lambda x: len(x) > 0,
                           map(lambda x: self._ftext(x).strip(), f)):
          chains = line.split()
          if chains[0] in self.chain_list:
            self.chain_list[chains[0]].append(chains[1:])
          else:
            self.chain_list[chains[0]] = [chains[1:]]

    self.fasta_dir = 'fasta'
    self.pdb_dir = 'npz'

    self.msa_list = ['BFD30_E-3']

  def __getstate__(self):
    d = self.__dict__
    if isinstance(self.data_dir, zipfile.ZipFile):
      d['data_dir'] = self.data_dir.filename
    logger.debug('%s is pickled ...', d['data_dir'])
    return d

  def __setstate__(self, d):
    logger.debug('%s is unpickled ...', d['data_dir'])
    if zipfile.is_zipfile(d['data_dir']):
      d['data_dir'] = zipfile.ZipFile(d['data_dir'])  # pylint: disable=consider-using-with
    self.__dict__ = d

  def __getitem__(self, idx):
    with timing(f'ProteinStructureDataset.__getitem__ {idx}', logger.debug):
      pids = self.pids[idx]
      pid = pids[np.random.randint(len(pids))]

      if np.random.random() < self.pseudo_linker_prob:
        chains = self.get_chain_list(pid)
      else:
        chains = None

      if not exists(chains) or len(chains) == 1:
        ret = self.get_monomer(pid, crop_fn=self.data_crop_fn)
      else:
        ret = self.get_complex(pid, chains)

      # We need all the amino acids!
      # if 'coord_mask' in ret:
      #   ret['mask'] = torch.sum(ret['coord_mask'], dim=-1) > 0
    return ret

  def __len__(self):
    return len(self.pids)

  def data_from_domain(self, item, domains):
    assert domains

    n, new_order = len(item['str_seq']), []
    for i, j in domains:
      assert j < n, (item['pid'], domains)
      for k in range(i, j + 1):
        new_order.append(k)

    assert 0 < len(new_order) <= n, (len(new_order), n)
    if len(new_order) < n:
      item = _make_feats_shrinked(item,
                                  new_order,
                                  seq_feats=('seq', 'seq_index', 'mask',
                                             'coord', 'coord_mask',
                                             'coord_plddt', 'seq_color'))
    return item

  def data_rm_mask(self, item):
    if 'coord_mask' in item:
      i, new_order = 0, []

      while i < item['coord_mask'].shape[0]:
        if torch.any(item['coord_mask'][i]):
          new_order.append(i)
        i += 1
      logger.debug('data_rm_mask: %s k=%d, i=%d', item['pid'], len(new_order),
                   i)

      assert 0 <= len(new_order) <= i, (len(new_order), i)
      if 0 < len(new_order) < i:
        item = _make_feats_shrinked(item,
                                    new_order,
                                    seq_feats=('seq', 'seq_index', 'mask',
                                               'coord', 'coord_mask',
                                               'coord_plddt'))
    return item

  def msa_as_seq(self, item, idx):
    assert idx > 0
    assert 'str_msa' in item and 'del_msa' in item
    assert len(item['str_seq']) == len(item['str_msa'][0]), (idx, item['pid'],
                                                             item['str_seq'],
                                                             item['str_msa'][0])

    if 'coord' in item:
      assert item['seq'].shape[0] == item['coord'].shape[0], (item['pid'],)

    # swap(msa[1], msa[idx])
    assert len(item['str_msa'][0]) == len(item['str_msa'][1])
    if item['str_seq'] != item['str_msa'][1]:
      assert item['str_seq'] != item['str_msa'][1], (item['pid'], idx,
                                                     item['str_msa'][0],
                                                     item['str_msa'][1])
      item['str_msa'][0] = item['str_msa'][1]
      item['str_msa'][1] = item['str_seq']
      assert item['str_seq'] != item['str_msa'][0], (item['pid'], idx,
                                                     item['str_msa'][0],
                                                     item['str_msa'][1])
      item['str_seq'] = item['str_msa'][0]
      assert item['str_seq'] == item['str_msa'][0]

      i, new_order = 0, []

      while i < len(item['str_seq']):
        if item['str_seq'][i] != residue_constants.restypes_with_x_and_gap[-1]:
          new_order.append(i)
        i += 1
      logger.debug('msa_as_seq: %s@%s k=%d, i=%d', item['pid'], idx,
                   len(new_order), i)
      assert 0 < len(new_order) <= i, (len(new_order), i)

      if len(new_order) < i:
        item = _make_feats_shrinked(item, new_order)

      # Renew seq related feats
      pid = item['pid']
      item['pid'] = f'{pid}@{idx}'
      item.update(
          _make_seq_features(item['str_seq'],
                             item['pid'],
                             seq_color=item['seq_color'][0],
                             max_seq_len=None))

      # Apply new coord_mask based on aatypes
      restype_atom14_mask = np.copy(residue_constants.restype_atom14_mask)
      includes = set(['N', 'CA', 'C', 'CB', 'O'])
      for i in range(residue_constants.restype_num):
        resname = residue_constants.restype_1to3[residue_constants.restypes[i]]
        atom_list = residue_constants.restype_name_to_atom14_names[resname]
        for j in range(restype_atom14_mask.shape[1]):
          if restype_atom14_mask[i, j] > 0 and atom_list[j] not in includes:
            restype_atom14_mask[i, j] = 0
      coord_exists = torch.gather(
          torch.from_numpy(restype_atom14_mask), 0,
          repeat(item['seq'].long(),
                 'i -> i n',
                 n=restype_atom14_mask.shape[-1]))
      for field in ('coord', 'coord_mask', 'coord_plddt'):
        if field in item:
          item[field] = torch.einsum('i n ...,i n -> i n ...', item[field],
                                     coord_exists)
      # Delete coords if all are invalid.
      # ca_idx = residue_constants.atom_order['CA']
      # FIXED: collate_fn may failed when batch_size > 1
      # if 'coord_mask' in item and not torch.any(item['coord_mask'][:, ca_idx]):
      #   for field in ('coord', 'coord_mask', 'coord_plddt'):
      #     if field in item:
      #       del item[field]

      # Fix seq_index
      del_seq = torch.cumsum(item['del_msa'][0],
                             dim=-1,
                             dtype=item['del_msa'].dtype)
      item['seq_index'] = item['seq_index'] + del_seq
      if 'resolu' in item:
        item['resolu'] = -1.  # delete it !

    return item

  @contextlib.contextmanager
  def _setattr(self, name, value):
    assert hasattr(self, name)
    old = getattr(self, name)
    setattr(self, name, value)
    yield self
    setattr(self, name, old)

  @contextlib.contextmanager
  def _fileobj(self, filename):
    if os.path.isabs(filename):
      with open(filename, mode='rb') as f:
        yield f
    elif isinstance(self.data_dir, zipfile.ZipFile):
      with self.data_dir.open(filename, 'r') as f:
        yield f
    else:
      with open(self.data_dir / filename, mode='rb') as f:
        yield f

  def _fstat(self, filename):
    if isinstance(self.data_dir, zipfile.ZipFile):
      try:
        self.data_dir.getinfo(filename)
        return True
      except KeyError as e:
        del e
        return False
    return (self.data_dir / filename).exists()

  def _ftext(self, line, encoding='utf-8'):
    if isinstance(line, bytes):
      return line.decode(encoding)
    return line

  def get_monomer(self, pid, seq_color=1, crop_fn=None):
    # CATH format pid
    pid, chain, domains = decompose_pid(pid, return_domain=True)
    if exists(domains):
      domains = list(seq_index_split(domains))
    pid = compose_pid(pid, chain)

    pkey = self.mapping[pid] if pid in self.mapping else pid
    seq_feats = self.get_seq_features(pkey, seq_color=seq_color)

    ret = dict(pid=pid,
               msa_idx=0,
               clip=None,
               resolu=self.get_resolution(pid),
               **seq_feats)
    if self.feat_flags & FEAT_PDB:
      ret.update(self.get_structure_label_npz(pid))
    if exists(domains):
      if self.feat_flags & FEAT_MSA:
        ret.update(self.get_msa_features_new(pkey))
      ret = self.data_from_domain(ret, domains)
    if exists(crop_fn):
      clip = crop_fn(ret)
      if exists(clip):
        ret = _protein_crop_fn(ret, clip)
        ret['clip'] = clip
    if not exists(domains) and (self.feat_flags & FEAT_MSA):
      ret.update(self.get_msa_features_new(pkey, ret.get('clip')))

    if exists(domains):
      # CATH update pid
      ret['pid'] = compose_pid(pid, None, seq_index_join(domains))

    if 'msa_idx' in ret and ret['msa_idx'] != 0:
      ret = self.msa_as_seq(ret, ret['msa_idx'])
    elif exists(domains):
      pass
    elif np.random.random() < self.data_rm_mask_prob:
      ret = self.data_rm_mask(ret)

    if 'clip' in ret and exists(ret['clip']):
      ret['clip'] = _protein_crop_fmt(ret['clip'])
    return ret

  def get_complex(self, protein_id, chains):
    assert len(chains) > 1

    pid, selected_chain = decompose_pid(protein_id)  # pylint: disable=unbalanced-tuple-unpacking
    assert selected_chain in chains

    # shuffle the chains
    if self.pseudo_linker_shuffle:
      np.random.shuffle(chains)

    seq_index_offset, seq_index_gap = 0, 128
    # Concat all the feats
    ret = {'clip': None}
    for idx, chain in enumerate(chains):
      with self._setattr(
          'msa_as_seq_prob',
          self.msa_as_seq_prob if chain == selected_chain else 0):
        feat = self.get_monomer(compose_pid(pid, chain), seq_color=idx + 1)
      # Sequence related
      for field in ('str_seq',):
        ret[field] = ret.get(field, '') + feat[field]
      assert 'seq' in feat
      for field in ('seq', 'seq_color', 'mask', 'coord', 'coord_mask'):
        assert field in feat, (field, pid, chain)
        if field not in ret:
          ret[field] = feat[field]
        else:
          ret[field] = torch.cat((ret[field], feat[field]), dim=0)
      for field in ('coord_plddt',):
        assert 'coord_mask' in feat, (pid, chain)
        if field not in feat:
          feat['coord_plddt'] = torch.ones_like(feat['coord_mask'],
                                                dtype=torch.float32)
        if field not in ret:
          ret[field] = feat[field]
        else:
          ret[field] = torch.cat((ret[field], feat[field]), dim=0)

      for field in ('seq_index',):
        seq_index = feat['seq_index'] + seq_index_offset
        if 'seq_index' not in ret:
          ret['seq_index'] = seq_index
        else:
          ret['seq_index'] = torch.cat((ret['seq_index'], seq_index), dim=0)
        seq_index_offset = seq_index[-1] + seq_index_gap
      ret['resolu'] = feat['resolu']
      # MSA related
      if self.feat_flags & FEAT_MSA:
        for field in ('msa_idx', 'num_msa'):
          ret[field] = max(ret.get(field, 0), feat[field])
        m, n = len(ret.get('str_msa', [])), len(feat['str_msa'])
        gap_idx = residue_constants.restypes_with_x_and_gap.index('-')
        if m < n and 'msa' in ret:
          seq_len = len(ret['str_msa'][0])
          ret['str_msa'] += ['-' * seq_len] * (n - m)
          ret['msa'] = torch.cat(
              (ret['msa'],
               torch.full((n - m, seq_len), gap_idx, dtype=ret['msa'].dtype)),
              dim=0)
          ret['del_msa'] = torch.cat(
              (ret['del_msa'], torch.zeros(n - m, seq_len)), dim=0)
        elif 0 <= n < m:
          seq_len = len(feat['str_msa'][0])
          feat['str_msa'] += ['-' * seq_len] * (m - n)
          feat['msa'] = torch.cat(
              (feat['msa'],
               torch.full((m - n, seq_len), gap_idx, dtype=feat['msa'].dtype)),
              dim=0)
          feat['del_msa'] = torch.cat(
              (feat['del_msa'], torch.zeros(m - n, seq_len)), dim=0)
        # Rand permute msa relate feat
        if 'str_msa' not in ret:
          ret['str_msa'] = feat['str_msa']
        else:
          for i in range(max(m, n)):
            ret['str_msa'][i] += feat['str_msa'][i]
        for field in ('msa', 'del_msa'):
          if field not in ret:
            ret[field] = feat[field]
          else:
            ret[field] = torch.cat((ret[field], feat[field]), dim=1)
        for field in ('msa_row_mask',):
          if field not in ret or m < n:
            ret[field] = feat[field]
        # Update chain_id
        if feat['msa_idx'] > 0:
          msa_idx = feat['msa_idx']
          chains[idx] = f'{chain}@{msa_idx}'
    ret['pid'] = compose_pid(pid, ','.join(chains))

    if exists(self.data_crop_fn):
      clip = self.data_crop_fn(ret)
      if exists(clip):
        ret = _protein_crop_fn(ret, clip)
        clip = _protein_crop_fmt(clip)
      ret['clip'] = clip

    return ret

  def get_chain_list(self, protein_id):
    pid, chain = decompose_pid(protein_id)  # pylint: disable=unbalanced-tuple-unpacking
    if pid in self.chain_list:
      chain_group = self.chain_list[pid]
      for g in chain_group:
        if chain in g:
          return list(g)  # shallow copy
      logger.error('get_chain_list: %s not found.', protein_id)
    return None

  def get_resolution(self, protein_id):
    pid, _ = decompose_pid(protein_id)  # pylint: disable=unbalanced-tuple-unpacking
    return self.resolu.get(pid[:4], -1.)

  def get_msa_features_new(self, protein_id, clip=None):

    def _aligned_ratio(msa, n):
      r = 1. - msa.count('-') / n
      if exists(self.msa_as_seq_min_alr) and r < self.msa_as_seq_min_alr:
        return 0
      return r

    def _ident_ratio(msa, seq):
      assert exists(self.msa_as_seq_min_ident) and self.msa_as_seq_min_ident > 0
      r = 0
      i, j = 0, 0
      while i < len(msa) and j < len(seq):
        if msa[i] == seq[j]:
          r += 1
        i, j = i + 1, j + 1
      r = r / max(min(i, j), 0.1)
      if r >= self.msa_as_seq_min_ident:
        return r
      return 0

    k = int(np.random.randint(len(self.msa_list)))
    source = self.msa_list[k]
    with self._fileobj(f'msa/{protein_id}/{source}/{protein_id}.a4m') as f:
      sequences = list(map(lambda x: self._ftext(x).strip(), f))

    if exists(clip):
      def _yield_msa_clip(msa):
        k = 0
        if 'd' in clip:
          new_order, j = clip['d'], 0
          for c in msa:
            if j >= len(new_order):
              break
            if not c.islower():
              if k >= new_order[j]:
                yield c
                j += 1
              k += 1
        else:
          i, j = clip['i'], clip['j']
          for c in msa:
            if k >= j:
              break
            if not c.islower():
              if k >= i:
                yield c
              k += 1

      sequences = [''.join(_yield_msa_clip(s)) for s in sequences]

    ret = {'msa_idx': 0}
    if len(sequences) > 1 and np.random.random() < self.msa_as_seq_prob:
      m, n = len(sequences[0]), len(sequences)
      if exists(self.msa_as_seq_topn):
        n = min(n, self.msa_as_seq_topn)
      assert n > 1

      clu_file_path = f'msa/{protein_id}/{source}/{protein_id}.clu'
      if self.msa_as_seq_clustering and self._fstat(clu_file_path):
        try:
          with self._fileobj(clu_file_path) as f:
            clu_list = list(map(lambda x: int(self._ftext(x).strip()), f))
          if len(clu_list) != len(sequences):
            raise ValueError('len(clu_list) != len(sequences)')

          clu_dict = defaultdict(int)
          for i, clu in enumerate(clu_list):
            clu_dict[clu] += 1
          clu_list = np.asarray([clu_dict[clu] for clu in clu_list[1:n]])
          del clu_dict
        except ValueError as e:
          clu_list = None
          logger.error('read clu faild: (%s) %s', protein_id, str(e))
      else:
        clu_list = None

      w = np.power(np.array([1.0 / p for p in range(1, n)]), 1.0 / 3.0)
      v = np.asarray([_aligned_ratio(s, m) for s in sequences[1:n]])
      w *= v
      if exists(clu_list):
        w /= clu_list
      if exists(self.msa_as_seq_min_ident) and self.msa_as_seq_min_ident > 0:
        v = np.asarray([_ident_ratio(s, sequences[0]) for s in sequences[1:n]])
        w *= v
      t = np.sum(w)
      if t > 0:
        w /= t
        ret['msa_idx'] = int(np.argmax(np.random.multinomial(1, w))) + 1
    ret.update(
        _make_msa_features(sequences,
                           msa_idx=ret['msa_idx'],
                           max_msa_depth=self.max_msa_depth))
    return ret

  def get_structure_label_npz(self, protein_id):
    ret = {}

    pdb_file = f'{self.pdb_dir}/{protein_id}.npz'
    if self._fstat(pdb_file):
      with self._fileobj(pdb_file) as f:
        structure = np.load(BytesIO(f.read()))
        ret = dict(coord=torch.from_numpy(structure['coord']),
                   coord_mask=torch.from_numpy(structure['coord_mask']))
        if 'bfactor' in structure:
          ret.update(coord_plddt=torch.from_numpy(structure['bfactor']))
        else:
          ret.update(
              coord_plddt=torch.ones_like(ret['coord_mask'], dtype=torch.float))
      pae_file = f'{self.pdb_dir}/{protein_id}-predicted_aligned_error.json'
      if self._fstat(pae_file):
        with self._fileobj(pae_file) as f:
          try:
            pae_obj = json.loads(f.read())
            assert len(pae_obj) == 1
            pae_obj = torch.as_tensor(pae_obj[0]['predicted_aligned_error'])
            coord = ret['coord']
            if coord.shape[0] == pae_obj.shape[0]:
              ret.update(coord_pae=pae_obj)
          except json.decoder.JSONDecodeError as e:
            logger.error('get_structure_label_npz.pae: %s|%s', protein_id, e)
    return ret

  def get_seq_features(self, protein_id, seq_color=1):
    """Runs alignment tools on the input sequence and creates features."""
    input_fasta_path = f'{self.fasta_dir}/{protein_id}.fasta'
    with self._fileobj(input_fasta_path) as f:
      input_fasta_str = self._ftext(f.read())
    input_seqs, input_descs = parse_fasta(input_fasta_str)
    if len(input_seqs) != 1:
      raise ValueError(
          f'More than one input sequence found in {input_fasta_path}.')
    input_sequence = input_seqs[0]
    input_description = input_descs[0]

    return _make_seq_features(input_sequence,
                              input_description,
                              seq_color=seq_color,
                              max_seq_len=None)

  @staticmethod
  def collate_fn(batch, feat_flags=None):
    fields = ('pid', 'resolu', 'seq', 'seq_index', 'seq_color', 'mask',
              'str_seq', 'clip')
    pids, resolutions, seqs, seqs_idx, seqs_color, masks, str_seqs, clips = list(  # pylint: disable=line-too-long
        zip(*[[b[k] for k in fields] for b in batch]))
    lengths = tuple(len(s) for s in str_seqs)
    max_batch_len = max(lengths)

    padded_seqs = pad_for_batch(seqs, max_batch_len, 'seq')
    padded_seqs_idx = pad_for_batch(seqs_idx, max_batch_len, 'seq_index')
    padded_seqs_clr = pad_for_batch(seqs_color, max_batch_len, 'seq_index')
    padded_masks = pad_for_batch(masks, max_batch_len, 'msk')

    ret = dict(pid=pids,
               resolution=torch.as_tensor(resolutions),
               seq=padded_seqs,
               seq_index=padded_seqs_idx,
               seq_color=padded_seqs_clr,
               mask=padded_masks,
               str_seq=str_seqs)

    feat_flags = default(feat_flags, FEAT_ALL)
    if feat_flags & FEAT_PDB and 'coord' in batch[0]:
      # required
      fields = ('coord', 'coord_mask')
      coords, coord_masks = list(zip(*[[b[k] for k in fields] for b in batch]))

      padded_coords = pad_for_batch(coords, max_batch_len, 'crd')
      padded_coord_masks = pad_for_batch(coord_masks, max_batch_len, 'crd_msk')
      ret.update(coord=padded_coords, coord_mask=padded_coord_masks)
      # optional
      fields = ('coord_plddt',)
      for field in fields:
        if not all(field in b for b in batch):
          continue
        padded_values = pad_for_batch([b[field] for b in batch], max_batch_len,
                                      field)
        ret[field] = padded_values

    if feat_flags & FEAT_MSA:
      fields = ('msa', 'msa_idx', 'msa_row_mask', 'str_msa', 'del_msa',
                'num_msa')
      msas, msa_idx, msa_row_msks, str_msas, del_msas, num_msa = list(
          zip(*[[b[k] for k in fields] for b in batch]))

      padded_msas = pad_for_batch(msas, max_batch_len, 'msa')
      msa_row_msks = pad_for_batch(msa_row_msks, max_batch_len, 'msa_row_mask')
      padded_dels = pad_for_batch(del_msas, max_batch_len, 'del_msa')
      ret.update(msa=padded_msas,
                 msa_idx=torch.as_tensor(msa_idx, dtype=torch.int),
                 msa_row_mask=msa_row_msks,
                 str_msa=str_msas,
                 del_msa=padded_dels,
                 num_msa=torch.as_tensor(num_msa, dtype=torch.int))

    ret['clips'] = clips

    return ret


def pad_for_batch(items, batch_length, dtype):
  """Pad a list of items to batch_len using values dependent on the item type.

    Args:
        items: List of items to pad (i.e. sequences or masks represented as
            arrays of numbers, angles, coordinates, pssms).
        batch_length: The integer maximum length of any of the items in the
            input. All items are padded so that their length matches this
            number.
        dtype: A string ('seq', 'msk', 'crd') reperesenting the type of
            data included in items.

    Returns:
         A padded list of the input items, all independently converted to Torch
           tensors.
    """
  batch = []
  if dtype == 'seq':
    for seq in items:
      z = torch.ones(batch_length - seq.shape[0],
                     dtype=seq.dtype) * residue_constants.unk_restype_index
      c = torch.cat((seq, z), dim=0)
      batch.append(c)
  elif dtype == 'seq_index':
    for idx in items:
      z = torch.zeros(batch_length - idx.shape[0], dtype=idx.dtype)
      c = torch.cat((idx, z), dim=0)
      batch.append(c)
  elif dtype == 'msk':
    # Mask sequences (1 if present, 0 if absent) are padded with 0s
    for msk in items:
      z = torch.zeros(batch_length - msk.shape[0], dtype=msk.dtype)
      c = torch.cat((msk, z), dim=0)
      batch.append(c)
  elif dtype == 'crd':
    for item in items:
      z = torch.zeros(
          (batch_length - item.shape[0], item.shape[-2], item.shape[-1]),
          dtype=item.dtype)
      c = torch.cat((item, z), dim=0)
      batch.append(c)
  elif dtype in ('crd_msk', 'coord_plddt'):
    for item in items:
      z = torch.zeros((batch_length - item.shape[0], item.shape[-1]),
                      dtype=item.dtype)
      c = torch.cat((item, z), dim=0)
      batch.append(c)
  elif dtype == 'msa':
    msa_depth = max(msa.shape[0] for msa in items)
    for msa in items:
      c = msa
      # Append columns
      z = torch.ones((c.shape[0], batch_length - c.shape[1]),
                     dtype=c.dtype) * residue_constants.HHBLITS_AA_TO_ID['-']
      c = torch.cat((c, z), dim=1)
      # Append rows
      z = torch.ones((msa_depth - c.shape[0], c.shape[1]),
                     dtype=c.dtype) * residue_constants.HHBLITS_AA_TO_ID['-']
      c = torch.cat((c, z), dim=0)
      batch.append(c)
  elif dtype == 'msa_row_mask':
    msa_depth = max(msa.shape[0] for msa in items)
    for msa in items:
      c = msa
      # Append rows
      z = torch.zeros((msa_depth - c.shape[0]), dtype=c.dtype)
      c = torch.cat((c, z), dim=0)
      batch.append(c)
  elif dtype == 'del_msa':
    msa_depth = max(msa.shape[0] for msa in items)
    for del_msa in items:
      c = del_msa
      # Append columns
      z = torch.zeros((c.shape[0], batch_length - c.shape[1]), dtype=c.dtype)
      c = torch.cat((c, z), dim=1)
      # Append rows
      z = torch.zeros((msa_depth - c.shape[0], c.shape[1]), dtype=c.dtype)
      c = torch.cat((c, z), dim=0)
      batch.append(c)
  else:
    raise ValueError('Not implement yet!')
  batch = torch.stack(batch, dim=0)
  return batch


def load(data_dir,
         data_idx=None,
         pseudo_linker_prob=0.0,
         data_rm_mask_prob=0.0,
         msa_as_seq_prob=0.0,
         min_crop_len=None,
         max_crop_len=None,
         min_crop_pae=False,
         max_crop_plddt=False,
         crop_probability=0,
         crop_algorithm='random',
         feat_flags=FEAT_ALL,
         **kwargs):
  max_msa_depth = kwargs.pop(
      'max_msa_depth') if 'max_msa_depth' in kwargs else 128
  msa_as_seq_topn = kwargs.pop(
      'msa_as_seq_topn') if 'msa_as_seq_topn' in kwargs else None
  msa_as_seq_min_alr = kwargs.pop(
      'msa_as_seq_min_alr') if 'msa_as_seq_min_alr' in kwargs else None
  msa_as_seq_clustering = kwargs.pop(
      'msa_as_seq_clustering') if 'msa_as_seq_clustering' in kwargs else False
  msa_as_seq_min_ident = kwargs.pop(
      'msa_as_seq_min_ident') if 'msa_as_seq_min_ident' in kwargs else None

  data_dir = data_dir.split(',')
  if exists(data_idx):
    data_idx = data_idx.split(',')
  else:
    data_idx = [None] * len(data_dir)
  assert len(data_dir) == len(data_idx)

  if 'data_crop_fn' not in kwargs:
    crop_fn_kwargs = {}
    if 'intra_domain_probability' in kwargs:
      crop_fn_kwargs['intra_domain_probability'] = kwargs.pop(
          'intra_domain_probability')
    data_crop_fn = functools.partial(_protein_clips_fn,
                                     min_crop_len=min_crop_len,
                                     max_crop_len=max_crop_len,
                                     min_crop_pae=min_crop_pae,
                                     max_crop_plddt=max_crop_plddt,
                                     crop_probability=crop_probability,
                                     crop_algorithm=crop_algorithm,
                                     **crop_fn_kwargs)
  else:
    data_crop_fn = kwargs.pop('data_crop_fn')

  dataset = torch.utils.data.ConcatDataset([
      ProteinStructureDataset(data_dir[i],
                              data_idx=data_idx[i],
                              data_crop_fn=data_crop_fn,
                              pseudo_linker_prob=pseudo_linker_prob,
                              data_rm_mask_prob=data_rm_mask_prob,
                              msa_as_seq_prob=msa_as_seq_prob,
                              msa_as_seq_topn=msa_as_seq_topn,
                              msa_as_seq_clustering=msa_as_seq_clustering,
                              msa_as_seq_min_alr=msa_as_seq_min_alr,
                              msa_as_seq_min_ident=msa_as_seq_min_ident,
                              max_msa_depth=max_msa_depth,
                              feat_flags=feat_flags)
      for i in range(len(data_dir))
  ])
  if 'collate_fn' not in kwargs:
    kwargs['collate_fn'] = functools.partial(ProteinStructureDataset.collate_fn,
                                             feat_flags=feat_flags)
  if 'weights' in kwargs:
    weights = kwargs.pop('weights')
    if weights:
      kwargs['sampler'] = WeightedRandomSampler(weights,
                                                num_samples=len(weights))
      if 'shuffle' in kwargs:
        kwargs.pop('shuffle')
  elif 'num_replicas' in kwargs and 'rank' in kwargs:
    num_replicas, rank = kwargs.pop('num_replicas'), kwargs.pop('rank')
    kwargs['sampler'] = DistributedSampler(dataset,
                                           num_replicas=num_replicas,
                                           rank=rank)
  return torch.utils.data.DataLoader(dataset, **kwargs)
