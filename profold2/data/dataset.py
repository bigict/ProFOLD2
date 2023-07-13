"""Dataset for structure
 """
import os
import contextlib
import functools
import logging
from io import BytesIO
import pathlib
import re
import string
import zipfile

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from einops import repeat

from profold2.common import residue_constants
from profold2.data.parsers import parse_fasta
from profold2.data.utils import domain_parser
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

  return dict(msa=torch.as_tensor(int_msa),
              str_msa=msa,
              del_msa=torch.as_tensor(del_matirx),
              num_msa=msa_depth)


def _parse_seq_index(description, input_sequence, seq_index):
  # description: pid field1 field2 ...
  seq_index_pattern = '(\\d+)-(\\d+)'

  def seq_index_split(text):
    for s in text.split(','):
      r = re.match(seq_index_pattern, s)
      assert r
      yield tuple(map(int, r.group(1, 2)))

  def seq_index_check(positions):
    for i in range(len(positions) - 1):
      p, q = positions[i]
      m, n = positions[i + 1]
      assert p < q and m < n
      assert q < m
    m, n = positions[-1]
    assert m <= n
    assert sum(map(lambda p: p[1] - p[0] + 1, positions)) == len(input_sequence)

  fields = description.split()
  for f in fields[1:]:
    r = re.match(f'.*:({seq_index_pattern}(,{seq_index_pattern})*)', f)
    if r:
      positions = list(seq_index_split(r.group(1)))
      seq_index_check(positions)
      p, q = positions[0]
      start, gap = p, 0
      for m, n in positions[1:]:
        gap += m - q - 1
        seq_index[m - start - gap:n - start - gap + 1] = torch.arange(
            m - start, n - start + 1)
        p, q = m, n
      logger.debug('_parse_seq_index: desc=%s, positions=%s', description,
                   positions)
      break

  return seq_index


def _make_seq_features(sequence, description, max_seq_len=None):
  residue_index = torch.arange(len(sequence), dtype=torch.int)
  residue_index = _parse_seq_index(description, sequence, residue_index)

  sequence = sequence[:max_seq_len]
  residue_index = residue_index[:max_seq_len]

  seq = torch.tensor(residue_constants.sequence_to_onehot(
      sequence=sequence,
      mapping=residue_constants.restype_order_with_x,
      map_unknown_to_x=True),
                     dtype=torch.int).argmax(-1)
  #residue_index = torch.arange(len(sequence), dtype=torch.int)
  str_seq = ''.join(
      map(
          lambda a: a if a in residue_constants.restype_order_with_x else
          residue_constants.restypes_with_x[-1], sequence))
  mask = torch.ones(len(sequence), dtype=torch.bool)

  return dict(seq=seq, seq_index=residue_index, str_seq=str_seq, mask=mask)


class ProteinSequenceDataset(torch.utils.data.Dataset):
  """Construct a `Dataset` from sequences
   """

  def __init__(self, sequences, descriptions=None, msa=None):
    self.sequences = sequences
    self.descriptions = descriptions
    self.msa = msa
    assert not exists(self.descriptions) or len(self.sequences) == len(
        self.descriptions)
    assert not exists(self.msa) or len(self.sequences) == len(self.msa)

  def __getitem__(self, idx):
    input_sequence = self.sequences[idx]
    seq = torch.tensor(residue_constants.sequence_to_onehot(
        sequence=input_sequence,
        mapping=residue_constants.restype_order_with_x,
        map_unknown_to_x=True),
                       dtype=torch.int).argmax(-1)
    residue_index = torch.arange(len(input_sequence), dtype=torch.int)
    str_seq = ''.join(
        map(
            lambda a: a if a in residue_constants.restype_order_with_x else
            residue_constants.restypes_with_x[-1], input_sequence))
    mask = torch.ones(len(input_sequence), dtype=torch.bool)
    if exists(self.descriptions) and exists(self.descriptions[idx]):
      desc = self.descriptions[idx]
      residue_index = _parse_seq_index(desc, input_sequence, residue_index)
    else:
      desc = str(idx)
    ret = dict(pid=desc,
               seq=seq,
               seq_index=residue_index,
               str_seq=str_seq,
               mask=mask)
    if exists(self.msa) and exists(self.msa[idx]):
      ret.update(_make_msa_features(self.msa[idx]))
    return ret

  def __len__(self):
    return len(self.sequences)

  @staticmethod
  def collate_fn(batch):
    fields = ('pid', 'seq', 'seq_index', 'mask', 'str_seq')
    pids, seqs, seqs_idx, masks, str_seqs = list(
        zip(*[[b[k] for k in fields] for b in batch]))
    lengths = tuple(len(s) for s in str_seqs)
    max_batch_len = max(lengths)

    padded_seqs = pad_for_batch(seqs, max_batch_len, 'seq')
    padded_seqs_idx = pad_for_batch(seqs_idx, max_batch_len, 'seq_index')
    padded_masks = pad_for_batch(masks, max_batch_len, 'msk')

    ret = dict(pid=pids,
               seq=padded_seqs,
               seq_index=padded_seqs_idx,
               mask=padded_masks,
               str_seq=str_seqs)

    fields = ('msa', 'str_msa', 'del_msa', 'num_msa')
    if all(all(field in b for field in fields) for b in batch):
      msas, str_msas, del_msas, num_msa = list(
          zip(*[[b[k] for k in fields] for b in batch]))

      padded_msas = pad_for_batch(msas, max_batch_len, 'msa')
      ret.update(msa=padded_msas,
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
               max_msa_depth=128,
               max_seq_len=None,
               data_rm_mask_prob=0.0,
               msa_as_seq_prob=0.0,
               feat_flags=FEAT_ALL & (~FEAT_MSA)):
    super().__init__()

    self.data_dir = pathlib.Path(data_dir)
    data_idx = default(data_idx, 'name.idx')
    if zipfile.is_zipfile(self.data_dir):
      self.data_dir = zipfile.ZipFile(self.data_dir)  # pylint: disable=consider-using-with
    self.max_msa_depth = max_msa_depth
    self.max_seq_len = max_seq_len
    self.data_rm_mask_prob = data_rm_mask_prob
    self.msa_as_seq_prob = msa_as_seq_prob
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

    self.fasta_dir = 'fasta'

    self.pdb_dir = 'npz'
    logger.info('load structure data from: %s', self.pdb_dir)

    self.msa_list = ['BFD30_E-3']

  def __getstate__(self):
    logger.debug('being pickled ...')
    d = self.__dict__
    if isinstance(self.data_dir, zipfile.ZipFile):
      d['data_dir'] = self.data_dir.filename
    return d

  def __setstate__(self, d):
    logger.debug('being unpickled ...')
    if zipfile.is_zipfile(d['data_dir']):
      d['data_dir'] = zipfile.ZipFile(d['data_dir'])  # pylint: disable=consider-using-with
    self.__dict__ = d

  def __getitem__(self, idx):
    with timing(f'ProteinStructureDataset.__getitem__ {idx}', logger.debug):
      pids = self.pids[idx]
      pid = pids[np.random.randint(len(pids))]

      pkey = self.mapping[pid] if pid in self.mapping else pid
      seq_feats = self.get_seq_features(pkey)

      ret = dict(pid=pid,
                 msa_idx=0,
                 resolu=self.get_resolution(pid),
                 **seq_feats)
      if self.feat_flags & FEAT_MSA:
        ret.update(self.get_msa_features_new(pkey))
      if self.feat_flags & FEAT_PDB:
        ret.update(self.get_structure_label_npz(pid))

      if 'msa_idx' in ret and ret['msa_idx'] != 0:
        ret = self.msa_as_seq(ret, ret['msa_idx'])
      elif np.random.random() < self.data_rm_mask_prob:
        ret = self.data_rm_mask(ret)

      # We need all the amino acids!
      # if 'coord_mask' in ret:
      #   ret['mask'] = torch.sum(ret['coord_mask'], dim=-1) > 0
    return ret

  def __len__(self):
    return len(self.pids)

  def data_rm_mask(self, item):
    i, k = 0, 0

    item['str_seq'] = list(item['str_seq'])
    for field in ('str_msa',):
      if field in item:
        for j in range(len(item['str_msa'])):
          item['str_msa'][j] = list(item['str_msa'][j])
    while i < item['mask'].shape[0]:
      if item['mask'][i]:
        item['str_seq'][k] = item['str_seq'][i]
        for field in ('seq', 'seq_index', 'mask', 'coord', 'coord_mask',
                      'coord_plddt'):
          if field in item:
            item[field][k] = item[field][i]
        for field in ('str_msa',):
          if field in item:
            for j in range(len(item['str_msa'])):
              item['str_msa'][j][k] = item['str_msa'][j][i]
        for field in ('msa', 'del_msa'):
          if field in item:
            item[field][:, k, ...] = item[field][:, i, ...]
        k += 1
      i += 1
    logger.debug('data_rm_mask: %s k=%d, i=%d', item['pid'], k, i)
    assert 0 < k <= i, (k, i)
    if k < i:
      item['str_seq'] = item['str_seq'][:k]
      for field in ('seq', 'seq_index', 'mask', 'coord', 'coord_mask',
                    'coord_plddt'):
        if field in item:
          item[field] = item[field][:k]
      for field in ('str_msa',):
        if field in item:
          for j in range(len(item['str_msa'])):
            item['str_msa'][j] = item['str_msa'][j][:k]
      for field in ('msa', 'del_msa'):
        if field in item:
          item[field] = item[field][:, :k, ...]
    item['str_seq'] = ''.join(item['str_seq'])
    for field in ('str_msa',):
      if field in item:
        for j in range(len(item['str_msa'])):
          item['str_msa'][j] = ''.join(item['str_msa'][j])
    return item

  def msa_as_seq(self, item, idx):
    assert idx > 0
    assert 'str_msa' in item and 'del_msa' in item
    assert len(item['str_seq']) == len(item['str_msa'][0]), (
        idx, item['pid'], item['str_seq'], item['str_msa'][0])

    if 'coord' in item:
      assert item['seq'].shape[0] == item['coord'].shape[0], (item['pid'],)

    # swap(msa[1], msa[idx])
    assert len(item['str_msa'][0]) == len(item['str_msa'][1])
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

    # Update seq related feats
    if len(new_order) < i:
      item['str_seq'] = ''.join(item['str_seq'][k] for k in new_order)

      for field in ('str_msa',):
        if field in item:
          for j in range(len(item['str_msa'])):
            item['str_msa'][j] = ''.join(
                item['str_msa'][j][k] for k in new_order)

    pid = item['pid']
    item['pid'] = f'{pid}@{idx}'
    item.update(
        _make_seq_features(item['str_seq'],
                           item['pid'],
                           max_seq_len=self.max_seq_len))
    # Update tensors
    if len(new_order) < i:
      new_order = torch.as_tensor(new_order)

      for field in ('coord', 'coord_mask', 'coord_plddt'):
        if field in item:
          item[field] = torch.index_select(item[field], 0, new_order)
      for field in ('msa', 'del_msa'):
        if field in item:
          item[field] = torch.index_select(item[field], 1, new_order)
    new_order = None

    # Apply new coord_mask based on aatypes
    coord_exists = torch.gather(
        torch.from_numpy(residue_constants.restype_atom14_mask), 0,
        repeat(item['seq'],
               'i -> i n',
               n=residue_constants.restype_atom14_mask.shape[-1]))
    for field in ('coord', 'coord_mask', 'coord_plddt'):
      if field in item:
        item[field] = torch.einsum('i n ...,i n -> i n ...', item[field],
                                   coord_exists)
    # Delete coords if all are invalid.
    ca_idx = residue_constants.atom_order['CA']
    if 'coord_mask' in item and not torch.any(item['coord_mask'][:,ca_idx]):
      for field in ('coord', 'coord_mask', 'coord_plddt'):
        if field in item:
          del item[field]

    # Fix seq_index
    del_seq = torch.cumsum(item['del_msa'][0], dim=-1)
    item['seq_index'] = item['seq_index'] + del_seq
    if 'resolu' in item:
      item['resolu'] = None  # delete it !

    return item

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

  def get_resolution(self, protein_id):
    pid = protein_id.split('_')
    return self.resolu.get(pid[0], None)

  def get_msa_features_new(self, protein_id):
    k = int(np.random.randint(len(self.msa_list)))
    source = self.msa_list[k]
    with self._fileobj(f'msa/{protein_id}/{source}/{protein_id}.a4m') as f:
      sequences = list(map(lambda x: self._ftext(x).strip(), f))

    ret = {'msa_idx': 0}
    if len(sequences) > 1 and np.random.random() < self.msa_as_seq_prob:
      n = len(sequences)
      if exists(self.max_msa_depth):
        n = min(n, self.max_msa_depth)
      w = np.power(np.array([1.0 / p for p in range(1, n)]), 0.75)
      w /= np.sum(w)
      ret['msa_idx'] = int(np.argmax(np.random.multinomial(1, w))) + 1
    ret.update(_make_msa_features(sequences,
                                  msa_idx=ret['msa_idx'],
                                  max_msa_depth=self.max_msa_depth))
    return ret

  def get_structure_label_npz(self, protein_id):
    if self._fstat(f'{self.pdb_dir}/{protein_id}.npz'):
      with self._fileobj(f'{self.pdb_dir}/{protein_id}.npz') as f:
        structure = np.load(BytesIO(f.read()))
        ret = dict(coord=torch.from_numpy(structure['coord']),
                   coord_mask=torch.from_numpy(structure['coord_mask']))
        if 'bfactor' in structure:
          ret.update(coord_plddt=torch.from_numpy(structure['bfactor']))
        return ret
    return {}

  def get_seq_features(self, protein_id):
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
                              max_seq_len=self.max_seq_len)

  @staticmethod
  def batch_clips_fn(batch,
                     min_crop_len=None,
                     max_crop_len=None,
                     crop_probability=0.0,
                     crop_algorithm='random',
                     **kwargs):

    def _random_sampler(b, n, batch):

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

      l = _crop_length(n, np.random.random() < crop_probability)
      logger.debug('min_crop_len=%s, max_crop_len=%s, n=%s, l=%s', min_crop_len,
                   max_crop_len, n, l)
      i, j = 0, l
      if not 'coord_mask' in batch[b] or torch.any(batch[b]['coord_mask']):
        while True:
          i = np.random.randint(n - l + 1)
          j = i + l
          if not 'coord_mask' in batch[b] or torch.any(
              batch[b]['coord_mask'][i:j]):
            break
      return dict(i=i, j=j, l=n)

    def _domain_sampler(b, n, batch):

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
      assert 'coord' in batch[b] and 'coord_mask' in batch[b]

      if exists(
          max_crop_len
      ) and n <= max_crop_len and crop_probability < np.random.random():
        assert not exists(min_crop_len) or min_crop_len < n
        return None

      ca_idx = residue_constants.atom_order['CA']
      ca_coord, ca_coord_mask = batch[b]['coord'][
          ..., ca_idx, :], batch[b]['coord_mask'][..., ca_idx]
      logger.debug('domain_sampler: batch=%d, seq_len=%d', b, n)
      weights = domain_parser(ca_coord, ca_coord_mask, max_len=max_crop_len)
      intra_domain_probability = kwargs.get('intra_domain_probability', 0)
      while True:
        i = _cascade_sampler(weights)
        if np.random.random() < intra_domain_probability:
          logger.debug('domain_intra: batch=%d, seq_len=%d, i=%d', b, n, i)
          half = max_crop_len // 2 + max_crop_len % 2
          if i + 1 < n - i:  # i <= n // 2
            i = max(0, i - half)
            j = min(i + max_crop_len, n)
          else:
            i = min(n, i + half)
            j = max(0, i - max_crop_len)
        else:
          j = _domain_next(weights,
                           i,
                           min_len=min_crop_len,
                           max_len=max_crop_len)
          logger.debug('domain_next: batch=%d, seq_len=%d, i=%d, j=%s', b, n, i,
                       str(j))
        if j is not None and torch.any(ca_coord_mask[min(i, j):max(i, j)]):
          break
      return dict(i=min(i, j), j=max(i, j), l=n)

    logger.debug('batch_clips_fn: crop_algorithm=%s', crop_algorithm)
    sampler_list = dict(random=_random_sampler, domain=_domain_sampler)

    assert crop_algorithm in sampler_list

    clips = {}

    for k in range(len(batch)):
      n = len(batch[k]['str_seq'])
      if (exists(max_crop_len) and
          max_crop_len < n) or (exists(min_crop_len) and min_crop_len < n and
                                crop_probability > 0):
        sampler_fn = sampler_list[crop_algorithm]
        if crop_algorithm == 'domain' and ('coord' not in batch[k] or
                                           'coord_mask' not in batch[k]):
          sampler_fn = sampler_list['random']
          logger.debug('batch_clips_fn: crop_algorithm=%s downgrad to: random',
                       crop_algorithm)
        clip = sampler_fn(k, n, batch)
        if clip:
          clips[k] = clip

    return clips

  @staticmethod
  def collate_fn(batch,
                 feat_flags=None,
                 min_crop_len=None,
                 max_crop_len=None,
                 crop_probability=0.0,
                 crop_algorithm='random',
                 **kwargs):
    if exists(max_crop_len) and exists(min_crop_len):
      assert max_crop_len >= min_crop_len

    clips = ProteinStructureDataset.batch_clips_fn(
        batch,
        min_crop_len=min_crop_len,
        max_crop_len=max_crop_len,
        crop_probability=crop_probability,
        crop_algorithm=crop_algorithm,
        **kwargs)
    for k, clip in clips.items():
      i, j = clip['i'], clip['j']

      batch[k]['str_seq'] = batch[k]['str_seq'][i:j]
      for field in ('seq', 'seq_index', 'mask', 'coord', 'coord_mask',
                    'coord_plddt'):
        if field in batch[k]:
          batch[k][field] = batch[k][field][i:j, ...]
      for field in ('str_msa',):
        if field in batch[k]:
          batch[k][field] = [v[i:j] for v in batch[k][field]]
      for field in ('msa', 'del_msa'):
        if field in batch[k]:
          batch[k][field] = batch[k][field][:, i:j, ...]

    fields = ('pid', 'resolu', 'seq', 'seq_index', 'mask', 'str_seq', 'msa_idx')
    pids, resolutions, seqs, seqs_idx, masks, str_seqs, msa_idx = list(
        zip(*[[b[k] for k in fields] for b in batch]))
    lengths = tuple(len(s) for s in str_seqs)
    max_batch_len = max(lengths)

    padded_seqs = pad_for_batch(seqs, max_batch_len, 'seq')
    padded_seqs_idx = pad_for_batch(seqs_idx, max_batch_len, 'seq_index')
    padded_masks = pad_for_batch(masks, max_batch_len, 'msk')

    ret = dict(pid=pids,
               msa_idx=torch.as_tensor(msa_idx),
               resolution=resolutions,
               seq=padded_seqs,
               seq_index=padded_seqs_idx,
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
        if not field in batch[0]:
          continue
        padded_values = pad_for_batch([b[field] for b in batch], max_batch_len,
                                      field)
        ret[field] = padded_values

    if feat_flags & FEAT_MSA:
      fields = ('msa', 'str_msa', 'del_msa', 'num_msa')
      msas, str_msas, del_msas, num_msa = list(
          zip(*[[b[k] for k in fields] for b in batch]))

      padded_msas = pad_for_batch(msas, max_batch_len, 'msa')
      padded_dels = pad_for_batch(del_msas, max_batch_len, 'del_msa')
      ret.update(msa=padded_msas,
                 str_msa=str_msas,
                 del_msa=padded_dels,
                 num_msa=torch.as_tensor(num_msa))

    if clips:
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
    for msa in items:
      z = torch.ones((msa.shape[0], batch_length - msa.shape[1]),
                     dtype=msa.dtype) * residue_constants.HHBLITS_AA_TO_ID['X']
      c = torch.cat((msa, z), dim=1)
      batch.append(c)
  elif dtype == 'del_msa':
    for del_msa in items:
      z = torch.zeros((del_msa.shape[0], batch_length - del_msa.shape[1]),
                      dtype=del_msa.dtype)
      c = torch.cat((del_msa, z), dim=1)
      batch.append(c)
  else:
    raise ValueError('Not implement yet!')
  batch = torch.stack(batch, dim=0)
  return batch


def load(data_dir,
         data_idx=None,
         data_rm_mask_prob=0.0,
         msa_as_seq_prob=0.0,
         min_crop_len=None,
         max_crop_len=None,
         crop_probability=0,
         crop_algorithm='random',
         feat_flags=FEAT_ALL,
         **kwargs):
  max_msa_depth = 128
  if 'max_msa_depth' in kwargs:
    max_msa_depth = kwargs.pop('max_msa_depth')

  data_dir = data_dir.split(',')
  if exists(data_idx):
    data_idx = data_idx.split(',')
  else:
    data_idx = [None] * len(data_dir)
  assert len(data_dir) == len(data_idx)

  dataset = torch.utils.data.ConcatDataset([
      ProteinStructureDataset(data_dir[i],
                              data_idx=data_idx[i],
                              data_rm_mask_prob=data_rm_mask_prob,
                              msa_as_seq_prob=msa_as_seq_prob,
                              max_msa_depth=max_msa_depth,
                              feat_flags=feat_flags)
      for i in range(len(data_dir))
  ])
  if 'collate_fn' not in kwargs:
    collate_fn_kwargs = {}
    if 'intra_domain_probability' in kwargs:
      collate_fn_kwargs['intra_domain_probability'] = kwargs.pop(
          'intra_domain_probability')
    kwargs['collate_fn'] = functools.partial(ProteinStructureDataset.collate_fn,
                                             feat_flags=feat_flags,
                                             min_crop_len=min_crop_len,
                                             max_crop_len=max_crop_len,
                                             crop_probability=crop_probability,
                                             crop_algorithm=crop_algorithm,
                                             **collate_fn_kwargs)
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
