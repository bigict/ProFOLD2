"""Dataset for structure
 """
import os
from collections import defaultdict
import contextlib
import functools
import io
import itertools
import json
import logging
from io import BytesIO
import pathlib
import pickle
import re
import string
import zipfile
import weakref

from Bio import PDB
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from einops import rearrange, repeat

from profold2.common import residue_constants
from profold2.data.parsers import parse_fasta
from profold2.data.padding import pad_sequential, pad_rectangle
from profold2.data.utils import (
    compose_pid, decompose_pid, parse_seq_index, seq_index_join, seq_index_split,
    str_seq_index
)
from profold2.utils import default, env, exists, timing

logger = logging.getLogger(__file__)

FEAT_PDB = 0x01
FEAT_MSA = 0x02
FEAT_VAR = 0x04
FEAT_ALL = 0xff


def _msa_aligned_ratio(msa, n, threshold=None):
  r = 1. - msa.count('-') / n
  if exists(threshold) and r < threshold:
    return 0
  return r


def _msa_ident_ratio(msa, seq, threshold=None):
  r = 0
  i, j = 0, 0
  while i < len(msa) and j < len(seq):
    if not msa[i].islower():  # NOTE: gap is not upper
      if msa[i] == seq[j]:
        r += 1
      j += 1
    i += 1
  r = r / max(min(i, j), 0.1)
  if exists(threshold) and r < threshold:
    return 0
  return r


def _msa_sample_weight(msa, min_alr=None, min_ident=None):
  m, n = len(msa[0]), len(msa)

  w = np.power(np.array([1.0 / p for p in range(1, n)]), 1.0 / 3.0)
  v = np.asarray([_msa_aligned_ratio(s, m, threshold=min_alr) for s in msa[1:n]])
  w *= v
  if exists(min_ident) and min_ident > 0:
    v = np.asarray([_msa_ident_ratio(s, msa[0], threshold=min_ident) for s in msa[1:n]])
    w *= v
  return w


def _msa_yield_from_clip(msa, clip):
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


def _msa_as_seq(item, idx, str_key='msa'):
  str_msa_key, del_msa_key = f'str_{str_key}', f'del_{str_key}'
  assert idx > 0
  assert str_msa_key in item and del_msa_key in item
  assert len(item['str_seq']) == len(
      item[str_msa_key][0]
  ), (idx, item['pid'], item['str_seq'], item[str_msa_key][0])

  if 'coord' in item:
    assert item['seq'].shape[0] == item['coord'].shape[0], (item['pid'], )

  # swap(msa[1], msa[idx])
  assert len(item[str_msa_key][0]) == len(item[str_msa_key][1])
  if item['str_seq'] != item[str_msa_key][1]:
    assert item['str_seq'] != item[str_msa_key][1], (
        item['pid'], idx, item[str_msa_key][0], item[str_msa_key][1]
    )
    item[str_msa_key][0] = item[str_msa_key][1]
    item[str_msa_key][1] = item['str_seq']
    assert item['str_seq'] != item[str_msa_key][0], (
        item['pid'], idx, item[str_msa_key][0], item[str_msa_key][1]
    )
    item['str_seq'] = item[str_msa_key][0]
    assert item['str_seq'] == item[str_msa_key][0]

    i, new_order = 0, []

    while i < len(item['str_seq']):
      if item['str_seq'][i] != residue_constants.restypes_with_x_and_gap[-1]:
        new_order.append(i)
      i += 1
    logger.debug(
        '%s_as_seq: %s@%s k=%d, i=%d', str_key, item['pid'], idx, len(new_order), i
    )
    assert 0 < len(new_order) <= i, (len(new_order), i)

    if len(new_order) < i:
      item = _make_feats_shrinked(item, new_order)

    # Renew pid
    pid = item['pid']
    item['pid'] = f'{pid}@{idx}'

    if 'coord' in item:
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
          repeat(item['seq'].long(), 'i -> i n', n=restype_atom14_mask.shape[-1])
      )
      for field in ('coord', 'coord_mask', 'coord_plddt'):
        if field in item:
          item[field] = torch.einsum(
              'i n ...,i n -> i n ...', item[field], coord_exists
          )
    # Delete coords if all are invalid.
    # ca_idx = residue_constants.atom_order['CA']
    # FIXED: collate_fn may failed when batch_size > 1
    # if 'coord_mask' in item and not torch.any(item['coord_mask'][:, ca_idx]):
    #   for field in ('coord', 'coord_mask', 'coord_plddt'):
    #     if field in item:
    #       del item[field]

    # Fix seq_index
    del_seq = torch.cumsum(item[del_msa_key][0], dim=-1, dtype=item[del_msa_key].dtype)
    item['seq_index'] = item['seq_index'] + del_seq
    if 'resolu' in item:
      item['resolu'] = -1.  # delete it !

  return item


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
                         n, replace=False) if max_msa_depth > n else []
    )
  msa, del_matirx = parse_a4m(sequences)

  int_msa = []
  for sequence in msa:
    int_msa.append(
        [
            residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE[
                residue_constants.HHBLITS_AA_TO_ID[res]] for res in sequence
        ]
    )
  int_msa = torch.as_tensor(int_msa, dtype=torch.int)

  return dict(
      msa=int_msa,
      msa_mask=torch.ones_like(int_msa, dtype=torch.bool),
      str_msa=msa,
      del_msa=torch.as_tensor(del_matirx, dtype=torch.int),
      num_msa=msa_depth
  )


_label_pattern, _label_mask_pattern = '^label=(.+)', '^label_mask=(.+)'


def _make_label_features(descriptions, attr_dict, task_num=1, defval=1.0):
  def _make_label(desc):
    label, label_mask = None, None

    pid, chain, _ = decompose_pid(desc.split()[0], return_domain=True)
    for k in (pid, compose_pid(pid, chain)):
      if k in attr_dict:
        if 'label' in attr_dict[k]:
          label = attr_dict[k]['label']
        if 'label_mask' in attr_dict[k]:
          label_mask = attr_dict[k]['label_mask']

    for s in desc.split():
      r = re.match(_label_pattern, s)
      if r:
        label = json.loads(r.group(1))
      r = re.match(_label_mask_pattern, s)
      if r:
        label_mask = json.loads(r.group(1))

    if not exists(label):
      label = [defval] * task_num
    if not exists(label_mask):
      label_mask = [True] * task_num
    if not isinstance(label, list):
      label = [label]
    assert len(label) == task_num, (pid, chain, label)

    if not isinstance(label_mask, list):
      label_mask = [label_mask]
    assert len(label_mask) == task_num, (pid, chain, label)

    return label, label_mask

  label, label_mask = [], []
  for l, m in map(_make_label, descriptions):
    label.append(l)
    label_mask.append(m)
  return label, label_mask


def _make_var_pid(desc):
  var_pid, c, _ = decompose_pid(desc.split()[0], return_domain=True)
  var_pid = compose_pid(var_pid, c)
  return var_pid


_weight_pattern = '^weight=(.+)'


def _make_var_choice(var_list, attr_list, max_var_depth, defval=1.0):
  assert max_var_depth > 0

  def _var_w(var_desc):
    # 1. parse description
    for s in var_desc.split():
      r = re.match(_weight_pattern, s)
      if r:
        return float(r.group(1))

    # 2.lookup dict
    var_pid = _make_var_pid(var_desc)
    if exists(attr_list) and var_pid in attr_list:
      return attr_list[var_pid].get('weight', defval)

    # 3. default value
    return defval

  w = np.asarray([_var_w(var_desc) for var_desc, *_ in var_list])
  w /= (np.sum(w) + 1e-8)
  new_order = np.random.choice(len(var_list), size=max_var_depth, replace=False, p=w)
  return new_order


def _make_var_features(
    sequences, descriptions, var_idx=0, attr_dict=None, max_var_depth=None
):
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

  var_depth = len(sequences)
  if 0 < var_idx < var_depth:
    t = sequences[var_idx]
    sequences[var_idx] = sequences[1]
    sequences[1] = t
  if exists(max_var_depth) and len(sequences) > max_var_depth:
    n = 2 if 0 < var_idx < var_depth else 1
    if max_var_depth > n:
      var_list = [(desc, ) for desc in descriptions[n:]]  # FIXME: tuple or list
      new_order = _make_var_choice(var_list, attr_dict, max_var_depth - n)
      sequences = sequences[:n] + [sequences[i + n] for i in new_order]
      descriptions = descriptions[:n] + [descriptions[i + n] for i in new_order]
  msa, del_matirx = parse_a4m(sequences)

  int_msa = []
  for sequence in msa:
    int_msa.append(
        [
            residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE[
                residue_constants.HHBLITS_AA_TO_ID[res]] for res in sequence
        ]
    )

  variant = torch.as_tensor(int_msa, dtype=torch.int)
  variant_mask = variant != residue_constants.restypes_with_x_and_gap.index('-')
  variant_pid = [_make_var_pid(desc) for desc in descriptions]
  return dict(
      variant=variant,
      variant_pid=variant_pid,
      variant_mask=variant_mask,
      variant_task_mask=variant_mask[..., None],
      str_var=msa,
      desc_var=descriptions,
      del_var=torch.as_tensor(del_matirx, dtype=torch.int),
      num_var=var_depth
  )


def _make_task_mask(
    mask, chain_name_list, chain_length_list, task_def=None, task_num=1
):
  variant_task_mask = [[] for _ in range(task_num)]
  if exists(task_def):
    task_idx = functools.reduce(lambda x, y: x + [x[-1] + y], chain_length_list, [0])
    for i, chain in enumerate(chain_name_list):
      chain, *_ = chain.split('@')
      task_list = set(task_def.get(chain, []))
      for j in range(task_num):
        if j in task_list:
          variant_task_mask[j].append(mask[task_idx[i]:task_idx[i + 1]])
        else:
          variant_task_mask[j].append(
              torch.zeros(chain_length_list[i], dtype=torch.bool, device=mask.device)
          )
    for j in range(task_num):
      variant_task_mask[j] = torch.cat(variant_task_mask[j], dim=0)
  else:
    for j in range(task_num):
      variant_task_mask[j] = mask
  return torch.stack(variant_task_mask, dim=-1)


def _make_seq_features(
    sequence,
    description,
    seq_color=1,
    seq_entity=None,
    seq_sym=None,
    max_seq_len=None
):
  residue_index = torch.arange(len(sequence), dtype=torch.int)
  residue_index = parse_seq_index(description, sequence, residue_index)

  sequence = sequence[:max_seq_len]
  residue_index = residue_index[:max_seq_len]
  seq_entity = torch.full(
      (len(sequence), ), default(seq_entity, seq_color), dtype=torch.int
  )
  seq_sym = torch.full((len(sequence), ), default(seq_sym, 1), dtype=torch.int)
  seq_color = torch.full((len(sequence), ), seq_color, dtype=torch.int)

  seq = torch.as_tensor(
      residue_constants.sequence_to_onehot(
          sequence=sequence,
          mapping=residue_constants.restype_order_with_x,
          map_unknown_to_x=True
      ),
      dtype=torch.int
  ).argmax(-1).to(torch.int)
  #residue_index = torch.arange(len(sequence), dtype=torch.int)
  str_seq = ''.join(
      map(
          lambda a: a if a in residue_constants.restype_order_with_x else
          residue_constants.restypes_with_x[-1], sequence
      )
  )
  mask = torch.ones(len(sequence), dtype=torch.bool)

  return dict(
      seq=seq,
      seq_index=residue_index,
      seq_color=seq_color,
      seq_entity=seq_entity,
      seq_sym=seq_sym,
      str_seq=str_seq,
      mask=mask
  )


def _make_pdb_features(
    pdb_id,
    pdb_string,
    pdb_type='pdb',
    seq_color=1,
    seq_entity=None,
    seq_sym=None,
    max_seq_len=None
):
  assert pdb_type in ('pdb', 'cif')
  if pdb_type == 'pdb':
    parser = PDB.PDBParser(QUIET=True)
  else:
    parser = PDB.MMCIFParser(QUIET=True)
  handle = io.StringIO(pdb_string)
  full_structure = parser.get_structure('', handle)
  model_structure = next(full_structure.get_models())
  chains = list(model_structure.get_chains())
  assert len(chains) == 1

  _unassigned = {'.', '?'}  # pylint: disable=invalid-name

  seq, domains = [], []
  coord_list, coord_mask_list, bfactor_list = [], [], []

  int_resseq_start, int_resseq_end = None, None

  for aa in chains[0].get_residues():
    hetero, int_resseq, icode = aa.id

    if hetero and hetero != ' ':
      continue
    if icode in _unassigned:
      icode = ' '
    if icode and icode != ' ':
      continue

    residue_id = aa.get_resname()
    if residue_id == 'MSE':
      residue_id = 'MET'

    if not exists(int_resseq_start):
      int_resseq_start = int_resseq
    if exists(int_resseq_end) and int_resseq - int_resseq_end > 1:
      domains.append((int_resseq_start, int_resseq_end))
      int_resseq_start = int_resseq
    if not exists(int_resseq_end) or int_resseq != int_resseq_end:
      # sequence
      resname = residue_constants.restype_3to1.get(
          residue_id, residue_constants.restypes_with_x[-1]
      )
      seq.append(resname)

      # cordinates
      labels = np.zeros((14, 3), dtype=np.float32)
      label_mask = np.zeros((14, ), dtype=np.bool_)
      bfactors = np.zeros((14, ), dtype=np.float32)

      if residue_id in residue_constants.restype_name_to_atom14_names:
        res_atom14_list = residue_constants.restype_name_to_atom14_names[residue_id]  # pylint: disable=line-too-long
      else:
        res_atom14_list = residue_constants.restype_name_to_atom14_names[residue_constants.unk_restype]  # pylint: disable=line-too-long
      for atom in aa.get_atoms():
        try:
          atom14idx = res_atom14_list.index(atom.id)
          coord = np.asarray(atom.get_coord())
          if np.any(np.isnan(coord)):
            continue
          labels[atom14idx] = coord
          bfactor = atom.get_bfactor() / 100.
          bfactors[atom14idx] = bfactor
          label_mask[atom14idx] = True
        except ValueError as e:
          logger.debug(e)
      coord_list.append(labels)
      coord_mask_list.append(label_mask)
      bfactor_list.append(bfactors)
    int_resseq_end = int_resseq

  domains.append((int_resseq_start, int_resseq_end))

  # sequence
  sequence = ''.join(seq)

  # description
  l = sum(map(lambda x: x[1] - x[0] + 1, domains))
  domain_str = ','.join(f'{i}-{j}' for i, j in domains)
  description = f'{pdb_id} domains:{domain_str} length={l}'

  ret = _make_seq_features(
      sequence,
      description,
      seq_color=seq_color,
      seq_entity=seq_entity,
      seq_sym=seq_sym,
      max_seq_len=max_seq_len
  )

  # make npz
  coord, coord_mask, bfactor = map(
      functools.partial(np.stack, axis=0), (coord_list, coord_mask_list, bfactor_list)
  )

  ret.update(
      sequence=seq,
      description=description,
      coord=torch.from_numpy(coord),
      coord_mask=torch.from_numpy(coord_mask),
      coord_plddt=torch.from_numpy(bfactor)
  )
  return ret


def _make_anchor_features(fgt_color, fgt_entity, feat):
  ret = {}

  entity_color_cnt, entity_length = {}, {}
  for entity_id in filter(lambda x: x > 0, torch.unique(fgt_entity)):
    input_color = torch.unique(fgt_color[fgt_entity == entity_id])
    if any(c in feat['seq_color'] for c in input_color):
      entity_color_cnt[int(entity_id)] = len(input_color)
      entity_length[int(entity_id)] = int(torch.sum(feat['seq_entity'] == entity_id))

  if any(n > 1 for n in entity_color_cnt.values()):  # is a homomer
    # filter entities by count
    if entity_color_cnt:
      group_by_cnt = lambda x: x[1]
      _, entity_color_cnt = next(
          itertools.groupby(
              sorted(entity_color_cnt.items(), key=group_by_cnt), key=group_by_cnt
          )
      )
      entity_color_cnt = dict(entity_color_cnt)

    # filter entities by length
    if len(entity_color_cnt) > 1:
      max_length = max(entity_length[entity_id] for entity_id in entity_color_cnt)
      entity_color_cnt = {
          entity_id: cnt
          for entity_id, cnt in entity_color_cnt.items()
          if entity_length[entity_id] == max_length
      }

    # random select one
    if len(entity_color_cnt) > 1:
      entity_id = np.random.choice(list(entity_color_cnt))
      entity_color_cnt = {entity_id: entity_color_cnt[entity_id]}

    if entity_color_cnt:
      assert len(entity_color_cnt) == 1
      entity_id = next(iter(entity_color_cnt))
      color_list = torch.unique(feat['seq_color'][feat['seq_entity'] == entity_id])

      if len(color_list) > 1:
        color_length = {}
        for c in color_list:
          color_length[int(c)] = int(torch.sum(feat['seq_color'] == c))
        max_length = max(color_length.values())
        color_list = [c for c in color_list if color_length[int(c)] == max_length]

      # random select one
      ret.update(seq_anchor=np.random.choice(color_list))
  logger.debug('_make_anchor_features: %s', int(ret.get('seq_anchor', 0)))

  return ret


def _make_feats_shrinked(
    item, new_order, seq_feats=None, msa_feats=None, var_feats=None
):
  # Update seq related feats
  item['str_seq'] = ''.join(item['str_seq'][k] for k in new_order)

  for field in ('str_msa', 'str_var'):
    if field in item:
      for j in range(len(item[field])):
        item[field][j] = ''.join(item[field][j][k] for k in new_order)

  # Update tensors
  new_order = torch.as_tensor(new_order)

  for field in default(
      seq_feats, (
          'seq', 'seq_index', 'seq_color', 'seq_entity', 'seq_sym', 'mask',
          'coord', 'coord_mask', 'coord_plddt'
      )
  ):
    if field in item:
      item[field] = torch.index_select(item[field], 0, new_order)
  for field in ('coord_pae', ):
    if field in item:
      item[field] = torch.index_select(item[field], 0, new_order)
      item[field] = torch.index_select(item[field], 1, new_order)
  for field in default(msa_feats, ('msa', 'msa_mask', 'del_msa')):
    if field in item:
      item[field] = torch.index_select(item[field], 1, new_order)
  for field in default(
      var_feats, ('variant', 'del_var', 'variant_mask', 'variant_task_mask')
  ):
    if field in item:
      item[field] = torch.index_select(item[field], 1, new_order)

  return item


def _protein_clips_fn(
    protein,
    min_crop_len=None,
    max_crop_len=None,
    min_crop_pae=False,
    max_crop_plddt=False,
    crop_probability=0.0,
    crop_algorithm='random',
    **kwargs
):
  def _crop_length(n, crop):
    assert exists(min_crop_len) or exists(max_crop_len)

    if not exists(max_crop_len):
      assert min_crop_len < n
      return np.random.randint(min_crop_len, n + 1) if crop else n
    elif not exists(min_crop_len):
      assert max_crop_len < n
      return max_crop_len
    assert min_crop_len <= max_crop_len and (min_crop_len < n or max_crop_len < n)
    return np.random.randint(min_crop_len,
                             min(n, max_crop_len) +
                             1) if crop else min(max_crop_len, n)

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
    ca_coord, ca_coord_mask = protein['coord'][...,
                                               ca_idx, :], protein['coord_mask'][...,
                                                                                 ca_idx]
    logger.debug('knn_sampler: seq_len=%d', n)

    min_len = 32  # default(min_crop_len, 32)
    # max_len = default(max_crop_len, 256)
    max_len = _crop_length(n, np.random.random() < crop_probability)
    gamma = 0.004

    ridx = np.random.randint(n)
    eps = 1e-1
    dist2 = torch.sum(
        torch.square(
            rearrange(ca_coord, 'i d -> i () d') - rearrange(ca_coord, 'j d -> () j d')
        ),
        dim=-1
    )
    mask = rearrange(ca_coord_mask, 'i -> i ()') * rearrange(ca_coord_mask, 'j -> () j')
    dist2 = dist2.masked_fill(~mask, torch.max(dist2))
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
    ) or n > env('profold2_data_knn_sampler_max_length', defval=65536, type=int):
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


def _protein_crop_fn(protein, clip):
  assert clip

  if 'd' in clip:
    return _make_feats_shrinked(protein, clip['d'])

  i, j = clip['i'], clip['j']
  protein['str_seq'] = protein['str_seq'][i:j]
  for field in (
      'seq', 'seq_index', 'mask', 'coord', 'coord_mask', 'coord_plddt', 'seq_color',
      'seq_entity', 'seq_sym'
  ):
    if field in protein:
      protein[field] = protein[field][i:j, ...]
  for field in ('coord_pae', ):
    if field in protein:
      protein[field] = protein[field][i:j, i:j]
  for field in ('str_msa', ):
    if field in protein:
      protein[field] = [v[i:j] for v in protein[field]]
  for field in ('msa', 'msa_mask', 'del_msa'):
    if field in protein:
      protein[field] = protein[field][:, i:j, ...]
  for field in ('str_var', ):
    if field in protein:
      protein[field] = [v[i:j] for v in protein[field]]
  for field in ('variant', 'del_var', 'variant_mask', 'variant_task_mask'):
    if field in protein:
      protein[field] = protein[field][:, i:j, ...]

  return protein


def _protein_crop_fmt(clip):
  assert exists(clip), clip
  if 'd' in clip:
    clip['d'] = str_seq_index(torch.as_tensor(clip['d']))
  return clip


class FileSystem(contextlib.AbstractContextManager):
  def __init__(self, data_dir):
    if zipfile.is_zipfile(data_dir):
      self.data_dir = zipfile.ZipFile(data_dir)
    else:
      self.data_dir = pathlib.Path(data_dir)

  def __enter__(self):
    return self

  def __exit__(self, *exc_details):
    del exc_details
    self.close()

  def abspath(self, filename):
    if os.path.isabs(filename):
      return filename
    elif isinstance(self.data_dir, zipfile.ZipFile):
      raise ValueError('Not implementation!')
    return self.data_dir / filename

  @contextlib.contextmanager
  def open(self, filename):
    if os.path.isabs(filename):
      with open(filename, mode='rb') as f:
        yield f
    elif isinstance(self.data_dir, zipfile.ZipFile):
      with self.data_dir.open(filename, 'r') as f:
        yield f
    else:
      with open(self.data_dir / filename, mode='rb') as f:
        yield f

  def close(self):
    if isinstance(self.data_dir, zipfile.ZipFile):
      self.data_dir.close()

  def exists(self, filename):
    if os.path.isabs(filename):
      return os.path.exists(filename)
    if isinstance(self.data_dir, zipfile.ZipFile):
      try:
        self.data_dir.getinfo(filename)
        return True
      except KeyError as e:
        del e
        return False
    return (self.data_dir / filename).exists()

  def textise(self, data, encoding='utf-8'):
    if isinstance(data, bytes):
      return data.decode(encoding)
    return data


class FoldcompDB(object):
  def __init__(self, fs, db_uri, db_idx, key_fmt=None, db_open=True):
    super().__init__()

    self.db_uri = fs.abspath(db_uri)
    self.db_idx = db_idx
    self.key_fmt = key_fmt

    self.db_hdr = None
    # if db_open:
    #   self.open()

  def open(self):
    if exists(self.db_hdr):
      self.close()
    assert not exists(self.db_hdr)
    with timing(f'FoldcompDB.open {self.db_uri}', logger.debug):
      import foldcomp  # pylint: disable=import-outside-toplevel
      self.db_hdr = foldcomp.open(self.db_uri)
    return self.db_hdr

  def close(self):
    if exists(self.db_hdr):
      logger.debug('FoldcompDB.close %s', self.db_uri)
      self.db_hdr.close()
      self.db_hdr = None

  def __getitem__(self, protein_id):
    if not exists(self.db_hdr):
      self.open()
    assert exists(self.db_hdr)

    if exists(self.key_fmt):
      protein_id = self.key_fmt.format(protein_id)

    idx = self.db_idx[protein_id]
    _, pdb_string = self.db_hdr[idx]
    pdb_type = 'pdb'
    return pdb_type, pdb_string


class ProteinSequenceDataset(torch.utils.data.Dataset):
  """Construct a `Dataset` from sequences
   """
  def __init__(
      self,
      sequences,
      descriptions=None,
      domain_as_seq=False,
      msa=None,
      msa_as_seq=False
  ):
    self.sequences = sequences
    self.domain_as_seq = domain_as_seq
    self.descriptions = descriptions
    self.msa = msa
    assert not exists(self.descriptions) or len(self.sequences
                                               ) == len(self.descriptions)
    assert not exists(self.msa) or len(self.sequences) == len(self.msa)
    assert (not msa_as_seq) or exists(msa)
    self.msa_depth = np.cumsum(
        np.asarray([len(m) for m in self.msa])
    ) if msa_as_seq else None

  def __getitem__(self, idx):
    seq_idx, msa_idx = idx, 0
    if exists(self.msa_depth):
      seq_idx = np.sum(self.msa_depth < idx)
      msa_idx = idx - (self.msa_depth[seq_idx - 1] if seq_idx > 0 else 0)  # pylint: disable=unsubscriptable-object
    input_sequence = self.sequences[seq_idx]
    seq = torch.as_tensor(
        residue_constants.sequence_to_onehot(
            sequence=input_sequence,
            mapping=residue_constants.restype_order_with_x,
            map_unknown_to_x=True
        ),
        dtype=torch.int
    ).argmax(-1).to(torch.int)
    residue_index = torch.arange(len(input_sequence), dtype=torch.int)
    str_seq = ''.join(
        map(
            lambda a: a if a in residue_constants.restype_order_with_x else
            residue_constants.restypes_with_x[-1], input_sequence
        )
    )
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
          (
              torch.zeros(1, dtype=torch.int),
              torch.cumsum(residue_index[:-1] + 1 != residue_index[1:], dim=-1)
          )
      )
    ret = dict(
        pid=desc,
        seq=seq,
        seq_index=residue_index,
        seq_color=seq_color,
        str_seq=str_seq,
        mask=mask
    )
    if exists(self.msa) and exists(self.msa[seq_idx]):
      ret.update(_make_msa_features(self.msa[seq_idx], msa_idx=msa_idx))
    if msa_idx > 0:
      ret = _msa_as_seq(ret, msa_idx)
    return ret

  def __len__(self):
    if exists(self.msa_depth):
      return int(self.msa_depth[-1])  # pylint: disable=unsubscriptable-object
    return len(self.sequences)

  @staticmethod
  def collate_fn(batch):
    return _collate_fn(batch, feat_flags=FEAT_ALL & (~FEAT_PDB))


class ProteinStructureDataset(torch.utils.data.Dataset):
  """Construct a `Dataset` from a zip or filesystem
   """
  def __init__(
      self,
      data_dir,
      data_idx=None,
      mapping_idx=None,
      chain_idx=None,
      attr_idx=None,
      var_dir=None,
      data_crop_fn=None,
      max_msa_depth=128,
      max_var_depth=1024,
      pseudo_linker_prob=0.0,
      pseudo_linker_shuffle=True,
      data_rm_mask_prob=0.0,
      msa_as_seq_prob=0.0,
      msa_as_seq_topn=None,
      msa_as_seq_clustering=False,
      msa_as_seq_min_alr=0.75,
      msa_as_seq_min_ident=0.0,
      var_task_num=1,
      var_as_seq_prob=0.0,
      var_as_seq_clustering=False,
      var_as_seq_min_alr=0.75,
      var_as_seq_min_ident=0.0,
      var_as_seq_min_label=None,
      feat_flags=FEAT_ALL & (~FEAT_MSA)
  ):
    super().__init__()

    self.data_dir = data_dir
    self.data_crop_fn = data_crop_fn
    self.max_msa_depth = max_msa_depth
    self.max_var_depth = max_var_depth
    self.pseudo_linker_prob = pseudo_linker_prob
    self.pseudo_linker_shuffle = pseudo_linker_shuffle
    self.data_rm_mask_prob = data_rm_mask_prob
    self.msa_as_seq_prob = msa_as_seq_prob
    self.msa_as_seq_topn = msa_as_seq_topn
    self.msa_as_seq_clustering = msa_as_seq_clustering
    self.msa_as_seq_min_alr = msa_as_seq_min_alr
    self.msa_as_seq_min_ident = msa_as_seq_min_ident
    self.var_task_num = var_task_num
    self.var_as_seq_prob = env(
        'profold2_data_var_as_seq_prob', defval=var_as_seq_prob, type=float
    )
    self.var_as_seq_clustering = var_as_seq_clustering
    self.var_as_seq_min_alr = env(
        'profold2_data_var_as_seq_min_alr', defval=var_as_seq_min_alr, type=float
    )
    self.var_as_seq_min_ident = env(
        'profold2_data_var_as_seq_min_ident', defval=var_as_seq_min_ident, type=float
    )
    self.var_as_seq_min_label = env(
        'profold2_data_var_as_seq_min_label', defval=var_as_seq_min_label, type=float
    )
    self.feat_flags = feat_flags
    with FileSystem(self.data_dir) as fs:
      data_idx = default(data_idx, 'name.idx')
      logger.info('load idx data from: %s', data_idx)
      with fs.open(data_idx) as f:
        self.pids = list(
            map(
                lambda x: x.split(),
                filter(
                    lambda x: len(x) > 0 and not x.startswith('#'),
                    map(lambda x: fs.textise(x).strip(), f)
                )
            )
        )

      self.mapping, self.cluster = {}, defaultdict(list)
      mapping_idx = default(mapping_idx, 'mapping.idx')
      if fs.exists(mapping_idx):
        with fs.open(mapping_idx) as f:
          for line in filter(
              lambda x: len(x) > 0, map(lambda x: fs.textise(x).strip(), f)
          ):
            v, k = line.split()
            self.mapping[k] = v
            self.cluster[v].append(k)

      self.resolu = {}
      if fs.exists('resolu.idx'):
        with fs.open('resolu.idx') as f:
          for line in filter(
              lambda x: len(x) > 0, map(lambda x: fs.textise(x).strip(), f)
          ):
            k, v = line.split()
            self.resolu[k] = float(v)

      self.chain_list = defaultdict(list)
      chain_idx = default(chain_idx, 'chain.idx')
      if fs.exists(chain_idx):
        logger.info('load chain data from: %s', chain_idx)
        with fs.open(chain_idx) as f:
          for line in filter(
              lambda x: len(x) > 0, map(lambda x: fs.textise(x).strip(), f)
          ):
            pid, *chains = line.split()
            self.chain_list[pid].append(chains)

      self.attr_list = {}
      attr_idx = default(attr_idx, 'attr.idx')
      if fs.exists(attr_idx):
        logger.info('load attr data from: %s', attr_idx)
        with fs.open(attr_idx) as f:
          for line in filter(
              lambda x: len(x) > 0, map(lambda x: fs.textise(x).strip(), f)
          ):
            k, v = line.split(maxsplit=1)
            self.attr_list[k] = json.loads(v)

      if fs.exists('pdb.uri'):
        with fs.open('pdb.uri') as f:
          pdb_uri = fs.textise(f.read()).strip()
        assert fs.exists(pdb_uri)

        assert fs.exists('pdb.pkl')
        logger.info('load pdb.idx from pdb.pkl')
        with fs.open('pdb.pkl') as f:
          pdb_idx = pickle.load(f)
        self.pdb_db = FoldcompDB(fs, pdb_uri, pdb_idx)
        weakref.finalize(self.pdb_db, self.pdb_db.close)

    self.fasta_dir = env('profold2_data_fasta_dir', defval='fasta')
    self.pdb_dir = env('profold2_data_pdb_dir', defval='npz')

    self.msa_list = ['BFD30_E-3']
    self.var_dir = env('profold2_data_var_dir', defval=default(var_dir, 'var'))

  def __getstate__(self):
    d = self.__dict__
    # if isinstance(self.data_dir, zipfile.ZipFile):
    #   d['data_dir'] = self.data_dir.filename
    logger.debug('%s is pickled ...', d['data_dir'])
    if 'pdb_db' in d:
      d['pdb_db'].close()
    return d

  def __setstate__(self, d):
    logger.debug('%s is unpickled ...', d['data_dir'])
    # if zipfile.is_zipfile(d['data_dir']):
    #   d['data_dir'] = zipfile.ZipFile(d['data_dir'])  # pylint: disable=consider-using-with
    if 'pdb_db' in d:
      d['pdb_db'].open()
    self.__dict__ = d

  def __getitem__(self, idx):
    idx, k = idx if isinstance(idx, tuple) else (idx, None)
    pids = self.pids[idx]
    if not exists(k):
      k = np.random.randint(len(pids))
    pid = pids[k]

    with timing(f'ProteinStructureDataset.__getitem__ {idx}', logger.debug):
      if np.random.random() < self.pseudo_linker_prob:
        chains = self.get_chain_list(pid)
      else:
        chains = None

      if not exists(chains) or len(chains) == 1:
        ret = self.get_monomer(pid, crop_fn=self.data_crop_fn)
      else:
        ret = self.get_multimer(pid, chains)

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
      item = _make_feats_shrinked(item, new_order)
    return item

  def data_rm_mask(self, item):
    if 'coord_mask' in item:
      i, new_order = 0, []

      while i < item['coord_mask'].shape[0]:
        if torch.any(item['coord_mask'][i]):
          new_order.append(i)
        i += 1
      logger.debug('data_rm_mask: %s k=%d, i=%d', item['pid'], len(new_order), i)

      assert 0 <= len(new_order) <= i, (len(new_order), i)
      if 0 < len(new_order) < i:
        item = _make_feats_shrinked(item, new_order)
    return item

  @contextlib.contextmanager
  def _setattr(self, **kwargs_new):
    kwargs_old = {}
    for name, value in kwargs_new.items():
      assert hasattr(self, name)
      kwargs_old[name] = getattr(self, name)
      setattr(self, name, value)
    yield self
    for name, value in kwargs_old.items():
      setattr(self, name, value)

  def get_monomer(self, pid, seq_color=1, seq_entity=None, seq_sym=None, crop_fn=None):
    # CATH format pid
    pid, chain, domains = decompose_pid(pid, return_domain=True)
    if exists(domains):
      domains = list(seq_index_split(domains))
    pid = compose_pid(pid, chain)

    pkey = self.mapping[pid] if pid in self.mapping else pid
    with FileSystem(self.data_dir) as fs:
      ret = dict(
          pid=pid, msa_idx=0, clip=None, resolu=self.get_resolution(pid), seq_anchor=0
      )
      ret.update(
          self.get_structure_label_npz(
              fs, pkey, pid, seq_color=seq_color, seq_entity=seq_entity, seq_sym=seq_sym
          )
      )
      if exists(domains):
        if self.feat_flags & FEAT_MSA:
          ret.update(self.get_msa_features_new(fs, pkey))
        if self.feat_flags & FEAT_VAR:
          ret.update(self.get_var_features_new(fs, pkey))
        ret = self.data_from_domain(ret, domains)
      if exists(crop_fn):
        clip = crop_fn(ret)
        if exists(clip):
          ret = _protein_crop_fn(ret, clip)
          ret['clip'] = clip
      if not exists(domains) and (self.feat_flags & FEAT_MSA):
        ret.update(self.get_msa_features_new(fs, pkey, ret.get('clip')))
      if not exists(domains) and (self.feat_flags & FEAT_VAR):
        ret.update(self.get_var_features_new(fs, pkey, ret.get('clip')))

    if exists(domains):
      # CATH update pid
      ret['pid'] = compose_pid(pid, None, seq_index_join(domains))

    if 'msa_idx' in ret and ret['msa_idx'] != 0:
      ret = _msa_as_seq(ret, ret['msa_idx'], str_key='msa')
    elif 'var_idx' in ret and ret['var_idx'] != 0:
      ret = _msa_as_seq(ret, ret['var_idx'], str_key='var')
    elif exists(domains):
      pass
    elif np.random.random() < self.data_rm_mask_prob:
      ret = self.data_rm_mask(ret)

    if 'clip' in ret and exists(ret['clip']):
      ret['clip'] = _protein_crop_fmt(ret['clip'])
    return ret

  def _multimer_yield_cluster(self, var_pid):
    for var_pid in set(self.cluster.get(var_pid, []) + [var_pid]):
      var_pid, c = decompose_pid(var_pid)
      if self.has_chain(var_pid, c):
        yield var_pid, c

  @functools.lru_cache(
      maxsize=env('profold2_data_build_chain_lru_maxsize', defval=0, type=int)
  )
  def _multimer_build_chain_list(self, protein_id, var_list):
    def _is_aligned(k, chain_list):
      if k != protein_id and k in self.attr_list:
        for c, *_ in chain_list:
          x = self.get_chain_list(compose_pid(k, c))
          # FIX: some chains may be removed from chain.idx
          if exists(x) and len(set(x) & set(chain_list)) == len(x):
            return True
      return False

    var_chain_list = defaultdict(list)
    for var_pid in var_list:
      for pid, c in self._multimer_yield_cluster(var_pid):
        var_chain_list[pid].append(c)

    with timing(
        f'ProteinStructureDataset.filter_chain_list {protein_id}', logger.debug
    ):
      var_chain_list = [(k, v) for k, v in var_chain_list.items() if _is_aligned(k, v)]
    return var_chain_list

  def get_multimer(self, protein_id, chains):
    assert len(chains) > 1

    pid, selected_chain = decompose_pid(protein_id)  # pylint: disable=unbalanced-tuple-unpacking
    assert selected_chain in chains
    # task_definition
    if pid in self.attr_list and 'task_def' in self.attr_list[pid]:
      task_def = self.attr_list[pid]['task_def']
    else:
      task_def = None

    # shuffle the chains
    if self.pseudo_linker_shuffle:
      np.random.shuffle(chains)

    seq_index_offset, seq_index_gap = 0, 128
    seq_entity_map, seq_sym_map = defaultdict(int), defaultdict(int)

    # Concat all the feats
    ret = {'seq_anchor': 0, 'clip': None}
    for idx, chain in enumerate(chains):
      with self._setattr(
          msa_as_seq_prob=self.msa_as_seq_prob if chain == selected_chain else 0,
          max_var_depth=None,
          var_as_seq_prob=0,
          attr_list=None
      ):
        feat = self.get_monomer(compose_pid(pid, chain), seq_color=idx + 1)
        # fix seq_entity
        assert 'str_seq' in feat
        if feat['str_seq'] not in seq_entity_map:
          seq_entity_map[feat['str_seq']] = len(seq_entity_map) + 1
        seq_sym_map[feat['str_seq']] += 1
        feat['seq_entity'] = torch.ones_like(feat['seq_entity']
                                            ) * seq_entity_map[feat['str_seq']]
        feat['seq_sym'] = torch.ones_like(feat['seq_sym']
                                            ) * seq_sym_map[feat['str_seq']]
      # Sequence related
      for field in ('str_seq', ):
        ret[field] = ret.get(field, '') + feat[field]
      assert 'seq' in feat
      for field in (
          'seq', 'seq_color', 'seq_entity', 'seq_sym', 'mask', 'coord', 'coord_mask'
      ):
        assert field in feat, (field, pid, chain)
        if field not in ret:
          ret[field] = feat[field]
        else:
          ret[field] = torch.cat((ret[field], feat[field]), dim=0)
      for field in ('coord_plddt', ):
        assert 'coord_mask' in feat, (pid, chain)
        if field not in feat:
          feat['coord_plddt'] = torch.ones_like(feat['coord_mask'], dtype=torch.float32)
        if field not in ret:
          ret[field] = feat[field]
        else:
          ret[field] = torch.cat((ret[field], feat[field]), dim=0)

      for field in ('seq_index', ):
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
              (
                  ret['msa'],
                  torch.full((n - m, seq_len), gap_idx, dtype=ret['msa'].dtype)
              ),
              dim=0
          )
          ret['msa_mask'] = torch.cat(
              (
                  ret['msa_mask'],
                  torch.zeros(n - m, seq_len, dtype=ret['msa_mask'].dtype)
              ),
              dim=0
          )
          ret['del_msa'] = torch.cat(
              (ret['del_msa'], torch.zeros(n - m, seq_len)), dim=0
          )
        elif 0 <= n < m:
          seq_len = len(feat['str_msa'][0])
          feat['str_msa'] += ['-' * seq_len] * (m - n)
          feat['msa'] = torch.cat(
              (
                  feat['msa'],
                  torch.full((m - n, seq_len), gap_idx, dtype=feat['msa'].dtype)
              ),
              dim=0
          )
          feat['msa_mask'] = torch.cat(
              (
                  feat['msa_mask'],
                  torch.zeros(m - n, seq_len, dtype=feat['msa_mask'].dtype)
              ),
              dim=0
          )
          feat['del_msa'] = torch.cat(
              (feat['del_msa'], torch.zeros(m - n, seq_len)), dim=0
          )
        # Rand permute msa relate feat
        if 'str_msa' not in ret:
          ret['str_msa'] = feat['str_msa']
        else:
          for i in range(max(m, n)):
            ret['str_msa'][i] += feat['str_msa'][i]
        for field in ('msa', 'msa_mask', 'del_msa'):
          if field not in ret:
            ret[field] = feat[field]
          else:
            ret[field] = torch.cat((ret[field], feat[field]), dim=1)
        # Update chain_id
        if feat['msa_idx'] > 0:
          msa_idx = feat['msa_idx']
          chains[idx] = f'{chain}@{msa_idx}'
      # Var related
      if self.feat_flags & FEAT_VAR:
        if 'var' not in ret:
          ret['var'] = defaultdict(dict)

        if 'length' not in ret:
          ret['length'] = []
        ret['length'].append(len(feat['str_seq']))

        if 'variant' in feat:
          for var_idx, desc in enumerate(feat['desc_var']):
            # remove domains pid/1-100
            var_pid = _make_var_pid(desc)
            ret['var'][var_pid] = (
                feat['variant'][var_idx],
                feat['variant_mask'][var_idx],
                chains[idx],
                feat['str_var'][var_idx],
                feat['del_var'][var_idx]
            )

    ret['pid'] = compose_pid(pid, ','.join(chains))
    if self.feat_flags & FEAT_VAR and 'var' in ret:
      assert 'length' in ret
      var_dict = ret['var']
      del ret['var']

      # filter complex with all chains aligned
      with timing(
          f'ProteinStructureDataset.build_chain_list {protein_id}', logger.debug
      ):
        var_chain_list = self._multimer_build_chain_list(
            pid, tuple(sorted(var_dict.keys()))
        )
      logger.debug('# of variants: %s', len(var_chain_list))

      if exists(self.max_var_depth) and self.max_var_depth < len(var_chain_list) + 1:
        if self.max_var_depth > 1:
          with timing(
              f'ProteinStructureDataset.sample_chain_list {protein_id}', logger.debug
          ):
            new_order = _make_var_choice(
                var_chain_list, self.attr_list, self.max_var_depth - 1
            )
            var_chain_list = [var_chain_list[i] for i in new_order]

      # realign the complex: iterate each target chain
      variant_label, variant_label_mask = _make_label_features(
          [k for k, _ in var_chain_list], self.attr_list, task_num=self.var_task_num
      )
      ret['variant_label'] = torch.as_tensor(
          [[1] * self.var_task_num] + variant_label, dtype=torch.float32
      )
      ret['variant_label_mask'] = torch.as_tensor(
          [[True] * self.var_task_num] + variant_label_mask, dtype=torch.bool
      )

      ret['variant'], ret['variant_mask'] = [ret['seq']], [ret['mask']]
      ret['str_var'] = [ret['str_seq']]
      ret['del_var'] = [torch.zeros((sum(ret['length']), ), dtype=torch.int)]
      ret['variant_pid'] = [ret['pid']]
      ret['variant_task_mask'] = [
          _make_task_mask(
              ret['mask'],
              chains,
              ret['length'],
              task_def=task_def,
              task_num=self.var_task_num
          )
      ]
      for var_pid, chain_list in var_chain_list:
        variant, variant_mask = [None] * len(chains), [None] * len(chains)
        str_var, del_var = [None] * len(chains), [None] * len(chains)
        for idx, chain in enumerate(chains):
          n = ret['length'][idx]
          for c, *_ in chain_list:
            cluster_id = compose_pid(var_pid, c)
            if cluster_id not in var_dict:
              cluster_id = self.mapping.get(cluster_id, cluster_id)
            hit_seq, hit_mask, target_chain, hit_str, hit_del = var_dict[cluster_id]
            if chains[idx] == target_chain:
              variant[idx], variant_mask[idx] = hit_seq, hit_mask
              str_var[idx], del_var[idx] = hit_str, hit_del
              break
          if not exists(variant[idx]):
            variant[idx] = torch.full(
                (n, ), residue_constants.restypes_with_x_and_gap.index('-')
            )
            variant_mask[idx] = torch.zeros((n, ), dtype=torch.bool)
            str_var[idx] = '-' * n
            del_var[idx] = torch.zeros((n, ), dtype=torch.int)
        ret['variant'].append(torch.cat(variant, dim=-1))
        ret['variant_mask'].append(torch.cat(variant_mask, dim=-1))
        ret['str_var'].append(''.join(str_var))
        ret['del_var'].append(torch.cat(del_var, dim=-1))
        ret['variant_pid'].append(var_pid)
        ret['variant_task_mask'].append(
            _make_task_mask(
                ret['variant_mask'][-1],
                chains,
                ret['length'],
                task_def=task_def,
                task_num=self.var_task_num
            )
        )
      for idx, field in enumerate(
          ('variant', 'variant_mask', 'variant_task_mask', 'del_var')
      ):
        ret[field] = torch.stack(ret[field], dim=0)

      ret['num_var'] = len(ret['variant'])

      var_idx, sequences = 0, ret['str_var']
      if ret.get('msa_idx', 0) == 0:
        if len(sequences) > 1 and np.random.random() < self.var_as_seq_prob:
          w = _msa_sample_weight(
              sequences,
              min_alr=self.var_as_seq_min_alr,
              min_ident=self.var_as_seq_min_ident
          )

          if exists(self.var_as_seq_min_label):
            v = torch.logical_or(
                torch.logical_and(
                    ret['variant_label'] >= self.var_as_seq_min_label,
                    ret['variant_label_mask']
                ),
                ~ret['variant_label_mask']
            )
            w *= np.all(v[1:].numpy(), axis=-1)

          t = np.sum(w)
          if t > 0:
            w /= t
            var_idx = int(np.argmax(np.random.multinomial(1, w))) + 1

      if var_idx > 0:
        var_depth = len(sequences)
        if 0 < var_idx < var_depth:
          t = sequences[var_idx]
          sequences[var_idx] = sequences[1]
          sequences[1] = t
          ret = _msa_as_seq(ret, var_idx, str_key='var')
      ret['var_idx'] = var_idx

    # for homomer: full ground truth
    for field in ('seq_index', 'seq_color', 'seq_entity', 'coord', 'coord_mask'):
      if field in ret:
        ret[f'{field}_fgt'] = ret[field]

    if exists(self.data_crop_fn):
      clip = self.data_crop_fn(ret)
      if exists(clip):
        ret = _protein_crop_fn(ret, clip)
        clip = _protein_crop_fmt(clip)
      ret['clip'] = clip

    # for homomer: select an anchor
    ret.update(_make_anchor_features(ret['seq_color_fgt'], ret['seq_entity_fgt'], ret))

    return ret

  def get_chain_list(self, protein_id):
    pid, chain = decompose_pid(protein_id)  # pylint: disable=unbalanced-tuple-unpacking
    if pid in self.chain_list:
      chain_group = self.chain_list[pid]
      # for g in chain_group:
      for idx in np.random.permutation(len(chain_group)):
        g = chain_group[idx]
        if chain in g:
          return list(g)  # shallow copy
      logger.warning('get_chain_list: %s not found.', protein_id)
    return None

  def has_chain(self, pid, chain):
    if pid in self.chain_list:
      chain_group = self.chain_list[pid]
      for g in chain_group:
        if chain in g:
          return True
    return False

  def get_resolution(self, protein_id):
    pid, _ = decompose_pid(protein_id)  # pylint: disable=unbalanced-tuple-unpacking
    return self.resolu.get(pid[:4], -1.)

  def get_msa_features_new(self, fs, protein_id, clip=None):
    k = int(np.random.randint(len(self.msa_list)))
    source = self.msa_list[k]
    with fs.open(f'msa/{protein_id}/{source}/{protein_id}.a4m') as f:
      sequences = list(map(lambda x: fs.textise(x).strip(), f))

    if exists(clip):
      sequences = [''.join(_msa_yield_from_clip(s, clip)) for s in sequences]

    ret = {'msa_idx': 0}
    if len(sequences) > 1 and np.random.random() < self.msa_as_seq_prob:
      n = len(sequences)
      if exists(self.msa_as_seq_topn):
        n = min(n, self.msa_as_seq_topn)
      assert n > 1

      w = _msa_sample_weight(
          sequences[:n],
          min_alr=self.msa_as_seq_min_alr,
          min_ident=self.msa_as_seq_min_ident
      )

      clu_file_path = f'msa/{protein_id}/{source}/{protein_id}.clu'
      if self.msa_as_seq_clustering and fs.exists(clu_file_path):
        try:
          with fs.open(clu_file_path) as f:
            clu_list = list(map(lambda x: int(fs.textise(x).strip()), f))
          if len(clu_list) != len(sequences):
            raise ValueError('len(clu_list) != len(sequences)')

          clu_dict = defaultdict(int)
          for i, clu in enumerate(clu_list):
            clu_dict[clu] += 1
          clu_list = np.asarray([clu_dict[clu] for clu in clu_list[1:n]])
          w /= clu_list
          del clu_dict
        except ValueError as e:
          logger.error('read clu faild: (%s) %s', protein_id, str(e))

      t = np.sum(w)
      if t > 0:
        w /= t
        ret['msa_idx'] = int(np.argmax(np.random.multinomial(1, w))) + 1
    ret.update(
        _make_msa_features(
            sequences, msa_idx=ret['msa_idx'], max_msa_depth=self.max_msa_depth
        )
    )
    return ret

  def get_var_features_new(self, fs, protein_id, clip=None):
    k = int(np.random.randint(len(self.msa_list)))
    source = self.msa_list[k]
    variant_path = f'{self.var_dir}/{protein_id}/msas/{protein_id}.a3m'
    if fs.exists(variant_path):
      with fs.open(variant_path) as f:
        sequences, descriptions = parse_fasta(fs.textise(f.read()))
      assert len(sequences) == len(descriptions)

      if self.attr_list:  # filter with attr_list, NOTE: keep the 1st one alway.

        def _is_aligned(desc):
          var_pid = _make_var_pid(desc)
          return var_pid in self.attr_list

        new_order = list(
            filter(
                lambda i: _is_aligned(descriptions[i + 1]), range(len(descriptions) - 1)
            )
        )
        sequences = sequences[:1] + [sequences[i + 1] for i in new_order]
        descriptions = descriptions[:1] + [descriptions[i + 1] for i in new_order]

      if exists(clip):
        sequences = [''.join(_msa_yield_from_clip(s, clip)) for s in sequences]

      ret = {'var_idx': 0}
      if len(sequences) > 1 and np.random.random() < self.var_as_seq_prob:
        w = _msa_sample_weight(
            sequences,
            min_alr=self.var_as_seq_min_alr,
            min_ident=self.var_as_seq_min_ident
        )

        t = np.sum(w)
        if t > 0:
          w /= t
          ret['var_idx'] = int(np.argmax(np.random.multinomial(1, w))) + 1

      ret.update(
          _make_var_features(
              sequences,
              descriptions,
              var_idx=ret['var_idx'],
              attr_dict=self.attr_list,
              max_var_depth=self.max_var_depth
          )
      )
      if exists(self.attr_list):
        variant_label, variant_label_mask = _make_label_features(
            ret['desc_var'], self.attr_list, task_num=self.var_task_num
        )
        ret['variant_label'] = torch.as_tensor(variant_label, dtype=torch.float32)
        ret['variant_label_mask'] = torch.as_tensor(
            variant_label_mask, dtype=torch.bool
        )
      return ret
    return {}

  def get_structure_label_npz(
      self, fs, protein_key, protein_id, seq_color=1, seq_entity=None, seq_sym=None
  ):
    ret = {}

    fasta_file = f'{self.fasta_dir}/{protein_key}.fasta'
    if fs.exists(fasta_file):
      with fs.open(fasta_file) as f:
        input_fasta_str = fs.textise(f.read())
      input_seqs, input_descs = parse_fasta(input_fasta_str)
      if len(input_seqs) != 1:
        raise ValueError(f'More than one input sequence found in {fasta_file}.')
      input_sequence = input_seqs[0]
      input_description = input_descs[0]

      ret.update(
          _make_seq_features(
              input_sequence,
              input_description,
              seq_color=seq_color,
              seq_entity=seq_entity,
              seq_sym=seq_sym,
              max_seq_len=None
          )
      )

      pdb_file = f'{self.pdb_dir}/{protein_id}.npz'
      if (self.feat_flags & FEAT_PDB) and fs.exists(pdb_file):
        with fs.open(pdb_file) as f:
          structure = np.load(BytesIO(f.read()))
          ret.update(
              coord=torch.from_numpy(structure['coord']),
              coord_mask=torch.from_numpy(structure['coord_mask'])
          )
          if 'bfactor' in structure:
            ret.update(coord_plddt=torch.from_numpy(structure['bfactor']))
          else:
            ret.update(
                coord_plddt=torch.ones_like(ret['coord_mask'], dtype=torch.float)
            )
    else:  # from pdb_db file
      for pdb_type in ('pdb', 'cif', None):  # None is a sentinal
        if exists(pdb_type):
          pdb_file = f'{self.pdb_dir}/{protein_id}.{pdb_type}'
          if fs.exists(pdb_file):
            with fs.open(pdb_file) as f:
              pdb_string = fs.textise(f.read())
            break
      if not exists(pdb_type):
        assert hasattr(self, 'pdb_db'), protein_id
        pdb_type, pdb_string = self.pdb_db[protein_id]
      ret.update(
          _make_pdb_features(
              protein_id,
              pdb_string,
              pdb_type=pdb_type,
              seq_color=seq_color,
              seq_entity=seq_entity,
              seq_sym=seq_sym,
              max_seq_len=None
          )
      )

    if 'coord' in ret:
      pae_file = f'{self.pdb_dir}/{protein_id}-predicted_aligned_error.json'
      if fs.exists(pae_file):
        with fs.open(pae_file) as f:
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

  @staticmethod
  def collate_fn(batch, feat_flags=None):
    return _collate_fn(batch, feat_flags=feat_flags)


def _collate_fn(batch, feat_flags=None):
  def _any(field):
    return any(exists(b.get(field)) for b in batch)

  def _to_list(field, defval=None):
    return [b.get(field, defval) for b in batch]

  def _to_tensor(field, defval=0, dtype=None):
    return torch.as_tensor(_to_list(field, defval=defval), dtype=dtype)

  ret = {}

  for field in ('pid', 'str_seq', 'clip'):
    ret[field] = _to_list(field)

  max_batch_len = max(len(s) for s in ret['str_seq'])

  ret['seq'] = pad_sequential(
      _to_list('seq'), max_batch_len, padval=residue_constants.unk_restype_index
  )
  for field in ('seq_index', 'mask'):
    ret[field] = pad_sequential(_to_list(field), max_batch_len)

  for field in ('seq_color', 'seq_entity', 'seq_sym'):
    if _any(field):
      ret[field] = pad_sequential(_to_list(field), max_batch_len)

  if _any('resolu'):
    ret['resolution'] = _to_tensor('resolu', -1.0)

  feat_flags = default(feat_flags, FEAT_ALL)
  if feat_flags & FEAT_PDB and _any('coord'):
    # required
    for field in ('coord', 'coord_mask'):
      ret[field] = pad_sequential(_to_list(field), max_batch_len)
    # optional
    if _any('coord_plddt'):
      ret['coord_plddt'] = pad_sequential(
          _to_list('coord_plddt'), max_batch_len, padval=1.0
      )

  # for homomer
  if _any('seq_anchor'):
    ret['seq_anchor'] = _to_tensor('seq_anchor', dtype=torch.int)
    for field in ('seq_index', 'seq_color', 'seq_entity', 'coord', 'coord_mask'):
      field = f'{field}_fgt'
      ret[field] = _to_list(field)

  if feat_flags & FEAT_MSA and _any('msa'):
    ret['str_msa'] = _to_list('str_msa')
    for field in ('msa_idx', 'num_msa'):
      ret[field] = _to_tensor(field, dtype=torch.int)
    ret['msa'] = pad_rectangle(
        _to_list('msa'), max_batch_len, padval=residue_constants.HHBLITS_AA_TO_ID['-']
    )
    for field in ('msa_mask', 'del_msa'):
      ret[field] = pad_rectangle(_to_list(field), max_batch_len)

  if feat_flags & FEAT_VAR and _any('variant'):
    ret['variant_pid'] = _to_list('variant_pid')
    for field in ('var_idx', 'num_var'):
      ret[field] = _to_tensor(field, dtype=torch.int)
    ret['variant'] = pad_rectangle(
        _to_list('variant'),
        max_batch_len,
        padval=residue_constants.HHBLITS_AA_TO_ID['-']
    )
    for field in ('variant_mask', 'variant_task_mask'):
      ret[field] = pad_rectangle(_to_list(field), max_batch_len)
    for field in ('variant_label', 'variant_label_mask'):
      items = _to_list(field)
      max_depth = max(item.shape[0] for item in items if exists(item))
      ret[field] = pad_sequential(items, max_depth)

  return ret


def load(
    data_dir,
    data_idx=None,
    pseudo_linker_prob=0.0,
    pseudo_linker_shuffle=True,
    data_rm_mask_prob=0.0,
    msa_as_seq_prob=0.0,
    min_crop_len=None,
    max_crop_len=None,
    min_crop_pae=False,
    max_crop_plddt=False,
    crop_probability=0,
    crop_algorithm='random',
    feat_flags=FEAT_ALL,
    **kwargs
):
  max_msa_depth = kwargs.pop('max_msa_depth') if 'max_msa_depth' in kwargs else 128
  msa_as_seq_topn = kwargs.pop(
      'msa_as_seq_topn'
  ) if 'msa_as_seq_topn' in kwargs else None
  msa_as_seq_min_alr = kwargs.pop(
      'msa_as_seq_min_alr'
  ) if 'msa_as_seq_min_alr' in kwargs else None
  msa_as_seq_clustering = kwargs.pop(
      'msa_as_seq_clustering'
  ) if 'msa_as_seq_clustering' in kwargs else False
  msa_as_seq_min_ident = kwargs.pop(
      'msa_as_seq_min_ident'
  ) if 'msa_as_seq_min_ident' in kwargs else None
  max_var_depth = kwargs.pop('max_var_depth') if 'max_var_depth' in kwargs else 1024
  var_task_num = kwargs.pop('var_task_num') if 'var_task_num' in kwargs else 1
  mapping_idx = kwargs.pop('mapping_idx') if 'mapping_idx' in kwargs else None
  chain_idx = kwargs.pop('chain_idx') if 'chain_idx' in kwargs else None
  attr_idx = kwargs.pop('attr_idx') if 'attr_idx' in kwargs else None
  var_dir = kwargs.pop('var_dir') if 'var_dir' in kwargs else None

  data_dir = data_dir.split(',')

  def _split_args(args):
    if exists(args):
      # let '' be None
      args = list(map(lambda x: x if x else None, args.split(',')))
    else:
      args = [None] * len(data_dir)
    assert len(data_dir) == len(args), (len(data_dir), args)
    return args

  data_idx, mapping_idx, chain_idx, attr_idx, var_dir = map(
      _split_args, (data_idx, mapping_idx, chain_idx, attr_idx, var_dir)
  )

  if 'data_crop_fn' not in kwargs:
    data_crop_fn = functools.partial(
        _protein_clips_fn,
        min_crop_len=min_crop_len,
        max_crop_len=max_crop_len,
        min_crop_pae=min_crop_pae,
        max_crop_plddt=max_crop_plddt,
        crop_probability=crop_probability,
        crop_algorithm=crop_algorithm
    )
  else:
    data_crop_fn = kwargs.pop('data_crop_fn')

  dataset = torch.utils.data.ConcatDataset(
      [
          ProteinStructureDataset(
              data_dir[i],
              data_idx=data_idx[i],
              mapping_idx=mapping_idx[i],
              chain_idx=chain_idx[i],
              attr_idx=attr_idx[i],
              var_dir=var_dir[i],
              data_crop_fn=data_crop_fn,
              pseudo_linker_prob=pseudo_linker_prob,
              pseudo_linker_shuffle=pseudo_linker_shuffle,
              data_rm_mask_prob=data_rm_mask_prob,
              msa_as_seq_prob=msa_as_seq_prob,
              msa_as_seq_topn=msa_as_seq_topn,
              msa_as_seq_clustering=msa_as_seq_clustering,
              msa_as_seq_min_alr=msa_as_seq_min_alr,
              msa_as_seq_min_ident=msa_as_seq_min_ident,
              max_msa_depth=max_msa_depth,
              max_var_depth=max_var_depth,
              var_task_num=var_task_num,
              feat_flags=feat_flags
          ) for i in range(len(data_dir))
      ]
  )
  if 'collate_fn' not in kwargs:
    kwargs['collate_fn'] = functools.partial(
        ProteinStructureDataset.collate_fn, feat_flags=feat_flags
    )
  if 'weights' in kwargs:
    weights = kwargs.pop('weights')
    if weights:
      kwargs['sampler'] = WeightedRandomSampler(weights, num_samples=len(weights))
      if 'shuffle' in kwargs:
        kwargs.pop('shuffle')
  elif 'num_replicas' in kwargs and 'rank' in kwargs:
    num_replicas, rank = kwargs.pop('num_replicas'), kwargs.pop('rank')
    kwargs['sampler'] = DistributedSampler(
        dataset, num_replicas=num_replicas, rank=rank
    )
  return torch.utils.data.DataLoader(dataset, **kwargs)
