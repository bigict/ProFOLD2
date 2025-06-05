"""Tools for check dataset file, run
     ```bash
     $python dataset.py -h
     ```
     for further help.
"""
import os
from collections import defaultdict
import functools
import itertools
import logging
import multiprocessing as mp

import numpy as np
import torch
from einops import rearrange

from profold2.common import protein, residue_constants
from profold2.data import dataset
from profold2.data.dataset import ProteinStructureDataset
from profold2.data.utils import compose_pid, decompose_pid, str_seq_index
from profold2.model.features import FeatureBuilder
from profold2.model.functional import sharded_apply, squared_cdist
from profold2.utils import exists, timing

logger = logging.getLogger(__file__)


def _to_chain_group(data, args, idx):
  def _chain_contact_num(datum, cx, cy):
    ca_idx = residue_constants.atom_order['CA']

    squared_mask = rearrange(
        datum[cx]['coord_mask'][..., ca_idx], '... i -> ... i ()'
    ) * rearrange(datum[cy]['coord_mask'][..., ca_idx], '... j -> ... () j')

    dist2 = squared_cdist(
        datum[cx]['coord'][..., ca_idx, :], datum[cy]['coord'][..., ca_idx, :]
    )

    return torch.sum((dist2 <= args.contact_cutoff**2) * squared_mask).item()

  def _chain_group_contact_num(contacts, cx_group, cy_group):
    return sum(
        contacts[cx][cy] for cx, cy in itertools.product(
            cx_group, cy_group
        ) if cy not in cx_group
    )

  def _chain_group_seq_len(datum, chain_group):
    return sum(datum[c]['seq'].shape[0] for c in chain_group)

  def _chain_group_new(datum, chain_group):
    contacts = defaultdict(dict)

    # compute contacts for each chain pair.
    for cx in chain_group:
      for cy in chain_group:
        if cx != cy:
          c = _chain_contact_num(datum, cx, cy)
          contacts[cx][cy] = c
          contacts[cy][cx] = c
        else:
          contacts[cx][cy] = 0

    # re-group monomers
    chain_group_list = defaultdict(list)
    for c in chain_group:
      chain_group_list[datum[c]['str_seq']].append(c)
    chain_group_list = list(chain_group_list.values())

    chain_seen = set()
    for cg in chain_group_list:
      new_group, new_len, new_added = cg, _chain_group_seq_len(datum, cg), True
      while new_added and new_len < args.max_sequence_length:
        weights = [
            (i, _chain_group_contact_num(contacts, new_group, x))
            for i, x in enumerate(chain_group_list)
        ]

        new_added = False
        for i, w in sorted(
            filter(lambda x: x[1] > 0, weights), key=lambda x: x[1], reverse=True
        ):
          cg_len = _chain_group_seq_len(datum, chain_group_list[i])
          if new_len + cg_len < args.max_sequence_length:
            new_len += cg_len
            new_group += chain_group_list[i]
            new_added = True
      new_group_str = ','.join(sorted(new_group))
      if new_group_str not in chain_seen:
        yield new_group
        chain_seen.add(new_group_str)

  chain_dict = defaultdict(list)

  for pid in data.pids[idx]:
    pid, chain = decompose_pid(pid)  # pylint: disable=unbalanced-tuple-unpacking
    if pid in data.chain_list:
      for chain_group in data.chain_list[pid]:
        if chain in chain_group:
          if len(chain_group) > 1:  # more than 1 chain
            chain_group_length = 0

            datum = {}
            for c in chain_group:
              datum[c] = data.get_monomer(compose_pid(pid, c), crop_fn=None)
              assert 'seq' in datum[c]
              assert 'coord' in datum[c] and 'coord_mask' in datum[c]

              chain_group_length += datum[c]['seq'].shape[0]
            if chain_group_length > args.max_sequence_length:  # sequence is too long
              logger.debug(
                  'NOTE: %s length: %d chains: %s', pid, chain_group_length, ','.join(chain_group)
              )
              for g in _chain_group_new(datum, chain_group):
                chain_dict[pid].append(g)
            else:
              chain_dict[pid].append(chain_group)
          else:
            chain_dict[pid].append(chain_group)

  return chain_dict


def to_chain_group(data, args):  # pylint: disable=redefined-outer-name
  work_fn = functools.partial(_to_chain_group, data, args)

  range_start, range_stop = args.range_start, args.range_stop
  if not exists(range_stop):
    range_stop = len(data)
  range_stop = min(range_stop, len(data))
  with open(args.output, 'w') as f:
    if args.num_workers == 1:
      for i in range(range_start, range_stop):
        chain_dict = work_fn(i)
        for pid, chain_group in chain_dict.items():
          for g in chain_group:
            g = ' '.join(g)
            f.write(f'{pid}\t{g}\n')
    else:
      with mp.Pool(args.num_workers) as p:
        for chain_dict in p.imap_unordered(
            work_fn, range(range_start, range_stop), chunksize=args.chunksize
        ):
          for pid, chain_group in chain_dict.items():
            for g in chain_group:
              g = ' '.join(g)
              f.write(f'{pid}\t{g}\n')


def to_chain_group_add_argument(parser):  # pylint: disable=redefined-outer-name
  parser.add_argument('-o', '--output', type=str, default=None, help='output file')
  parser.add_argument(
      '--max_sequence_length',
      type=int,
      default=2048,
      help='maximum sequence length for each chain group'
  )
  parser.add_argument(
      '--contact_cutoff',
      type=float,
      default=8,
      help='maximum residue distance for each contact'
  )
  parser.add_argument(
      '--num_workers', type=int, default=None, help='num of workers.'
  )
  parser.add_argument(
      '--chunksize', type=int, default=10, help='chunk size of each worker.'
  )
  parser.add_argument(
      '--range_start', type=int, default=0, help='data index start.'
  )
  parser.add_argument(
      '--range_stop', type=int, default=None, help='data index stop.'
  )
  return parser


def to_fasta(data, args):  # pylint: disable=redefined-outer-name
  for prot in iter(data):
    if args.dump_keys:
      print(prot.keys())
    assert 'pid' in prot and 'str_seq' in prot
    if args.print_fasta:
      pid = prot['pid']
      domains = str_seq_index(prot['seq_index'])
      print(f'>{pid} domains:{domains}')
      print(prot['str_seq'])
    if args.print_first_only:
      print(prot)
      break


def to_fasta_add_argument(parser):  # pylint: disable=redefined-outer-name
  parser.add_argument('--dump_keys', action='store_true', help='dump keys')
  parser.add_argument('--print_fasta', action='store_true', help='print fasta')
  parser.add_argument(
      '--print_first_only', action='store_true', help='print first only'
  )
  return parser


def to_pdb(data, args):  # pylint: disable=redefined-outer-name
  os.makedirs(args.output, exist_ok=True)

  cb_idx = residue_constants.atom_order["CB"]
  for feat in iter(data):
    coord_mask = feat['coord_mask']
    if args.coord_pad:
      coord_mask = torch.clone(coord_mask)
      coord_mask[:, :cb_idx] = True
    prot = protein.Protein(
        aatype=np.array(feat['seq']),
        atom_positions=np.array(feat['coord']),
        atom_mask=np.array(coord_mask),
        residue_index=np.array(feat['seq_index']) + 1,
        chain_index=np.array(feat['seq_color']) - 1,
        b_factors=np.zeros_like(feat['coord_mask'])
    )
    pid = feat['pid']
    with open(os.path.join(args.output, f'{pid}.pdb'), 'w') as f:
      if args.verbose:
        logger.debug('to pdb: %s', pid)
      f.write(protein.to_pdb(prot))


def to_pdb_add_argument(parser):  # pylint: disable=redefined-outer-name
  parser.add_argument('-o', '--output', type=str, default=None, help='output file.')
  parser.add_argument('--coord_pad', action='store_true', help='coord padded.')
  return parser


def checksum(data, args):  # pylint: disable=redefined-outer-name
  for prot in iter(data):
    n = len(prot['str_seq'])
    if 'msa' in prot:
      if n != prot['msa'].shape[1]:
        print(prot['pid'], n, prot['msa'].shape)
      assert prot['msa'].shape == prot['msa_mask'].shape
    elif args.msa_required:
      print(prot['pid'], 'MSA required')
    if 'coord' in prot:
      if n != prot['coord'].shape[0]:
        print(prot['pid'], n, prot['coord'].shape)
    elif args.coord_required:
      print(prot['pid'], 'coord required')
    if 'coord_pae' in prot:
      if n != prot['coord_pae'].shape[0] or prot['coord_pae'].shape[1] != n:
        print(prot['pid'], n, prot['coord_pae'].shape)
    if 'coord_mask' in prot:
      if n != prot['coord_mask'].shape[0]:
        print(prot['pid'], n, prot['coord_mask'].shape)
    elif args.coord_required:
      print(prot['pid'], 'coord_mask required')
    if 'variant' in prot:
      if n != prot['variant'].shape[1]:
        print(prot['pid'], n, prot['variant'].shape)
      assert prot['variant'].shape == prot['variant_mask'].shape


def checksum_add_argument(parser):  # pylint: disable=redefined-outer-name
  parser.add_argument('--msa_required', action='store_true', help='MSA required')
  parser.add_argument('--coord_required', action='store_true', help='coord required')
  return parser


def _plddt_mean(args, feat, data, idx):  # pylint: disable=redefined-outer-name
  del args

  prot = data[idx]
  pid = prot['pid']
  with timing(f'process {idx} {pid}', logger.info):
    prot = feat(prot)
    if 'plddt_mean' in prot:
      return pid, prot['plddt_mean'], len(prot['str_seq'])
    return pid, None, len(prot['str_seq'])


def plddt(data, args):  # pylint: disable=redefined-outer-name
  feat = FeatureBuilder([('make_coord_plddt', {})])
  work_fn = functools.partial(_plddt_mean, args, feat, data)

  range_start, range_stop = args.range_start, args.range_stop
  if not exists(range_stop):
    range_stop = len(data)
  range_stop = min(range_stop, len(data))
  with open(args.output, 'w') as f:
    if args.num_workers == 1:
      for i in range(range_start, range_stop):
        pid, score = work_fn(i)
        if exists(score):
          f.write(f'plddt\t{pid}\t{score.item()}\n')
    else:
      with mp.Pool(args.num_workers) as p:
        for pid, score, seq_len in p.imap_unordered(
            work_fn, range(range_start, range_stop), chunksize=args.chunksize
        ):
          if exists(score):
            f.write(f'plddt\t{pid}\t{score.item()}\t{seq_len}\n')


def plddt_add_argument(parser):  # pylint: disable=redefined-outer-name
  parser.add_argument('-o', '--output', type=str, default=None, help='output file.')
  parser.add_argument(
      '--num_workers', type=int, default=None, help='num of workers.'
  )
  parser.add_argument(
      '--chunksize', type=int, default=10, help='chunk size of each worker.'
  )
  parser.add_argument('--msa_required', action='store_true', help='MSA required')
  parser.add_argument('--coord_required', action='store_true', help='coord required')
  parser.add_argument(
      '--range_start', type=int, default=0, help='data index start.'
  )
  parser.add_argument(
      '--range_stop', type=int, default=None, help='data index stop.'
  )
  return parser


def contacts_func(args, feat, data, idx):  # pylint: disable=redefined-outer-name
  prot = data[idx]
  pid = prot['pid']
  seq_index = prot['seq_index']
  with timing(f'process {idx} {seq_index.shape} {pid}', logger.info):
    prot = feat(prot)
    seq_color = prot['seq_color']
    positions = prot['pseudo_beta']
    mask = prot['pseudo_beta_mask']

    n = torch.amax(seq_color)

    def _calc_contacts(seq_index_i, seq_color_i, positions_i, mask_i):
      squared_idx = torch.abs(
          rearrange(seq_index_i, '... i -> ... i ()') -
          rearrange(seq_index, '... j -> ... () j')
      )
      squared_mask = rearrange(mask_i, '... i -> ... i ()'
                              ) * rearrange(mask, '... j -> ... () j')
      dist2 = squared_cdist(positions_i, positions)
      all_contacts = (dist2 <= args.contact_cutoff**2) * squared_mask

      pair_clr_mask = (
          rearrange(seq_color_i,
                    '... i -> ... i ()') == rearrange(seq_color, '... j -> ... () j')
      )
      intra_contacts = torch.sum(
          all_contacts * pair_clr_mask * (squared_idx >= args.contact_range_min),
          dim=-1
      )

      pair_clr_mask = (
          rearrange(seq_color_i, '... i -> ... i ()') !=
          rearrange(seq_color, '... j -> ... () j')
      )
      inter_contacts = torch.sum(all_contacts * pair_clr_mask, dim=-1)

      return intra_contacts, inter_contacts

    def _cat_contacts(chunk_iter):
      intra_contacts, inter_contacts = [], []
      for x, y in chunk_iter:
        intra_contacts.append(x)
        inter_contacts.append(y)
      return torch.cat(intra_contacts, dim=-1), torch.cat(inter_contacts, dim=-1)

    intra_contacts, inter_contacts = sharded_apply(
        _calc_contacts, [seq_index, seq_color, positions, mask],
        shard_size=args.shard_size,
        shard_dim=0,
        cat_dim=_cat_contacts
    )
    intra_contacts = torch.scatter_add(
        torch.zeros(n, ), -1, seq_color - 1, (intra_contacts > 0).float()
    )
    inter_contacts = torch.scatter_add(
        torch.zeros(n, ), -1, seq_color - 1, (inter_contacts > 0).float()
    )
  return pid, intra_contacts, inter_contacts


def contacts(data, args):  # pylint: disable=redefined-outer-name
  feat = FeatureBuilder([('make_pseudo_beta', {})])
  work_fn = functools.partial(contacts_func, args, feat, data)

  with open(args.output, 'w') as f:
    if args.num_workers == 1:
      for idx in range(len(data)):
        pid, intra_contacts, inter_contacts = work_fn(idx)
        if not (exists(intra_contacts) and exists(inter_contacts)):
          continue
        pid, chains, *_ = decompose_pid(pid)
        for i, chain in enumerate(chains.split(',')):
          f.write(
              f'contacts\t{compose_pid(pid, chain)}\t{intra_contacts[i].item()}\t{inter_contacts[i].item()}\n'  # pylint: disable=line-too-long
          )
    else:
      with mp.Pool(args.num_workers) as p:
        for pid, intra_contacts, inter_contacts in p.imap_unordered(
            work_fn, range(len(data)), chunksize=args.chunksize
        ):
          if not (exists(intra_contacts) and exists(inter_contacts)):
            continue
          pid, chains, *_ = decompose_pid(pid)
          for i, chain in enumerate(chains.split(',')):
            f.write(
                f'contacts\t{compose_pid(pid, chain)}\t{intra_contacts[i].item()}\t{inter_contacts[i].item()}\n'  # pylint: disable=line-too-long
            )


def contacts_add_argument(parser):  # pylint: disable=redefined-outer-name
  parser.add_argument('-o', '--output', type=str, default=None, help='output file.')
  parser.add_argument(
      '--num_workers', type=int, default=None, help='num of workers.'
  )
  parser.add_argument(
      '--chunksize', type=int, default=10, help='chunk size of each worker.'
  )
  parser.add_argument(
      '--shard_size',
      type=int,
      default=None,
      help='shard size of each protein.'
  )
  parser.add_argument('--msa_required', action='store_true', help='MSA required')
  parser.add_argument(
      '--contact_cutoff', type=float, default=8, help='Contact cutoff.'
  )
  parser.add_argument(
      '--contact_range_min', type=int, default=6, help='Contact range start.'
  )
  parser.add_argument(
      '--contact_range_max',
      type=int,
      default=-1,
      help='Contact range stop.'
  )
  return parser


def main(work_fn, args):  # pylint: disable=redefined-outer-name
  logger.info(args)
  # get data
  feat_flags = dataset.FEAT_ALL & (~dataset.FEAT_MSA)
  if hasattr(args, 'msa_required') and args.msa_required:
    feat_flags = feat_flags | dataset.FEAT_MSA
  data = ProteinStructureDataset(
      data_dir=args.data_dir,
      data_idx=args.data_idx,
      pseudo_linker_prob=args.pseudo_linker_prob,
      pseudo_linker_shuffle=False,
      data_rm_mask_prob=args.data_rm_mask_prob,
      msa_as_seq_prob=args.msa_as_seq_prob,
      feat_flags=feat_flags
  )
  with timing(f'{args.command}', logger.info):
    work_fn(data, args)


if __name__ == '__main__':
  import argparse

  commands = {
      'checksum': (checksum, checksum_add_argument),
      'contacts': (contacts, contacts_add_argument),
      'plddt': (plddt, plddt_add_argument),
      'to_chain_group': (to_chain_group, to_chain_group_add_argument),
      'to_fasta': (to_fasta, to_fasta_add_argument),
      'to_pdb': (to_pdb, to_pdb_add_argument),
  }

  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

  sub_parsers = parser.add_subparsers(dest='command', required=True)
  for cmd, (_, add_argument) in commands.items():
    cmd_parser = sub_parsers.add_parser(
        cmd, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    cmd_parser.add_argument(
        '--data_dir', type=str, default=None, help='train dataset dir.'
    )
    cmd_parser.add_argument(
        '--data_idx', type=str, default=None, help='dataset idx.'
    )
    cmd_parser.add_argument(
        '--data_rm_mask_prob', type=float, default=0.0, help='data_rm_mask'
    )
    cmd_parser.add_argument(
        '--msa_as_seq_prob', type=float, default=0.0, help='msa_as_mask'
    )
    cmd_parser.add_argument(
        '--pseudo_linker_prob', type=float, default=0.0, help='pseudo_linker_prob'
    )
    cmd_parser.add_argument('-v', '--verbose', action='store_true', help='verbose')

    add_argument(cmd_parser)

  args = parser.parse_args()

  logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

  main(commands[args.command][0], args)
