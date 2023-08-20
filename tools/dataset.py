"""Tools for check dataset file, run
     ```bash
     $python dataset.py -h
     ```
     for further help.
"""
import functools
import logging
import multiprocessing as mp

import torch
from einops import rearrange, repeat

from profold2.data import dataset
from profold2.data.dataset import ProteinStructureDataset
from profold2.data.utils import compose_pid, decompose_pid
from profold2.model.features import FeatureBuilder
from profold2.model.functional import sharded_apply, squared_cdist
from profold2.utils import exists, timing


logger = logging.getLogger(__file__)

def to_fasta(data, args):  # pylint: disable=redefined-outer-name
  for prot in iter(data):
    if args.dump_keys:
      print(prot.keys())
    assert 'pid' in prot and 'str_seq' in prot
    assert len(prot['pid']) == len(prot['str_seq'])
    if args.print_fasta:
      for i, pid in enumerate(prot['pid']):
        print(f'>{pid}')
        print(prot['str_seq'][i], prot['str_msa'][i][0])
    if args.print_first_only:
      print(prot)
      break


def to_fasta_add_argument(parser):  # pylint: disable=redefined-outer-name
  parser.add_argument('--dump_keys', action='store_true', help='dump keys')
  parser.add_argument('--checksum', action='store_true', help='dump keys')
  parser.add_argument('--print_fasta', action='store_true', help='print fasta')
  parser.add_argument('--print_first_only',
                      action='store_true',
                      help='print first only')
  return parser


def checksum(data, args):  # pylint: disable=redefined-outer-name
  for prot in iter(data):
    n = len(prot['str_seq'][0])
    if 'msa' in prot:
      if n != prot['msa'].shape[2]:
        print(prot['pid'], n, prot['msa'].shape)
    elif args.msa_required:
      print(prot['pid'], 'MSA required')
    if 'coord' in prot:
      if n != prot['coord'].shape[1]:
        print(prot['pid'], n, prot['coord'].shape)
    elif args.coord_required:
      print(prot['pid'], 'coord required')
    if 'coord_mask' in prot:
      if n != prot['coord_mask'].shape[1]:
        print(prot['pid'], n, prot['coord_mask'].shape)
    elif args.coord_required:
      print(prot['pid'], 'coord_mask required')


def checksum_add_argument(parser):  # pylint: disable=redefined-outer-name
  parser.add_argument('--msa_required',
                      action='store_true',
                      help='MSA required')
  parser.add_argument('--coord_required',
                      action='store_true',
                      help='coord required')
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
          rearrange(seq_index, '... j -> ... () j'))
      squared_mask = rearrange(mask_i, '... i -> ... i ()') * rearrange(
          mask, '... j -> ... () j')
      dist2 = squared_cdist(positions_i, positions)
      contacts = (dist2 <= args.contact_cutoff**2) * squared_mask

      pair_clr_mask = (rearrange(seq_color_i, '... i -> ... i ()') == rearrange(
          seq_color, '... j -> ... () j'))

      intra_contacts = torch.sum(contacts * pair_clr_mask *
                                 (squared_idx >= args.contact_range_min),
                                 dim=-1)
      pair_clr_mask = (rearrange(seq_color_i, '... i -> ... i ()') != rearrange(
          seq_color, '... j -> ... () j'))
      inter_contacts = torch.sum(contacts * pair_clr_mask, dim=-1)

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
        cat_dim=_cat_contacts)
    intra_contacts = torch.scatter_add(torch.zeros(n,), -1, seq_color - 1,
                                       (intra_contacts > 0).float())
    inter_contacts = torch.scatter_add(torch.zeros(n,), -1, seq_color - 1,
                                         (inter_contacts > 0).float())
    # shard_size = 128
    # if exists(args.shard_size):
    #   pass
    # else:
    #   squared_idx = torch.abs(
    #       rearrange(seq_index, '... i -> ... i ()') -
    #       rearrange(seq_index, '... j -> ... () j'))
    #   squared_mask = rearrange(mask, '... i -> ... i ()') * rearrange(
    #       mask, '... j -> ... () j')
    #   dist2 = squared_cdist(positions, positions)
    #   contacts = (dist2 <= args.contact_cutoff**2) * squared_mask

    #   pair_clr_mask = (rearrange(seq_color, '... i -> ... i ()') == rearrange(
    #       seq_color, '... j -> ... () j'))
    #   intra_contacts = torch.sum(contacts * pair_clr_mask *
    #                              (squared_idx >= args.contact_range_min),
    #                              dim=-1)
    #   intra_contacts = torch.scatter_add(torch.zeros(n,), -1, seq_color - 1,
    #                                      (intra_contacts > 0).float())

    #   pair_clr_mask = (rearrange(seq_color, '... i -> ... i ()') != rearrange(
    #       seq_color, '... j -> ... () j'))
    #   inter_contacts = torch.sum(contacts * pair_clr_mask, dim=-1)
    #   inter_contacts = torch.scatter_add(torch.zeros(n,), -1, seq_color - 1,
    #                                      (inter_contacts > 0).float())

  return pid, intra_contacts, inter_contacts


def contacts(data, args):  # pylint: disable=redefined-outer-name
  feat = FeatureBuilder([('make_pseudo_beta', {})])
  work_fn = functools.partial(contacts_func, args, feat, data)

  with open(args.output, 'w') as f:
    if args.num_workers == 1:
      for i in range(len(data)):
        pid, intra_contacts, inter_contacts = work_fn(i)
        if not (exists(intra_contacts) and exists(inter_contacts)):
          continue
        pid, chains = decompose_pid(pid)
        for i, chain in enumerate(chains.split(',')):
          f.write(
              f'contacts\t{compose_pid(pid, chain)}\t{intra_contacts[i].item()}\t{inter_contacts[i].item()}\n'
          )
    else:
      with mp.Pool(args.num_workers) as p:
        for pid, intra_contacts, inter_contacts in p.imap_unordered(
            work_fn, range(len(data)), chunksize=args.chunksize):
          if not (exists(intra_contacts) and exists(inter_contacts)):
            continue
          pid, chains = decompose_pid(pid)
          for i, chain in enumerate(chains.split(',')):
            f.write(
                f'contacts\t{compose_pid(pid, chain)}\t{intra_contacts[i].item()}\t{inter_contacts[i].item()}\n'
            )


def contacts_add_argument(parser):  # pylint: disable=redefined-outer-name
  parser.add_argument('-o', '--output',
                      type=str,
                      default=None,
                      help='output file')
  parser.add_argument('--num_workers',
                      type=int,
                      default=None,
                      help='num of workers, default=#cpus')
  parser.add_argument('--chunksize',
                      type=int,
                      default=10,
                      help='chunk size of each worker, default=1')
  parser.add_argument('--shard_size',
                      type=int,
                      default=None,
                      help='shard size of each protein, default=None')
  parser.add_argument('--msa_required',
                      action='store_true',
                      help='MSA required')
  parser.add_argument('--contact_cutoff',
                      type=float,
                      default=8,
                      help='Contact cutoff, default=8')
  parser.add_argument('--contact_range_min',
                      type=int,
                      default=6,
                      help='Contact range start, default=6')
  parser.add_argument('--contact_range_max',
                      type=int,
                      default=-1,
                      help='Contact range start, default=-1')
  return parser


def main(work_fn, args):  # pylint: disable=redefined-outer-name
  # get data
  feat_flags = dataset.FEAT_ALL & (~dataset.FEAT_MSA)
  if hasattr(args, 'msa_required') and args.msa_required:
    feat_flags = feat_flags | dataset.FEAT_MSA
  data = ProteinStructureDataset(data_dir=args.data_dir,
                                 data_idx=args.data_idx,
                                 pseudo_linker_prob=args.pseudo_linker_prob,
                                 data_rm_mask_prob=args.data_rm_mask_prob,
                                 msa_as_seq_prob=args.msa_as_seq_prob,
                                 feat_flags=feat_flags)
  with timing(f'{args.command}', logging.info):
    work_fn(data, args)


if __name__ == '__main__':
  import argparse

  commands = {
      'checksum': (checksum, checksum_add_argument),
      'contacts': (contacts, contacts_add_argument),
      'to_fasta': (to_fasta, to_fasta_add_argument),
  }

  parser = argparse.ArgumentParser()

  sub_parsers = parser.add_subparsers(dest='command', required=True)
  for cmd, (_, add_argument) in commands.items():
    cmd_parser = sub_parsers.add_parser(cmd)

    cmd_parser.add_argument('--data_dir',
                            type=str,
                            default=None,
                            help='train dataset dir, default=None')
    cmd_parser.add_argument('--data_idx',
                            type=str,
                            default=None,
                            help='dataset idx, default=None')
    cmd_parser.add_argument('--data_rm_mask_prob',
                            type=float,
                            default=0.0,
                            help='data_rm_mask')
    cmd_parser.add_argument('--msa_as_seq_prob',
                            type=float,
                            default=0.0,
                            help='msa_as_mask')
    cmd_parser.add_argument('--pseudo_linker_prob',
                            type=float,
                            default=0.0,
                            help='pseudo_linker_prob')
    cmd_parser.add_argument('-v', '--verbose', action='store_true', help='verbose')

    add_argument(cmd_parser)

  args = parser.parse_args()

  logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

  main(commands[args.command][0], args)
