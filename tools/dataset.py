"""Tools for check dataset file, run
     ```bash
     $python dataset.py -h
     ```
     for further help.
"""
import logging

from profold2.data import dataset
from profold2.data.utils import weights_from_file
from profold2.utils import timing


def to_fasta(data, args):  # pylint: disable=redefined-outer-name
  for prot in iter(data):
    if args.dump_keys:
      print(prot.keys())
    assert 'pid' in prot and 'str_seq' in prot
    assert len(prot['pid']) == len(prot['str_seq'])
    if args.print_fasta:
      for i, pid in enumerate(prot['pid']):
        print(f'>{pid}')
        print(prot['str_seq'][i])
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
    if 'coord_mask' in prot:
      if n != prot['coord_mask'].shape[1]:
        print(prot['pid'], n, prot['coord_mask'].shape)

def checksum_add_argument(parser):  # pylint: disable=redefined-outer-name
  parser.add_argument('--msa_required', action='store_true',
                      help='MSA required')
  return parser

def main(work_fn, args):  # pylint: disable=redefined-outer-name
  # get data
  data_loader = dataset.load(data_dir=args.data_dir,
                             data_idx=args.data_idx,
                             feat_flags=dataset.FEAT_ALL & (~dataset.FEAT_MSA),
                             weights=list(weights_from_file(args.data_weights)))
  with timing(f'{args.command}', print):
    work_fn(data_loader, args)


if __name__ == '__main__':
  import argparse

  commands = {
    'checksum': (checksum, checksum_add_argument),
    'to_fasta': (to_fasta, to_fasta_add_argument),
  }

  parser = argparse.ArgumentParser()

  sub_parsers = parser.add_subparsers(dest='command', required=True)
  for cmd, (_, add_argument) in commands.items():
    cmd_parser = sub_parsers.add_parser(cmd)
    add_argument(cmd_parser)

  parser.add_argument('--data_dir',
                      type=str,
                      default=None,
                      help='train dataset dir, default=None')
  parser.add_argument('--data_idx',
                      type=str,
                      default=None,
                      help='dataset idx, default=None')
  parser.add_argument('--data_weights',
                      type=str,
                      default=None,
                      help='sample data by weights, default=None')
  parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
  args = parser.parse_args()

  logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

  main(commands[args.command], args)
