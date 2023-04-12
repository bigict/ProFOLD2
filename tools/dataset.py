"""Tools for check dataset file, run
     ```bash
     $python dataset.py -h
     ```
     for further help.
"""
import logging

from profold2.data import dataset


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
    if args.checksum:
      #print(prot['pid'], 'Cheking...')
      n = len(prot['str_seq'][0])
      assert 'msa' in prot
      if n != prot['msa'].shape[2]:
        print(prot['pid'], n, prot['msa'].shape)
      if 'coord' in prot:
        if n != prot['coord'].shape[1]:
          print(prot['pid'], n, prot['coord'].shape)
      if 'coord_mask' in prot:
        if n != prot['coord_mask'].shape[1]:
          print(prot['pid'], n, prot['coord_mask'].shape)


def main(args):  # pylint: disable=redefined-outer-name
  # get data
  data_loader = dataset.load(data_dir=args.data, data_idx=args.name_idx)
  to_fasta(data_loader, args)


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--data',
                      type=str,
                      default=None,
                      help='train dataset dir, default=None')
  parser.add_argument('--name_idx',
                      type=str,
                      default='name.idx',
                      help='train dataset idx, default=\'name.idx\'')
  parser.add_argument('--dump_keys', action='store_true', help='dump keys')
  parser.add_argument('--checksum', action='store_true', help='dump keys')
  parser.add_argument('--print_fasta', action='store_true', help='print fasta')
  parser.add_argument('--print_first_only',
                      action='store_true',
                      help='print first only')
  parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
  args = parser.parse_args()

  logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

  main(args)
