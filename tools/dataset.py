# data
from profold2.data import dataset

def to_fasta(data, args):
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

def main(args):
  # get data
  data_loader = dataset.load(
      data_dir=args.data,
      data_idx=args.name_idx)
  to_fasta(data_loader, args)

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--data', type=str, default=None,
      help='train dataset dir, default=None')
  parser.add_argument('--name_idx', type=str, default='name.idx',
      help='train dataset idx, default=\'name.idx\'')
  parser.add_argument('--dump_keys', action='store_true', help='dump keys')
  parser.add_argument('--print_fasta', action='store_true', help='print fasta')
  parser.add_argument('--print_first_only', action='store_true', help='print first only')
  args = parser.parse_args()

  main(args)
