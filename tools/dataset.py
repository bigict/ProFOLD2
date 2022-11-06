# data
from profold2.data import dataset

def to_fasta(data):
  for prot in iter(data):
    assert 'pid' in prot and 'str_seq' in prot
    assert len(prot['pid']) == len(prot['str_seq'])
    for i, pid in enumerate(prot['pid']):
        print(f'>{pid}')
        print(prot['str_seq'][i])

def main(args):
  # get data
  data_loader = dataset.load(
      data_dir=args.data,
      data_idx=args.name_idx)
  to_fasta(data_loader)

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--data', type=str, default=None,
      help='train dataset dir, default=None')
  parser.add_argument('--name_idx', type=str, default='name.idx',
      help='train dataset idx, default=\'name.idx\'')
  args = parser.parse_args()

  main(args)
