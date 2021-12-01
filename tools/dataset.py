import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange

# data
from profold2.data import scn, custom

def data_loader(args):
  # get data
  if args.casp_version > 12:
    return custom.load(
                    data_dir=args.casp_data,
                    batch_size=1,
                    num_workers=0,
                    is_training=False)
  data = scn.load(casp_version=args.casp_version,
                  thinning=args.casp_thinning,
                  batch_size=1,
                  num_workers=0,
                  is_training=False)
  return data[args.casp_data]

def to_fasta(data):
  for prot in iter(data):
    assert 'pid' in prot and 'str_seq' in prot
    assert len(prot['pid']) == len(prot['str_seq'])
    for i, pid in enumerate(prot['pid']):
        print(f'>{pid}')
        print(prot['str_seq'][i])

def main(args):
  # get data
  data = data_loader(args)
  to_fasta(data)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-C', '--casp_version', type=int, default=12,
      help='CASP version, default=12')
  parser.add_argument('-T', '--casp_thinning', type=int, default=30,
      help='CASP version, default=30')
  parser.add_argument('-k', '--casp_data', type=str, default='test',
      help='CASP dataset, default=\'test\'')
  args = parser.parse_args()

  main(args)
