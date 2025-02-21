"""Tools for construct BLOSUM from si, run
     ```bash
     $python blosum_from_si.py -h
     ```
     for further help.
"""
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from einops import rearrange
from Bio.Align import substitution_matrices

from profold2.common import residue_constants


def si_from_prediction(state_dict):
  a = state_dict.headers['profile']['logits']
  a = torch.softmax(rearrange(a, '... i a -> (... i) a'), dim=-1)
  return a


def confidence_from_prediction(state_dict):
  a = state_dict.headers['confidence']['loss'].item()
  return a


def evolution_from_prediction(state_dict):
  a = state_dict.headers['metric']['loss']['profile']['cosine'].item()
  return a


def similarity(x, y):
  x, y = torch.as_tensor(x), torch.as_tensor(y)
  p = torch.cat([torch.diagonal(x, offset=i) for i in (0, 1)], dim=-1)
  q = torch.cat([torch.diagonal(y, offset=i) for i in (0, 1)], dim=-1)
  a = F.cosine_similarity(p, q, dim=0)
  return a.item()


def main(args):  # pylint: disable=redefined-outer-name
  blosum62 = substitution_matrices.load('BLOSUM62')

  state_dict_list = [torch.load(f, map_location='cpu') for f in args.input_files]

  plddt_list = [
      confidence_from_prediction(state_dict) for state_dict in state_dict_list
  ]
  profile_list = [
      evolution_from_prediction(state_dict) for state_dict in state_dict_list
  ]

  a = torch.cat(
      [si_from_prediction(state_dict) for state_dict in state_dict_list], dim=0
  )
  b = torch.sum(a, dim=0)
  num = torch.sum(rearrange(a, 'i a -> i a ()') * rearrange(a, 'i b -> i () b'), dim=0)
  den = rearrange(b, 'a -> a ()') * rearrange(b, 'b -> () b') + args.epsilon
  s = args.gamma * torch.log2(args.eta * num / den)
  # print(s)

  blosum62_aa = 'CSTAGPDEQNHRKMILVWYF'
  if args.verbose:
    print('  '.join(f'    {a}' for a in blosum62_aa))

  x = [[0] * len(blosum62_aa) for _ in blosum62_aa]
  y = [[0] * len(blosum62_aa) for _ in blosum62_aa]
  for i, aa in enumerate(blosum62_aa):
    t = [f'{aa}']
    for j, bb in enumerate(blosum62_aa):
      p = residue_constants.restype_order[aa]
      q = residue_constants.restype_order[bb]
      v = s[p, q].item()
      t.append(f'{v:+4.2f}')
      x[i][j] = v
      y[i][j] = blosum62[(aa, bb)]
    if args.verbose:
      print('  '.join(t))

  #print(sum(plddt_list)/len(plddt_list))
  #for i in range(len(blosum62_aa)):
  #  print(i, x[i][i], y[i][i])
  fields = [
      f'{sum(plddt_list)/len(plddt_list)}',
      f'{similarity(x, y)}',
      f'{sum(profile_list)/len(profile_list)}',
  ]
  print('\t'.join(fields))
  if (args.plot_svg_file):
    with open(args.plot_svg_file, 'w') as f:
      plt.matshow(-true_bins, **kwargs)
      plt.savefig(-x, format='svg', dpi=100)
      plt.close()


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument(
      'input_files', metavar='file', type=str, nargs='+', help='input files (.pth)'
  )
  parser.add_argument('--gamma', type=float, default=1.0, help='input files (.pth)')
  parser.add_argument('--eta', type=float, default=100, help='input files (.pth)')
  parser.add_argument('--epsilon', type=float, default=1e-9, help='input files (.pth)')
  parser.add_argument(
      '--plot_svg_file', type=str, default=None, help='output svg file (.svg)'
  )
  parser.add_argument('-v', '--verbose', action='store_true', help='verbose')

  args = parser.parse_args()

  main(args)
