import os
import logging

import matplotlib.pyplot as plt
import torch
from einops import rearrange

from profold2.common import protein, residue_constants
from profold2.model.features import pseudo_beta_fn

logger = logging.getLogger(__file__)

def main(args):  # pylint: disable=redefined-outer-name
  os.makedirs(args.prefix, exist_ok=True)
  
  breaks = torch.linspace(2.3125, 21.6875, steps=37-1)

  for pdb_file in args.pdb_files:
    with open(pdb_file, 'r') as f:
      pdb_str = f.read()
    prot = protein.from_pdb_string(pdb_str)

    positions, mask = pseudo_beta_fn(
        torch.from_numpy(prot.aatype),
        torch.from_numpy(prot.atom_positions),
        torch.from_numpy(prot.atom_mask))
    sq_breaks = torch.square(breaks)

    dist2 = torch.sum(
        torch.square(
            rearrange(positions, '... i c -> ... i () c') -
            rearrange(positions, '... j c -> ... () j c')),
        dim=-1,
        keepdims=True)

    sq_mask = rearrange(mask, '... i -> ... i ()') * rearrange(mask, '... j -> ... () j')
    true_bins = torch.sum(dist2 > sq_breaks, dim=-1) * sq_mask

    p, _ = os.path.splitext(os.path.basename(pdb_file))
    with open(os.path.join(args.prefix, f'{p}.svg'), 'w') as f:
        plt.matshow(-true_bins)
        plt.savefig(f, format='svg', dpi=100)
        plt.close()

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('-o', '--prefix', type=str, default='.',
      help='prefix of out directory, default=\'.\'')
  parser.add_argument('pdb_files', type=str, nargs='+',
      help='list of pdf files')
  parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
  args = parser.parse_args()

  main(args)
