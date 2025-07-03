"""Tools for plot contact, run
     ```bash
     $python contact_plot.py -h
     ```
     for further help.
"""
import os
import math

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from einops import rearrange

from profold2.common import protein, residue_constants
from profold2.data.dataset import ProteinStructureDataset
from profold2.model.features import pseudo_beta_fn
from profold2.utils import exists

b_colors = [
    (168 / 256, 3 / 256, 38 / 256),
    (218 / 256, 56 / 256, 42 / 256),
    (246 / 256, 121 / 256, 72 / 256),
    (253 / 256, 185 / 256, 107 / 256),
    (254 / 256, 233 / 256, 157 / 256),
    (244 / 256, 251 / 256, 211 / 256),
    (1, 1, 1),
    (202 / 256, 232 / 256, 242 / 256),
    (146 / 256, 197 / 256, 222 / 256),
    (92 / 256, 144 / 256, 194 / 256),
    (57 / 256, 81 / 256, 162 / 256),
]
gulf_colors = [
    (1, 1, 1),
    (227 / 256, 246 / 256, 253 / 256),
    (187 / 256, 222 / 256, 251 / 256),
    (144 / 256, 202 / 256, 249 / 256),
    (100 / 256, 181 / 256, 246 / 256),
    (66 / 256, 165 / 256, 245 / 256),
    (33 / 256, 150 / 256, 243 / 256),
    (30 / 256, 136 / 256, 229 / 256),
    (25 / 256, 118 / 256, 210 / 256),
    (21 / 256, 101 / 256, 192 / 256),
    (13 / 256, 71 / 256, 161 / 256),
    (4 / 256, 20 / 256, 50 / 256),
]
cmap = dict(
    gulf=mcolors.LinearSegmentedColormap.from_list('my_colormap', gulf_colors, N=256)
)


def _dataset_load(data_dir, data_idx='name.idx'):
  data = ProteinStructureDataset(data_dir, data_idx=data_idx)
  pid_dict = {}
  for idx, cluster in enumerate(data.pids):
    for k, pid in enumerate(cluster):
      pid_dict[pid] = (idx, k)
  return data, pid_dict


def main(args):  # pylint: disable=redefined-outer-name
  os.makedirs(args.prefix, exist_ok=True)

  breaks = torch.linspace(2.3125, 21.6875, steps=37 - 1)

  if exists(args.data_dir):
    datum, pid_dict = _dataset_load(args.data_dir, args.data_idx)
  for pdb_file in args.pdb_files:
    if exists(args.data_dir):
      idx = pid_dict[pdb_file]
      prot = datum[idx]
      positions, mask = pseudo_beta_fn(prot['seq'], prot['coord'], prot['coord_mask'])
      seq = prot['seq']
    else:
      with open(pdb_file, 'r') as f:
        pdb_str = f.read()
      prot = protein.from_pdb_string(pdb_str)

      positions, mask = pseudo_beta_fn(
          torch.from_numpy(prot.aatype), torch.from_numpy(prot.atom_positions),
          torch.from_numpy(prot.atom_mask)
      )
      seq = prot.aatype
    sq_breaks = torch.square(breaks)

    dist2 = torch.sum(
        torch.square(
            rearrange(positions, '... i c -> ... i () c') -
            rearrange(positions, '... j c -> ... () j c')
        ),
        dim=-1,
        keepdims=True
    )

    sq_mask = rearrange(mask,
                        '... i -> ... i ()') * rearrange(mask, '... j -> ... () j')
    true_bins = torch.sum(dist2 > sq_breaks, dim=-1) * sq_mask

    if args.verbose:
      m, n = dist2.shape[-3:-1]
      for i in range(m):
        for j in range(i + 1, n):
          if sq_mask[i, j]:
            d = math.sqrt(dist2[i, j].item())
            si = residue_constants.restypes_with_x[seq[i]]
            sj = residue_constants.restypes_with_x[seq[j]]
            print(f'{i+1}\t{j+1}\t{j-i}\t{si}\t{sj}\t{d}')

    p, _ = os.path.splitext(os.path.basename(pdb_file))
    with open(os.path.join(args.prefix, f'{p}.svg'), 'w') as f:
      kwargs = {}
      if exists(args.cmap):
        kwargs['cmap'] = cmap[args.cmap]
      plt.matshow(-true_bins, **kwargs)
      plt.savefig(f, format='svg', dpi=100)
      plt.close()


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-o',
      '--prefix',
      type=str,
      default='.',
      help='prefix of out directory, default=\'.\''
  )
  parser.add_argument(
      '-d', '--data_dir', type=str, default=None, help='dataset dir, default=None'
  )
  parser.add_argument(
      '--data_idx',
      type=str,
      default='name.idx',
      help='dataset index, default=\'name.idx\''
  )
  parser.add_argument('--cmap', type=str, default='gulf', help='The colormap')
  parser.add_argument('pdb_files', type=str, nargs='+', help='list of pdf files')
  parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
  args = parser.parse_args()

  main(args)
