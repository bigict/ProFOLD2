"""Tools for computing enery, run
     ```bash
     $python energy.py -h
     ```
     for further help.
"""
import os
import sys
import math
import pickle

from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

from profold2.common import residue_constants
from profold2.data.dataset import _make_var_features
from profold2.data.parsers import parse_fasta
from profold2.data.utils import decompose_pid
from profold2.model import functional, potts
from profold2.utils import exists, version_cmp


def model_from_ckpt(model_file, device=None):
  map_location = 'cpu'
  if version_cmp(torch.__version__, '2.0.0') >= 0:  # disable=warning
    checkpoint = torch.load(model_file, map_location=map_location, weights_only=False)
  else:
    checkpoint = torch.load(model_file, map_location=map_location)
  sigma = checkpoint['model']['impl.head_fitness.sigma']
  config = {}
  for name, c, _ in checkpoint['headers']:
    if name == 'fitness':
      config = c
      break
  return sigma.to(device=device), config


def model_from_pkl(model_file, mask=None, device=None):
  with open(model_file, 'rb') as f:
    pkl = pickle.load(f)

  assert 'coevolution' in pkl
  wij = torch.from_numpy(pkl['coevolution']['wij']).to(device=device)
  bi = torch.from_numpy(pkl['coevolution']['bi']).to(device=device)
  if 'wab' in pkl['coevolution']:
    wab = torch.from_numpy(pkl['coevolution']['wab']).to(device=device)
    wij = torch.einsum('b i j q,q c d -> b i j c d', wij, wab)
  if exists(mask):
    wij = torch.einsum('b i j c d,d -> b i j c d', wij, mask)

  return wij, bi


def elo_score(U, U0, sigma, config):  # pylint: disable=invalid-name
  softplus = config.get('softplus', False)
  prior_b = config.get('prior_b', None)
  w = sigma

  if softplus:
    w = F.softplus(w)
    if exists(prior_b):
      w = w + prior_b
  elif exists(prior_b):
    w = torch.clamp(w, min=prior_b)
  return F.sigmoid(w * (U0 - U))  # H + U = 0


def main(args):  # pylint: disable=redefined-outer-name
  config = {'pooling': 'sum'}
  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  m = functional.make_mask(
      residue_constants.restypes_with_x_and_gap, args.mask, device=device
  )
  wij, bi = model_from_pkl(args.model_file, mask=m, device=device)

  if exists(args.model_ckpt):
    sigma, config = model_from_ckpt(args.model_ckpt, device=device)
  if config.get('pooling', 'sum') == 'mean':
    print('Using mean pooling')

  pid, _ = os.path.splitext(os.path.basename(args.model_file))
  if pid.endswith('_var'):
    pid = pid[:-4]
  pid, _ = decompose_pid(pid)

  if exists(args.output_file):
    out = open(args.output_file, 'w')  # pylint: disable=consider-using-with
  else:
    out = sys.stdout

  def output(b, U, sequences, descriptions):  # pylint: disable=invalid-name
    if exists(args.model_ckpt):
      X = elo_score(U[b], U0[b], sigma, config)  # pylint: disable=invalid-name
      for _, (s, d, u, x) in enumerate(zip(sequences, descriptions, U[b].tolist(), X)):
        out.write(f'>{d} U:{u} X:{math.exp(u)} Elo_score:{x}\n')
        out.write(f'{s}\n')
    else:
      for _, (s, d, u) in enumerate(zip(sequences, descriptions, U[b].tolist())):
        out.write(f'>{d} U:{u} X:{math.exp(u)}\n')
        out.write(f'{s}\n')

  for a3m_file in args.a3m_file:
    with open(a3m_file, 'r') as f:
      a3m_string = f.read()
    sequences, descriptions = parse_fasta(a3m_string)

    if exists(args.model_ckpt):
      feats = _make_var_features(sequences[:1], descriptions[:1])
      S = feats['variant'].to(device=device)[None]  # pylint: disable=invalid-name
      mask = feats['variant_mask'].to(device=device)[None]
      U0, U_i = potts.energy(S, -bi, -wij, mask)  # pylint: disable=invalid-name
      if config.get('pooling', 'sum') == 'mean':
        U_i = torch.sum(U_i * F.one_hot(S.long(), num_classes=bi.shape[-1]), dim=-1)  # pylint: disable=invalid-name
        U0 = functional.masked_mean(value=U_i, mask=mask, dim=-1)  # pylint: disable=invalid-name

    if not exists(args.chunksize):
      args.chunksize = len(sequences)

    for cidx, cstart in enumerate(
        tqdm(range(0, len(sequences), args.chunksize), desc='Chunked Energy')
    ):
      num_seqs = min(args.chunksize, len(sequences) - cidx * args.chunksize)
      cend = cstart + num_seqs

      feats = _make_var_features(sequences[cstart:cend], descriptions[cstart:cend])
      # S = torch.randint(0, c, size=(b, num_seqs, n), device=device)
      S = feats['variant'].to(device=device)[None]  # pylint: disable=invalid-name
      mask = feats['variant_mask'].to(device=device)[None]

      U, U_i = potts.energy(S, -bi, -wij, mask)  # pylint: disable=invalid-name
      if config.get('pooling', 'sum') == 'mean':
        U_i = torch.sum(U_i * F.one_hot(S.long(), num_classes=bi.shape[-1]), dim=-1)  # pylint: disable=invalid-name
        U = functional.masked_mean(value=U_i, mask=mask, dim=-1)  # pylint: disable=invalid-name

      output(0, U, sequences[cstart:cend], descriptions[cstart:cend])

  if exists(args.output_file):
    out.close()


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

  parser.add_argument(
      'a3m_file', type=str, nargs='+', help='list of model files (.pkl)'
  )
  parser.add_argument(
      '--model_ckpt', type=str, default=None, help='profold model file (.pth)'
  )
  parser.add_argument(
      '--model_file', type=str, default=None, help='fitness model file (.pkl)'
  )
  parser.add_argument(
      '-o',
      '--output_file',
      type=str,
      default=None,
      help='list of amino acides to be masked.'
  )
  parser.add_argument(
      '--chunksize',
      type=int,
      default=None,
      help='number of seqences to design for each model.'
  )
  parser.add_argument(
      '-m', '--mask', type=str, default='-', help='list of amino acides to be masked.'
  )
  parser.add_argument(
      '--field', type=str, default='U', help='list of amino acides to be masked.'
  )

  args = parser.parse_args()

  main(args)
