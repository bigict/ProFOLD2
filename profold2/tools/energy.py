"""Tools for computing enery, run
     ```bash
     $python energy.py -h
     ```
     for further help.
"""
import os
import sys
import json
import math
import pickle

from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from einops import repeat

from profold2.common import residue_constants
from profold2.data.dataset import _make_var_features, _make_task_mask
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

  def _pkl_get(key):
    value = pkl.get(key)
    if exists(value):
      value = torch.from_numpy(value).to(device=device)
    return value

  gating, seq_color = map(_pkl_get, ('gating', 'seq_color'))

  return wij, bi, gating, seq_color


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
  dG = U0 - U
  return F.sigmoid(w * dG)  # H + U = 0


def main(args):  # pylint: disable=redefined-outer-name
  config = {'pooling': 'sum'}
  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  m = functional.make_mask(
      residue_constants.restypes_with_x_and_gap, args.mask, device=device
  )
  wij, bi, gating, seq_color = model_from_pkl(args.model_file, mask=m, device=device)

  if exists(args.model_ckpt):
    sigma, config = model_from_ckpt(args.model_ckpt, device=device)
  if config.get('pooling', 'sum') == 'mean':
    print('Using mean pooling')

  pid, _ = os.path.splitext(os.path.basename(args.model_file))
  if pid.endswith('_var'):
    pid = pid[:-4]
  pid, chains = decompose_pid(pid)
  chains = chains.split(',')
  if exists(seq_color):
    chain_length_list = [
        torch.sum(seq_color[0] == idx + 1).item() for idx in range(len(chains))
    ]
  else:
    chain_length_list = None
  task_def = json.loads(args.task_def) if exists(args.task_def) else None
  task_mask = _make_task_mask(
      torch.ones(seq_color.shape[-1], dtype=torch.bool, device=seq_color.device),
      chains,
      chain_length_list,
      task_def=task_def,
      task_num=config.get('task_num', 1)
  )
  task_mask = repeat(task_mask, 'i c -> b () i c', b=seq_color.shape[0])

  if exists(args.output_file):
    out = open(args.output_file, 'w')  # pylint: disable=consider-using-with
  else:
    out = sys.stdout

  def _output(b, U, sequences, descriptions, U0=None):  # pylint: disable=invalid-name
    if exists(U0):
      X = elo_score(U[b], U0[b], sigma, config)  # pylint: disable=invalid-name
      for _, (s, d, u, x) in enumerate(zip(sequences, descriptions, U[b], X)):
        out.write(f'>{d}\tU:{u.tolist()}\tX:{torch.exp(u).tolist()}\tElo_score:{x.tolist()}\n'
        )
        out.write(f'{s}\n')
    else:
      for _, (s, d, u) in enumerate(zip(sequences, descriptions, U[b])):
        out.write(f'>{d}\tU:{u.tolist()}\tX:{torch.exp(u).tolist()}\n')
        out.write(f'{s}\n')

  def _energy(S, mask):
    U0, U_i = potts.energy(S, -bi, -wij, mask)  # pylint: disable=invalid-name
    if config.get('log_softmax', False):
      U_i = F.log_softmax(U_i, dim=-1)
    U_i = torch.sum(
        U_i * F.one_hot(S.long(), num_classes=bi.shape[-1]), dim=-1, keepdim=True  # pylint: disable=invalid-name
    )
    U_i = U_i * mask[..., None]
    if exists(gating):
      U_i = U_i * gating
    if config.get('pooling', 'sum') == 'mean':
      U0 = functional.masked_mean(value=U_i, mask=task_mask, dim=-2)  # pylint: disable=invalid-name
    else:
      U0 = torch.sum(U_i, dim=-2)
    return U0


  for a3m_file in args.a3m_file:
    with open(a3m_file, 'r') as f:
      a3m_string = f.read()
    sequences, descriptions = parse_fasta(a3m_string)

    if exists(args.model_ckpt):
      feats = _make_var_features(sequences[:1], descriptions[:1])
      S = feats['variant'].to(device=device)[None]  # pylint: disable=invalid-name
      mask = feats['variant_mask'].to(device=device)[None]
      U0 = _energy(S, mask)
    else:
      U0 = None

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
      U = _energy(S, mask)

      _output(0, U, sequences[cstart:cend], descriptions[cstart:cend], U0=U0)

  if exists(args.output_file):
    out.close()


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

  parser.add_argument(
      'a3m_file',
      type=argparse.FileType('r'),
      nargs='+',
      help='list of a3m files (.a3m)'
  )
  parser.add_argument(
      '--model_ckpt', type=str, default=None, help='profold model file (.pth)'
  )
  parser.add_argument(
      '--model_file', type=str, default=None, help='fitness model file (.pkl)'
  )
  parser.add_argument(
      '--task_def', type=str, default=None, help='task definition (json)'
  )
  parser.add_argument(
      '-o',
      '--output_file',
      type=argparse.FileType('w'),
      default=sys.stdout,
      help='output file'
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

  args = parser.parse_args()

  main(args)
