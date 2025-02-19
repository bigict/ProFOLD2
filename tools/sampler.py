import os
import sys
import pickle

from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

from profold2.common import residue_constants
from profold2.data.utils import decompose_pid
from profold2.model import complexity, functional, sampler
from profold2.utils import exists

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

def main(args):
  if not exists(args.chunksize):
    args.chunksize = args.num_seqs

  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  m = functional.make_mask(
      residue_constants.restypes_with_x_and_gap, args.mask, device=device
  )
  ban_S = torch.argwhere(m == 0)

  for result_file in args.model_file:
    pid, _ = os.path.splitext(os.path.basename(result_file))
    if pid.endswith('_var'):
      pid = pid[:-4]
    pid, chain = decompose_pid(pid)

    wij, bi = model_from_pkl(result_file, mask=m, device=device)

    b, n, c = bi.shape

    if exists(args.output_file):
      f = open(args.output_file, 'w')
    else:
      f = sys.stdout

    for cidx, _ in enumerate(
        tqdm(range(0, args.num_seqs, args.chunksize), desc='Chunked Sampling')
    ):
      num_seqs = min(args.chunksize, args.num_seqs - cidx * args.chunksize)

      S = torch.randint(0, c, size=(b, num_seqs, n), device=device)
      C = torch.ones(b, n, device=device)
      mask = torch.ones(b, n, device=device)

      penalty_func = lambda _S: complexity.complexity_lcp(_S, C, mask)
      X, U = sampler.from_potts(
          -bi,
          -wij,
          mask,
          S=S,
          mask_ban=ban_S,
          num_sweeps=args.num_sweeps,
          temperature=args.temperature,
          temperature_init=args.temperature_init,
          penalty_func=penalty_func
      )

      def output(b, X, U):
        for i, (x, u) in enumerate(sorted(zip(X[b], U[b]), key=lambda x: x[1])):
          str_seq = ''.join(
              residue_constants.restypes_with_x_and_gap[a] for a in x
          )
          f.write(f'>{pid}_{b}_{i}_{cidx}_{chain} U:{u} chunk:{cidx}\n')
          f.write(f'{str_seq}\n')

      X, U = X.tolist(), U.tolist()
      for k in range(len(X)):
        output(k, X, U)

    if exists(args.output_file):
      f.close()

if __name__ == '__main__':
  import argparse

  formatter_class = argparse.ArgumentDefaultsHelpFormatter
  parser = argparse.ArgumentParser(formatter_class=formatter_class)

  parser.add_argument('model_file', type=str, nargs='+',
                      help='list of Potts Model files (.pkl)')
  parser.add_argument('-o', '--output_file', type=str, default=None,
      help='write sampled sequences to output file.')
  parser.add_argument('-n', '--num_seqs', type=int, default=1,
      help='number of seqences to design for each model.')
  parser.add_argument('--chunksize', type=int, default=None,
      help='split num_seqs to chunks.')
  parser.add_argument('-m', '--mask', type=str, default=None,
      help='list of amino acides to be masked.')
  parser.add_argument('--num_sweeps', type=int, default=100,
      help='number of sweeps of MCMC to perform.')
  parser.add_argument('--temperature', type=float, default=0.1,
      help='final sampling temperature.')
  parser.add_argument('--temperature_init', type=float, default=1.0,
      help='initial sampling temperature.')

  args = parser.parse_args()

  main(args)
