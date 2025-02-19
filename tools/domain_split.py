"""Tools for split fasta to single sequence file, run
     ```bash
     $python fasta_split.py -h
     ```
     for further help.
"""
import os
import functools
import glob
import multiprocessing as mp
import re
import logging

from profold2.data.parsers import parse_fasta

logger = logging.getLogger(__file__)

seq_index_pattern = '(\\d+)-(\\d+)'


def domain_index_split(text):
  for s in text.split(','):
    r = re.match(seq_index_pattern, s)
    assert r
    yield tuple(map(int, r.group(1, 2)))


def domain_index_find(description):
  positions = []

  fields = description.split()
  for f in fields[1:]:
    r = re.match(f'.*:({seq_index_pattern}(,{seq_index_pattern})*)', f)
    if r:
      positions = list(domain_index_split(r.group(1)))
      break

  return positions


def process(item, args=None):  # pylint: disable=redefined-outer-name
  if args.domain_index_from_fasta:
    input_file = item
  else:
    input_file, domain_index = item
    positions = list(domain_index_split(domain_index))

  logger.info('process: %s', input_file)

  filename, _ = os.path.splitext(os.path.basename(input_file))
  with open(input_file, 'r') as f:
    fasta_string = f.read()
  sequences, descriptions = parse_fasta(fasta_string)
  assert len(sequences) == 1
  for seq, pid in zip(sequences, descriptions):
    if args.domain_index_from_fasta:
      positions = domain_index_find(pid)
    print(positions)
    pid = pid.split()[0]
    if args.filename_as_pid:
      pid = filename

    delta = 0
    if positions and args.domain_from_compat_fasta:
      delta = positions[0][0] - 1
    for k, (i, j) in enumerate(positions):
      if args.domain_from_compat_fasta and k > 0:
        delta += positions[k][0] - positions[k - 1][1] - 1
      with open(os.path.join(args.prefix, f'{pid}_{k}.fasta'), 'w') as f:
        f.write(f'>{pid}@{k} {pid}:{i}-{j}\n')
        f.write(seq[i - 1 - delta:j - delta])
      logger.info('pid: %s@%s->%s', pid, k, input_file)
    break

  return input_file


def main(args):  # pylint: disable=redefined-outer-name
  logger.info('args - %s', args)

  os.makedirs(args.prefix, exist_ok=True)
  if args.domain_index_from_fasta:
    inputs = args.inputs
  else:
    inputs = [
        (args.inputs[i], args.inputs[i + 1]) for i in range(0, len(args.inputs), 2)
    ]

  with mp.Pool() as p:
    for _ in p.imap(functools.partial(process, args=args), inputs):
      pass


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('inputs', metavar='file', type=str, nargs='+', help='inputs')
  parser.add_argument('-o', '--prefix')
  parser.add_argument('--filename_as_pid', action='store_true', help='')
  parser.add_argument('--domain_index_from_fasta', action='store_true', help='')
  parser.add_argument('--domain_from_compat_fasta', action='store_true', help='')
  parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
  args = parser.parse_args()

  logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

  main(args)
