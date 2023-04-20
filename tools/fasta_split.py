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

def yield_rcsb_chain(desc):
  # 8HC4_2|Chains D, J[auth K], N|Heavy chain of R1-32 Fab|Homo sapiens (9606) 
  chain_p = '([0-9a-zA-Z]+)(\\[auth ([0-9a-zA-Z]+)\\])?'
  m = re.match(f'Chain[s]? ({chain_p}(, {chain_p})*)', desc)
  if m:
    for item in map(lambda x: x.strip(), m.group(1).split(',')):
      m = re.match(chain_p, item)
      if m.group(3):
        yield m.group(3)
      else:
        yield m.group(1)

def pid_split(pid):
  k = pid.find('_')
  if k != -1:
    return pid[:k], pid[k+1:]
  return pid, None

def pid_join(pid, chain):
  if chain:
    return f'{pid}_{chain}'
  return pid

def parse_desc(desc, fmt):
  if fmt == 'rcsb_new':
    desc = desc.split('|')
    pid, chain = pid_split(desc[0])
    pid = pid.lower()

    chains = []
    for d in desc[1:]:
      chains += list(yield_rcsb_chain(d))
    if chains:
      return [(pid, c) for c in chains]
    return [(pid, chain)]
  else:
    desc = desc.split()
    pid, chain = pid_split(desc[0])
  return [(pid, chain)]

def process(input_file, args=None):  # pylint: disable=redefined-outer-name
  logger.info('process: %s', input_file)

  with open(input_file, 'r') as f:
    fasta_string = f.read()
  try:
    sequences, descriptions = parse_fasta(fasta_string)
    for seq, desc in zip(sequences, descriptions):
      for pid, chain in parse_desc(desc, args.fasta_desc_fmt):
        n = pid_join(pid, chain)
        with open(os.path.join(args.prefix, f'{n}.fasta'), 'w') as f:
          f.write(f'>{n}\n')
          f.write(seq)
      logger.info('pid: %s@%s', pid, input_file)
  except:
    pass
  return input_file

def main(args):  # pylint: disable=redefined-outer-name
  logger.info('args - %s', args)

  os.makedirs(args.prefix, exist_ok=True)
  input_files = functools.reduce(lambda x,y: x+y,
      [glob.glob(input_file) for input_file in args.input_files])

  with mp.Pool() as p:
    for _ in p.imap(
        functools.partial(process, args=args), input_files):
      pass


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('input_files', metavar='file', type=str, nargs='+',
      help='input files')
  parser.add_argument('--fasta_desc_fmt', type=str,
      help='fasta format')
  parser.add_argument('-o', '--prefix')
  args = parser.parse_args()

  main(args)
