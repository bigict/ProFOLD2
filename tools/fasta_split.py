#!/Users/zhangyingchen/opt/anaconda3/envs/pytorch/bin/python3
import os
import functools
import glob
import multiprocessing as mp
import logging

from profold2.data.parsers import parse_fasta


logger = logging.getLogger(__file__)

def process(input_file, args=None):  # pylint: disable=redefined-outer-name
  logger.info('process: %s', input_file)

  with open(input_file, 'r') as f:
    fasta_string = f.read()
  sequences, descriptions = parse_fasta(fasta_string)
  for seq, pid in zip(sequences, descriptions):
    pid = pid.split()[0]
    with open(os.path.join(args.prefix, f'{pid}.fasta'), 'w') as f:
      f.write(f'>{pid}\n')
      f.write(seq)
    logger.info('pid: %s@%s', pid, input_file)
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
  parser.add_argument('-o', '--prefix')
  args = parser.parse_args()

  main(args)
