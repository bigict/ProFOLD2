import logging

from profold2.data.parsers import parse_fasta
from profold2.utils import exists

logger = logging.getLogger(__file__)


def fasta_length_main(args):
  if exists(args.file_list):
    with open(args.file_list, 'r') as f:
      for line in filter(lambda x: len(x) > 0, map(lambda x: x.strip(), f)):
        args.input_files.append(line)
  for fasta_file in args.input_files:
    with open(fasta_file, 'r') as f:
      sequences, descriptions = parse_fasta(f.read())
    for seq, desc in zip(sequences, descriptions):
      print(f'{fasta_file}\t{desc}\t{len(seq)}')


def fasta_length_add_argument(parser):
  parser.add_argument('input_files', type=str, nargs='+', help='list of fasta files')
  parser.add_argument('-l', '--file_list', type=str, default=None, help='list file')
  return parser


if __name__ == '__main__':
  import argparse

  commands = {
      'fasta_length': (fasta_length_main, fasta_length_add_argument),
  }

  parser = argparse.ArgumentParser()

  sub_parsers = parser.add_subparsers(dest='command', required=True)
  for cmd, (_, add_argument) in commands.items():
    cmd_parser = sub_parsers.add_parser(cmd)
    add_argument(cmd_parser)
    cmd_parser.add_argument('-v', '--verbose', action='store_true', help='verbose')

  args = parser.parse_args()

  logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

  work_fn, _ = commands[args.command]
  work_fn(args)
