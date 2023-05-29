"""Tools for profold2, run
     ```bash
     $python main.py -h
     ```
     for further help.
"""
import os
import argparse

from profold2.command import (
    evaluator,
    msa_processor,
    predictor,
    trainer,
    worker)

_COMMANDS = [
    ('msa_process', msa_processor.msa_process, msa_processor.add_arguments),
    ('train', trainer.train, trainer.add_arguments),
    ('evaluate', evaluator.evaluate, evaluator.add_arguments),
    ('predict', predictor.predict, predictor.add_arguments),
]

def create_args():
  parser = argparse.ArgumentParser()

  # distributed args
  parser.add_argument('--nnodes', type=int, default=None,
      help='number of nodes.')
  parser.add_argument('--node_rank', type=int, default=0,
      help='rank of the node, default=0.')
  parser.add_argument('--local_rank', type=int,
      default=int(os.environ.get('LOCAL_RANK', 0)),
      help='local rank of xpu, default=0.')
  parser.add_argument('--init_method', type=str, default=None,
      help='method to initialize the process group, default=None')

  # command args
  subparsers = parser.add_subparsers(dest='command', required=True)
  for cmd, _, add_arguments in _COMMANDS:
    cmd_parser = subparsers.add_parser(cmd)

    # output dir
    cmd_parser.add_argument('-o', '--prefix', type=str, default='.',
        help='prefix of out directory, default=\'.\'')
    add_arguments(cmd_parser)
    # verbose
    cmd_parser.add_argument('-v', '--verbose', action='store_true',
        help='verbose')

  return parser.parse_args()

def create_fn(args):  # pylint: disable=redefined-outer-name
  for cmd, fn, _ in _COMMANDS:
    if cmd == args.command:
      return fn
  return None

if __name__ == '__main__':
  args = create_args()
  work_fn = create_fn(args)
  worker.main(args, work_fn)
