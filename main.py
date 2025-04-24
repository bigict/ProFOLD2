"""Tools for profold2, run
     ```bash
     $python main.py -h
     ```
     for further help.
"""
import argparse

from profold2.command import (evaluator, predictor, trainer, worker)
from profold2.utils import env

_COMMANDS = [
    ('train', trainer.train, trainer.add_arguments),
    ('evaluate', evaluator.evaluate, evaluator.add_arguments),
    ('predict', predictor.predict, predictor.add_arguments),
]


def create_args():
  formatter_class = argparse.ArgumentDefaultsHelpFormatter
  parser = argparse.ArgumentParser(formatter_class=formatter_class)

  # distributed args
  parser.add_argument(
      '--nnodes',
      type=int,
      default=env('SLURM_NNODES', defval=None, func=int),
      help='number of nodes.'
  )
  parser.add_argument(
      '--node_rank',
      type=int,
      default=env('SLURM_NODEID', defval=0, func=int),
      help='rank of the node.'
  )
  parser.add_argument(
      '--local_rank',
      type=int,
      default=int(env('LOCAL_RANK', defval=0, func=int)),
      help='local rank of xpu.'
  )
  parser.add_argument(
      '--init_method',
      type=str,
      default=None,
      help='method to initialize the process group.'
  )

  # command args
  subparsers = parser.add_subparsers(dest='command', required=True)
  for cmd, _, add_arguments in _COMMANDS:
    cmd_parser = subparsers.add_parser(cmd, formatter_class=formatter_class)

    # output dir
    cmd_parser.add_argument(
        '-o', '--prefix', type=str, default='.', help='prefix of out directory.'
    )
    add_arguments(cmd_parser)
    # verbose
    cmd_parser.add_argument('-v', '--verbose', action='store_true', help='verbose')

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
