"""Tools for profold2, run
     ```bash
     $python main.py -h
     ```
     for further help.
"""
import os
import argparse

import hydra

from profold2.command import (evaluator, predictor, trainer, worker)
from profold2.utils import env, exists

_COMMANDS = [('train', trainer), ('evaluate', evaluator), ('predict', predictor)]


def create_args():
  formatter_class = argparse.ArgumentDefaultsHelpFormatter
  parser = argparse.ArgumentParser(formatter_class=formatter_class)

  # command args
  subparsers = parser.add_subparsers(dest='command', required=True)
  for cmd, _ in _COMMANDS:
    cmd_parser = subparsers.add_parser(cmd, formatter_class=formatter_class)
    cmd_parser.add_argument(
        '-c', '--config', type=str, default=None, help='config file.'
    )
    cmd_parser.add_argument(
        'overrides',
        nargs='*',
        metavar='KEY=VAL',
        help='override configs, see: https://hydra.cc'
    )

  return parser.parse_args()


def create_task(args):  # pylint: disable=redefined-outer-name
  for cmd, task in _COMMANDS:
    if cmd == args.command:
      return task
  return None


def main():
  args = create_args()
  config_dir, config_name = os.path.split(
      os.path.abspath(args.config)
  ) if exists(args.config) else (os.getcwd(), None)

  with hydra.initialize_config_dir(
      version_base=None, config_dir=config_dir, job_name=args.command
  ):
    task = create_task(args)
    worker.main(task, hydra.compose(config_name, args.overrides))


if __name__ == '__main__':
  main()
