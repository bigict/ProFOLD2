from functools import reduce
import logging
import re

import torch

PREFIX = 'module.'


def params_count_do(params, pattern=None, verbose=False, verbose_w=False):
  n = 0
  if isinstance(params, list):
    for p in params:
      n += params_count_do(p,
                           pattern=pattern,
                           verbose=verbose,
                           verbose_w=verbose_w)
  if isinstance(params, dict):
    for k, p in params.items():
      if pattern is None or pattern.match(k):
        n += params_count_do(p, pattern=pattern)
        if verbose:
          print(k, p.shape)
        if verbose_w:
          print(f'verbose_w\t{k}\t{p.tolist()}')
  elif isinstance(params, torch.Tensor):
    n += reduce(lambda a, b: a * b, params.shape)
  else:
    print(type(params))
    n += 1
    #assert False, params
  return n


def params_count_main(args):
  p = re.compile(args.grep) if args.grep else None
  for model_file in args.model_file:
    m = torch.load(model_file, map_location='cpu')
    assert 'model' in m
    c = params_count_do(m['model'],
                        pattern=p,
                        verbose=args.verbose,
                        verbose_w=args.verbose_w)
    print(f'{c}\t{model_file}')


def params_count_add_argument(parser):
  parser.add_argument('model_file',
                      type=str,
                      nargs='+',
                      help='list of model files')
  parser.add_argument('-E',
                      '--grep',
                      type=str,
                      default=None,
                      help='parameter patterns, default=None')
  parser.add_argument('-w',
                      '--verbose-w',
                      action='store_true',
                      help='verbose w')
  return parser


def params_modify_main(args):
  assert len(args.grep) % 2 == 0
  logging.debug(args.grep)

  x = torch.load(args.model_files[0], map_location='cpu')
  logging.debug(x.keys())

  if 'optimizer' in x:
    del x['optimizer']

  o = {}
  for key, val in x['model'].items():
    key_new = key
    for i in range(0, len(args.grep), 2):
      key_new = re.sub(args.grep[i], args.grep[i + 1], key_new)
    if key_new != key:
      logging.debug('from <%s> to <%s>', key, key_new)
    o[key_new] = val
  x['model'] = o

  torch.save(x, args.model_files[1])
  logging.info('done.')

def params_modify_add_argument(parser):
  parser.add_argument('model_files',
                      type=str,
                      nargs=2,
                      help='list of model files')
  parser.add_argument('-E',
                      '--grep',
                      type=str,
                      nargs='+',
                      default=None,
                      help='parameter patterns, default=None')
  return parser


def strip_optim_main(args):
  x = torch.load(args.model_files[0], map_location='cpu')
  logging.debug(x.keys())
  if 'optimizer' in x:
    del x['optimizer']
  logging.debug(x.keys())
  if args.grep:
    p = re.compile(args.grep)
    keys = [k for k in x['model'] if p.match(k)]
    logging.debug(keys)
    for k in keys:
      del x['model'][k]

  torch.save(x, args.model_files[1])
  logging.info('done.')


def strip_optim_add_argument(parser):
  parser.add_argument('model_files',
                      type=str,
                      nargs=2,
                      help='list of model files')
  parser.add_argument('-E',
                      '--grep',
                      type=str,
                      default=None,
                      help='parameter patterns, default=None')
  return parser


def to_state_dict_main(args):
  x = torch.load(args.model_files[0], map_location='cpu')
  logging.debug(x.keys())

  d = {}
  for k, v in x['model'].items():
    d[f'{PREFIX}{k}'] = v

  torch.save({'model': d}, args.model_files[1])
  logging.info('done.')


def to_state_dict_add_argument(parser):
  parser.add_argument('model_files',
                      type=str,
                      nargs=2,
                      help='list of model files')
  return parser


if __name__ == '__main__':
  import argparse

  commands = {
      'params_count': (params_count_main, params_count_add_argument),
      'params_modify': (params_modify_main, params_modify_add_argument),
      'strip_optim': (strip_optim_main, strip_optim_add_argument),
      'to_state_dict': (to_state_dict_main, to_state_dict_add_argument),
  }

  parser = argparse.ArgumentParser()

  sub_parsers = parser.add_subparsers(dest='command', required=True)
  for cmd, (_, add_argument) in commands.items():
    cmd_parser = sub_parsers.add_parser(cmd)
    add_argument(cmd_parser)
    cmd_parser.add_argument('-v',
                            '--verbose',
                            action='store_true',
                            help='verbose')

  args = parser.parse_args()

  logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

  work_fn, _ = commands[args.command]
  work_fn(args)
