from functools import reduce
import logging
import re

import torch
from einops import rearrange

PREFIX = 'module.'


def params_count_do(params, pattern=None, verbose=False, verbose_w=False):
  n = 0
  if isinstance(params, list):
    for p in params:
      n += params_count_do(p, pattern=pattern, verbose=verbose, verbose_w=verbose_w)
  if isinstance(params, dict):
    for k, p in params.items():
      if pattern is None or pattern.match(k):
        n += params_count_do(p, pattern=pattern)
        if verbose:
          print(k, p.shape)
        if verbose_w:
          print(f'verbose_w\t{k}\t{p.tolist()}')
  elif isinstance(params, torch.Tensor):
    if params.shape:
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
    c = params_count_do(
        m['model'], pattern=p, verbose=args.verbose, verbose_w=args.verbose_w
    )
    print(f'{c}\t{model_file}')


def params_count_add_argument(parser):
  parser.add_argument('model_file', type=str, nargs='+', help='list of model files')
  parser.add_argument(
      '-E', '--grep', type=str, default=None, help='parameter patterns, default=None'
  )
  parser.add_argument('-w', '--verbose-w', action='store_true', help='verbose w')
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
  parser.add_argument('model_files', type=str, nargs=2, help='list of model files')
  parser.add_argument(
      '-E',
      '--grep',
      type=str,
      nargs='+',
      default=None,
      help='parameter patterns, default=None'
  )
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
  parser.add_argument('model_files', type=str, nargs=2, help='list of model files')
  parser.add_argument(
      '-E', '--grep', type=str, default=None, help='parameter patterns, default=None'
  )
  return parser


def relpos_modify_main(args):
  def _linear_key(key):
    return f'{key}.weight', f'{key}.bias'

  x = torch.load(args.model_files[0], map_location='cpu')
  logging.debug(x.keys())

  aa_num, base_num = 21, 5
  mol_num = aa_num + base_num * 2
  gap_num = 2

  # relpos embed.
  # impl.embedder.to_pairwise_emb.relative_pos_emb.embedding.weight
  token_emb_key = f'{args.key_prefix}relative_pos_emb.embedding.weight'
  print(token_emb_key)
  assert token_emb_key in x['model']
  m = x['model'][token_emb_key]
  assert len(m.shape) == 2
  print(m.shape)

  padd_dim = 1
  if args.s_max:
    padd_dim += 2 * args.r_max + 2 * args.s_max + 5
  padd = torch.zeros(padd_dim, m.shape[1]).float()
  m = torch.cat((m, padd), dim=0)
  m = rearrange(m, 'c d -> d c')
  print(m.shape)
  x['model'][token_emb_key] = m

  if 'optimizer' in x:
    del x['optimizer']

  torch.save(x, args.model_files[1])
  logging.info('done.')

def relpos_modify_add_argument(parser):
  parser.add_argument('model_files',
                      type=str,
                      nargs=2,
                      help='list of model files')
  parser.add_argument('-k',
                      '--key_prefix',
                      type=str,
                      default='module.impl.',
                      help='prefix of key.')
  parser.add_argument('--r_max',
                      type=int,
                      default=32,
                      help='prefix of key.')
  parser.add_argument('--s_max',
                      type=int,
                      default=None,
                      help='prefix of key.')
  return parser

def tokens_modify_main(args):
  def _linear_key(key):
    return f'{key}.weight', f'{key}.bias'

  x = torch.load(args.model_files[0], map_location='cpu')
  logging.debug(x.keys())

  aa_num, base_num = 21, 5
  mol_num = aa_num + base_num * 2
  gap_num = 2

  # token embed.
  token_emb_key = f'{args.key_prefix}token_emb.weight'
  assert token_emb_key in x['model']
  m = x['model'][token_emb_key]
  assert len(m.shape) == 2
  print(m.shape)
  assert m.shape[0] == aa_num + 1

  padd = torch.empty(mol_num + 1 - m.shape[0], m.shape[1])
  torch.nn.init.uniform_(padd)
  m = torch.cat((m[:-1], padd, m[-1:]), dim=0)
  x['model'][token_emb_key] = m

  """
  module.impl.head_folding.struct_module.to_angles.to_groups.weight torch.Size([14, 128])
  module.impl.head_folding.struct_module.to_angles.to_groups.bias torch.Size([14])
  chi_angles_num

  """
  # angle_net
  angle_net_key = f'{args.key_prefix}head_folding.struct_module.to_angles.to_groups'
  angle_net_weight_key, angle_net_bias_key = _linear_key(angle_net_key)
  if angle_net_weight_key in x['model']:
    m = x['model'][angle_net_weight_key]
    assert m.shape[0] == 7 * 2
    padd = torch.zeros(11 * 2 - m.shape[0], m.shape[1], dtype=m.dtype)
    torch.nn.init.uniform_(padd)
    m = torch.cat((m, padd), dim=0)
    x['model'][angle_net_weight_key] = m
  if angle_net_bias_key in x['model']:
    m = x['model'][angle_net_bias_key]
    assert m.shape[0] == 7 * 2
    padd = torch.zeros(11 * 2 - m.shape[0], dtype=m.dtype)
    torch.nn.init.uniform_(padd)
    m = torch.cat((m, padd), dim=0)
    x['model'][angle_net_bias_key] = m

  # sequence profile
  profile_key = f'{args.key_prefix}head_profile.project.3'
  profile_weight_key, profile_bias_key = _linear_key(profile_key)
  if profile_weight_key in x['model']:
    m = x['model'][profile_weight_key]
    assert m.shape[0] == aa_num + 1
    padd = torch.zeros(mol_num + 1 + gap_num - m.shape[0], m.shape[1], dtype=m.dtype)
    torch.nn.init.uniform_(padd)
    m = torch.cat((m[:-2], padd[:-gap_num], m[-2:], padd[-gap_num:]), dim=0)
    x['model'][profile_weight_key] = m
  if profile_bias_key in x['model']:
    m = x['model'][profile_bias_key]
    assert m.shape[0] == aa_num + 1
    padd = torch.zeros(mol_num + 1 + gap_num - m.shape[0], dtype=m.dtype)
    torch.nn.init.uniform_(padd)
    m = torch.cat((m[:-2], padd[:-gap_num], m[-2:], padd[-gap_num:]), dim=0)
    x['model'][profile_bias_key] = m

  # coevolution single
  coevolution_single_key = f'{args.key_prefix}head_coevolution.single.3'
  coevolution_single_weight_key, coevolution_single_bias_key = _linear_key(
      coevolution_single_key)
  if coevolution_single_weight_key in x['model']:
    m = x['model'][coevolution_single_weight_key]
    assert m.shape[0] == aa_num + 1
    padd = torch.zeros(mol_num + 1 + gap_num - m.shape[0], m.shape[1], dtype=m.dtype)
    torch.nn.init.uniform_(padd)
    m = torch.cat((m[:-2], padd[:-gap_num], m[-2:], padd[-gap_num:]), dim=0)
    x['model'][coevolution_single_weight_key] = m
  if coevolution_single_bias_key in x['model']:
    m = x['model'][coevolution_single_bias_key]
    assert m.shape[0] == aa_num + 1
    padd = torch.zeros(mol_num + 1 + gap_num - m.shape[0], dtype=m.dtype)
    torch.nn.init.uniform_(padd)
    m = torch.cat((m[:-2], padd[:-gap_num], m[-2:], padd[-gap_num:]), dim=0)
    x['model'][coevolution_single_bias_key] = m

  # coevolution pairwize
  # module.impl.head_coevolution.states
  coevolution_states_key = f'{args.key_prefix}head_coevolution.states'
  if coevolution_states_key in x['model']:
    m = x['model'][coevolution_states_key]
    assert m.shape[1] == aa_num + 1, m.shape[2] == aa_num + 1
    padd = torch.zeros(m.shape[0], mol_num + 1 + gap_num - m.shape[1], m.shape[2])
    torch.nn.init.uniform_(padd)
    m = torch.cat((m[:,:-2,:], padd[:,:-gap_num,:], m[:, -2:,:], padd[:,-gap_num:,:]), dim=1)
    padd = torch.zeros(m.shape[0], m.shape[1], mol_num + 1 + gap_num - m.shape[2])
    torch.nn.init.uniform_(padd)
    m = torch.cat((m[:,:,:-2], padd[:,:,:-gap_num], m[:,:,-2:], padd[:,:,-gap_num:]), dim=2)
    x['model'][coevolution_states_key] = m
  else:
    coevolution_pair_key = f'{args.key_prefix}head_coevolution.pairwize.1'
    coevolution_pair_weight_key, coevolution_pair_bias_key = _linear_key(
        coevolution_pair_key)

    if coevolution_pair_weight_key in x['model']:
      m = x['model'][coevolution_pair_weight_key]
      assert m.shape[0] == (aa_num + 1) * (aa_num + 1)
      m = rearrange(m, '(c d) e -> c d e', c=aa_num + 1, d=aa_num + 1)
      padd = torch.zeros(mol_num + 1 + gap_num - m.shape[0], m.shape[1], m.shape[2])
      m = torch.cat((m[:-1], padd[:-gap_num], m[-1:], padd[-gap_num:]), dim=0)
      padd = torch.zeros(m.shape[0], mol_num + 1 + gap_num - m.shape[1], m.shape[2])
      m = torch.cat((m[:,:-1], padd[:,:-gap_num], m[:,-1:], padd[:,-gap_num:]), dim=1)
      m = rearrange(m, 'c d e -> (c d) e')
      x['model'][coevolution_pair_weight_key] = m
    if coevolution_pair_bias_key in x['model']:
      m = x['model'][coevolution_pair_bias_key]
      assert m.shape[0] == (aa_num + 1) * (aa_num + 1)
      m = rearrange(m, '(c d) -> c d', c=aa_num + 1, d=aa_num + 1)
      padd = torch.zeros(mol_num + 1 + gap_num - m.shape[0], m.shape[1])
      m = torch.cat((m[:-1], padd[:-gap_num], m[-1:], padd[-gap_num:]), dim=0)
      padd = torch.zeros(m.shape[0], mol_num + 1 + gap_num - m.shape[1])
      m = torch.cat((m[:,:-1], padd[:,:-gap_num], m[:,-1:], padd[:,-gap_num:]), dim=1)
      m = rearrange(m, 'c d -> (c d)')
      x['model'][coevolution_pair_bias_key] = m

  if 'optimizer' in x:
    del x['optimizer']

  torch.save(x, args.model_files[1])
  logging.info('done.')

def tokens_modify_add_argument(parser):
  parser.add_argument('model_files',
                      type=str,
                      nargs=2,
                      help='list of model files')
  parser.add_argument('-k',
                      '--key_prefix',
                      type=str,
                      default='module.impl.',
                      help='prefix of key.')
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
  parser.add_argument('model_files', type=str, nargs=2, help='list of model files')
  return parser


if __name__ == '__main__':
  import argparse

  commands = {
      'params_count': (params_count_main, params_count_add_argument),
      'params_modify': (params_modify_main, params_modify_add_argument),
      'strip_optim': (strip_optim_main, strip_optim_add_argument),
      'tokens_modify': (tokens_modify_main, tokens_modify_add_argument),
      'relpos_modify': (relpos_modify_main, relpos_modify_add_argument),
      'to_state_dict': (to_state_dict_main, to_state_dict_add_argument),
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
