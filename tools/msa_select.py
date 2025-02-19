"""Tools to select msa from dataset randomly, run
     ```bash
     $python msa_select.py -h
     ```
     for further help.
"""
import os
import contextlib
import functools
from inspect import isfunction
import logging
import multiprocessing as mp
import pathlib
import re
import string
import zipfile

import numpy as np
try:
  import jax.numpy as jnp
except:
  jnp = np

logger = logging.getLogger(__file__)

MSA_LIST = ['BFD30_E-3']


# helpers
def exists(val):
  return val is not None


def default(val, d):
  if exists(val):
    return val
  return d() if isfunction(d) else d


_seq_index_pattern = '(\\d+)-(\\d+)'


def _seq_index_split(text):
  for s in text.split(','):
    r = re.match(_seq_index_pattern, s)
    assert r
    yield tuple(map(int, r.group(1, 2)))


def _seq_index_join(seq_index):
  return ','.join(f'{i}-{j}' for i, j in seq_index)


def _make_feats_shrinked(item, new_order, seq_feats=None, msa_feats=None):
  # Update seq related feats
  if 'str_seq' in item:
    item['str_seq'] = ''.join(item['str_seq'][k] for k in new_order)

  for field in ('str_msa', ):
    if field in item:
      for j in range(len(item['str_msa'])):
        item['str_msa'][j] = ''.join(item['str_msa'][j][k] for k in new_order)

  # Update tensors
  new_order = np.asarray(new_order)

  for field in default(seq_feats, ('coord', 'coord_mask', 'coord_plddt')):
    if field in item:
      item[field] = np.take(item[field], new_order, axis=0)
  for field in default(msa_feats, ('msa', 'del_msa')):
    if field in item:
      item[field] = np.take(item[field], new_order, axis=1)

  return item


def _data_from_domain(item, domains):
  assert domains
  domains = list(_seq_index_split(domains))

  if 'str_seq' in item:
    n, new_order = len(item['str_seq']), []
  else:
    assert 'str_msa' in item
    n, new_order = len(item['str_msa'][0]), []
  for i, j in domains:
    assert j < n, (item['pid'], domains)
    for k in range(i, j + 1):
      new_order.append(k)

  assert 0 <= len(new_order) <= n, (len(new_order), n)
  if len(new_order) < n:
    item = _make_feats_shrinked(item, new_order, seq_feats=('seq', 'seq_index', 'mask'))
  return item


def decompose_pid(pid, return_domain=False):
  k = pid.find('/')
  if k != -1:
    pid, domains = pid[:k], pid[k + 1:]
  else:
    domains = None

  k = pid.find('_')
  if k != -1:
    pid, chain = pid[:k], pid[k + 1:]
  else:
    chain = None

  if return_domain:
    return pid, chain, domains
  return pid, chain


def compose_pid(pid, chain, domains=None):
  if chain is not None:
    pid = f'{pid}_{chain}'
  if domains is not None:
    pid = f'{pid}/{domains}'
  return pid


class MsaDataset:
  """Read msa from dataset file
    """
  def __init__(self, data_dir, data_idx='name.idx', deletion_table=None):
    self.data_dir = pathlib.Path(data_dir)
    if zipfile.is_zipfile(self.data_dir):
      self.data_dir = zipfile.ZipFile(self.data_dir)  # pylint: disable=consider-using-with

    logger.info('load idx data from: %s@%s', data_dir, data_idx)
    with self._fileobj(data_idx) as f:
      self.pids = list(map(lambda x: self._ftext(x).strip().split(), f))

    self.mapping = {}
    if self._fstat('mapping.idx'):
      with self._fileobj('mapping.idx') as f:
        for line in filter(
            lambda x: len(x) > 0, map(lambda x: self._ftext(x).strip(), f)
        ):
          v, k = line.split()
          self.mapping[k] = v
    self.deletion_table = '-' if deletion_table is None else deletion_table

  def __getstate__(self):
    logger.debug('being pickled ...')
    d = self.__dict__
    if isinstance(self.data_dir, zipfile.ZipFile):
      d['data_dir'] = self.data_dir.filename
    return d

  def __setstate__(self, d):
    logger.debug('being unpickled ...')
    if zipfile.is_zipfile(d['data_dir']):
      d['data_dir'] = zipfile.ZipFile(d['data_dir'])  # pylint: disable=consider-using-with
    self.__dict__ = d

  @contextlib.contextmanager
  def _fileobj(self, filename):
    if os.path.isabs(filename):
      with open(filename, mode='rb') as f:
        yield f
    elif isinstance(self.data_dir, zipfile.ZipFile):
      with self.data_dir.open(filename, 'r') as f:
        yield f
    else:
      with open(self.data_dir / filename, mode='rb') as f:
        yield f

  def _fstat(self, filename):
    if isinstance(self.data_dir, zipfile.ZipFile):
      try:
        self.data_dir.getinfo(filename)
        return True
      except KeyError:
        return False
    logger.debug('test file: %s/%s', self.data_dir, filename)
    return (self.data_dir / filename).exists()

  def _ftext(self, line, encoding='utf-8'):
    if isinstance(line, bytes):
      return line.decode(encoding)
    return line

  def __len__(self):
    return len(self.pids)

  def __getitem__(self, idx):
    pids = self.pids[idx]
    pid = pids[np.random.randint(len(pids))]

    pid, chain, domains = decompose_pid(pid, return_domain=True)
    pid = compose_pid(pid, chain)

    pkey = self.mapping[pid] if pid in self.mapping else pid
    ret = dict(pid=pkey)
    ret.update(self.get_msa_features_new(pkey))

    if domains:
      ret = _data_from_domain(ret, domains)
      pid = compose_pid(pid, None, domains)
      ret.update(pid=pid, pkey=pkey)
    return ret

  def get_msa_features_new(self, protein_id):
    """Constructs a feature dict of MSA features."""
    def parse_a4m(sequences):
      deletion_matrix = []
      for msa_sequence in sequences:
        deletion_vec = []
        deletion_count = 0
        for j in msa_sequence:
          if j.islower():
            deletion_count += 1
          else:
            deletion_vec.append(deletion_count)
            deletion_count = 0
        deletion_matrix.append(deletion_vec)
      # Make the MSA matrix out of aligned (deletion-free) sequences
      deletion_table = str.maketrans('', '', self.deletion_table)
      aligned_sequences = [s.translate(deletion_table).upper() for s in sequences]
      return aligned_sequences, deletion_matrix

    k = int(np.random.randint(len(MSA_LIST)))
    source = MSA_LIST[k]

    msa_file = f'msa/{protein_id}/{source}/{protein_id}.a4m'
    if self._fstat(msa_file):
      with self._fileobj(msa_file) as f:
        sequences = list(map(lambda x: self._ftext(x).strip(), f))
      msa, _ = parse_a4m(sequences)
      return dict(str_msa=msa)
    return {}


def msa_select_args(parser):  # pylint: disable=redefined-outer-name
  parser.add_argument(
      '-w', '--weights', type=str, default=None, help='weights, default=None'
  )
  parser.add_argument('-n', '--count', type=int, default=10, help='weights, default=10')
  return parser


def msa_select_input(args):  # pylint: disable=redefined-outer-name
  def weights_from_file(filename):
    if filename:
      with open(filename, 'r', encoding='utf-8') as f:
        for line in filter(lambda x: len(x) > 0, map(lambda x: x.strip(), f)):
          items = line.split()
          yield float(items[0])

  dataset = MsaDataset(args.data, data_idx=args.data_idx, deletion_table='-')
  if args.weights:
    weights = np.array(list(weights_from_file(args.weights)))
    weights = weights / np.sum(weights)
    assert len(weights) == len(dataset)
    idx_list = np.random.multinomial(len(dataset) * args.count, weights)
  else:
    idx_list = [args.count] * len(dataset)
  return dataset, enumerate(idx_list)


def msa_select_process(item, dataset=None):  # pylint: disable=redefined-outer-name
  idx, c = item

  pid, xlist = None, []
  if c > 0:
    data = dataset[idx]
    pid, str_msa = data['pid'], data['str_msa']
    w = np.array([1e-8] + [1.0 / x for x in range(1, min(len(str_msa), 1000))])
    w = np.power(w, 0.75)
    w /= np.sum(w)
    for i, x in enumerate(np.random.multinomial(c, w)):
      if i > 0 and x > 0:
        xlist.append((i, x, str_msa[i]))
  return idx, pid, xlist


def msa_select_result(item, args):  # pylint: disable=redefined-outer-name
  idx, pid, xlist = item
  if xlist:
    output_dir = pathlib.Path(args.output_dir)
    if not output_dir.exists():
      output_dir.mkdir(parents=True, exist_ok=True)

    for i, c, seq in xlist:
      assert i > 0
      with open(output_dir / f'{pid}@{i}.fasta', 'w') as f:
        print(f'>{pid}@{i}', file=f)
        print(seq, file=f)
      print(idx, pid, i, c)


def aligned_ratio_args(parser):  # pylint: disable=redefined-outer-name
  return parser


def aligned_ratio_input(args):  # pylint: disable=redefined-outer-name
  dataset = MsaDataset(args.data, data_idx=args.data_idx, deletion_table='')
  return dataset, range(len(dataset))


def aligned_ratio_process(item, dataset=None):  # pylint: disable=redefined-outer-name
  def aligned_ratio(msa, n):
    c = sum(1 for s in msa if s == '-')
    return 1. - c / n

  def aligned_ident(msa, seq):
    c = 0
    i, j = 0, 0
    while i < len(msa) and j < len(seq):
      if msa[i].islower():
        i += 1
      elif seq[j].islower():
        j += 1
      else:
        if msa[i] == seq[j] and msa[i] != '-':
          c += 1
        i += 1
        j += 1
    return c

  idx = item
  data = dataset[idx]
  if 'str_msa' in data:
    pid, str_msa = data['pid'], data['str_msa']
    assert len(str_msa) > 0
    n = len(str_msa[0])
    ratio = [aligned_ratio(msa_i, n) for msa_i in str_msa]
    ident = [aligned_ident(msa_i, str_msa[0]) for msa_i in str_msa]
    return idx, pid, (ratio, ident, n)
  return idx, None, None


def aligned_ratio_result(item, args):  # pylint: disable=redefined-outer-name
  idx, pid, result = item
  if pid:
    logger.debug('output %d:%s', idx, pid)
    pid = pid.replace('/', '-')
    output_dir = pathlib.Path(os.path.join(args.output_dir, pid, MSA_LIST[0]))
    if not output_dir.exists():
      output_dir.mkdir(parents=True, exist_ok=True)

    ratio, ident, n = result
    ratio = '\n'.join(f'{r}\t{c}\t{c/n}' for r, c in zip(ratio, ident))
    with open(output_dir / f'{pid}.alr', 'w') as f:
      f.write(ratio)


def msa_cluster_args(parser):  # pylint: disable=redefined-outer-name
  parser.add_argument('--skip_done', action='store_true', help='skip done')
  parser.add_argument(
      '--num_workers', type=int, default=1, help='num_workers, default=1'
  )
  return parser


def _msa_cluster_read(args, dataset, idx):
  from profold2.common import residue_constants

  def _have_done(pid, str_msa):
    output_dir = pathlib.Path(os.path.join(args.output_dir, pid, MSA_LIST[0]))
    if not output_dir.exists():
      return False

    output_file = output_dir / f'{pid}.clu'
    if output_file.exists():
      with open(output_file, 'r') as f:
        return len(f.readlines()) == len(str_msa)
    return False

  data = dataset[idx]
  if 'str_msa' in data and (
      not args.skip_done or not _have_done(data['pid'], data['str_msa'])
  ):
    msa_depth = len(data['str_msa'])
    assert msa_depth >= 1
    seq_len = len(data['str_msa'][0])
    num_class = len(residue_constants.restypes_with_x_and_gap)

    logger.info('_msa_cluster_read: %s|%s start', idx, data.get('pid'))
    # to int repr msa
    # msa = np.zeros((msa_depth, seq_len, num_class), dtype=np.int32)
    # for i, sequence in enumerate(data['str_msa']):
    #   for j, res in enumerate(sequence):
    #     if res != residue_constants.restypes_with_x_and_gap[-1]:
    #       k = residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE[
    #           residue_constants.HHBLITS_AA_TO_ID[res]]
    #       msa[i, j, k] = 1
    #
    msa = np.zeros((msa_depth, seq_len), dtype=np.int32)
    for i, sequence in enumerate(data['str_msa']):
      for j, res in enumerate(sequence):
        k = residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE[
            residue_constants.HHBLITS_AA_TO_ID[res]]
        msa[i, j] = k

    logger.info('_msa_cluster_read: %s|%s done', idx, data.get('pid'))
    data['msa'] = msa
  else:
    logger.info('_msa_cluster_read: %s|%s skiped', idx, data.get('pid'))
  return idx, data


def _msa_cluster_iter(args, dataset):
  with mp.Pool(args.num_workers) as p:
    for item in p.imap(
        functools.partial(_msa_cluster_read, args, dataset),
        range(len(dataset)),
        chunksize=100
    ):
      idx, data = item
      if 'msa' in data:
        logger.info('_msa_cluster_iter: %d', idx)
        yield item


def msa_cluster_input(args):  # pylint: disable=redefined-outer-name
  dataset = MsaDataset(
      args.data, data_idx=args.data_idx, deletion_table=string.ascii_lowercase
  )
  return dataset, _msa_cluster_iter(args, dataset)


def msa_cluster_process(item, dataset=None):  # pylint: disable=redefined-outer-name
  from profold2.common import residue_constants

  # idx = item
  idx, data = item  # dataset[idx]

  if 'msa' in data:
    pass
    # # msa_depth = len(data['str_msa'])
    # # assert msa_depth >= 1
    # # seq_len = len(data['str_msa'][0])
    # # num_class = len(residue_constants.restypes_with_x_and_gap)

    # # # to int repr msa
    # # msa = np.zeros((msa_depth, seq_len, num_class), dtype=np.int32)
    # # for i, sequence in enumerate(data['str_msa']):
    # #   for j, res in enumerate(sequence):
    # #     if res != residue_constants.restypes_with_x_and_gap[-1]:
    # #       k = residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE[
    # #           residue_constants.HHBLITS_AA_TO_ID[res]]
    # #       msa[i, j, k] = 1
    # #
    # num_class = len(residue_constants.restypes_with_x_and_gap)

    # # disable gap
    # tbl = np.eye(num_class, num_class)
    # gap_idx = len(residue_constants.restypes_with_x_and_gap) - 1
    # tbl[gap_idx][gap_idx] = 0

    # msa = tbl[data['msa']]
    # msa_depth = msa.shape[0]

    # num_cluster = 512
    # max_msa_depth = min(num_cluster, msa_depth)

    # index_order = jnp.full((1,), 0)
    # if msa_depth > 1:
    #   index_order = jnp.concatenate((
    #       index_order, np.random.permutation(msa_depth - 1) + 1))
    # sel_msa, not_sel_msa = index_order[:max_msa_depth], index_order[
    #     max_msa_depth:]
    # msa_cencter = jnp.take(msa, sel_msa, axis=0)
    # m, n = msa_cencter.shape[0], msa_depth

    # logger.info('msa_cluster_process: %s m=%s, n=%s starting', idx, m, n)
    # agreement = jnp.matmul(msa_cencter.reshape(m, -1),
    #                       msa.reshape(n, -1).transpose())

    # agreement = jnp.argmax(agreement, axis=0)
    # agreement = sel_msa[agreement]
    # logger.info('msa_cluster_process: %d m=%s, n=%s done', idx, m, n)
    # return idx, data['pid'], agreement.tolist()
  return idx, None, None


def msa_cluster_result(item, args):  # pylint: disable=redefined-outer-name
  idx, pid, agreement = item
  if pid:
    logger.debug('output %d:%s', idx, pid)
    output_dir = pathlib.Path(os.path.join(args.output_dir, pid, MSA_LIST[0]))
    if not output_dir.exists():
      output_dir.mkdir(parents=True, exist_ok=True)

    agreement = '\n'.join(f'{r}' for r in agreement)
    with open(output_dir / f'{pid}.clu', 'w') as f:
      f.write(agreement)


def main(args, work_fn):  # pylint: disable=redefined-outer-name
  fmt = '%(asctime)-15s [%(levelname)s] (%(filename)s:%(lineno)d) %(message)s'
  level = logging.DEBUG if args.verbose else logging.INFO
  logging.basicConfig(format=fmt, level=level)

  work_input, work_process, work_result = work_fn

  dataset, idx_list = work_input(args)
  with mp.Pool(args.processes) as p:
    for item in p.imap(
        functools.partial(work_process, dataset=dataset), idx_list, chunksize=10
    ):
      work_result(item, args)


if __name__ == '__main__':
  import argparse

  commands = {
      'msa_select':
          (msa_select_args, msa_select_input, msa_select_process, msa_select_result),
      'aligned_ratio':
          (
              aligned_ratio_args, aligned_ratio_input, aligned_ratio_process,
              aligned_ratio_result
          ),
      'msa_cluster':
          (
              msa_cluster_args, msa_cluster_input, msa_cluster_process,
              msa_cluster_result
          ),
  }

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-p',
      '--processes',
      type=int,
      default=None,
      help='number of worker processes to use, default=None'
  )
  # command args
  subparsers = parser.add_subparsers(dest='command', required=True)
  for cmd, (add_argument, *_) in commands.items():
    cmd_parser = subparsers.add_parser(cmd)
    cmd_parser.add_argument(
        '-o',
        '--output_dir',
        type=str,
        default='.',
        help='Output directory, default=\'.\''
    )
    cmd_parser.add_argument('-t', '--data', type=str, required=True, help='dataset')
    cmd_parser.add_argument(
        '--data_idx', type=str, default='name.idx', help='dataset idx'
    )

    add_argument(cmd_parser)

    cmd_parser.add_argument('-v', '--verbose', action='store_true', help='verbose')

  args = parser.parse_args()
  assert args.command in commands
  main(args, commands[args.command][1:])
