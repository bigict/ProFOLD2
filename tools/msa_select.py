"""Tools to select msa from dataset randomly, run
     ```bash
     $python msa_select.py -h
     ```
     for further help.
"""
import contextlib
import functools
import logging
import multiprocessing as mp
import pathlib
import zipfile

import numpy as np

logger = logging.getLogger(__file__)


class MsaDataset:
  """Read msa from dataset file
    """

  MSA_LIST = ['BFD30_E-3']

  def __init__(self, data_dir, data_idx='name.idx'):
    self.data_dir = pathlib.Path(data_dir)
    if zipfile.is_zipfile(self.data_dir):
      self.data_dir = zipfile.ZipFile(self.data_dir)  # pylint: disable=consider-using-with

    logger.info('load idx data from: %s', data_idx)
    with self._fileobj(data_idx) as f:
      self.pids = list(map(lambda x: self._ftext(x).strip().split(), f))

    self.mapping = {}
    if self._fstat('mapping.idx'):
      with self._fileobj('mapping.idx') as f:
        for line in filter(lambda x: len(x) > 0,
                           map(lambda x: self._ftext(x).strip(), f)):
          v, k = line.split()
          self.mapping[k] = v

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
    if isinstance(self.data_dir, zipfile.ZipFile):
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

    pkey = self.mapping[pid] if pid in self.mapping else pid
    ret = dict(pid=pkey)
    ret.update(self.get_msa_features_new(pkey))
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
      deletion_table = str.maketrans('', '', '-')
      aligned_sequences = [
          s.translate(deletion_table).upper() for s in sequences
      ]
      return aligned_sequences, deletion_matrix

    k = int(np.random.randint(len(MsaDataset.MSA_LIST)))
    source = MsaDataset.MSA_LIST[k]

    msa_file = f'msa/{protein_id}/{source}/{protein_id}.a4m'
    if self._fstat(msa_file):
      with self._fileobj(msa_file) as f:
        sequences = list(map(lambda x: self._ftext(x).strip(), f))
      msa, _ = parse_a4m(sequences)
      return dict(str_msa=msa)
    return {}


def process(item, dataset=None):  # pylint: disable=redefined-outer-name
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


def main(args):  # pylint: disable=redefined-outer-name

  def weights_from_file(filename):
    if filename:
      with open(filename, 'r', encoding='utf-8') as f:
        for line in filter(lambda x: len(x) > 0, map(lambda x: x.strip(), f)):
          items = line.split()
          yield float(items[0])

  fmt = '%(asctime)-15s [%(levelname)s] (%(filename)s:%(lineno)d) %(message)s'
  level = logging.DEBUG if args.verbose else logging.INFO
  logging.basicConfig(format=fmt, level=level)

  dataset = MsaDataset(args.data)
  if args.weights:
    weights = np.array(list(weights_from_file(args.weights)))
    weights = weights / np.sum(weights)
    assert len(weights) == len(dataset)
    idx_list = np.random.multinomial(len(dataset) * args.count, weights)
  else:
    idx_list = [args.count] * len(dataset)

  output_dir = pathlib.Path(args.output_dir)
  if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)

  with mp.Pool(args.processes) as p:
    for idx, pid, xlist in p.imap(functools.partial(process,
                                                    dataset=dataset),
                                  enumerate(idx_list),
                                  chunksize=100):
      if xlist:
        for i, c, seq in xlist:
          assert i > 0
          with open(output_dir / f'{pid}@{i}.fasta', 'w') as f:
            print(f'>{pid}@{i}', file=f)
            print(seq, file=f)
          print(idx, pid, i, c)


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('-p',
                      '--processes',
                      type=int,
                      default=None,
                      help='number of worker processes to use, default=None')
  parser.add_argument('-o',
                      '--output_dir',
                      type=str,
                      default='.',
                      help='Output directory, default=\'.\'')
  parser.add_argument('-t', '--data', type=str, required=True, help='dataset')
  parser.add_argument('-w',
                      '--weights',
                      type=str,
                      default=None,
                      help='weights, default=None')
  parser.add_argument('-n',
                      '--count',
                      type=int,
                      default=10,
                      help='weights, default=10')
  parser.add_argument('-v', '--verbose', action='store_true', help='verbose')

  args = parser.parse_args()
  main(args)
