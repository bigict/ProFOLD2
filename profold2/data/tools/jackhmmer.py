# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Library to run Jackhmmer from Python."""

from concurrent import futures
import glob
import os
import pathlib
import subprocess
from typing import Any, Callable, Mapping, Optional, Sequence
from urllib import request
import logging

from profold2.data.tools import utils
# Internal import (7716).


class Jackhmmer:
  """Python wrapper of the Jackhmmer binary."""
  def __init__(
      self,
      *,
      binary_path: str,
      database_path: str,
      n_cpu: int = 4,
      n_iter: int = 1,
      e_value: float = 0.0001,
      z_value: Optional[int] = None,
      get_tblout: bool = False,
      filter_f1: float = 0.0005,
      filter_f2: float = 0.00005,
      filter_f3: float = 0.0000005,
      incdom_e: Optional[float] = None,
      dom_e: Optional[float] = None,
      num_streamed_chunks: Optional[int] = None,
      streaming_callback: Optional[Callable[[int], None]] = None
  ):
    """Initializes the Python Jackhmmer wrapper.

    Args:
      binary_path: The path to the jackhmmer executable.
      database_path: The path to the jackhmmer database (FASTA format).
      n_cpu: The number of CPUs to give Jackhmmer.
      n_iter: The number of Jackhmmer iterations.
      e_value: The E-value, see Jackhmmer docs for more details.
      z_value: The Z-value, see Jackhmmer docs for more details.
      get_tblout: Whether to save tblout string.
      filter_f1: MSV and biased composition pre-filter, set to >1.0 to turn off.
      filter_f2: Viterbi pre-filter, set to >1.0 to turn off.
      filter_f3: Forward pre-filter, set to >1.0 to turn off.
      incdom_e: Domain e-value criteria for inclusion of domains in MSA/next
        round.
      dom_e: Domain e-value criteria for inclusion in tblout.
      num_streamed_chunks: Number of database chunks to stream over.
      streaming_callback: Callback function run after each chunk iteration with
        the iteration number as argument.
    """
    self.binary_path = binary_path
    self.database_path = database_path
    self.num_streamed_chunks = num_streamed_chunks

    if not os.path.exists(self.database_path) and num_streamed_chunks is None:
      logging.error('Could not find Jackhmmer database %s', database_path)
      raise ValueError(f'Could not find Jackhmmer database {database_path}')

    self.n_cpu = n_cpu
    self.n_iter = n_iter
    self.e_value = e_value
    self.z_value = z_value
    self.filter_f1 = filter_f1
    self.filter_f2 = filter_f2
    self.filter_f3 = filter_f3
    self.incdom_e = incdom_e
    self.dom_e = dom_e
    self.get_tblout = get_tblout
    self.streaming_callback = streaming_callback

  def _query_chunk(self, input_fasta_path: str,
                   database_path: str) -> Mapping[str, Any]:
    """Queries the database chunk using Jackhmmer."""
    with utils.tmpdir_manager() as query_tmp_dir:
      sto_path = os.path.join(query_tmp_dir, 'output.sto')

      # The F1/F2/F3 are the expected proportion to pass each of the filtering
      # stages (which get progressively more expensive), reducing these
      # speeds up the pipeline at the expensive of sensitivity.  They are
      # currently set very low to make querying Mgnify run in a reasonable
      # amount of time.
      cmd_flags = [
          # Don't pollute stdout with Jackhmmer output.
          '-o',
          '/dev/null',
          '-A',
          sto_path,
          '--noali',
          '--F1',
          str(self.filter_f1),
          '--F2',
          str(self.filter_f2),
          '--F3',
          str(self.filter_f3),
          '--incE',
          str(self.e_value),
          # Report only sequences with E-values <= x in per-sequence output.
          '-E',
          str(self.e_value),
          '--cpu',
          str(self.n_cpu),
          '-N',
          str(self.n_iter)
      ]
      if self.get_tblout:
        tblout_path = os.path.join(query_tmp_dir, 'tblout.txt')
        cmd_flags.extend(['--tblout', tblout_path])

      if self.z_value:
        cmd_flags.extend(['-Z', str(self.z_value)])

      if self.dom_e is not None:
        cmd_flags.extend(['--domE', str(self.dom_e)])

      if self.incdom_e is not None:
        cmd_flags.extend(['--incdomE', str(self.incdom_e)])

      cmd = [self.binary_path] + cmd_flags + [input_fasta_path, database_path]

      logging.info('Launching subprocess "%s"', ' '.join(cmd))
      process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      with utils.timing(f'Jackhmmer ({os.path.basename(database_path)}) query'):
        _, stderr = process.communicate()
        retcode = process.wait()

      if retcode:
        raise RuntimeError(
            'Jackhmmer failed\ninput_fasta_path: %s\nstderr:\n%s\n' %
            (input_fasta_path, stderr.decode('utf-8'))
        )

      # Get e-values for each target name
      tbl = ''
      if self.get_tblout:
        with open(tblout_path) as f:
          tbl = f.read()

      with open(sto_path) as f:
        sto = f.read()

    raw_output = dict(
        sto=sto, tbl=tbl, stderr=stderr, n_iter=self.n_iter, e_value=self.e_value
    )

    return raw_output

  def query(self, input_fasta_path: str) -> Sequence[Mapping[str, Any]]:
    """Queries the database using Jackhmmer."""
    if self.num_streamed_chunks is None:
      return [self._query_chunk(input_fasta_path, self.database_path)]

    db_basename = os.path.basename(self.database_path)
    db_remote_chunk = lambda db_idx: f'{self.database_path}.{db_idx}'
    db_local_chunk = lambda db_idx: f'/tmp/ramdisk/{db_basename}.{db_idx}'

    # Remove existing files to prevent OOM
    for f in glob.glob(db_local_chunk('[0-9]*')):
      try:
        os.remove(f)
      except OSError:
        print(f'OSError while deleting {f}')

    # Download the (i+1)-th chunk while Jackhmmer is running on the i-th chunk
    with futures.ThreadPoolExecutor(max_workers=2) as executor:
      chunked_output = []
      for i in range(1, self.num_streamed_chunks + 1):
        # Copy the chunk locally
        if i == 1:
          future = executor.submit(
              request.urlretrieve, db_remote_chunk(i), db_local_chunk(i)
          )
        if i < self.num_streamed_chunks:
          next_future = executor.submit(
              request.urlretrieve, db_remote_chunk(i + 1), db_local_chunk(i + 1)
          )

        # Run Jackhmmer with the chunk
        future.result()
        chunked_output.append(self._query_chunk(input_fasta_path, db_local_chunk(i)))

        # Remove the local copy of the chunk
        os.remove(db_local_chunk(i))
        future = next_future
        if self.streaming_callback:
          self.streaming_callback(i)
    return chunked_output


def main(args):
  fmt = '%(asctime)-15s [%(levelname)s] (%(filename)s:%(lineno)d) %(message)s'
  level = logging.DEBUG if args.verbose else logging.INFO
  handlers = [logging.StreamHandler()]
  logging.basicConfig(format=fmt, level=level, handlers=handlers)

  for tool_name in (  # pylint: disable=redefined-outer-name
      'jackhmmer', 'hmmsearch', 'hmmbuild'):
    if not getattr(args, f'{tool_name}_binary_path'):
      raise ValueError(
          f'Could not find path to the "{tool_name}" binary. Make '
          'sure it is installed on your system.'
      )
  # Check for duplicate FASTA file names.
  fasta_names = [pathlib.Path(p).stem for p in args.fasta_paths]
  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')

  jackhmmer_uniref90_runner = Jackhmmer(
      binary_path=args.jackhmmer_binary_path, database_path=args.uniref90_database_path
  )
  jackhmmer_mgnify_runner = Jackhmmer(
      binary_path=args.jackhmmer_binary_path, database_path=args.mgnify_database_path
  )

  for fasta_path, fasta_name in zip(args.fasta_paths, fasta_names):
    msa_output_dir = os.path.join(args.output_dir, fasta_name, 'msas')
    if not os.path.exists(msa_output_dir):
      os.makedirs(msa_output_dir, exist_ok=True)

    jackhmmer_uniref90_result = jackhmmer_uniref90_runner.query(fasta_path)[0]
    jackhmmer_mgnify_result = jackhmmer_mgnify_runner.query(fasta_path)[0]

    jackhmmer_uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits.sto')
    with open(jackhmmer_uniref90_out_path, 'w') as f:
      f.write(jackhmmer_uniref90_result['sto'])

    jackhmmer_mgnify_out_path = os.path.join(msa_output_dir, 'mgnify_hits.sto')
    with open(jackhmmer_mgnify_out_path, 'w') as f:
      f.write(jackhmmer_mgnify_result['sto'])


if __name__ == '__main__':
  import argparse
  import shutil

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-o', '--output_dir', type=str, default='.', help='Output directory'
  )
  for tool_name in ('jackhmmer', 'hmmsearch', 'hmmbuild'):
    parser.add_argument(
        f'--{tool_name}_binary_path',
        type=str,
        default=shutil.which(tool_name),
        help=f'path to the `{tool_name}` executable.'
    )
  for database_name in (
      'uniref90', 'mgnify', 'bfd', 'small_bfd', 'uniclust30', 'pdb70'
  ):
    parser.add_argument(
        f'--{database_name}_database_path',
        type=str,
        default=None,
        help=f'path to database {database_name}'
    )
  parser.add_argument('--fasta_paths', type=str, nargs='+', help='list of fasta files')
  parser.add_argument(
      '--use_small_bfd', action='store_true', help='use small bfd database or not'
  )
  parser.add_argument('-v', '--verbose', action='store_true', help='verbose')

  args = parser.parse_args()
  main(args)
