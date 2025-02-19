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
"""Library to run HHblits from Python."""

import glob
import os
import pathlib
import subprocess
from typing import Any, Mapping, Optional, Sequence
import logging

from profold2.data.tools import utils
from profold2.data import parsers
# Internal import (7716).

_HHBLITS_DEFAULT_P = 20
_HHBLITS_DEFAULT_Z = 500


class HHBlits:
  """Python wrapper of the HHblits binary."""
  def __init__(
      self,
      *,
      binary_path: str,
      databases: Sequence[str],
      n_cpu: int = 32,
      n_iter: int = 3,
      e_value: float = 0.001,
      maxseq: int = 1_000_000,
      realign_max: int = 100_000,
      maxfilt: int = 100_000,
      min_prefilter_hits: int = 1000,
      all_seqs: bool = False,
      alt: Optional[int] = None,
      p: int = _HHBLITS_DEFAULT_P,
      z: int = _HHBLITS_DEFAULT_Z
  ):
    """Initializes the Python HHblits wrapper.

    Args:
      binary_path: The path to the HHblits executable.
      databases: A sequence of HHblits database paths. This should be the
        common prefix for the database files (i.e. up to but not including
        _hhm.ffindex etc.)
      n_cpu: The number of CPUs to give HHblits.
      n_iter: The number of HHblits iterations.
      e_value: The E-value, see HHblits docs for more details.
      maxseq: The maximum number of rows in an input alignment. Note that this
        parameter is only supported in HHBlits version 3.1 and higher.
      realign_max: Max number of HMM-HMM hits to realign. HHblits default: 500.
      maxfilt: Max number of hits allowed to pass the 2nd prefilter.
        HHblits default: 20000.
      min_prefilter_hits: Min number of hits to pass prefilter.
        HHblits default: 100.
      all_seqs: Return all sequences in the MSA / Do not filter the result MSA.
        HHblits default: False.
      alt: Show up to this many alternative alignments.
      p: Minimum Prob for a hit to be included in the output hhr file.
        HHblits default: 20.
      z: Hard cap on number of hits reported in the hhr file.
        HHblits default: 500. NB: The relevant HHblits flag is -Z not -z.

    Raises:
      RuntimeError: If HHblits binary not found within the path.
    """
    self.binary_path = binary_path
    self.databases = databases

    for database_path in self.databases:
      if not glob.glob(database_path + '_*'):
        logging.error('Could not find HHBlits database %s', database_path)
        raise ValueError(f'Could not find HHBlits database {database_path}')

    self.n_cpu = n_cpu
    self.n_iter = n_iter
    self.e_value = e_value
    self.maxseq = maxseq
    self.realign_max = realign_max
    self.maxfilt = maxfilt
    self.min_prefilter_hits = min_prefilter_hits
    self.all_seqs = all_seqs
    self.alt = alt
    self.p = p
    self.z = z

  def query(self, input_fasta_path: str) -> Mapping[str, Any]:
    """Queries the database using HHblits."""
    with utils.tmpdir_manager(base_dir='./tmp') as query_tmp_dir:
      a3m_path = os.path.join(query_tmp_dir, 'output.a3m')

      db_cmd = []
      for db_path in self.databases:
        db_cmd.append('-d')
        db_cmd.append(db_path)
      cmd = [
          self.binary_path, '-i', input_fasta_path, '-cpu',
          str(self.n_cpu), '-oa3m', a3m_path, '-o', '/dev/null', '-n',
          str(self.n_iter), '-e',
          str(self.e_value), '-maxseq',
          str(self.maxseq), '-realign_max',
          str(self.realign_max), '-maxfilt',
          str(self.maxfilt), '-min_prefilter_hits',
          str(self.min_prefilter_hits)
      ]
      if self.all_seqs:
        cmd += ['-all']
      if self.alt:
        cmd += ['-alt', str(self.alt)]
      if self.p != _HHBLITS_DEFAULT_P:
        cmd += ['-p', str(self.p)]
      if self.z != _HHBLITS_DEFAULT_Z:
        cmd += ['-Z', str(self.z)]
      cmd += db_cmd

      logging.info('Launching subprocess "%s"', ' '.join(cmd))
      process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

      with utils.timing('HHblits query'):
        stdout, stderr = process.communicate()
        retcode = process.wait()

      if retcode:
        # Logs have a 15k character limit, so log HHblits error line by line.
        logging.error('HHblits failed. HHblits stderr begin:')
        for error_line in stderr.decode('utf-8').splitlines():
          if error_line.strip():
            logging.error(error_line.strip())
        logging.error('HHblits stderr end')
        raise RuntimeError(
            'HHblits failed\nstdout:\n%s\n\nstderr:\n%s\n' %
            (stdout.decode('utf-8'), stderr[:500_000].decode('utf-8'))
        )

      with open(a3m_path) as f:
        a3m = f.read()

    raw_output = dict(
        a3m=a3m, output=stdout, stderr=stderr, n_iter=self.n_iter, e_value=self.e_value
    )
    return raw_output


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

  hhblits_bfd_uniclust_runner = HHBlits(
      binary_path=args.hhblits_binary_path,
      databases=[args.bfd_database_path, args.uniclust30_database_path]
  )

  for input_fasta_path, fasta_name in zip(args.fasta_paths, fasta_names):
    msa_output_dir = os.path.join(args.output_dir, fasta_name, 'msas')
    if not os.path.exists(msa_output_dir):
      os.makedirs(msa_output_dir, exist_ok=True)

    with open(input_fasta_path) as f:
      input_fasta_str = f.read()

    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    if len(input_seqs) != 1:
      raise ValueError(f'More than one input sequence found in {input_fasta_path}.')
    input_sequence = input_seqs[0]
    input_description = input_descs[0]
    num_res = len(input_sequence)

    # uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits.sto')
    # with open(uniref90_out_path, 'r') as f:
    #   jackhmmer_uniref90_result = dict(sto=f.read())

    # mgnify_out_path = os.path.join(msa_output_dir, 'mgnify_hits.sto')
    # with open(mgnify_out_path, 'r') as f:
    #   jackhmmer_mgnify_result = dict(sto=f.read())

    # uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(
    #     jackhmmer_uniref90_result['sto'], max_sequences=args.uniref_max_hits)

    # uniref90_msa, uniref90_deletion_matrix, _ = parsers.parse_stockholm(
    #     jackhmmer_uniref90_result['sto'])
    # mgnify_msa, mgnify_deletion_matrix, _ = parsers.parse_stockholm(
    #     jackhmmer_mgnify_result['sto'])
    # mgnify_msa = mgnify_msa[:args.mgnify_max_hits]
    # mgnify_deletion_matrix = mgnify_deletion_matrix[:args.mgnify_max_hits]

    hhblits_bfd_uniclust_result = hhblits_bfd_uniclust_runner.query(input_fasta_path)

    bfd_out_path = os.path.join(msa_output_dir, 'bfd_uniclust_hits.a3m')
    with open(bfd_out_path, 'w') as f:
      f.write(hhblits_bfd_uniclust_result['a3m'])

    bfd_msa, bfd_deletion_matrix = parsers.parse_a3m(hhblits_bfd_uniclust_result['a3m'])

    # sequence_features = make_sequence_features(
    #     sequence=input_sequence,
    #     description=input_description,
    #     num_res=num_res)

    # msa_features = make_msa_features(
    #     msas=(uniref90_msa, bfd_msa, mgnify_msa),
    #     deletion_matrices=(uniref90_deletion_matrix,
    #                        bfd_deletion_matrix,
    #                        mgnify_deletion_matrix))

    # logging.info('Uniref90 MSA size: %d sequences.', len(uniref90_msa))
    logging.info('BFD MSA size: %d sequences.', len(bfd_msa))
    # logging.info('MGnify MSA size: %d sequences.', len(mgnify_msa))
    # logging.info('Final (deduplicated) MSA size: %d sequences.',
    #              msa_features['num_alignments'][0])

    # ret = {**sequence_features, **msa_features}


if __name__ == '__main__':
  import argparse
  import shutil

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-o', '--output_dir', type=str, default='.', help='Output directory'
  )
  for tool_name in ('jackhmmer', 'hhblits', 'hhsearch', 'hmmsearch', 'hmmbuild'):
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
  parser.add_argument(
      '--uniref_max_hits', type=int, default=10000, help='max hits of uniref'
  )
  parser.add_argument(
      '--mgnify_max_hits', type=int, default=501, help='max hits of mgnify'
  )
  parser.add_argument('-v', '--verbose', action='store_true', help='verbose')

  args = parser.parse_args()
  main(args)
