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

"""Functions for building the input features for the AlphaFold model."""

import os
import gzip
import pathlib
from typing import Mapping, Optional, Sequence
import logging

import numpy as np

# Internal import (7716).

from profold2.common import residue_constants
from profold2.data import parsers
from profold2.data import templates
from profold2.data.tools import hhblits
from profold2.data.tools import hhsearch
from profold2.data.tools import jackhmmer

FeatureDict = Mapping[str, np.ndarray]


def make_sequence_features(
    sequence: str, description: str, num_res: int) -> FeatureDict:
  """Constructs a feature dict of sequence features."""
  features = {}
  features['aatype'] = residue_constants.sequence_to_onehot(
      sequence=sequence,
      mapping=residue_constants.restype_order_with_x,
      map_unknown_to_x=True)
  features['between_segment_residues'] = np.zeros((num_res,), dtype=np.int32)
  features['domain_name'] = np.array([description.encode('utf-8')],
                                     dtype=np.object_)
  features['residue_index'] = np.array(range(num_res), dtype=np.int32)
  features['seq_length'] = np.array([num_res] * num_res, dtype=np.int32)
  features['sequence'] = np.array([sequence.encode('utf-8')], dtype=np.object_)
  return features


def make_msa_features(
    msas: Sequence[Sequence[str]],
    deletion_matrices: Sequence[parsers.DeletionMatrix]) -> FeatureDict:
  """Constructs a feature dict of MSA features."""
  if not msas:
    raise ValueError('At least one MSA must be provided.')

  int_msa = []
  deletion_matrix = []
  seen_sequences = set()
  for msa_index, msa in enumerate(msas):
    if not msa:
      raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
    for sequence_index, sequence in enumerate(msa):
      if sequence in seen_sequences:
        continue
      seen_sequences.add(sequence)
      int_msa.append(
          [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
      deletion_matrix.append(deletion_matrices[msa_index][sequence_index])

  num_res = len(msas[0][0])
  num_alignments = len(int_msa)
  features = {}
  features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
  features['msa'] = np.array(int_msa, dtype=np.int32)
  features['num_alignments'] = np.array(
      [num_alignments] * num_res, dtype=np.int32)
  return features


class DataPipeline:
  """Runs the alignment tools and assembles the input features."""

  def __init__(self,
               jackhmmer_binary_path: str,
               hhblits_binary_path: str,
               hhsearch_binary_path: str,
               uniref90_database_path: str,
               mgnify_database_path: str,
               bfd_database_path: Optional[str],
               uniclust30_database_path: Optional[str],
               small_bfd_database_path: Optional[str],
               pdb70_database_path: str,
               template_featurizer: templates.TemplateHitFeaturizer,
               use_small_bfd: bool,
               mgnify_max_hits: int = int(os.environ.get('PIPELINE_MGNIFY_MAX_HITS', 501)),
               uniref_max_hits: int = int(os.environ.get('PIPELINE_UNIREF_MAX_HITS', 10000))):
    """Constructs a feature dict for a given FASTA file."""
    self._use_small_bfd = use_small_bfd
    # self.jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
    #     binary_path=jackhmmer_binary_path,
    #     database_path=uniref90_database_path)
    # if use_small_bfd:
    #   self.jackhmmer_small_bfd_runner = jackhmmer.Jackhmmer(
    #       binary_path=jackhmmer_binary_path,
    #       database_path=small_bfd_database_path)
    # else:
    #   self.hhblits_bfd_uniclust_runner = hhblits.HHBlits(
    #       binary_path=hhblits_binary_path,
    #       databases=[bfd_database_path, uniclust30_database_path])
    # self.jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
    #     binary_path=jackhmmer_binary_path,
    #     database_path=mgnify_database_path)
    # self.hhsearch_pdb70_runner = hhsearch.HHSearch(
    #     binary_path=hhsearch_binary_path,
    #     databases=[pdb70_database_path])
    # self.template_featurizer = template_featurizer
    self.mgnify_max_hits = mgnify_max_hits
    self.uniref_max_hits = uniref_max_hits

  def process(self, input_fasta_path: str, msa_output_dir: str) -> FeatureDict:
    """Runs alignment tools on the input sequence and creates features."""
    with open(input_fasta_path) as f:
      input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    if len(input_seqs) != 1:
      raise ValueError(
          f'More than one input sequence found in {input_fasta_path}.')
    input_sequence = input_seqs[0]
    input_description = input_descs[0]
    num_res = len(input_sequence)

    # jackhmmer_uniref90_result = self.jackhmmer_uniref90_runner.query(
    #     input_fasta_path)[0]
    # jackhmmer_mgnify_result = self.jackhmmer_mgnify_runner.query(
    #     input_fasta_path)[0]

    # uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(
    #     jackhmmer_uniref90_result['sto'], max_sequences=self.uniref_max_hits)
    # hhsearch_result = self.hhsearch_pdb70_runner.query(uniref90_msa_as_a3m)

    uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits.sto')
    # with open(uniref90_out_path, 'w') as f:
    #   f.write(jackhmmer_uniref90_result['sto'])
    if os.path.exists(uniref90_out_path):
      with open(uniref90_out_path, 'r') as f:
        jackhmmer_uniref90_result = {'sto':f.read()}
    else:
      uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits.sto.gz')
      if os.path.exists(uniref90_out_path):
        with gzip.open(uniref90_out_path, 'rt') as f:
          jackhmmer_uniref90_result = {'sto':f.read()}
      else:
        jackhmmer_uniref90_result = {}

    mgnify_out_path = os.path.join(msa_output_dir, 'mgnify_hits.sto')
    # with open(mgnify_out_path, 'w') as f:
    #   f.write(jackhmmer_mgnify_result['sto'])
    if os.path.exists(mgnify_out_path):
      with open(mgnify_out_path, 'r') as f:
        jackhmmer_mgnify_result = {'sto':f.read()}
    else:
      mgnify_out_path = os.path.join(msa_output_dir, 'mgnify_hits.sto.gz')
      if os.path.exists(mgnify_out_path):
        with gzip.open(mgnify_out_path, 'rt') as f:
          jackhmmer_mgnify_result = {'sto':f.read()}
      else:
        jackhmmer_mgnify_result = {}

    # uniref90_msa, uniref90_deletion_matrix, _ = parsers.parse_stockholm(
    #     jackhmmer_uniref90_result['sto'])
    if 'sto' in jackhmmer_uniref90_result:
      uniref90_msa, _, uniref90_name_list = parsers.parse_stockholm(
          jackhmmer_uniref90_result['sto'])
      uniref90_msa, uniref90_name_list = (
          uniref90_msa[:self.uniref_max_hits], uniref90_name_list[:self.uniref_max_hits])
    else:
      uniref90_msa, uniref90_name_list = [], []
    if 'sto' in jackhmmer_mgnify_result:
      mgnify_msa, _, mgnify_name_list = parsers.parse_stockholm(
          jackhmmer_mgnify_result['sto'])
      mgnify_msa, mgnify_name_list = (
          mgnify_msa[:self.mgnify_max_hits], mgnify_name_list[:self.mgnify_max_hits])
    else:
      mgnify_msa, mgnify_name_list = [], []
    # hhsearch_hits = parsers.parse_hhr(hhsearch_result)

    if self._use_small_bfd:
      # jackhmmer_small_bfd_result = self.jackhmmer_small_bfd_runner.query(
      #     input_fasta_path)[0]

      bfd_out_path = os.path.join(msa_output_dir, 'small_bfd_hits.a3m')
      # with open(bfd_out_path, 'w') as f:
      #   f.write(jackhmmer_small_bfd_result['sto'])
      if os.path.exists(bfd_out_path):
        with open(bfd_out_path, 'r') as f:
          jackhmmer_small_bfd_result = {'sto':f.read()}

        bfd_msa, _, bfd_name_list = parsers.parse_stockholm(
            jackhmmer_small_bfd_result['sto'])
      else:
        bfd_msa, bfd_name_list = [], []
    else:
      # hhblits_bfd_uniclust_result = self.hhblits_bfd_uniclust_runner.query(
      #     input_fasta_path)

      bfd_out_path = os.path.join(msa_output_dir, 'bfd_uniclust_hits.a3m')
      # with open(bfd_out_path, 'w') as f:
      #   f.write(hhblits_bfd_uniclust_result['a3m'])
      if os.path.exists(bfd_out_path):
        with open(bfd_out_path, 'r') as f:
          hhblits_bfd_uniclust_result = {'a3m':f.read()}

        bfd_msa, bfd_name_list = parsers.parse_fasta(
            hhblits_bfd_uniclust_result['a3m'])
      else:
        bfd_msa, bfd_name_list = [], []

    # templates_result = self.template_featurizer.get_templates(
    #     query_sequence=input_sequence,
    #     query_pdb_code=None,
    #     query_release_date=None,
    #     hits=hhsearch_hits)

    # sequence_features = make_sequence_features(
    #     sequence=input_sequence,
    #     description=input_description,
    #     num_res=num_res)

    # msa_features = make_msa_features(
    #     msas=(uniref90_msa, bfd_msa, mgnify_msa),
    #     deletion_matrices=(uniref90_deletion_matrix,
    #                        bfd_deletion_matrix,
    #                        mgnify_deletion_matrix))

    # return {**sequence_features, **msa_features, **templates_result.features}

    seq_msa = []
    seen_sequences = set()
    for msa_index, (msa, name_list) in enumerate(
        [(uniref90_msa, uniref90_name_list), (mgnify_msa, mgnify_name_list), (bfd_msa, bfd_name_list)]):
      # if not msa:
      #   raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
      for sequence_index, (sequence, name) in enumerate(zip(msa, name_list)):
        if sequence in seen_sequences:
          continue
        seen_sequences.add(sequence)
        seq_msa.append((sequence, name))

    return seq_msa

def main(args):
  fmt = '%(asctime)-15s [%(levelname)s] (%(filename)s:%(lineno)d) %(message)s'
  level=logging.DEBUG if args.verbose else logging.INFO
  handlers = [
      logging.StreamHandler()]
  logging.basicConfig(
      format=fmt,
      level=level,
      handlers=handlers)

  for tool_name in (  # pylint: disable=redefined-outer-name
      'jackhmmer', 'hhblits', 'hhsearch', 'hmmsearch', 'hmmbuild'):
    if not getattr(args, f'{tool_name}_binary_path'):
      raise ValueError(f'Could not find path to the "{tool_name}" binary. Make '
                       'sure it is installed on your system.')
  # Check for duplicate FASTA file names.
  fasta_names = [pathlib.Path(p).stem for p in args.fasta_paths]
  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')

  pipeline = DataPipeline(args.jackhmmer_binary_path,
        args.hhblits_binary_path,
        args.hhsearch_binary_path,
        args.uniref90_database_path,
        args.mgnify_database_path,
        args.bfd_database_path,
        args.uniclust30_database_path,
        args.small_bfd_database_path,
        args.pdb70_database_path,
        None, args.use_small_bfd)

  for input_fasta_path, fasta_name in zip(args.fasta_paths, fasta_names):
    msa_output_dir = os.path.join(args.output_dir, fasta_name, 'msas')
    if not os.path.exists(msa_output_dir):
      os.makedirs(msa_output_dir, exist_ok=True)

    seq_msa = pipeline.process(input_fasta_path, msa_output_dir)
    if seq_msa:
      with open(os.path.join(msa_output_dir, f'{fasta_name}.a3m'), 'w') as f:
        for seq, desc in seq_msa:
          f.write(f'>{desc}\n{seq}\n')

if __name__ == '__main__':
  import argparse
  import shutil

  parser = argparse.ArgumentParser()
  parser.add_argument('-o', '--output_dir', type=str, default='.',
      help='Output directory')
  for tool_name in (
      'jackhmmer', 'hhblits', 'hhsearch', 'hmmsearch', 'hmmbuild'):
    parser.add_argument(f'--{tool_name}_binary_path', type=str,
        default=shutil.which(tool_name),
        help=f'path to the `{tool_name}` executable.')
  for database_name in (
      'uniref90', 'mgnify', 'bfd', 'small_bfd', 'uniclust30', 'pdb70'):
    parser.add_argument(f'--{database_name}_database_path', type=str,
        default=None,
        help=f'path to database {database_name}')
  parser.add_argument('--fasta_paths', type=str, nargs='+',
      help='list of fasta files')
  parser.add_argument('--use_small_bfd', action='store_true',
      help='use small bfd database or not')
  parser.add_argument('-v', '--verbose', action='store_true', help='verbose')

  args = parser.parse_args()
  main(args)
