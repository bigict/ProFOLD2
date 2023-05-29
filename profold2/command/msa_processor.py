"""Full ProFOLD msa search pipeline script. run
     ```bash
     $python msa_processor.py -h
     ```
     for further help.
"""

import os
import pathlib
import pickle
import shutil
import time
import logging

from profold2.data import pipeline
from profold2.data import templates
from profold2.data.tools import hhsearch

MAX_TEMPLATE_HITS = 20

def run_msa_tool(
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    data_pipeline: pipeline.DataPipeline):
  """Predicts structure using ProFOLD for the given sequence."""
  logging.info('Querying %s', fasta_name)
  timings = {}
  output_dir = os.path.join(output_dir_base, fasta_name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  msa_output_dir = os.path.join(output_dir, 'msas')
  if not os.path.exists(msa_output_dir):
    os.makedirs(msa_output_dir)

  # Get features.
  t_0 = time.time()
  feature_dict = data_pipeline.process(
      input_fasta_path=fasta_path,
      msa_output_dir=msa_output_dir)
  timings['features'] = time.time() - t_0

  # Write out features as a pickled dictionary.
  features_output_path = os.path.join(output_dir, 'features.pkl')
  with open(features_output_path, 'wb') as f:
    pickle.dump(feature_dict, f, protocol=4)


def msa_process(rank, args):  # pylint: disable=redefined-outer-name
  del rank  # cpu only

  # Check for duplicate FASTA file names.
  fasta_names = [pathlib.Path(p).stem for p in args.fasta_files]
  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')

  template_searcher = hhsearch.HHSearch(
      binary_path=args.hhsearch_binary_path,
      databases=[args.pdb70_database_path])
  template_featurizer = templates.HhsearchHitFeaturizer(
      mmcif_dir=args.template_mmcif_dir,
      max_template_date=args.max_template_date,
      max_hits=MAX_TEMPLATE_HITS,
      kalign_binary_path=args.kalign_binary_path,
      release_dates_path=None,
      obsolete_pdbs_path=args.obsolete_pdbs_path)

  data_pipeline = pipeline.DataPipeline(
      jackhmmer_binary_path=args.jackhmmer_binary_path,
      hhblits_binary_path=args.hhblits_binary_path,
      uniref90_database_path=args.uniref90_database_path,
      mgnify_database_path=args.mgnify_database_path,
      bfd_database_path=args.bfd_database_path,
      uniref30_database_path=args.uniref30_database_path,
      small_bfd_database_path=args.small_bfd_database_path,
      template_searcher=template_searcher,
      template_featurizer=template_featurizer,
      use_small_bfd=args.use_small_bfd,
      use_precomputed_msas=True)

  # Predict structure for each of the sequences.
  for i, fasta_path in enumerate(args.fasta_files):
    fasta_name = fasta_names[i]
    run_msa_tool(
        fasta_path=fasta_path,
        fasta_name=fasta_name,
        output_dir_base=args.prefix,
        data_pipeline=data_pipeline)

def add_arguments(parser):  # pylint: disable=redefined-outer-name
  parser.add_argument('fasta_files', type=str, nargs='*',
      help='Paths to FASTA files, each containing a prediction '
           'target that will be folded one after another.')
  for tool_name in (
      'jackhmmer', 'hhblits', 'hhsearch', 'hmmsearch', 'hmmbuild', 'kalign'):
    parser.add_argument(f'--{tool_name}_binary_path', type=str,
        default=shutil.which(tool_name),
        help=f'path to the `{tool_name}` executable.')
  for database_name in (
      'uniref90', 'mgnify', 'bfd', 'small_bfd', 'uniref30', 'pdb70'):
    parser.add_argument(f'--{database_name}_database_path', type=str,
        default=None,
        help=f'path to database {database_name}')
  parser.add_argument('--use_small_bfd', action='store_true',
      help='use small bfd database or not')
  parser.add_argument('--template_mmcif_dir', type=str, default=None,
      help='Path to a directory with '
           'template mmCIF structures, each named <pdb_id>.cif')
  parser.add_argument('--max_template_date', type=str,default=None,
      help='Maximum template release date '
           'to consider. Important if folding historical test sets.')
  parser.add_argument('--obsolete_pdbs_path', type=str, default=None,
      help='Path to file containing a '
           'mapping from obsolete PDB IDs to the PDB IDs of their '
           'replacements.')
