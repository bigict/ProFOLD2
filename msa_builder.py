"""Tools to build msa, run
     ```bash
     $python msa_builder.py -h
     ```
     for further help.
"""
import os
import pathlib
import logging

from profold2.data.pipeline import DataPipeline

def main(args):  # pylint: disable=redefined-outer-name
  fmt = '%(asctime)-15s [%(levelname)s] (%(filename)s:%(lineno)d) %(message)s'
  level=logging.DEBUG if args.verbose else logging.INFO
  handlers = [
      logging.StreamHandler()]
  logging.basicConfig(
      format=fmt,
      level=level,
      handlers=handlers)

  for tool_name in (  # pylint: disable=redefined-outer-name
      'jackhmmer', 'hhblits', 'hhsearch', 'hmmsearch', 'hmmbuild', 'kalign'):
    if not getattr(args, f'{tool_name}_binary_path'):
      raise ValueError(f'Could not find path to the "{tool_name}" binary. Make '
                       'sure it is installed on your system.')
  # Check for duplicate FASTA file names.
  fasta_names = [pathlib.Path(p).stem for p in args.fasta_paths]
  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')

  data_pipeline = DataPipeline(
      jackhmmer_binary_path=args.jackhmmer_binary_path,
      hhblits_binary_path=args.hhblits_binary_path,
      hhsearch_binary_path=args.hhsearch_binary_path,
      uniref90_database_path=args.uniref90_database_path,
      mgnify_database_path=args.mgnify_database_path,
      bfd_database_path=args.bfd_database_path,
      uniclust30_database_path=args.uniclust30_database_path,
      small_bfd_database_path=args.small_bfd_database_path,
      pdb70_database_path=args.pdb70_database_path,
      template_featurizer=None,
      use_small_bfd=args.use_small_bfd)

  for fasta_path, fasta_name in zip(args.fasta_paths, fasta_names):
    output_dir = os.path.join(args.output_dir, fasta_name)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    msa_output_dir = os.path.join(output_dir, 'msas')
    if not os.path.exists(msa_output_dir):
      os.makedirs(msa_output_dir)

    feature_dict = data_pipeline.process(
        input_fasta_path=fasta_path,
        msa_output_dir=msa_output_dir)
    logging.debug(feature_dict)

if __name__ == '__main__':
  import argparse
  import shutil

  parser = argparse.ArgumentParser()
  parser.add_argument('-o', '--output_dir', type=str, default='.',
      help='Output directory')
  for tool_name in (
      'jackhmmer', 'hhblits', 'hhsearch', 'hmmsearch', 'hmmbuild', 'kalign'):
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
