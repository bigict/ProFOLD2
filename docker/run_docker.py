"""Docker launch script for ProFold docker image."""

import os
import logging
import pathlib
import signal

import docker
from docker import types

_ROOT_MOUNT_DIRECTORY = '/mnt/'

def _create_mount(mount_name, path, read_only=True):
  path = os.path.abspath(path)
  source_path = path if os.path.isdir(path) else os.path.dirname(path)
  target_path = os.path.join(_ROOT_MOUNT_DIRECTORY, mount_name)
  if not os.path.exists(source_path):
    raise ValueError(f'Failed to find source directory "{source_path}" to '
                     'mount in Docker container.')
  logging.info('Mounting %s -> %s', source_path, target_path)
  mount = types.Mount(target_path, source_path, type='bind',
      read_only=read_only)
  return mount, (target_path if os.path.isdir(path)
                 else os.path.join(target_path, os.path.basename(path)))

def main(args):  # pylint: disable=redefined-outer-name
  # You can individually override the following paths if you have placed the
  # data in locations other than the FLAGS.data_dir.

  # Path to the Uniref90 database for use by JackHMMER.
  uniref90_database_path = os.path.join(
      args.data_dir, 'uniref90', 'uniref90.fasta')

  # Path to the MGnify database for use by JackHMMER.
  mgnify_database_path = os.path.join(
      args.data_dir, 'mgnify', 'mgy_clusters.fa')

  # Path to the BFD database for use by HHblits.
  bfd_database_path = os.path.join(
      args.data_dir, 'bfd',
      'bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt')

  # Path to the Uniclust30 database for use by HHblits.
  uniclust30_database_path = os.path.join(
      args.data_dir, 'uniclust30', 'uniclust30_2018_08', 'uniclust30_2018_08')

  # Path to the PDB70 database for use by HHsearch.
  pdb70_database_path = os.path.join(args.data_dir, 'pdb70', 'pdb70')

  # Path to a directory with template mmCIF structures, each named <pdb_id>.cif.
  template_mmcif_dir = os.path.join(args.data_dir, 'pdb_mmcif', 'mmcif_files')

  # Path to a file mapping obsolete PDB IDs to their replacements.
  obsolete_pdbs_path = os.path.join(args.data_dir, 'pdb_mmcif', 'obsolete.dat')

  profold_path = pathlib.Path(__file__).parent.parent
  data_dir_path = pathlib.Path(args.data_dir)
  if profold_path == data_dir_path or profold_path in data_dir_path.parents:
    raise ValueError(
        f'The download directory {args.data_dir} should not be a subdirectory '
        f'in the ProFOLD repository directory. If it is, the Docker build is '
        f'slow since the large databases are copied during the image creation.')

  command_args = []

  mounts = []

  # Mount the data directory
  mount, data_target_path = _create_mount('data_dir', args.data_dir)
  mounts.append(mount)

  database_paths = [
      ('uniref90_database_path', uniref90_database_path),
      ('mgnify_database_path', mgnify_database_path),
      ('template_mmcif_dir', template_mmcif_dir),
      ('obsolete_pdbs_path', obsolete_pdbs_path),
      ('pdb70_database_path', pdb70_database_path),
      ('uniref30_database_path', uniclust30_database_path),
      ('bfd_database_path', bfd_database_path),
  ]

  for name, path in database_paths:
    if path:
      mount, target_path = _create_mount(name, path)
      mounts.append(mount)
      command_args.append(f'--{name}={target_path}')

  output_target_path = os.path.join(_ROOT_MOUNT_DIRECTORY, 'output')

  # Mount output directory
  mount, output_target_path = _create_mount(
                              'output', args.output_dir, read_only=False)
  mounts.append(mount)

  command_args += [
      f'--max_template_date={args.max_template_date}',
  ]
  if args.use_precomputed_msas:
    command_args += ['--use_precomputed_msas']
  command_args += [
      f'--prefix={output_target_path}',
      '--map_location=cpu',
  ]

  if args.models:
    def _to_model_location(model_name):
      if not model_name.endswith('.pth'):
        model_name = f'{model_name}.pth'
      return os.path.join(data_target_path, 'params', model_name)
    models = ' '.join([_to_model_location(model) for model in args.models])
    command_args += [f'--models {models}']

  command_args += [
      f'--fasta_fmt={args.fasta_fmt}',
      f'--model_recycles={args.model_recycles}',
      f'--model_shard_size={args.model_shard_size}',
  ]
  if args.no_relaxer:
    command_args += ['--no_relaxer']
  if args.verbose:
    command_args += ['--verbose']

  # Mount each fasta path as a unique target directory.
  target_fasta_paths = []
  for i, fasta_path in enumerate(args.fasta_files):
    mount, target_path = _create_mount(f'fasta_path_{i}', fasta_path)
    mounts.append(mount)
    target_fasta_paths.append(target_path)
  command_args += target_fasta_paths
  logging.debug('command_args: %s', command_args)

  # Run docker
  client = docker.from_env()
  device_requests = [
      docker.types.DeviceRequest(driver='nvidia', capabilities=[['gpu']])
  ]
  container = client.containers.run(
      image=args.docker_image_name,
      command=command_args,
      device_requests=device_requests,
      remove=True,
      detach=True,
      mounts=mounts,
      user=f'{os.geteuid()}:{os.getegid()}',
      environment={
          'NVIDIA_VISIBLE_DEVICES': 'all',
      })

  # Add signal handler to ensure CTRL+C also stops the running container.
  signal.signal(signal.SIGINT,
                lambda unused_sig, unused_frame: container.kill())

  for line in container.logs(stream=True):
    logging.info(line.strip().decode('utf-8'))

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--docker_image_name', type=str, default='profold2',
      help='name of the ProFold docker image, default=\'profold2\'')

  parser.add_argument('--output_dir', type=str, default='/tmp/profold',
      help='prefix of output directory, default=/tmp/profold')
  parser.add_argument('--data_dir', type=str, default=None,
      help='path to directory with models, default=None')
  parser.add_argument('--models', type=str, nargs='+',
      help='models to be loaded')
  parser.add_argument('--fasta_fmt', type=str, default='single',
      choices=['single', 'a3m', 'a4m', 'pkl'],
      help='format of fasta files, default=\'single\'')

  parser.add_argument('--max_template_date', type=str,default=None,
      help='Maximum template release date '
           'to consider. Important if folding historical test sets.')
  parser.add_argument('--use_precomputed_msas', action='store_true',
      help='Whether to read MSAs that '
           'have been written to disk instead of running the MSA '
           'tools. The MSA files are looked up in the output '
           'directory, so it must stay the same between multiple '
           'runs that are to reuse the MSAs. WARNING: This will not '
           'check if the sequence, database or configuration have '
           'changed.')

  parser.add_argument('--model_recycles', type=int, default=2,
      help='number of recycles in profold2, default=2')
  parser.add_argument('--model_shard_size', type=int, default=4,
      help='shard size in evoformer model, default=4')

  parser.add_argument('--no_relaxer', action='store_true',
      help='do NOT run relaxer')

  parser.add_argument('fasta_files', type=str, nargs='+',
      help='fasta files')

  parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
  args = parser.parse_args()

  logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

  main(args)
