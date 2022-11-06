"""Docker launch script for ProFold docker image."""

import os
import logging
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
  command_args = []
  if args.gpu_list:
    command_args += [f'--gpu_list {" ".join(args.gpu_list)}']
  command_args += ['--map_location=cpu']

  mounts = []

  # Mount data_dir directory
  mount, data_target_path = _create_mount('data', args.data_dir)
  mounts.append(mount)
  if args.models:
    def _to_model_location(model_name):
      if not model_name.endswith('.pth'):
        model_name = f'{model_name}.pth'
      return os.path.join(data_target_path, model_name)
    models = ' '.join([_to_model_location(model) for model in args.models])
    command_args += [f'--models {models}']

  # Mount output directory
  mount, output_target_path = _create_mount(
                              'output', args.output_dir, read_only=False)
  mounts.append(mount)
  command_args += [f'--prefix={output_target_path}']

  command_args += [
          f'--model_sequence_max_input_len={args.model_sequence_max_input_len}',
          f'--model_sequence_max_step_len={args.model_sequence_max_step_len}',
          f'--model_recycles={args.model_recycles}',
          f'--model_shard_size={args.model_shard_size}',
      ]
  if args.no_relaxer:
    command_args += ['--no_relaxer']
  if args.gpu_list:
    command_args += ['--use_gpu_relax']

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
  container = client.containers.run(
      image=args.docker_image_name,
      command=command_args,
      runtime='nvidia' if args.gpu_list else None,
      remove=True,
      detach=True,
      mounts=mounts,
      user=f'{os.geteuid()}:{os.getegid()}',
      environment={
          'TORCH_HOME': os.path.join(data_target_path, 'torch'),
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

  parser.add_argument('--gpu_list', type=str, nargs='+',
      help='list of GPU IDs')
  parser.add_argument('--output_dir', type=str, default='/tmp/profold',
      help='prefix of output directory, default=/tmp/profold')
  parser.add_argument('--data_dir', type=str, default=None,
      help='path to directory with models, default=None')
  parser.add_argument('--models', type=str, nargs='+',
      help='models to be loaded')

  parser.add_argument('--model_sequence_max_input_len', type=int, default=384,
      help='predict sequence embedding segment by seqment, default=384')
  parser.add_argument('--model_sequence_max_step_len', type=int, default=128,
      help='predict sequence embedding segment by seqment, default=128')
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
