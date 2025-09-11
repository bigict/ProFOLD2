"""Docker launch script for ProFOLD docker image."""

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
    raise ValueError(
        f'Failed to find source directory "{source_path}" to mount in Docker container.'
    )
  logging.info('Mounting %s -> %s', source_path, target_path)
  mount = types.Mount(target_path, source_path, type='bind', read_only=read_only)
  return mount, (
      target_path
      if os.path.isdir(path) else os.path.join(target_path, os.path.basename(path))
  )


def main(args):  # pylint: disable=redefined-outer-name
  command_args = ['--platform=local', '--']

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
  # Mount torch_extension directory
  torch_extensions_path = os.path.expanduser('~/.cache/torch_extensions_profold2')
  os.makedirs(torch_extensions_path, exist_ok=True)
  mount, torch_extensions_path = _create_mount(
      '.cache/torch_extensions_path', torch_extensions_path, read_only=False
  )
  mounts.append(mount)

  # Mount output directory
  mount, output_target_path = _create_mount('output', args.output_dir, read_only=False)
  mounts.append(mount)
  command_args += [f'--prefix={output_target_path}']

  command_args += [
      f'--model_recycles={args.model_recycles}',
      f'--model_shard_size={args.model_shard_size}',
  ]
  if args.add_pseudo_linker:
    command_args += ['--add_pseudo_linker']
  if args.no_relaxer:
    command_args += ['--no_relaxer']

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
      runtime='nvidia',
      remove=True,
      detach=True,
      mounts=mounts,
      user=f'{os.geteuid()}:{os.getegid()}',
      environment={
          'NVIDIA_VISIBLE_DEVICES': ','.join(args.gpu_list) if args.gpu_list else 'all',
          'TORCH_EXTENSIONS_DIR': torch_extensions_path,
          'TORCH_HOME': os.path.join(data_target_path, 'torch'),
      }
  )

  # Add signal handler to ensure CTRL+C also stops the running container.
  signal.signal(signal.SIGINT, lambda unused_sig, unused_frame: container.kill())

  for line in container.logs(stream=True):
    logging.info(line.strip().decode('utf-8'))


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument(
      '--docker_image_name',
      type=str,
      default='profold2',
      help='name of the ProFold docker image.'
  )
  parser.add_argument('--gpu_list', type=str, nargs='+', help='list of GPU IDs')
  parser.add_argument(
      '--output_dir',
      type=str,
      default='/tmp/profold',
      help='prefix of output directory.'
  )
  parser.add_argument(
      '--data_dir', type=str, default=None, help='path to directory with models.'
  )
  parser.add_argument('--models', type=str, nargs='+', help='models to be loaded.')
  parser.add_argument(
      '--model_recycles', type=int, default=2, help='number of recycles in profold2.'
  )
  parser.add_argument(
      '--model_shard_size',
      type=int,
      default=256,
      help='shard size in evoformer model.'
  )
  parser.add_argument(
      '--add_pseudo_linker', action='store_true', help='enable loading complex data.'
  )
  parser.add_argument('--no_relaxer', action='store_true', help='do NOT run relaxer.')
  parser.add_argument('--no_gpu_relax', action='store_true', help='run relax on cpu.')
  parser.add_argument('fasta_files', type=str, nargs='+', help='fasta files.')
  parser.add_argument('-v', '--verbose', action='store_true', help='verbose.')
  args = parser.parse_args()

  logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

  main(args)
