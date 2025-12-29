"""Tools for inference, run
        ```bash
        $python predictor.py -h
        ```
        for further help.
"""
import os
import functools
import glob
import json
import logging

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler

# models & data
from profold2.common import protein
from profold2.data import dataset
from profold2.data.dataset import ProteinSequenceDataset
from profold2.data.parsers import parse_fasta
from profold2.data.utils import pdb_from_prediction
from profold2.model import accelerator, profiler, snapshot, FeatureBuilder, ReturnValues
from profold2.utils import exists, timing

from profold2.command import worker


def _read_fasta(args):  # pylint: disable=redefined-outer-name
  def filename_get(fasta_file):
    fasta_file = os.path.basename(fasta_file)
    pid, _ = os.path.splitext(fasta_file)
    return pid

  if exists(args.fasta_file_list):
    with open(args.fasta_file_list, 'r') as f:
      for fasta_file in filter(lambda x: len(x) > 0, map(lambda x: x.strip(), f)):
        args.fasta_files.append(fasta_file)
  for fasta_glob in args.fasta_files:
    for fasta_file in glob.glob(fasta_glob):
      fasta_name = filename_get(fasta_file)
      with open(fasta_file, 'r') as f:
        fasta_str = f.read()
      yield fasta_name, fasta_str


def _create_dataloader(xpu, args):  # pylint: disable=redefined-outer-name
  kwargs = {'pin_memory': True, 'shuffle': False}
  if exists(args.data_dir):
    if xpu.is_available() and accelerator.world_size(args.nnodes) > 1:
      kwargs['num_replicas'] = accelerator.world_size(args.nnodes)
      kwargs['rank'] = xpu.rank
    return dataset.load(
        data_dir=args.data_dir,
        data_idx=args.data_idx,
        pseudo_linker_prob=1.0 if args.add_pseudo_linker else 0.0,
        pseudo_linker_shuffle=False,
        max_msa_depth=args.max_msa_size,
        num_workers=args.num_workers,
        **kwargs
    )

  sequences, descriptions, msa = [], [], []
  for fasta_name, fasta_str in _read_fasta(args):
    s, d = parse_fasta(fasta_str)
    d[0] = f'{fasta_name} {d[0]}' if exists(d[0]) else fasta_name
    if args.fasta_fmt == 'single':
      sequences += [s]
      descriptions += [d]
      msa += [[None] * len(s)]
    else:
      sequences += [s[:1]]
      descriptions += [d[:1]]
      if len(s) > args.max_msa_size:
        s = s[:1] + list(
            np.random.choice(
                s, size=args.max_msa_size - 1, replace=False
            ) if args.max_msa_size > 1 else []
        )
      msa += [[s]]
  data = ProteinSequenceDataset(sequences, descriptions, msa=msa)
  if xpu.is_available() and accelerator.world_size(args.nnodes) > 1:
    kwargs['sampler'] = DistributedSampler(
        data,
        num_replicas=accelerator.world_size(args.nnodes),
        rank=xpu.rank,
        shuffle=False
    )
  return torch.utils.data.DataLoader(
      data,
      collate_fn=ProteinSequenceDataset.collate_fn,
      num_workers=args.num_workers,
      **kwargs
  )


def _create_relaxer(use_gpu_relax=False):
  from profold2.relax import relax  # pylint: disable=import-outside-toplevel

  return relax.AmberRelaxation(
      max_iterations=relax.RELAX_MAX_ITERATIONS,
      tolerance=relax.RELAX_ENERGY_TOLERANCE,
      stiffness=relax.RELAX_STIFFNESS,
      exclude_residues=relax.RELAX_EXCLUDE_RESIDUES,
      max_outer_iterations=relax.RELAX_MAX_OUTER_ITERATIONS,
      use_gpu=use_gpu_relax
  )


def _load_models(rank, args):  # pylint: disable=redefined-outer-name
  def _location_split(model_location):
    k = model_location.find('=')
    if k != -1:
      return model_location.split('=', 1)
    model_name = os.path.basename(model_location)
    model_name, _ = os.path.splitext(model_name)
    return model_name, model_location

  wm = worker.WorkerModel(rank, args)
  for i, model_location in enumerate(args.models):
    model_name, model_location = _location_split(model_location)
    logging.info(
        'Load model [%d/%d] %s from %s', i, len(args.models), model_name, model_location
    )

    feats, model = wm.load(model_location)
    features = FeatureBuilder(feats).to(wm.device())
    yield model_name, (features, model)


def predict(rank, args):  # pylint: disable=redefined-outer-name
  model_runners = dict(_load_models(rank, args))
  logging.info('Have %d models: %s', len(model_runners), list(model_runners.keys()))

  test_loader = _create_dataloader(rank, args)
  amber_relaxer = (
      None if args.no_relaxer else
      _create_relaxer(use_gpu_relax=rank.is_available() and not args.no_gpu_relax)
  )

  def timing_callback(timings, key, tic, toc):
    timings[key] = toc - tic

  def predict_structure(idx, batch):
    assert len(batch['pid']) == 1
    timings = {}

    fasta_name = ','.join(batch['pid'])
    with timing(
        f'Predicting {fasta_name}',
        print_fn=logging.info,
        callback_fn=functools.partial(timing_callback, timings, 'predict_structure')
    ):
      logging.debug('Sequence [%d] %s shape : %s', idx, fasta_name, batch['seq'].shape)
      if args.fasta_fmt in ('a3m', 'a4m'):
        logging.debug('msa shape %s: %s', fasta_name, batch['msa'].shape)

      output_dir = os.path.join(args.prefix, fasta_name)
      os.makedirs(output_dir, exist_ok=True)

      unrelaxed_pdbs, relaxed_pdbs = {}, {}
      ranking_scores = {}

      def process_structure(model_name, pdb_str, plddt=None):
        ranking_scores[model_name] = 0
        if exists(plddt):
          ranking_scores[model_name] = plddt.item()

        unrelaxed_pdbs[model_name] = pdb_str
        unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}.pdb')
        with open(unrelaxed_pdb_path, 'w') as f:
          f.write(unrelaxed_pdbs[model_name])

        if exists(amber_relaxer):
          # Relax the prediction.
          with timing(
              f'Relax pdb from model {model_name} on {fasta_name}',
              print_fn=logging.info,
              callback_fn=functools.partial(
                  timing_callback, timings, f'relax_{model_name}'
              )
          ):
            retry = 2
            while retry > 0:
              retry -= 1
              try:
                relaxed_pdb_str, _, _ = amber_relaxer.process(
                    prot=protein.from_pdb_string(unrelaxed_pdbs[model_name])
                )
                break
              except ValueError as e:
                logging.error('Relax throw an exception: %s', e)
                if retry <= 0:
                  relaxed_pdb_str = unrelaxed_pdbs[model_name]
                  logging.error('Using unrelaxed pdb instead.')
                  # raise e

          relaxed_pdbs[model_name] = relaxed_pdb_str

          # Save the relaxed PDB.
          relaxed_output_path = os.path.join(output_dir, f'relaxed_{model_name}.pdb')
          with open(relaxed_output_path, 'w') as f:
            f.write(relaxed_pdb_str)

      for model_name, (features, model) in model_runners.items():
        # Build features.
        with timing(
            f'Building features for model {model_name} on {fasta_name}',
            print_fn=logging.info,
            callback_fn=functools.partial(
                timing_callback, timings, f'build_features_{model_name}'
            )
        ):
          feats = features(batch, is_training=False)

        # Predict - out is (b, i, c, 3)
        with torch.no_grad():
          with timing(
              f'Running model {model_name} on {fasta_name}',
              print_fn=logging.info,
              callback_fn=functools.partial(
                  timing_callback, timings, f'predict_{model_name}'
              )
          ):
            with accelerator.amp(args.amp_enabled):  # Automatic Mixed Precision
              r = ReturnValues(
                  **model(
                      batch=feats,
                      num_recycle=args.model_recycles,
                      shard_size=args.model_shard_size
                  )
              )

        # Save the model outputs.
        if not args.no_pth:
          torch.save(r, os.path.join(output_dir, f'result_{model_name}.pth'))

        plddt = None
        if 'confidence' in r.headers:
          plddt = r.headers['confidence']['loss'][0]  # idx = 0
        process_structure(
            model_name, pdb_from_prediction(batch, r.headers, idx=0), plddt=plddt
        )
      # Rank by model confidence and write out relaxed PDBs in rank order.
      ranked_order = []
      for i, (model_name, _) in enumerate(
          sorted(ranking_scores.items(), key=lambda x: x[1], reverse=True)
      ):
        ranked_order.append(model_name)
        ranked_output_path = os.path.join(output_dir, f'ranked_{i}.pdb')
        with open(ranked_output_path, 'w') as f:
          if exists(amber_relaxer):
            f.write(relaxed_pdbs[model_name])
          else:
            f.write(unrelaxed_pdbs[model_name])

      ranking_output_path = os.path.join(output_dir, 'ranking_debug.json')
      with open(ranking_output_path, 'w') as f:
        f.write(
            json.dumps(
                {
                    'confidences': ranking_scores,
                    'order': ranked_order
                }, indent=4
            )
        )

    logging.info('Final timings for %s: %s', fasta_name, timings)

    timings_output_path = os.path.join(output_dir, 'timings.json')
    with open(timings_output_path, 'w') as f:
      f.write(json.dumps(timings, indent=4))

  # Predict structure
  with profiler.profile(
      enabled=args.enable_profiler,
      record_shapes=True,
      profile_memory=True,
      with_stack=True
  ) as prof:
    with snapshot.memory_snapshot(
        enabled=args.enable_memory_snapshot, device=rank.device
    ):
      for idx, batch in enumerate(iter(test_loader)):
        try:
          predict_structure(idx, batch)
        except RuntimeError as e:
          logging.error('%d %s', idx, str(e))

        if hasattr(prof, 'step'):
          prof.step()

  if hasattr(prof, 'key_averages'):
    logging.debug('%s', prof.key_averages().table(sort_by='cuda_time_total'))
  logging.debug('memory_summary: \n%s', rank.memory_summary())


def add_arguments(parser):  # pylint: disable=redefined-outer-name
  parser.add_argument(
      '--map_location',
      type=str,
      default=None,
      help='remapped to an alternative set of devices.'
  )
  parser.add_argument('fasta_files', type=str, nargs='*', help='fasta files.')
  parser.add_argument(
      '--fasta_file_list', type=str, default=None, help='fasta file list.'
  )
  parser.add_argument(
      '--fasta_fmt',
      type=str,
      default='single',
      choices=['single', 'a3m'],
      help='format of fasta files.'
  )

  parser.add_argument(
      '--data_dir', type=str, default=None, help='load data from dataset.'
  )
  parser.add_argument('--data_idx', type=str, default=None, help='dataset idx.')
  parser.add_argument(
      '--add_pseudo_linker', action='store_true', help='enable loading complex data.'
  )

  parser.add_argument(
      '--models',
      type=str,
      nargs='+',
      required=True,
      metavar='[MODEL_NAME=]MODEL_PATH',
      help=' Models to be loaded using [model_name=]model_location format.'
  )
  parser.add_argument(
      '--model_recycles', type=int, default=0, help='number of recycles in profold2.'
  )
  parser.add_argument(
      '--model_shard_size',
      type=int,
      default=None,
      help='shard size in evoformer model.'
  )
  parser.add_argument(
      '--max_msa_size', type=int, default=1024, help='filter out msas whose size>SIZE.'
  )

  parser.add_argument('--num_workers', type=int, default=1, help='number of workers.')
  parser.add_argument('--no_relaxer', action='store_true', help='do NOT run relaxer.')
  parser.add_argument(
      '--no_pth', action='store_true', help='do NOT save prediction header.'
  )
  parser.add_argument('--no_gpu_relax', action='store_true', help='run relax on cpu.')
  parser.add_argument(
      '--amp_enabled', action='store_true', help='enable automatic mixed precision.'
  )
  parser.add_argument('--enable_profiler', action='store_true', help='enable profiler.')
  parser.add_argument(
      '--enable_memory_snapshot', action='store_true', help='enable memory snapshot.'
  )


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()

  # init distributed env
  parser.add_argument('--nnodes', type=int, default=None, help='number of nodes.')
  parser.add_argument('--node_rank', type=int, default=0, help='rank of the node.')
  parser.add_argument(
      '--local_rank', type=int, default=None, help='local rank of xpu, default=None'
  )
  parser.add_argument(
      '--init_method',
      type=str,
      default='file:///tmp/profold2.dist',
      help='method to initialize the process group, '
      'default=\'file:///tmp/profold2.dist\''
  )

  # output dir
  parser.add_argument(
      '-o',
      '--prefix',
      type=str,
      default='.',
      help='prefix of out directory, default=\'.\''
  )
  add_arguments(parser)
  # verbose
  parser.add_argument('-v', '--verbose', action='store_true', help='verbose')

  args = parser.parse_args()

  worker.main(args, predict)
