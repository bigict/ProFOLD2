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
import pickle
import shutil
import time
import logging

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler

# models & data
from profold2.common import protein
from profold2.data import dataset
from profold2.data import pipeline
from profold2.data import templates
from profold2.data.dataset import ProteinSequenceDataset, ProteinPklDataset
from profold2.data.parsers import parse_fasta
from profold2.data.tools import hhsearch
from profold2.data.utils import pdb_from_prediction
from profold2.model import FeatureBuilder, ReturnValues
from profold2.utils import exists, timing

from profold2.command.worker import main, WorkerModel, WorkerXPU

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

  return feature_dict

def _read_fasta(args):  # pylint: disable=redefined-outer-name
  def filename_get(fasta_file):
    fasta_file = os.path.basename(fasta_file)
    pid, _ = os.path.splitext(fasta_file)
    return pid

  if hasattr(args, 'fasta_data'):
    for fasta_name, fasta_str in args.fasta_data:
      yield fasta_name, fasta_str
  else:
    if args.fasta_fmt == 'pkl':
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
          use_precomputed_msas=args.use_precomputed_msas)
    for fasta_glob in args.fasta_files:
      for fasta_file in glob.glob(fasta_glob):
        fasta_name = filename_get(fasta_file)
        with open(fasta_file, 'r', encoding='utf-8') as f:
          fasta_str = f.read()
        if args.fasta_fmt == 'pkl':
          run_msa_tool(
              fasta_path=fasta_file,
              fasta_name=fasta_name,
              output_dir_base=args.prefix,
              data_pipeline=data_pipeline)
        yield fasta_name, fasta_str

def _create_dataloader(xpu, args):  # pylint: disable=redefined-outer-name
  kwargs = {'pin_memory': True}
  if exists(args.data_dir):
    if xpu.is_available() and WorkerXPU.world_size(args.nnodes) > 1:
      kwargs['num_replicas'] = WorkerXPU.world_size(args.nnodes)
      kwargs['rank'] = xpu.rank
    return dataset.load(
        data_dir=args.data_dir,
        data_idx=args.data_idx,
        max_msa_size=args.max_msa_size,
        num_workers=args.num_workers, **kwargs)

  sequences, descriptions, msa = [], [], []
  for fasta_name, fasta_str in _read_fasta(args):
    if args.fasta_fmt == 'a4m':
      s = fasta_str.splitlines()
      d = [None] * len(s)
    else:
      s, d = parse_fasta(fasta_str)
    d[0] = fasta_name
    if args.fasta_fmt == 'single':
      sequences += s
      descriptions += d
      msa += [None] * len(s)
    elif args.fasta_fmt == 'pkl':
      sequences += [os.path.join(args.prefix, fasta_name, 'features.pkl')]
      descriptions += d
    else:
      sequences += s[:1]
      descriptions += d[:1]
      if len(s) > args.max_msa_size:
        s = s[:1] + list(np.random.choice(
            s,
            size=args.max_msa_size - 1,
            replace=False) if args.max_msa_size > 1 else [])
      msa += [s]
  if args.fasta_fmt == 'pkl':
    data = ProteinPklDataset(sequences, descriptions)
  else:
    data = ProteinSequenceDataset(sequences, descriptions, msa=msa)
    kwargs['collate_fn'] = ProteinSequenceDataset.collate_fn
  if xpu.is_available() and WorkerXPU.world_size(args.nnodes) > 1:
    kwargs['sampler'] = DistributedSampler(data,
        num_replicas=WorkerXPU.world_size(args.nnodes), rank=xpu.rank)
  return torch.utils.data.DataLoader(data,
                                     num_workers=args.num_workers, **kwargs)

def _create_relaxer(use_gpu_relax=False):
  from profold2.relax import relax  # pylint: disable=import-outside-toplevel

  return relax.AmberRelaxation(
      max_iterations=relax.RELAX_MAX_ITERATIONS,
      tolerance=relax.RELAX_ENERGY_TOLERANCE,
      stiffness=relax.RELAX_STIFFNESS,
      exclude_residues=relax.RELAX_EXCLUDE_RESIDUES,
      max_outer_iterations=relax.RELAX_MAX_OUTER_ITERATIONS,
      use_gpu=use_gpu_relax)

def _load_models(rank, args):  # pylint: disable=redefined-outer-name
  def _location_split(model_location):
    k = model_location.find('=')
    if k != -1:
      return model_location.split('=', 1)
    model_name = os.path.basename(model_location)
    model_name, _ = os.path.splitext(model_name)
    return model_name, model_location

  worker = WorkerModel(rank, args)
  for i, model_location in enumerate(args.models):
    model_name, model_location = _location_split(model_location)
    logging.info('Load model [%d/%d] %s from %s',
        i, len(args.models), model_name, model_location)

    feats, model = worker.load(model_location)
    features = FeatureBuilder(feats).to(worker.device())
    yield model_name, (features, model)

def predict(rank, args):  # pylint: disable=redefined-outer-name
  model_runners = dict(_load_models(rank, args))
  logging.info('Have %d models: %s', len(model_runners),
              list(model_runners.keys()))

  test_loader = _create_dataloader(rank, args)
  amber_relaxer = (None
      if args.no_relaxer
      else _create_relaxer(
          use_gpu_relax=rank.is_available() and not args.no_gpu_relax))

  def timing_callback(timings, key, tic, toc):
    timings[key] = toc - tic

  # Predict structure
  for idx, batch in enumerate(iter(test_loader)):
    assert len(batch['pid']) == 1
    timings = {}

    fasta_name = ','.join(batch['pid'])
    with timing(f'Predicting {fasta_name}',
        print_fn=logging.info,
        callback_fn=functools.partial(timing_callback,
            timings, 'predict_structure')):
      logging.debug('Sequence %d shape %s: %s',
                    idx, fasta_name, batch['seq'].shape)
      if args.fasta_fmt in ('a3m', 'a4m'):
        logging.debug('msa shape %s: %s', fasta_name, batch['msa'].shape)

      output_dir = os.path.join(args.prefix, fasta_name)
      os.makedirs(output_dir, exist_ok=True)

      unrelaxed_pdbs, relaxed_pdbs = {}, {}
      ranking_scores = {}
      for model_name, (features, model) in model_runners.items():
        # Build features.
        with timing(f'Building features for model {model_name} on {fasta_name}',
            print_fn=logging.info,
            callback_fn=functools.partial(timing_callback,
                timings, f'build_features_{model_name}')):
          feats = features(batch, is_training=False)

        # Predict - out is (batch, L * 3, 3)
        with torch.no_grad():
          with timing(f'Running model {model_name} on {fasta_name}',
              print_fn=logging.info,
              callback_fn=functools.partial(timing_callback,
                  timings, f'predict_{model_name}')):
            r = ReturnValues(**model(batch=feats,
                num_recycle=args.model_recycles,
                shard_size=args.model_shard_size))

        ranking_scores[model_name] = 0
        if 'confidence' in r.headers:
          ranking_scores[model_name] = r.headers['confidence']['loss'].item()

        # Save the model outputs.
        if not args.no_pth:
          torch.save(r, os.path.join(output_dir, f'result_{model_name}.pth'))

        unrelaxed_pdbs[model_name] = pdb_from_prediction(batch,
                                                         r.headers, idx=0)
        unrelaxed_pdb_path = os.path.join(output_dir,
                                          f'unrelaxed_{model_name}.pdb')
        with open(unrelaxed_pdb_path, 'w', encoding='utf-8') as f:
          f.write(unrelaxed_pdbs[model_name])

        if exists(amber_relaxer):
          # Relax the prediction.
          with timing(f'Relax pdb from model {model_name} on {fasta_name}',
              print_fn=logging.info,
              callback_fn=functools.partial(timing_callback,
                  timings, f'relax_{model_name}')):
            retry = 2
            while retry > 0:
              retry -= 1
              try:
                relaxed_pdb_str, _, _ = amber_relaxer.process(
                    prot=protein.from_pdb_string(unrelaxed_pdbs[model_name]))
                break
              except ValueError as e:
                if retry <= 0:
                  raise e
                logging.error('Relax throw an exception: %s', e)

          relaxed_pdbs[model_name] = relaxed_pdb_str

          # Save the relaxed PDB.
          relaxed_output_path = os.path.join(
              output_dir, f'relaxed_{model_name}.pdb')
          with open(relaxed_output_path, 'w', encoding='utf-8') as f:
            f.write(relaxed_pdb_str)

      # Rank by model confidence and write out relaxed PDBs in rank order.
      ranked_order = []
      for i, (model_name, _) in enumerate(
          sorted(ranking_scores.items(), key=lambda x: x[1], reverse=True)):
        ranked_order.append(model_name)
        ranked_output_path = os.path.join(output_dir, f'ranked_{i}.pdb')
        with open(ranked_output_path, 'w', encoding='utf-8') as f:
          if exists(amber_relaxer):
            f.write(relaxed_pdbs[model_name])
          else:
            f.write(unrelaxed_pdbs[model_name])

      ranking_output_path = os.path.join(output_dir, 'ranking_debug.json')
      with open(ranking_output_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(
            {'confidences': ranking_scores, 'order': ranked_order}, indent=4))

    logging.info('Final timings for %s: %s', fasta_name, timings)

    timings_output_path = os.path.join(output_dir, 'timings.json')
    with open(timings_output_path, 'w', encoding='utf-8') as f:
      f.write(json.dumps(timings, indent=4))

def add_arguments(parser):  # pylint: disable=redefined-outer-name
  parser.add_argument('--map_location', type=str, default=None,
      help='remapped to an alternative set of devices, default=None')
  parser.add_argument('fasta_files', type=str, nargs='*',
      help='fasta files')
  parser.add_argument('--fasta_fmt', type=str, default='single',
      choices=['single', 'a3m', 'a4m', 'pkl'],
      help='format of fasta files, default=\'single\'')

  parser.add_argument('--data_dir', type=str, default=None,
      help='load data from dataset, default=None')
  parser.add_argument('--data_idx', type=str, default=None,
      help='dataset idx, default=None')

  parser.add_argument('--models', type=str, nargs='+',
      metavar='[MODEL_NAME=]MODEL_PATH',
      help=' Models to be loaded using [model_name=]model_location format')
  parser.add_argument('--model_recycles', type=int, default=0,
      help='number of recycles in profold2, default=0')
  parser.add_argument('--model_shard_size', type=int, default=None,
      help='shard size in evoformer model, default=None')
  parser.add_argument('--max_msa_size', type=int, default=1024,
      help='filter out msas whose size>SIZE, default=1024')

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
  parser.add_argument('--pdb_seqres_database_path', type=str, default=None,
      help='Path to the PDB '
           'seqres database for use by hmmsearch.')
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
  parser.add_argument('--use_precomputed_msas', action='store_true',
      help='Whether to read MSAs that '
           'have been written to disk instead of running the MSA '
           'tools. The MSA files are looked up in the output '
           'directory, so it must stay the same between multiple '
           'runs that are to reuse the MSAs. WARNING: This will not '
           'check if the sequence, database or configuration have '
           'changed.')

  parser.add_argument('--num_workers', type=int, default=1,
      help='number of workers, default=1')
  parser.add_argument('--no_relaxer', action='store_true',
      help='do NOT run relaxer')
  parser.add_argument('--no_pth', action='store_true',
      help='do NOT save prediction header')
  parser.add_argument('--no_gpu_relax', action='store_true',
      help='run relax on cpu')

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()

  # init distributed env
  parser.add_argument('--nnodes', type=int, default=None,
      help='number of nodes.')
  parser.add_argument('--node_rank', type=int, default=0,
      help='rank of the node.')
  parser.add_argument('--local_rank', type=int, default=None,
      help='local rank of xpu, default=None')
  parser.add_argument('--init_method', type=str,
      default='file:///tmp/profold2.dist',
      help='method to initialize the process group, '
           'default=\'file:///tmp/profold2.dist\'')

  # output dir
  parser.add_argument('-o', '--prefix', type=str, default='.',
      help='prefix of out directory, default=\'.\'')
  add_arguments(parser)
  # verbose
  parser.add_argument('-v', '--verbose', action='store_true', help='verbose')

  args = parser.parse_args()

  main(args, predict)
