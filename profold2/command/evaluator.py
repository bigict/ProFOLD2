"""Tools for inference, run
        ```bash
        $python evaluator.py -h
        ```
        for further help.
"""
import os
import logging

import torch
from einops import rearrange

# models & data
from profold2.data import dataset
from profold2.data.utils import pdb_save
from profold2.model import FeatureBuilder, ReturnValues
from profold2.utils import Kabsch, TMscore, timing

from profold2.command.worker import main, WorkerModel, WorkerXPU

def preprocess(args):  # pylint: disable=redefined-outer-name
  if args.save_pdb:
    os.makedirs(os.path.abspath(os.path.join(args.prefix, 'pdbs')),
                exist_ok=True)

def evaluate(rank, args):  # pylint: disable=redefined-outer-name
  worker = WorkerModel(rank, args)
  feats, model = worker.load(args.model)
  features = FeatureBuilder(feats).to(worker.device())
  logging.info('feats: %s', feats)
  logging.info('model: %s', model)

  kwargs = {}
  if rank.is_available() and WorkerXPU.world_size(args.nnodes) > 1:
    kwargs['num_replicas'] = WorkerXPU.world_size(args.nnodes)
    kwargs['rank'] = rank.rank
  test_loader = dataset.load(
      data_dir=args.eval_data,
      data_idx=args.eval_idx,
      max_msa_size=args.max_msa_size,
      min_crop_len=args.min_crop_len,
      max_crop_len=args.max_crop_len,
      crop_algorithm=args.crop_algorithm,
      crop_probability=args.crop_probability,
      feat_flags=(~dataset.ProteinStructureDataset.FEAT_PDB
                  if args.eval_without_pdb
                  else dataset.ProteinStructureDataset.FEAT_ALL),
      batch_size=args.batch_size,
      num_workers=args.num_workers, **kwargs)

  def data_cond(batch):
    return (args.min_protein_len <= batch['seq'].shape[1] and
        batch['seq'].shape[1] < args.max_protein_len)

  tmscore, n = 0, 0
  # eval loop
  for i, batch in enumerate(filter(data_cond, iter(test_loader))):
    fasta_name, fasta_len = ','.join(batch['pid']), batch['seq'].shape[1]
    with timing(f'Building features for model on {fasta_name} {fasta_len}',
        logging.debug):
      batch = features(batch, is_training=False)

    # predict - out is (batch, L * 3, 3)
    with timing(f'Running model on {fasta_name} {fasta_len}', logging.debug):
      with torch.no_grad():
        r = ReturnValues(**model(batch=batch,  # pylint: disable=not-callable
            sequence_max_input_len=args.model_sequence_max_input_len,
            sequence_max_step_len=args.model_sequence_max_step_len,
            num_recycle=args.model_recycles,
            shard_size=args.model_shard_size))

    metric_dict = {}
    if 'confidence' in r.headers:
      metric_dict['confidence'] = r.headers['confidence']['loss'].item()
      logging.debug('%d pid: %s Confidence: %s',
            i, fasta_name, r.headers['confidence']['loss'].item())
    if 'metric' in r.headers:
      metrics = r.headers['metric']['loss']
      if 'contact' in metrics:
        if '[24,inf)_1' in metrics['contact']:
          metric_dict['P@L'] = metrics['contact']['[24,inf)_1'].item()
      if 'coevolution' in metrics:
        if 'perplexity' in metrics['coevolution']:
          metric_dict['perplexity'] = metrics['coevolution']['perplexity']
    if 'folding' in r.headers:
      assert 'coords' in r.headers['folding']
      if 'coord' in batch:
        coords = r.headers['folding']['coords']  # (b l c d)
        _, _, num_atoms, _ = coords.shape

        labels = batch['coord'][...,:num_atoms,:]
        flat_cloud_mask = rearrange(
            batch['coord_mask'][...,:num_atoms], 'b l c -> b (l c)')

        # rotate / align
        coords_aligned, labels_aligned = Kabsch(
            rearrange(
                rearrange(coords,
                          'b l c d -> b (l c) d')[flat_cloud_mask],
                'c d -> d c'),
            rearrange(
                rearrange(labels,
                          'b l c d -> b (l c) d')[flat_cloud_mask],
                'c d -> d c'))
        logging.debug('coords_aligned: %s', coords_aligned.shape)
        logging.debug('labels_aligned: %s', labels_aligned.shape)

        tms = TMscore(rearrange(coords_aligned, 'd l -> () d l'),
                      rearrange(labels_aligned, 'd l -> () d l'),
                      L=torch.sum(batch['mask'], dim=-1))
        metric_dict['tmscore'] = tms.item()
        logging.debug('%d pid: %s TM-score: %f',
            i, fasta_name, tms.item())

        tmscore, n = tmscore + tms.item(), n + 1

      logging.info('no: %d pid: %s, %s', i, fasta_name,
                   ', '.join(f'{k}: {v}' for k, v in metric_dict.items()))
      if args.save_pdb:
        pdb_save(batch, r.headers, os.path.join(args.prefix, 'pdbs'), step=i)
    else:
      raise ValueError('folding are not implemented yet!')

  if n > 0:
    logging.info('%d TM-score: %f (average)', n, tmscore / n)

setattr(evaluate, 'preprocess', preprocess)

def add_arguments(parser):  # pylint: disable=redefined-outer-name
  parser.add_argument('--map_location', type=str, default=None,
      help='remapped to an alternative set of devices, default=None')
  parser.add_argument('--model', type=str, default='model.pth',
      help='model of profold2, default=\'model.pth\'')

  parser.add_argument('--eval_data', type=str, default=None,
      help='eval dataset, default=None')
  parser.add_argument('--eval_idx', type=str, default='name.idx',
      help='eval dataset idx, default=\'name.idx\'')
  parser.add_argument('--eval_without_pdb', action='store_true',
      help='DO NOT load pdb data')
  parser.add_argument('--min_protein_len', type=int, default=0,
      help='filter out proteins whose length<LEN, default=0')
  parser.add_argument('--max_protein_len', type=int, default=1024,
      help='filter out proteins whose length>LEN, default=1024')
  parser.add_argument('--max_msa_size', type=int, default=512,
      help='filter out msas whose size>SIZE, default=512')
  parser.add_argument('--min_crop_len', type=int, default=None,
      help='filter out proteins whose length<LEN, default=None')
  parser.add_argument('--max_crop_len', type=int, default=None,
      help='filter out proteins whose length>LEN, default=None')
  parser.add_argument('--crop_algorithm', type=str, default='random',
      choices=['random', 'domain'],
      help='type of crop algorithm')
  parser.add_argument('--crop_probability', type=float, default=0.0,
      help='crop protein with probability CROP_PROBABILITY when it\'s '
          'length>MIN_CROP_LEN, default=0.0')

  parser.add_argument('-b', '--batch_size', type=int, default=1,
      help='batch size, default=1')
  parser.add_argument('--num_workers', type=int, default=1,
      help='number of workers, default=1')

  parser.add_argument('--model_sequence_max_input_len', type=int, default=None,
      help='predict sequence embedding segment by seqment, default=None')
  parser.add_argument('--model_sequence_max_step_len', type=int, default=None,
      help='predict sequence embedding segment by seqment, default=None')
  parser.add_argument('--model_recycles', type=int, default=0,
      help='number of recycles in profold2, default=0')
  parser.add_argument('--model_shard_size', type=int, default=None,
      help='shard size in evoformer model, default=None')

  parser.add_argument('--save_pdb', action='store_true', help='save pdb files')

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
  parser.add_argument('-v', '--verbose', action='store_true', help='verbose')

  args = parser.parse_args()

  main(args, evaluate)
