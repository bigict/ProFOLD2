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
from worker import main, WorkerModel

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
  if args.gpu_list and len(args.gpu_list) > 1:
    kwargs['num_replicas'] = len(args.gpu_list)
    kwargs['rank'] = rank
  test_loader = dataset.load(
      data_dir=args.casp_data,
      max_msa_size=args.max_msa_size,
      min_crop_len=args.min_crop_len,
      max_crop_len=args.max_crop_len,
      crop_algorithm=args.crop_algorithm,
      crop_probability=args.crop_probability,
      feat_flags=(~dataset.ProteinStructureDataset.FEAT_PDB
                  if args.casp_without_pdb
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

    if 'confidence' in r.headers:
      logging.info('%d pid: %s Confidence: %s',
            i, fasta_name, r.headers['confidence']['loss'].item())
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
        logging.info('%d pid: %s TM-score: %f',
            i, fasta_name, tms.item())

        tmscore, n = tmscore + tms.item(), n + 1

      if args.save_pdb:
        pdb_save(batch, r.headers, os.path.join(args.prefix, 'pdbs'), step=i)
    else:
      raise ValueError('folding are not implemented yet!')

  if n > 0:
    logging.info('%d TM-score: %f (average)', n, tmscore / n)

setattr(evaluate, 'preprocess', preprocess)

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('-g', '--gpu_list', type=int, nargs='+',
      help='list of GPU IDs')
  parser.add_argument('--ipc_file', type=str, default='/tmp/profold2.dist',
      help='ipc file to initialize the process group, '
           'default="/tmp/profold2.dist"')
  parser.add_argument('--map_location', type=str, default=None,
      help='remapped to an alternative set of devices, default=None')
  parser.add_argument('-X', '--model', type=str, default='model.pth',
      help='model of profold2, default=\'model.pth\'')
  parser.add_argument('-o', '--prefix', type=str, default='.',
      help='prefix of out directory, default=\'.\'')
  parser.add_argument('-k', '--casp_data', type=str, default='test',
      help='CASP dataset, default=\'test\'')
  parser.add_argument('--casp_without_pdb', action='store_true',
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
  parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
  args = parser.parse_args()

  main(args, evaluate)
