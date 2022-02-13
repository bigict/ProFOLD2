"""Tools for inference, run
        ```bash
        $python evaluator.py -h
        ```
        for further help.
"""
import os
import argparse
import logging
from logging.handlers import QueueHandler, QueueListener
import resource

import torch
import torch.multiprocessing as mp
from einops import rearrange

# models & data
from profold2.data import dataset
from profold2.data.utils import pdb_save
from profold2.model import ReturnValues
from profold2.utils import Kabsch, TMscore

class WorkerLogFilter(logging.Filter):
  def __init__(self, rank=-1):
    super().__init__()
    self._rank = rank

  def filter(self, record):
    if self._rank != -1:
      record.msg = f'Rank {self._rank} | {record.msg}'
    return True

def worker_setup(rank, log_queue, args):  # pylint: disable=redefined-outer-name
  # logging
  logger = logging.getLogger()
  ctx_handler = QueueHandler(log_queue)
  if args.gpu_list:
    ctx_filter = WorkerLogFilter(args.gpu_list[rank])
    ctx_handler.addFilter(ctx_filter)
  logger.addHandler(ctx_handler)

  level=logging.DEBUG if args.verbose else logging.INFO
  logger.setLevel(level)

  if (args.gpu_list or args.map_location) and torch.cuda.is_available():
    world_size = len(args.gpu_list) if args.gpu_list else 1
    logging.info(
            'torch.distributed.init_process_group: rank=%d@%d, world_size=%d',
            rank, args.gpu_list[rank] if args.gpu_list else 0, world_size)
    torch.distributed.init_process_group(
            backend='nccl',
            init_method=f'file://{args.ipc_file}',
            rank=rank,
            world_size=world_size)

def worker_cleanup(args):  # pylint: disable=redefined-outer-name
  if (args.gpu_list or args.map_location) and torch.cuda.is_available():
    torch.distributed.destroy_process_group()

def worker_device(rank, args):  # pylint: disable=redefined-outer-name
  if args.gpu_list and rank < len(args.gpu_list):
    assert args.gpu_list[rank] < torch.cuda.device_count()
    #torch.cuda.set_device(args.gpu_list[rank])
    return args.gpu_list[rank]
  elif args.map_location:
    return torch.device(args.map_location)
  return torch.device('cpu')

def worker_load(rank, args):  # pylint: disable=redefined-outer-name
  def _feats_gen(feats, device):
    for fn, opts in feats:
      if 'device' in opts:
        opts['device'] = device
      yield fn, opts

  device = worker_device(rank, args)
  checkpoint = torch.load(args.model, map_location=args.map_location)
  feats, model = checkpoint['feats'], checkpoint['model']

  model = model.to(device=device)
  model.eval()

  return list(_feats_gen(feats, device)), model

def evaluate(rank, log_queue, args):  # pylint: disable=redefined-outer-name
  worker_setup(rank, log_queue, args)

  feats, model = worker_load(rank, args)
  logging.info('feats: %s', feats)
  logging.info('model: %s', model)

  test_loader = dataset.load(
                  data_dir=args.casp_data,
                  feat_flags=~dataset.ProteinStructureDataset.FEAT_PDB
                          if args.casp_without_pdb
                          else dataset.ProteinStructureDataset.FEAT_ALL,
                  batch_size=args.batch_size,
                  feats=feats,
                  is_training=False,
                  num_workers=args.num_workers)

  data_cond = lambda x: args.min_protein_len <= x['seq'].shape[1] and x['seq'].shape[1] < args.max_protein_len  # pylint: disable=line-too-long

  tmscore, n = 0, 0
  # eval loop
  for i, batch in enumerate(filter(data_cond, iter(test_loader))):
    if args.num_batches <= 0:
      break
    args.num_batches -= 1

    logging.debug('seq.pids: %s', ','.join(batch['pid']))
    logging.debug('seq.shape: %s', batch['seq'].shape)

    # predict - out is (batch, L * 3, 3)
    with torch.no_grad():
      r = ReturnValues(**model(batch=batch,
                               num_recycle=args.model_recycles))

    if 'confidence' in r.headers:
      logging.info('%d pid: %s Confidence: %s',
            i, ','.join(batch['pid']), r.headers['confidence'])
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
            i, ','.join(batch['pid']), tms.item())

        tmscore, n = tmscore + tms.item(), n + 1

      if args.save_pdb:
        pdb_save(i, batch, r.headers, os.path.join(args.prefix, 'pdbs'))
    else:
      raise ValueError('folding are not implemented yet!')

  if n > 0:
    logging.info('%d TM-score: %f (average)', n, tmscore / n)

  worker_cleanup(args)

def main(args):  # pylint: disable=W0621
  mp.set_start_method('spawn', force=True)

  # logging
  os.makedirs(os.path.abspath(args.prefix), exist_ok=True)
  if args.save_pdb:
    os.makedirs(os.path.abspath(os.path.join(args.prefix, 'pdbs')),
                exist_ok=True)
  handlers = [
      logging.StreamHandler(),
      logging.FileHandler(
          os.path.join(
              args.prefix,
              f'{os.path.splitext(os.path.basename(__file__))[0]}.log'))]

  def handler_apply(h, f, *arg):
    f(*arg)
    return h
  level=logging.DEBUG if args.verbose else logging.INFO
  handlers = list(map(lambda x: handler_apply(x, x.setLevel, level), handlers))
  fmt = '%(asctime)-15s [%(levelname)s] (%(filename)s:%(lineno)d) %(message)s'
  handlers = list(map(lambda x: handler_apply(
      x, x.setFormatter, logging.Formatter(fmt)), handlers))

  logging.basicConfig(
      format=fmt,
      level=level,
      handlers=handlers)

  log_queue = mp.Queue(-1)
  listener = QueueListener(log_queue, *handlers,
      respect_handler_level=True)
  listener.start()

  logging.info('-----------------')
  logging.info('Arguments: %s', args)
  logging.info('-----------------')

  mp.spawn(evaluate, args=(log_queue, args),
          nprocs=len(args.gpu_list) if args.gpu_list else 1,
          join=True)

  logging.info('-----------------')
  logging.info('Resources(myself): %s',
      resource.getrusage(resource.RUSAGE_SELF))
  logging.info('Resources(children): %s',
      resource.getrusage(resource.RUSAGE_CHILDREN))
  logging.info('-----------------')

  listener.stop()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-g', '--gpu_list', type=int, nargs='+',
      help='list of GPU IDs')
  parser.add_argument('--gpu_type', type=str, default='gpu',
      choices=['gpu', 'mlu'],
      help='type of GPUs, one of gpu or mlu')
  parser.add_argument('--ipc_file', type=str, default='/tmp/profold2.dist',
      help='ipc file to initialize the process group')
  parser.add_argument('--map_location', type=str, default=None,
      help='prefix of out directory, default=\'.\'')
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

  parser.add_argument('-n', '--num_batches', type=int, default=100000,
      help='number of batches, default=10^5')
  parser.add_argument('-b', '--batch_size', type=int, default=1,
      help='batch size, default=1')
  parser.add_argument('--num_workers', type=int, default=1,
      help='number of workers, default=1')

  parser.add_argument('--model_recycles', type=int, default=0,
      help='number of recycles in profold2, default=0')

  parser.add_argument('--save_pdb', action='store_true', help='save pdb files')
  parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
  args = parser.parse_args()

  main(args)
