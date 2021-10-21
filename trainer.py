"""Tools for train, run
     ```bash
     $python trainer.py -h
     ```
     for further help.
"""
import os
import argparse
from contextlib import contextmanager
import functools
import logging
import random
import resource

import torch
from torch import nn
import torch.multiprocessing as mp
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from profold2 import constants
from profold2.data import esm, scn
from profold2.data.utils import embedding_get_labels, pdb_save
from profold2.model import Alphafold2, ReturnValues
from profold2.model.utils import CheckpointManager

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
  ctx_handler = logging.handlers.QueueHandler(log_queue)
  if args.gpu_list:
    ctx_filter = WorkerLogFilter(args.gpu_list[rank])
    ctx_handler.addFilter(ctx_filter)
  logger.addHandler(ctx_handler)

  level=logging.DEBUG if args.verbose else logging.INFO
  logger.setLevel(level)

  if args.gpu_list:
    logging.info(
            'torch.distributed.init_process_group: rank=%d@%d, world_size=%d',
            rank, args.gpu_list[rank], len(args.gpu_list))
    torch.distributed.init_process_group(
            backend='nccl',
            init_method=f'file://{args.ipc_file}',
            rank=rank,
            world_size=len(args.gpu_list))

def worker_cleanup(args):  # pylint: disable=redefined-outer-name
  if args.gpu_list:
    torch.distributed.destroy_process_group()

def worker_device(rank, args):  # pylint: disable=redefined-outer-name
  if args.gpu_list and rank < len(args.gpu_list):
    assert args.gpu_list[rank] < torch.cuda.device_count()
    #torch.cuda.set_device(args.gpu_list[rank])
    return args.gpu_list[rank]
  return torch.device('cpu')

def worker_model(rank, model, args):  # pylint: disable=redefined-outer-name
  if args.gpu_list and rank < len(args.gpu_list):
    device = worker_device(rank, args)
    model.to(device)

    logging.info('wrap model with nn.parallel.DistributedDataParallel class')
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_list[rank]], output_device=args.gpu_list[rank],
        find_unused_parameters=False)
    model._set_static_graph()  # pylint: disable=protected-access
  return model

@contextmanager
def worker_no_sync(no_sync, model, args):  # pylint: disable=redefined-outer-name
  del args  # pylint: disable=unused-argument
  if no_sync and isinstance(model, nn.parallel.DistributedDataParallel):
    old_require_backward_grad_sync = model.require_backward_grad_sync
    model.require_backward_grad_sync = False
    try:
      yield
    finally:
      model.require_backward_grad_sync = old_require_backward_grad_sync
  else:
    yield

def worker_data_init_fn(rank, args=None):  # pylint: disable=redefined-outer-name
  del rank
  if args:
    random.seed(args.random_seed)

def train(rank, log_queue, args):  # pylint: disable=redefined-outer-name
  random.seed(args.random_seed)

  worker_setup(rank, log_queue, args)

  # helpers
  def cycling(loader, cond=lambda x: True):
    epoch = 0
    while True:
      logging.info('epoch: %d', epoch)

      data_iter = iter(loader)
      for data in data_iter:
        if cond(data):
          yield data

      epoch += 1

  # get data
  device = worker_device(rank, args)
  feats = [('make_pseudo_beta', {}),
           ('make_to_device',
            dict(fields=[
                'seq', 'mask', 'coord', 'coord_mask', 'pseudo_beta',
                'pseudo_beta_mask'
            ], device=device)),
           ('make_esm_embedd',
            dict(model=esm.ESM_MODEL_PATH, repr_layer=esm.ESM_EMBED_LAYER,
                device=device))]
  data = scn.load(casp_version=args.casp_version,
                  thinning=args.casp_thinning,
                  batch_size=args.batch_size,
                  num_workers=args.num_workers,
                  filter_by_resolution=args.filter_by_resolution,
                  feats=feats,
                  dynamic_batching=False,
                  scn_dir=args.scn_dir)

  train_loader = data['train']
  if not train_loader.worker_init_fn:
    logging.info('set worker_init_fn')
    train_loader.worker_init_fn = functools.partial(
                                            worker_data_init_fn, args=args)

  data_cond = lambda x: args.min_protein_len <= x['seq'].shape[1] and x['seq'].shape[1] < args.max_protein_len  # pylint: disable=line-too-long
  dl = cycling(train_loader, data_cond)

  # model
  headers = [
      ('distogram',
          dict(buckets_first_break=2.3125,
              buckets_last_break=21.6875,
              buckets_num=constants.DISTOGRAM_BUCKETS), dict(weight=1.0)),
      ('folding',
          dict(structure_module_depth=args.alphafold2_structure_module_depth,
              structure_module_heads=4,
              fape_min=args.alphafold2_fape_min,
              fape_max=args.alphafold2_fape_max,
              fape_z=args.alphafold2_fape_z), dict(weight=0.1)),
      ('tmscore', {}, {})]

  logging.info('Alphafold2.feats: %s', feats)
  logging.info('Alphafold2.headers: %s', headers)

  model = worker_model(rank, Alphafold2(dim=args.alphafold2_dim,
                     depth=args.alphafold2_evoformer_depth,
                     heads=8,
                     dim_head=64,
                     predict_angles=False,
                     headers=headers), args)

  # optimizer
  optim = Adam(model.parameters(), lr=args.learning_rate)

  # tensorboard
  writer = SummaryWriter(os.path.join(args.prefix, 'runs', 'eval'))
  def model_add_embeddings(writer, model, it):
    def add_embeddings(embedds, prefix=''):
      for k, v in embedds.items():
        if isinstance(v, dict):
          add_embeddings(v, prefix=f'{prefix}{k}_')
        else:
          writer.add_embedding(v, metadata=embedding_get_labels(k, v),
              global_step=it, tag=f'{prefix}{k}')

    if isinstance(model, nn.parallel.DistributedDataParallel):
      embeddings = model.module.embeddings()
    else:
      embeddings = model.embeddings()
    add_embeddings(embeddings)

  global_step = 0
  # CheckpointManager
  if args.checkpoint_every > 0:
    checkpoint_manager = CheckpointManager(
        os.path.join(args.prefix, 'checkpoints'),
        max_to_keep=args.checkpoint_max_to_keep,
        model=model,
        optimizer=optim)
    global_step = checkpoint_manager.restore_or_initialize() + 1
    model.train()

  # training loop
  for it in range(global_step, args.num_batches):
    optim.zero_grad()

    running_loss = {}
    for jt in range(args.gradient_accumulate_every):
      batch = next(dl)

      seq = batch['seq']
      logging.debug('%d %d seq.shape: %s', it, jt, seq.shape)

      # sequence embedding (msa / esm / attn / or nothing)
      r = ReturnValues(**model(batch=batch,
          num_recycle=args.alphafold2_recycles))

      if it == 0 and jt == 0 and args.tensorboard_add_graph:
        with SummaryWriter(os.path.join(args.prefix, 'runs', 'network')) as w:
          w.add_graph(model, (batch, ), verbose=True)

      # running loss
      running_loss['all'] = running_loss.get('all', 0) + r.loss.item()
      for h, v in r.headers.items():
        if 'loss' in v:
          running_loss[h] = running_loss.get(h, 0) + v['loss'].item()

      r.loss.backward()

      if ('tmscore' in r.headers and
          r.headers['tmscore']['loss'].item() >= args.save_pdb):
        pdb_save(it, batch, r.headers, os.path.join(args.prefix, 'pdbs'))

    for k, v in running_loss.items():
      v /= (args.batch_size * args.gradient_accumulate_every)
      logging.info('%d loss@%s: %s', it, k, v)
      writer.add_scalar(f'Loss/train@{k}', v, it)

    optim.step()

    if (args.checkpoint_every > 0 and (it + 1) % args.checkpoint_every == 0 and
        (not args.gpu_list or rank == 0)):
      # Save a checkpoint every N iters.
      checkpoint_manager.save(it)

      # Add embeddings
      model_add_embeddings(writer, model, it)

  writer.close()

  # latest checkpoint
  if (global_step < args.num_batches and
      args.checkpoint_every > 0 and (it + 1) % args.checkpoint_every != 0 and
      (not args.gpu_list or rank == 0)):
    checkpoint_manager.save(it)

    # Add embeddings
    model_add_embeddings(writer, model, it)

  # save model
  if not args.gpu_list or rank == 0:
    torch.save(dict(feats=feats,
            model=model.module
                if isinstance(model, nn.parallel.DistributedDataParallel)
                else model),
        os.path.join(args.prefix, args.model))

  worker_cleanup(args)

def main(args):  # pylint: disable=redefined-outer-name
  # set torch local cache home
  if args.torch_home:
    os.environ['TORCH_HOME'] = args.torch_home

  mp.set_start_method('spawn', force=True)

  # logging
  os.makedirs(os.path.abspath(args.prefix), exist_ok=True)
  if args.checkpoint_every > 0:
    os.makedirs(os.path.abspath(os.path.join(args.prefix, 'checkpoints')),
                exist_ok=True)
  if args.save_pdb <= 1.0:
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
  listener = logging.handlers.QueueListener(log_queue, *handlers,
      respect_handler_level=True)
  listener.start()

  logging.info('-----------------')
  logging.info('Arguments: %s', args)
  logging.info('-----------------')

  mp.spawn(train, args=(log_queue, args),
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
  parser.add_argument('--ipc_file', type=str, default='/tmp/profold2.dist',
      help='ipc file to initialize the process group')
  parser.add_argument('-X', '--model', type=str, default='model.pth',
      help='model of alphafold2, default=\'model.pth\'')
  parser.add_argument('-o', '--prefix', type=str, default='.',
      help='prefix of out directory, default=\'.\'')
  parser.add_argument('-C', '--casp_version', type=int, default=12,
      help='CASP version, default=12')
  parser.add_argument('-T', '--casp_thinning', type=int, default=30,
      help='CASP version, default=30')
  parser.add_argument('-m', '--min_protein_len', type=int, default=50,
      help='filter out proteins whose length<LEN, default=50')
  parser.add_argument('-M', '--max_protein_len', type=int, default=1024,
      help='filter out proteins whose length>LEN, default=1024')
  parser.add_argument('-r', '--filter_by_resolution', type=float, default=0,
      help='filter by resolution<=RES')
  parser.add_argument('--random_seed', type=int, default=None,
      help='random seed')

  parser.add_argument('--torch_home', type=str, help='set env `TORCH_HOME`')
  parser.add_argument('--scn_dir', type=str, default='./sidechainnet_data',
      help='specify scn_dir')

  parser.add_argument('-n', '--num_batches', type=int, default=100000,
      help='number of batches, default=10^5')
  parser.add_argument('-N', '--checkpoint_max_to_keep', type=int, default=5,
      help='the maximum number of checkpoints to keep, default=5')
  parser.add_argument('-K', '--checkpoint_every', type=int, default=100,
      help='save a checkpoint every K times, default=100')
  parser.add_argument('-k',
      '--gradient_accumulate_every', type=int, default=16,
      help='accumulate grads every k times, default=16')
  parser.add_argument('-b', '--batch_size', type=int, default=1,
      help='batch size, default=1')
  parser.add_argument('--num_workers', type=int, default=1,
      help='number of workers, default=1')
  parser.add_argument('-l', '--learning_rate', type=float, default='3e-4',
      help='learning rate, default=3e-4')

  parser.add_argument('--alphafold2_recycles', type=int, default=0,
      help='number of recycles in alphafold2, default=0')
  parser.add_argument('--alphafold2_dim', type=int, default=256,
      help='dimension of alphafold2, default=256')
  parser.add_argument('--alphafold2_evoformer_depth', type=int, default=1,
      help='depth of evoformer in alphafold2, default=1')
  parser.add_argument('--alphafold2_structure_module_depth',
      type=int, default=1,
      help='depth of structure module in alphafold2, default=1')
  parser.add_argument('--alphafold2_fape_min', type=float, default=1e-4,
      help='minimum of dij in alphafold2, default=1e-4')
  parser.add_argument('--alphafold2_fape_max', type=float, default=10.0,
      help='maximum of dij in alphafold2, default=10.0')
  parser.add_argument('--alphafold2_fape_z', type=float, default=10.0,
      help='Z of dij in alphafold2, default=10.0')

  parser.add_argument('--save_pdb', type=float, default=1.0,
      help='save pdb files when TMscore>=VALUE, default=1.0')
  parser.add_argument('--tensorboard_add_graph', action='store_true',
      help='call tensorboard.add_graph')
  parser.add_argument('-v', '--verbose', action='store_true', help='verbose')

  args = parser.parse_args()

  main(args)
