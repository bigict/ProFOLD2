"""Tools for train, run
     ```bash
     $python trainer.py -h
     ```
     for further help.
"""
import os
import argparse
import copy
import functools
import json
import logging
from logging.handlers import QueueHandler, QueueListener
import random
import resource

import numpy as np
import torch
from torch import nn
import torch.multiprocessing as mp
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from profold2.data import dataset, esm
from profold2.data.utils import (
    cycling,
    embedding_get_labels,
    pdb_save,
    weights_from_file)
from profold2.model import Alphafold2, MetricDict, ReturnValues
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
  ctx_handler = QueueHandler(log_queue)
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
    torch.cuda.set_device(args.gpu_list[rank])

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

def worker_data_init_fn(rank, args=None):  # pylint: disable=redefined-outer-name
  del rank
  if args:
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

def train(rank, log_queue, args):  # pylint: disable=redefined-outer-name
  random.seed(args.random_seed)
  np.random.seed(args.random_seed)

  worker_setup(rank, log_queue, args)

  # get data
  device = worker_device(rank, args)
  with open(args.model_features, 'r', encoding='utf-8') as f:
    feats = json.loads(f.read())
    for i in range(len(feats)):
      feat_name, feat_args = feats[i]
      if 'device' in feat_args and feat_args['device'] == '%(device)s':
        feat_args['device'] = device
        feats[i] = (feat_name, feat_args)

  train_loader = dataset.load(
      data_dir=args.train_data,
      data_idx=args.train_idx,
      max_msa_size=args.max_msa_size,
      min_crop_len=args.min_crop_len,
      max_crop_len=args.max_crop_len,
      crop_algorithm=args.crop_algorithm,
      crop_probability=args.crop_probability,
      feats=feats,
      worker_init_fn=functools.partial(
          worker_data_init_fn, args=args),
      batch_size=args.batch_size,
      weights=list(weights_from_file(args.sampling_by_weights)),
      shuffle=True,
      num_workers=args.num_workers)
  if args.tuning_data:
    tuning_loader = dataset.load(
        data_dir=args.tuning_data,
        max_msa_size=args.max_msa_size,
        min_crop_len=args.min_crop_len,
        max_crop_len=args.max_crop_len,
        feats=feats,
        batch_size=args.batch_size,
        is_training=True,
        shuffle=True,
        num_workers=0)

  if args.eval_data:
    eval_loader = dataset.load(
        data_dir=args.eval_data,
        max_msa_size=args.max_msa_size,
        min_crop_len=args.min_crop_len,
        max_crop_len=args.max_crop_len,
        crop_algorithm=args.crop_algorithm,
        feats=feats,
        batch_size=args.batch_size,
        is_training=False,
        num_workers=0)

    fake_loader = dataset.load(
        data_dir=args.eval_data,
        max_msa_size=args.max_msa_size,
        min_crop_len=args.min_crop_len,
        max_crop_len=args.max_crop_len,
        crop_algorithm=args.crop_algorithm,
        feats=feats,
        batch_size=args.batch_size,
        is_training=False,
        shuffle=True,
        num_workers=0)

  data_cond = lambda x: args.min_protein_len <= x['seq'].shape[1] and x['seq'].shape[1] < args.max_protein_len  # pylint: disable=line-too-long
  train_data = cycling(train_loader, data_cond)
  if args.tuning_data:
    tuning_data = cycling(tuning_loader, data_cond)

  # model
  with open(args.model_headers, 'r', encoding='utf-8') as f:
    headers = json.loads(f.read())

  logging.info('Alphafold2.feats: %s', feats)
  logging.info('Alphafold2.headers: %s', headers)

  model = worker_model(rank, Alphafold2(dim=args.model_dim,
                     depth=args.model_evoformer_depth,
                     heads=8,
                     dim_head=64,
                     embedd_dim=args.model_embedd_dim,
                     headers=headers), args)

  # optimizer
  optim = Adam(model.parameters(), lr=args.learning_rate)

  # tensorboard
  writer = SummaryWriter(os.path.join(args.prefix, 'runs', 'eval'))
  def writer_add_embeddings(writer, model, it):
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

  def writer_add_scalars(writer, loss, it, prefix=''):
    if isinstance(loss, MetricDict):
      if prefix:
        prefix = f'{prefix}.'
      for k, v in loss.items():
        writer_add_scalars(writer, v, it, prefix=f'{prefix}{k}')
    else:
      if isinstance(loss, torch.Tensor):
        loss = loss.item()
      logging.info('%d loss@%s: %s', it, prefix, loss)
      if writer:
        writer.add_scalar(prefix, loss, it)

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

  def _step(data_loader, it, writer, stage='train', batch_callback=None):
    optim.zero_grad()

    running_loss = MetricDict()
    for jt in range(args.gradient_accumulate_every):
      epoch, batch = next(data_loader)
      if batch_callback:
        batch = batch_callback(batch)

      seq = batch['seq']
      logging.debug('%d %d %d seq.shape: %s pid: %s clips: %s',
          epoch, it, jt, seq.shape, ','.join(batch['pid']), batch.get('clips'))

      # sequence embedding (msa / esm / attn / or nothing)
      r = ReturnValues(**model(batch=batch, num_recycle=args.model_recycles))

      # running loss
      running_loss += MetricDict({'all':r.loss})
      for h, v in r.headers.items():
        if 'loss' in v:
          running_loss += MetricDict({h:v['loss']})

      r.loss.backward()

      if ('tmscore' in r.headers and
          r.headers['tmscore']['loss'].item() >= args.save_pdb):
        pdb_save(it, batch, r.headers, os.path.join(args.prefix, 'pdbs'))

    for k, v in running_loss.items():
      v /= (args.batch_size * args.gradient_accumulate_every)
      writer_add_scalars(writer, v, it, prefix=f'Loss/{stage}@{k}')
      #writer.add_scalar(f'Loss/train@{k}', v, it)

    optim.step()

  def batch_seq_only(batch):
    batch = copy.copy(batch)
    for field in ('coord', 'coord_mask', 'backbone_affine', 'backbone_affine_mask', 'pseudo_beta', 'pseudo_beta_mask'):
      if field in batch:
        del batch[field]
    return batch

  # training loop
  for it in range(global_step, args.num_batches):
    _step(train_data, it, writer, stage='train')

    if (args.checkpoint_every > 0 and (it + 1) % args.checkpoint_every == 0 and
        (not args.gpu_list or rank == 0)):
      # Save a checkpoint every N iters.
      checkpoint_manager.save(it)

      # Add embeddings
      writer_add_embeddings(writer, model, it)

    if (args.tuning_data and 
        args.tuning_every > 0 and (it + 1) % args.tuning_every == 0):
      _step(tuning_data, it, writer, stage='tuning', batch_callback=batch_seq_only)

    if (args.eval_data and 
        args.eval_every > 0 and (it + 1) % args.eval_every == 0):
      _step(cycling(fake_loader), it, None, stage='fake', batch_callback=batch_seq_only)

    if (args.eval_data and (not args.gpu_list or rank == 0) and
        args.eval_every > 0 and (it + 1) % args.eval_every == 0):
      
      model.eval()
      with torch.no_grad():
        # eval loss
        n, eval_loss = 0, MetricDict()
        for data in iter(eval_loader):
          r = ReturnValues(**model(batch=data, num_recycle=args.model_recycles))
          for h, v in r.headers.items():
            if 'loss' in v:
              eval_loss += MetricDict({h:v['loss']})
          n += 1
        for k, v in eval_loss.items():
          v /= (args.batch_size * n)
          writer_add_scalars(writer, v, it, prefix=f'Loss/eval@{k}')
          #writer.add_scalar(f'Loss/eval@{k}', v, it)

      model.train()

  writer.close()

  # latest checkpoint
  if (global_step < args.num_batches and
      args.checkpoint_every > 0 and (it + 1) % args.checkpoint_every != 0 and
      (not args.gpu_list or rank == 0)):
    checkpoint_manager.save(it)

    # Add embeddings
    writer_add_embeddings(writer, model, it)

  # save model
  if not args.gpu_list or rank == 0:
    torch.save(dict(feats=feats,
            model=model.module
                if isinstance(model, nn.parallel.DistributedDataParallel)
                else model),
        os.path.join(args.prefix, 'model.pth'))

  worker_cleanup(args)

def main(args):  # pylint: disable=redefined-outer-name
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
  listener = QueueListener(log_queue, *handlers,
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
  parser.add_argument('-o', '--prefix', type=str, default='.',
      help='prefix of out directory, default=\'.\'')
  parser.add_argument('-t', '--train_data', type=str, default='train',
      help='train dataset dir, default=\'train\'')
  parser.add_argument('--train_idx', type=str, default='name.idx',
      help='train dataset idx, default=\'name.idx\'')
  parser.add_argument('-n', '--num_batches', type=int, default=100000,
      help='number of batches, default=10^5')
  parser.add_argument('-e', '--eval_data', type=str, default=None,
      help='eval dataset dir, default=None')
  parser.add_argument('--tuning_data', type=str, default=None,
      help='eval dataset dir, default=None')
  parser.add_argument('--sampling_by_weights', type=str, default=None,
      help='sample train data by weights, default=None')
  parser.add_argument('--min_protein_len', type=int, default=50,
      help='filter out proteins whose length<LEN, default=50')
  parser.add_argument('--max_protein_len', type=int, default=1024,
      help='filter out proteins whose length>LEN, default=1024')
  parser.add_argument('--max_msa_size', type=int, default=128,
      help='filter out msas whose size>SIZE, default=128')
  parser.add_argument('--min_crop_len', type=int, default=80,
      help='filter out proteins whose length<LEN, default=80')
  parser.add_argument('--max_crop_len', type=int, default=255,
      help='filter out proteins whose length>LEN, default=255')
  parser.add_argument('--crop_algorithm', type=str, default='random',
      choices=['random', 'domain'],
      help='type of crop algorithm')
  parser.add_argument('--crop_probability', type=float, default=0.0,
      help='crop protein with probability CROP_PROBABILITY when it\'s '
          'length>MIN_CROP_LEN, default=0.0')
  parser.add_argument('--random_seed', type=int, default=None,
      help='random seed')

  parser.add_argument('--checkpoint_max_to_keep', type=int, default=5,
      help='the maximum number of checkpoints to keep, default=5')
  parser.add_argument('--checkpoint_every', type=int, default=100,
      help='save a checkpoint every K times, default=100')
  parser.add_argument('--tuning_every', type=int, default=10,
      help='eval model every K times, default=1000')
  parser.add_argument('--eval_every', type=int, default=1000,
      help='eval model every K times, default=1000')
  parser.add_argument(
      '--gradient_accumulate_every', type=int, default=16,
      help='accumulate grads every k times, default=16')
  parser.add_argument('-b', '--batch_size', type=int, default=1,
      help='batch size, default=1')
  parser.add_argument('--num_workers', type=int, default=1,
      help='number of workers, default=1')
  parser.add_argument('-l', '--learning_rate', type=float, default='3e-4',
      help='learning rate, default=3e-4')

  parser.add_argument('--model_features', type=str,
      default='model_features_main.json',
      help='json format features of model, default=model_features_main.json')
  parser.add_argument('--model_headers', type=str,
      default='model_headers_main.json',
      help='json format headers of model, default=model_headers_main.json')
  parser.add_argument('--model_recycles', type=int, default=0,
      help='number of recycles in model, default=0')
  parser.add_argument('--model_dim', type=int, default=256,
      help='dimension of model, default=256')
  parser.add_argument('--model_embedd_dim', type=int,
      default=esm.ESM_EMBED_DIM,
      help=f'dimension of alphafold2, default={esm.ESM_EMBED_DIM}')
  parser.add_argument('--model_evoformer_depth', type=int, default=1,
      help='depth of evoformer in model, default=1')

  parser.add_argument('--save_pdb', type=float, default=1.0,
      help='save pdb files when TMscore>=VALUE, default=1.0')
  parser.add_argument('-v', '--verbose', action='store_true', help='verbose')

  args = parser.parse_args()

  main(args)
