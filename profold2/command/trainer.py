"""Tools for train, run
     ```bash
     $python trainer.py -h
     ```
     for further help.
"""
import os
import contextlib
import copy
import json
import logging
import random
import re
from urllib.parse import urlparse, parse_qsl

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim import Adam

from profold2.common import residue_constants
from profold2.data import dataset
from profold2.data.utils import (
    embedding_get_labels, tensor_to_numpy, weights_from_file
)
from profold2.model import FeatureBuilder, MetricDict, ReturnValues
from profold2.model.utils import CheckpointManager
from profold2.utils import exists
from profold2.command.worker import main, autocast_ctx, WorkerModel, WorkerXPU


def wandb_setup(args):  # pylint: disable=redefined-outer-name
  import wandb  # pylint: disable=import-outside-toplevel

  if exists(args.wandb_dir):
    os.makedirs(args.wandb_dir, exist_ok=True)
  run = wandb.init(
      project=args.wandb_project,
      dir=args.wandb_dir,
      name=args.wandb_name,
      mode=args.wandb_mode
  )
  run.config.update(args)
  return run


def backward_hook_wrap(name, param, wandb_run=None):
  def backward_hook_print(grad):
    logging.debug('After backard of %s, grad: %s', name, grad)
    if exists(wandb_run):
      import wandb  # pylint: disable=import-outside-toplevel
      if exists(grad):
        wandb_run.log({f'{name}.grad': wandb.Histogram(tensor_to_numpy(grad))})

  logging.debug('Register backward hook: %s', name)
  param.register_hook(backward_hook_print)


@contextlib.contextmanager
def no_sync_ctx(cond, module):
  if cond and isinstance(module, nn.parallel.DistributedDataParallel):
    with module.no_sync():
      yield
  else:
    yield


def preprocess(args):  # pylint: disable=redefined-outer-name
  assert args.model_evoformer_accept_msa_attn or args.model_evoformer_accept_frame_attn  # pylint: disable=line-too-long
  if args.checkpoint_every > 0:
    os.makedirs(os.path.join(args.prefix, 'checkpoints'), exist_ok=True)


def train(rank, args):  # pylint: disable=redefined-outer-name
  from torch.utils.tensorboard import SummaryWriter  # pylint: disable=import-outside-toplevel

  random.seed(args.random_seed)
  np.random.seed(args.random_seed)

  # get data
  worker = WorkerModel(rank, args)
  device = worker.device()
  with open(args.model_features, 'r', encoding='utf-8') as f:
    feats = json.loads(f.read())

  def data_cond(batch):
    return (
        args.min_protein_len <= batch['seq'].shape[1] and
        batch['seq'].shape[1] < args.max_protein_len
    )

  def create_cycling_data(
      data_dir,
      weights=None,
      data_idx=None,
      chain_idx=None,
      attr_idx=None,
      pseudo_linker_prob=0.0,
      crop_probability=0.0,
      data_msa_as_seq_prob=0.0,
      data_msa_as_seq_topn=None,
      data_msa_as_seq_clustering=False,
      data_msa_as_seq_min_alr=None,
      data_msa_as_seq_min_ident=None,
      data_filter=data_cond
  ):
    data_loader = dataset.load(
        data_dir=data_dir,
        data_idx=data_idx,
        chain_idx=chain_idx,
        attr_idx=attr_idx,
        pseudo_linker_prob=pseudo_linker_prob,
        data_rm_mask_prob=args.data_rm_mask_prob,
        msa_as_seq_prob=data_msa_as_seq_prob,
        msa_as_seq_topn=data_msa_as_seq_topn,
        msa_as_seq_clustering=data_msa_as_seq_clustering,
        msa_as_seq_min_alr=data_msa_as_seq_min_alr,
        msa_as_seq_min_ident=data_msa_as_seq_min_ident,
        max_msa_depth=args.max_msa_size,
        max_var_depth=args.max_var_size,
        var_task_num=args.num_var_task,
        min_crop_len=args.min_crop_len,
        max_crop_len=args.max_crop_len,
        min_crop_pae=True,
        max_crop_plddt=True,
        crop_algorithm=args.crop_algorithm,
        crop_probability=crop_probability,
        batch_size=args.batch_size,
        weights=list(weights_from_file(weights)),
        shuffle=True,
        prefetch_factor=args.prefetch_factor,
        drop_last=True,
        pin_memory=True,
        num_workers=args.num_workers
    )
    epoch = 0
    while True:
      logging.info('epoch: %d', epoch)

      data_iter = iter(data_loader)
      for data in data_iter:
        if data_filter(data):
          yield epoch, data

      epoch += 1

  train_data = create_cycling_data(
      args.train_data,
      data_idx=args.train_idx,
      chain_idx=args.train_chain,
      attr_idx=args.train_attr,
      weights=args.train_data_weights,
      pseudo_linker_prob=args.train_pseudo_linker_prob,
      crop_probability=args.train_crop_probability,
      data_msa_as_seq_prob=args.train_msa_as_seq_prob,
      data_msa_as_seq_topn=args.train_msa_as_seq_topn,
      data_msa_as_seq_clustering=args.train_msa_as_seq_clustering,
      data_msa_as_seq_min_alr=args.train_msa_as_seq_min_alr,
      data_msa_as_seq_min_ident=args.train_msa_as_seq_min_ident
  )
  if args.tuning_data:
    tuning_data = create_cycling_data(
        args.tuning_data,
        data_idx=args.tuning_idx,
        chain_idx=args.tuning_chain,
        attr_idx=args.tuning_attr,
        weights=args.tuning_data_weights,
        pseudo_linker_prob=args.tuning_pseudo_linker_prob,
        crop_probability=args.tuning_crop_probability,
        data_msa_as_seq_prob=args.tuning_msa_as_seq_prob,
        data_msa_as_seq_topn=args.tuning_msa_as_seq_topn,
        data_msa_as_seq_clustering=args.tuning_msa_as_seq_clustering,
        data_msa_as_seq_min_alr=args.tuning_msa_as_seq_min_alr,
        data_msa_as_seq_min_ident=args.tuning_msa_as_seq_min_ident
    )

  if args.eval_data:
    eval_loader = dataset.load(
        data_dir=args.eval_data,
        data_idx=args.eval_idx,
        chain_idx=args.eval_chain,
        attr_idx=args.eval_attr,
        max_msa_depth=args.max_msa_size,
        min_crop_len=args.min_crop_len,
        max_crop_len=args.max_crop_len,
        crop_algorithm=args.crop_algorithm,
        batch_size=args.batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=args.num_workers
    )

  # model
  with open(args.model_headers, 'r', encoding='utf-8') as f:
    headers = json.loads(f.read())

  # wandb
  wandb_run = wandb_setup(args) if args.wandb_enabled and worker.is_master() else None
  if exists(wandb_run):
    wandb_run.config.update({'feats': feats, 'headers': headers})

  logging.info('AlphaFold2.feats: %s', feats)
  logging.info('AlphaFold2.headers: %s', headers)

  features = FeatureBuilder(feats).to(device)
  model = worker.wrap(
      dim=args.model_dim,
      evoformer_depth=args.model_evoformer_depth,
      evoformer_head_num=args.model_evoformer_head_num,
      evoformer_head_dim=args.model_evoformer_head_dim,
      num_tokens=args.model_num_tokens,
      attn_dropout=args.model_dropout,
      accept_msa_attn=args.model_evoformer_accept_msa_attn,
      accept_frame_attn=args.model_evoformer_accept_frame_attn,
      accept_frame_update=args.model_evoformer_accept_frame_update,
      headers=headers
  )
  ####
  # HACK
  if exists(args.model_params_requires_grad):
    params_requires_grad_pattern = re.compile(args.model_params_requires_grad)
    for name, param in model.named_parameters():
      if not params_requires_grad_pattern.match(name):
        param.requires_grad = False
      else:
        logging.info('name: %s, param: %s', name, param)
  if exists(args.model_params_requires_hook):
    # torch.set_printoptions(profile='full')
    params_requires_hook = re.compile(args.model_params_requires_hook)
    for name, param in model.named_parameters():
      if params_requires_hook.match(name):
        backward_hook_wrap(name, param, wandb_run=wandb_run)
  ####

  # optimizer
  if exists(args.model_params_optim_option):

    def optim_option_parse(optim_option):
      o = urlparse(optim_option)
      assert o.scheme == 'optim'
      q = {'params': []}
      for k, v in parse_qsl(o.query):
        q[k] = float(v)
      logging.debug('pattern: %s, options: %s', o.fragment, q)
      return re.compile(o.fragment), q

    def model_params_groups(optim_options):
      optim_options = [
          optim_option_parse(optim_option) for optim_option in optim_options.split(',')
      ]
      optim_options.append((re.compile('.*'), {'params': [], 'lr': args.learning_rate}))
      patterns, params = zip(*optim_options)

      for name, param in model.named_parameters():
        for i, p in enumerate(patterns):
          if p.match(name):
            logging.debug('name: %s, pattern_idx: %d', name, i)
            params[i]['params'].append(param)
            break
      return params

    optim = Adam(
        model_params_groups(args.model_params_optim_option), lr=args.learning_rate
    )
  else:
    optim = Adam(model.parameters(), lr=args.learning_rate)

  # tensorboard
  writer = SummaryWriter(os.path.join(args.prefix, 'runs', 'eval')
                        ) if worker.is_master() else None

  def writer_add_embeddings(writer, model, it):
    def add_embeddings(embedds, prefix=''):
      for k, v in embedds.items():
        if isinstance(v, dict):
          add_embeddings(v, prefix=f'{prefix}{k}_')
        elif exists(writer):
          writer.add_embedding(
              v,
              metadata=embedding_get_labels(k, v),
              global_step=it,
              tag=f'{prefix}{k}'
          )

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
    elif exists(writer):
      if isinstance(loss, torch.Tensor):
        loss = torch.nanmean(loss).item()
      logging.info('%d loss@%s: %s', it, prefix, loss)
      writer.add_scalar(prefix, loss, it)
      if exists(wandb_run):
        wandb_run.log({prefix: loss}, step=it)

  global_step = 0
  # CheckpointManager
  if args.checkpoint_every > 0:
    checkpoint_manager = CheckpointManager(
        os.path.join(args.prefix, 'checkpoints'),
        max_to_keep=args.checkpoint_max_to_keep,
        model=model,
        optimizer=optim
    )
    global_step = checkpoint_manager.restore_or_initialize() + 1
    logging.info('checkpoint_manager.global_step: %d', global_step)
    model.train()

  # .. note:: When a model is trained on ``M`` nodes with ``batch=N``, the
  #     gradient will be ``M`` times smaller when compared to the same model
  #     trained on a single node with ``batch=M*N`` if the loss is summed (NOT
  #     averaged as usual) across instances in a batch (because the gradients
  #     between different nodes are averaged). You should take this into
  #     consideration when you want to obtain a mathematically equivalent
  #     training process compared to the local training counterpart. But in most
  #     cases, you can just treat a DistributedDataParallel wrapped model, a
  #     DataParallel wrapped model and an ordinary model on a single GPU as the
  #     same (E.g. using the same learning rate for equivalent batch size).
  grad_scaler = GradScaler(enabled=args.amp_enabled)
  loss_scaler = (WorkerXPU.world_size(args.nnodes) or
                 1) / (args.gradient_accumulate_every or 1.0)

  def _step(data_loader, it, writer, stage='train', batch_callback=None):
    optim.zero_grad(set_to_none=True)

    logging.debug('_step it: %d, loss_scaler: %f', it, loss_scaler)

    running_loss = MetricDict()
    for jt in range(args.gradient_accumulate_every):
      epoch, batch = next(data_loader)
      batch = features(batch, is_training=True)
      if batch_callback:
        batch = batch_callback(batch)

      seq = batch['seq']
      logging.debug(
          '%d %d %d seq.shape: %s pid: %s, clips: %s', epoch, it, jt, seq.shape,
          ','.join(batch['pid']), batch.get('clip')
      )

      # maybe sync or not
      with no_sync_ctx(
          it != global_step and jt + 1 != args.gradient_accumulate_every, model
      ):
        with autocast_ctx(grad_scaler.is_enabled()):
          r = ReturnValues(
              **model(
                  batch=batch,
                  num_recycle=args.model_recycles,
                  shard_size=args.model_shard_size
              )
          )
        grad_scaler.scale(r.loss * loss_scaler).backward()

      # running loss
      running_loss += MetricDict({'all': r.loss})
      for h, v in r.headers.items():
        if 'loss' in v:
          running_loss += MetricDict({h: v['loss']})

    for k, v in running_loss.items():
      v /= args.gradient_accumulate_every
      writer_add_scalars(writer, v, it, prefix=f'Loss/{stage}@{k}')
      # writer.add_scalar(f'Loss/train@{k}', v, it)

    # optim.step()
    grad_scaler.step(optim)
    grad_scaler.update()

  def batch_seq_only(batch):
    batch = copy.copy(batch)
    for field in ('coord', 'coord_alt', 'coord_mask', 'coord_alt_mask', 'coord_plddt', 'backbone_affine', 'backbone_affine_mask', 'atom_affine', 'atom_affine_mask', 'pseudo_beta', 'pseudo_beta_mask', 'torsion_angles', 'torsion_angles_mask', 'torsion_angles_alt'):  # pylint: disable=line-too-long
      if field in batch:
        del batch[field]
    return batch

  def batch_with_coords(batch):
    return batch

  # training loop
  for it in range(global_step, args.num_batches):
    _step(train_data, it, writer, stage='train')

    if (
        args.tuning_data and args.tuning_every > 0 and (it + 1) % args.tuning_every == 0
    ):
      _step(
          tuning_data,
          it,
          writer,
          stage='tuning',
          batch_callback=(
              batch_with_coords if args.tuning_with_coords else batch_seq_only
          )
      )

    if (
        args.checkpoint_every > 0 and (it + 1) % args.checkpoint_every == 0 and
        worker.is_master()
    ):
      # Save a checkpoint every N iters.
      checkpoint_manager.save(it)

      # Add embeddings
      writer_add_embeddings(writer, model, it)

    if (
        args.eval_data and worker.is_master() and args.eval_every > 0 and
        (it + 1) % args.eval_every == 0
    ):

      model.eval()
      with torch.no_grad():
        # eval loss
        n, eval_loss = 0, MetricDict()
        for jt, data in enumerate(iter(eval_loader)):
          seq = data['seq']
          logging.debug(
              '%d %d %d seq.shape: %s, pid: %s, clips: %s', 0, it, jt, seq.shape,
              ','.join(data['pid']), data.get('clip')
          )
          data = features(data, is_training=False)
          r = ReturnValues(**model(batch=data, num_recycle=args.model_recycles))
          for h, v in r.headers.items():
            if 'loss' in v:
              eval_loss += MetricDict({h: v['loss']})
          n += 1
        for k, v in eval_loss.items():
          v /= n
          writer_add_scalars(writer, v, it, prefix=f'Loss/eval@{k}')
          #writer.add_scalar(f'Loss/eval@{k}', v, it)

      model.train()

  # latest checkpoint
  if (
      global_step < args.num_batches and args.checkpoint_every > 0 and
      (it + 1) % args.checkpoint_every != 0 and worker.is_master()
  ):
    checkpoint_manager.save(it)

    # Add embeddings
    writer_add_embeddings(writer, model, it)

  if exists(writer):
    writer.close()

  # save model
  if worker.is_master():
    torch.save(
        dict(
            dim=args.model_dim,  # pylint: disable=use-dict-literal)
            evoformer_depth=args.model_evoformer_depth,
            evoformer_head_num=args.model_evoformer_head_num,
            evoformer_head_dim=args.model_evoformer_head_dim,
            evoformer_accept_msa_attn=args.model_evoformer_accept_msa_attn,
            evoformer_accept_frame_attn=args.model_evoformer_accept_frame_attn,
            evoformer_accept_frame_update=args.model_evoformer_accept_frame_update,  # pylint: disable=line-too-long
            num_tokens=args.model_num_tokens,
            headers=headers,
            feats=feats,
            model=model.module.state_dict()
            if isinstance(model, nn.parallel.DistributedDataParallel) else
            model.state_dict()
        ),
        os.path.join(args.prefix, 'model.pth')
    )


setattr(train, 'preprocess', preprocess)


def add_arguments(parser):  # pylint: disable=redefined-outer-name
  parser.add_argument(
      '-t', '--train_data', type=str, default='train.zip', help='train dataset dir.'
  )
  parser.add_argument('--train_idx', type=str, default=None, help='train dataset idx.')
  parser.add_argument(
      '--train_chain', type=str, default=None, help='train dataset chain idx.'
  )
  parser.add_argument(
      '--train_attr', type=str, default=None, help='train dataset attr idx.'
  )
  parser.add_argument(
      '--train_data_weights',
      type=str,
      default=None,
      help='sample train data by weights.'
  )
  parser.add_argument(
      '-n', '--num_batches', type=int, default=100000, help='number of batches.'
  )
  parser.add_argument(
      '-e', '--eval_data', type=str, default=None, help='eval dataset dir.'
  )
  parser.add_argument('--eval_idx', type=str, default=None, help='eval dataset idx.')
  parser.add_argument(
      '--eval_chain', type=str, default=None, help='eval dataset chain idx.'
  )
  parser.add_argument(
      '--eval_attr', type=str, default=None, help='eval dataset attr idx.'
  )
  parser.add_argument('--tuning_data', type=str, default=None, help='eval dataset dir.')
  parser.add_argument(
      '--tuning_idx', type=str, default=None, help='tuning dataset idx.'
  )
  parser.add_argument(
      '--tuning_chain', type=str, default=None, help='tuning dataset chain idx.'
  )
  parser.add_argument(
      '--tuning_attr', type=str, default=None, help='tuning dataset attr idx.'
  )
  parser.add_argument(
      '--tuning_data_weights',
      type=str,
      default=None,
      help='sample tuning data by weights.'
  )
  parser.add_argument(
      '--tuning_with_coords', action='store_true', help='use `coord` when tuning.'
  )
  parser.add_argument(
      '--min_protein_len',
      type=int,
      default=50,
      help='filter out proteins whose length<LEN.'
  )
  parser.add_argument(
      '--max_protein_len',
      type=int,
      default=1024,
      help='filter out proteins whose length>LEN.'
  )
  parser.add_argument(
      '--max_msa_size', type=int, default=1024, help='sampling MSAs with depth<=SIZE.'
  )
  parser.add_argument(
      '--max_var_size', type=int, default=8192, help='sampling VARs with depth<=SIZE.'
  )
  parser.add_argument(
      '--num_var_task', type=int, default=1, help='number of tasks in VARs.'
  )
  parser.add_argument(
      '--min_crop_len',
      type=int,
      default=80,
      help='do not crop protein whose length<LEN.'
  )
  parser.add_argument(
      '--max_crop_len', type=int, default=255, help='crop protein whose length>LEN.'
  )
  parser.add_argument(
      '--crop_algorithm',
      type=str,
      default='auto',
      choices=['auto', 'random', 'knn'],
      help='type of crop algorithm.'
  )
  parser.add_argument(
      '--train_crop_probability',
      type=float,
      default=0.0,
      help='crop protein with probability CROP_PROBABILITY when it\'s '
      'length>MIN_CROP_LEN.'
  )
  parser.add_argument(
      '--train_pseudo_linker_prob',
      type=float,
      default=0.0,
      help='enable loading complex data.'
  )
  parser.add_argument(
      '--data_rm_mask_prob',
      type=float,
      default=0.0,
      help='remove masked amino acid with probability DATA_RM_MASK_PROB.'
  )
  parser.add_argument(
      '--train_msa_as_seq_prob',
      type=float,
      default=0.0,
      help='take msa_{i} as sequence with probability DATA_MSA_AS_SEQ_PROB.'
  )
  parser.add_argument(
      '--train_msa_as_seq_topn',
      type=int,
      default=None,
      help='take msa_{i} as sequence belongs to DATA_MSA_AS_SEQ_TOPN.'
  )
  parser.add_argument(
      '--train_msa_as_seq_clustering',
      action='store_true',
      help='take msa_{i} as sequence sampling from clusters.'
  )
  parser.add_argument(
      '--train_msa_as_seq_min_alr',
      type=float,
      default=None,
      help='take msa_{i} as sequence with alr <= DATA_MSA_AS_SEQ_MIN_ALR.'
  )
  parser.add_argument(
      '--train_msa_as_seq_min_ident',
      type=float,
      default=None,
      help='take msa_{i} as sequence with ident <= DATA_MSA_AS_SEQ_MIN_IDENT.'
  )
  parser.add_argument(
      '--tuning_pseudo_linker_prob',
      type=float,
      default=0.0,
      help='enable loading complex data.'
  )
  parser.add_argument(
      '--tuning_crop_probability',
      type=float,
      default=0.0,
      help='crop protein with probability CROP_PROBABILITY when it\'s '
      'length>MIN_CROP_LEN.'
  )
  parser.add_argument(
      '--tuning_msa_as_seq_prob',
      type=float,
      default=0.0,
      help='take msa_{i} as sequence with probability DATA_MSA_AS_SEQ_PROB.'
  )
  parser.add_argument(
      '--tuning_msa_as_seq_topn',
      type=int,
      default=None,
      help='take msa_{i} as sequence belongs to DATA_MSA_AS_SEQ_TOPN.'
  )
  parser.add_argument(
      '--tuning_msa_as_seq_clustering',
      action='store_true',
      help='take msa_{i} as sequence sampling from clusters.'
  )
  parser.add_argument(
      '--tuning_msa_as_seq_min_alr',
      type=float,
      default=None,
      help='take msa_{i} as sequence with alr <= DATA_MSA_AS_SEQ_MIN_ALR.'
  )
  parser.add_argument(
      '--tuning_msa_as_seq_min_ident',
      type=float,
      default=None,
      help='take msa_{i} as sequence with ident <= DATA_MSA_AS_SEQ_MIN_IDENT.'
  )
  parser.add_argument('--random_seed', type=int, default=None, help='random seed.')

  parser.add_argument(
      '--checkpoint_max_to_keep',
      type=int,
      default=5,
      help='the maximum number of checkpoints to keep.'
  )
  parser.add_argument(
      '--checkpoint_every',
      type=int,
      default=100,
      help='save a checkpoint every K times.'
  )
  parser.add_argument(
      '--tuning_every', type=int, default=10, help='tuning model every K times.'
  )
  parser.add_argument(
      '--eval_every', type=int, default=100, help='eval model every K times.'
  )
  parser.add_argument(
      '--gradient_accumulate_every',
      type=int,
      default=16,
      help='accumulate grads every k times.'
  )
  parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch size')
  parser.add_argument('--num_workers', type=int, default=1, help='number of workers.')
  parser.add_argument(
      '--prefetch_factor',
      type=int,
      default=2,
      help='number of batches loaded in advance by each worker.'
  )
  parser.add_argument(
      '-l', '--learning_rate', type=float, default='1e-3', help='learning rate.'
  )

  parser.add_argument(
      '--model_features',
      type=str,
      default='model_features_main.json',
      help='json format features of model.'
  )
  parser.add_argument(
      '--model_headers',
      type=str,
      default='model_headers_main.json',
      help='json format headers of model.'
  )
  parser.add_argument(
      '--model_recycles', type=int, default=2, help='number of recycles in model.'
  )
  parser.add_argument(
      '--model_recycling_frames', action='store_true', help='enable frame recycling.'
  )
  parser.add_argument(
      '--model_recycling_pos',
      action='store_true',
      help='enable backbone atom position recycling.'
  )
  parser.add_argument(
      '--model_dim',
      type=int,
      nargs=3,
      default=(384, 256, 128),
      help='dimension of model.'
  )
  parser.add_argument(
      '--model_num_tokens',
      type=int,
      default=len(residue_constants.restypes_with_x),
      help='number of tokens in the model.'
  )
  parser.add_argument(
      '--model_evoformer_depth',
      type=int,
      default=1,
      help='depth of evoformer in model.'
  )
  parser.add_argument(
      '--model_evoformer_head_num',
      type=int,
      default=48,
      help='number of heads in evoformer model.'
  )
  parser.add_argument(
      '--model_evoformer_head_dim',
      type=int,
      default=32,
      help='dimensions of each head in evoformer model.'
  )
  parser.add_argument(
      '--model_shard_size',
      type=int,
      default=None,
      help='shard size in evoformer model.'
  )
  parser.add_argument(
      '--model_dropout',
      type=float,
      nargs=2,
      default=(0.15, 0.25),
      help='dropout of evoformer(single & pair) in model.'
  )
  parser.add_argument(
      '--model_evoformer_accept_msa_attn',
      action='store_true',
      help='enable MSATransformer in evoformer.'
  )
  parser.add_argument(
      '--model_evoformer_accept_frame_attn',
      action='store_true',
      help='enable FrameTransformer in evoformer.'
  )
  parser.add_argument(
      '--model_evoformer_accept_frame_update',
      action='store_true',
      help='enable FrameUpdater in evoformer.'
  )
  parser.add_argument(
      '--model_params_requires_grad',
      type=str,
      default=None,
      help='learn partial parameters only.'
  )
  parser.add_argument(
      '--model_params_requires_hook',
      type=str,
      default=None,
      help='hook partial parameters.'
  )
  parser.add_argument(
      '--model_params_optim_option',
      type=str,
      default=None,
      help='optimizer arguments accepted by partial parameters only.'
  )

  parser.add_argument(
      '--wandb_enabled',
      action='store_true',
      help='Enable wandb for experient tracking'
  )
  parser.add_argument(
      '--wandb_project', type=str, default='profold2', help='Wandb project name'
  )
  parser.add_argument('--wandb_dir', type=str, default=None, help='Wandb dir')
  parser.add_argument('--wandb_name', type=str, default=None, help='Wandb name name')
  parser.add_argument(
      '--wandb_mode',
      type=str,
      default='online',
      choices=['online', 'offline', 'disabled'],
      help='Wandb mode'
  )
  parser.add_argument(
      '--amp_enabled', action='store_true', help='enable automatic mixed precision.'
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

  main(args, train)
