"""Tools for train, run
     ```bash
     $python trainer.py -h
     ```
     for further help.
"""
import os
import contextlib
import copy
from dataclasses import dataclass, make_dataclass
import json
import logging
import random
import re
from typing import Any, Optional
from urllib.parse import urlparse, parse_qsl

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from profold2.common import residue_constants
from profold2.data import dataset
from profold2.data.utils import (
    embedding_get_labels, tensor_to_numpy, weights_from_file
)
from profold2.model import accelerator, optim, FeatureBuilder, MetricDict, ReturnValues
from profold2.model.utils import CheckpointManager
from profold2.utils import default, exists

from profold2.command import worker


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


@dataclass
class Args(worker.Args):
  model_dim: tuple[int, int, int] = (384, 256, 128)
  model_num_tokens: int = len(residue_constants.restypes_with_x)
  model_evoformer_depth: int = 48
  model_evoformer_head_num: int = 8
  model_evoformer_head_dim: int = 32
  model_evoformer_accept_msa_attn: bool = True
  model_evoformer_accept_frame_attn: bool = False
  model_evoformer_accept_frame_update: bool = False
  model_dropout: tuple[float, float] = (0.15, 0.25)
  model_shard_size: Optional[int] = None
  model_recycles: int = 2  # number of recycles in model
  model_recycling_pos: bool = False
  model_recycling_frames: bool = False
  model_features: Optional[str] = None
  model_headers: Optional[str] = None
  model_params_requires_grad: Optional[str] = None
  model_params_requires_hook: Optional[str] = None
  model_params_optim_option: Optional[str] = None

  train_data: str = 'train.zip'  # train dataset dir
  train_idx: Optional[str] = None  # train name idx
  train_chain: Optional[str] = None  #  train dataset chain idx
  train_attr: Optional[str] = None  # train dataset attr idx
  train_data_weights: Optional[str] = None
  train_crop_probability: float = 0.0
  train_pseudo_linker_prob: float = 0.0
  train_msa_as_seq_prob: float = 0.0
  train_msa_as_seq_topn: Optional[int] = None
  train_msa_as_seq_clustering: bool = False
  train_msa_as_seq_min_alr: Optional[float] = None
  train_msa_as_seq_min_ident: Optional[float] = None

  tuning_data: Optional[str] = None  # tuning dataset dir
  tuning_idx: Optional[str] = None  # tuning name idx
  tuning_chain: Optional[str] = None  #  tuning dataset chain idx
  tuning_attr: Optional[str] = None  # tuning dataset attr idx
  tuning_data_weights: Optional[str] = None

  eval_data: Optional[str] = None  # eval dataset dir
  eval_idx: Optional[str] = None  # eval name idx
  eval_chain: Optional[str] = None  #  eval dataset chain idx
  eval_attr: Optional[str] = None  # eval dataset attr idx

  min_protein_len: int = 50
  max_protein_len: int = 1024
  min_crop_len: int = 80
  max_crop_len: int = 256
  crop_algorithm: str = 'auto'
  crop_probability: float = 0.0
  data_rm_mask_prob: float = 0.0
  max_msa_size: int = 1024
  max_var_size: int = 8192

  num_var_task: int = 1

  num_batches: int = 100000  # number of batches
  batch_size: int = 1  # batch size
  num_workers: int = 1  # num of workers
  prefetch_factor: int = 2  # num of batches loaded in advance by each worker
  gradient_accumulate_every: int = 16  # accumulate grads every k times
  learning_rate: float = 1e-3  # learning rate
  amp_enabled: bool = False  # enable automatic mixed precision
  random_seed: Optional[int] = None  # random seed
  checkpoint_every: int = 100
  checkpoint_max_to_keep: int = 5  # the maximum number of checkpoints to keep

  wandb_enabled: bool = False  # enable wandb for experient tracking
  wandb_project: str = 'profold2'  # wandb project name
  wandb_mode: str = 'online'


def preprocess(args):  # pylint: disable=redefined-outer-name
  assert args.model_evoformer_accept_msa_attn or args.model_evoformer_accept_frame_attn  # pylint: disable=line-too-long
  if args.checkpoint_every > 0:
    os.makedirs(os.path.join(args.prefix, 'checkpoints'), exist_ok=True)


def run(rank, args):  # pylint: disable=redefined-outer-name
  from torch.utils.tensorboard import SummaryWriter  # pylint: disable=import-outside-toplevel

  random.seed(args.random_seed)
  np.random.seed(args.random_seed)

  # get data
  wm = worker.WorkerModel(rank, args)
  device = wm.device()
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
  wandb_run = wandb_setup(args) if args.wandb_enabled and wm.is_master() else None
  if exists(wandb_run):
    wandb_run.config.update({'feats': feats, 'headers': headers})

  logging.info('AlphaFold2.feats: %s', feats)
  logging.info('AlphaFold2.headers: %s', headers)

  features = FeatureBuilder(feats).to(device)
  model = wm.wrap(
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
        logging.info('params_requires_grad: name=%s', name)
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

    optimizer = Adam(
        model_params_groups(args.model_params_optim_option), lr=args.learning_rate
    )
  else:
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

  # tensorboard
  writer = SummaryWriter(os.path.join(args.prefix, 'runs', 'eval')
                        ) if wm.is_master() else None

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
        optimizer=optimizer
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
  # .. on small technicality:: If you are using an optimizer like Adam, the scaling
  #     isn't always perfectly linear because Adam normalizes gradients by their
  #     moving average of squared values (the variance). For Adam, sometimes scaling
  #     by ``sqrt{M}`` is more effective than scaling by ``M``.
  scheduler = optim.get_scheduler(
      args.lr_scheduler,
      optimizer,
      num_warmup_steps=args.lr_scheduler_warmup_steps,
      num_training_steps=default(args.lr_scheduler_training_steps, args.num_batches),
      factor=(accelerator.world_size(args.nnodes) or 1)**.5,
      eta_min=args.lr_scheduler_eta_min,
      last_global_step=global_step
  )

  grad_scaler = accelerator.GradScaler(enabled=args.amp_enabled)
  loss_scaler = 1 / (args.gradient_accumulate_every or 1.0)

  def _step(data_loader, it, writer, stage='train', batch_callback=None):
    optimizer.zero_grad(set_to_none=True)

    logging.debug(
        '_step it: %d, loss_scaler: %f, lr: %s', it, loss_scaler,
        scheduler.get_last_lr()
    )

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
      length_scaler = 1.0
      if args.train_apply_sqrt_length_scale:
        length_scaler = torch.sqrt(
            (torch.mean(torch.sum(batch['mask'], dim=-1) + 1e-6)) / args.max_crop_len
        )

      # maybe sync or not
      with no_sync_ctx(
          it != global_step and jt + 1 != args.gradient_accumulate_every, model
      ):
        with accelerator.amp(args.amp_enabled):  # Automatic Mixed Precision
          r = ReturnValues(
              **model(
                  batch=batch,
                  num_recycle=args.model_recycles,
                  shard_size=args.model_shard_size
              )
          )
        # NOTE: do GradScaler.scale first!
        (grad_scaler.scale(r.loss) * loss_scaler * length_scaler).backward()

      # running loss
      running_loss += MetricDict({'all': r.loss})
      for h, v in r.headers.items():
        if 'loss' in v:
          running_loss += MetricDict({h: v['loss']})

    for k, v in running_loss.items():
      v /= args.gradient_accumulate_every
      writer_add_scalars(writer, v, it, prefix=f'Loss/{stage}@{k}')
      # writer.add_scalar(f'Loss/train@{k}', v, it)

    # optimizer.step()
    grad_scaler.step(optimizer)
    grad_scaler.update()
    scheduler.step()

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
        wm.is_master()
    ):
      # Save a checkpoint every N iters.
      checkpoint_manager.save(it)

      # Add embeddings
      writer_add_embeddings(writer, model, it)

    if (
        args.eval_data and wm.is_master() and args.eval_every > 0 and
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
      (it + 1) % args.checkpoint_every != 0 and wm.is_master()
  ):
    checkpoint_manager.save(it)

    # Add embeddings
    writer_add_embeddings(writer, model, it)

  if exists(writer):
    writer.close()

  # save model
  if wm.is_master():
    wm.save(os.path.join(args.prefix, 'model.pth'), feats, model)


if __name__ == '__main__':
  import argparse
  import hydra

  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

  parser.add_argument('-c', '--config', type=str, default=None, help='config file.')
  parser.add_argument(
      'overrides',
      nargs='*',
      metavar='KEY=VAL',
      help='override configs, see: https://hydra.cc'
  )

  args = parser.parse_args()
  config_dir, config_name = os.path.split(
      os.path.abspath(args.config)
  ) if exists(args.config) else (os.getcwd(), None)

  with hydra.initialize_config_dir(
      version_base=None, config_dir=config_dir, job_name=__file__
  ):
    worker.main(
        make_dataclass('t', [], namespace={
            'Args': Args,
            'run': run
        }), hydra.compose(config_name, args.overrides)
    )
