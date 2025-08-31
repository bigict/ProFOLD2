"""Wrap distibuted env
"""
import os
import contextlib
from datetime import timedelta
import functools
import logging
from logging.handlers import QueueHandler, QueueListener
import re
import resource

import torch
from torch.cuda.amp import autocast
import torch.multiprocessing as mp
from torch import nn

from profold2.model import AlphaFold2
from profold2.model.commons import torch_allow_tf32
from profold2.utils import default, env, exists, version_cmp


@contextlib.contextmanager
def autocast_ctx(cond):
  if cond:
    dtype = torch.float16
    if hasattr(torch.cuda, 'is_bf16_supported'):
      if torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    ctx = functools.partial(autocast, dtype=dtype)
    # FIXED ME: cache_enabled=True will crash :(
    if version_cmp(torch.__version__, '1.10.0') >= 0:
      ctx = functools.partial(ctx, cache_enabled=False)
    with ctx():
      yield
  else:
    yield


class _WorkerLogRecordFactory(object):
  """Preprocess tensor args before creating a LogRecord
  """
  def __init__(self, record_factory):
    self.record_factory = record_factory

  def __call__(
      self, name, level, fn, lno, msg, args, exc_info, func=None, sinfo=None, **kwargs
  ):
    def _tensor_to_list(x):
      if isinstance(x, torch.Tensor):
        if len(x.shape) > 1 or (len(x.shape) == 1 and x.shape[0] != 1):
          return x.tolist()
        return x.item()
      return x

    if exists(args):
      args = tuple(map(_tensor_to_list, args))
    return self.record_factory(
        name, level, fn, lno, msg, args, exc_info, func=func, sinfo=sinfo, **kwargs
    )


class _WorkerLogFilter(logging.Filter):
  def __init__(self, rank=-1):
    super().__init__()
    self._rank = rank

  def filter(self, record):
    if self._rank != -1:
      record.msg = f'Rank {self._rank} | {record.msg}'
    return True


class _WorkerLogging(object):
  """Initialize distibuted logger
  """
  def __init__(self, work_fn, args):  # pylint: disable=redefined-outer-name
    # logging
    os.makedirs(args.prefix, exist_ok=True)

    local_rank = f'-{args.local_rank}' if exists(args.local_rank) else ''
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(
                args.prefix, f'{work_fn.__name__}_{args.node_rank}{local_rank}.log'
            )
        )
    ]

    level = logging.DEBUG if args.verbose else logging.INFO
    fmt = '%(asctime)-15s [%(levelname)s] (%(filename)s:%(lineno)d) %(message)s'

    logging.basicConfig(format=fmt, level=level, handlers=handlers)

    if exists(args.nnodes):
      self.queue = mp.Queue(-1)
      self.listener = QueueListener(self.queue, *handlers, respect_handler_level=True)
    else:
      self.queue = None
      self.listener = None

  def start(self):
    if exists(self.listener):
      self.listener.start()

  def stop(self):
    if exists(self.listener):
      self.listener.stop()


class WorkerXPU(object):
  """Wrap distibuted XPU(GPU,MLU etc)
  """
  def __init__(self, rank, args):
    self.local_rank = rank
    self.args = args

  def is_available(self):
    if self.local_rank != -1:
      return torch.cuda.is_available()
    return False

  def is_master(self):
    if self.is_available():
      return self.args.node_rank == 0 and self.local_rank == 0
    return True

  @property
  def device(self):
    if self.is_available():
      return self.local_rank
    return torch.device('cpu')

  @staticmethod
  def device_count():
    return torch.cuda.device_count()

  @property
  def rank(self):
    return env(
        'RANK',
        defval=WorkerXPU.device_count() * self.args.node_rank + self.local_rank,
        dtype=int
    )

  @staticmethod
  def world_size(nnodes=None):
    return env(
        'WORLD_SIZE', defval=WorkerXPU.device_count() * default(nnodes, 1), dtype=int
    )

  def memory_summary(self):
    if self.is_available() and hasattr(torch.cuda, 'memory_summary'):
      return torch.cuda.memory_summary()
    return 'only cuda supported.'

  def init_process_group(self):
    if self.is_available():
      timeout = env('NCCL_TIMEOUT', defval=1800, dtype=int)
      logging.info(
          'distributed.init_process_group: rank=%s@%s, world_size=%s@%s, '
          'init_method=%s, timeout=%s(s)',
          self.device, WorkerXPU.device_count(), self.rank,
          WorkerXPU.world_size(self.args.nnodes), self.args.init_method, timeout
      )
      torch.distributed.init_process_group(
          backend='nccl',
          init_method=self.args.init_method,
          timeout=timedelta(seconds=timeout),
          rank=self.rank,
          world_size=WorkerXPU.world_size(self.args.nnodes)
      )
      torch.cuda.set_device(self.local_rank)

  def destroy_process_group(self):
    if self.is_available():
      torch.distributed.destroy_process_group()


class WorkerModel(object):
  """Wrap distibuted model
  """
  def __init__(self, xpu, args):  # pylint: disable=redefined-outer-name
    self.xpu = xpu
    self.args = args

  def is_master(self):
    return self.xpu.is_master()

  def device(self):
    return self.xpu.device

  def hook(self, model):
    def _load_state_dict_pre_hook(state_dict, *args, **kwargs):
      key_modifier_list = [
          ('(.*)impl.token_emb.(.*)', '\\1impl.embedder.to_single_emb.\\2'),
          ('(.*)impl.to_pairwise_repr.(.*)', '\\1impl.embedder.to_pairwise_emb.\\2')
      ]
      key_modifier_list = [
          functools.partial(re.sub, pattern, repl)
          for pattern, repl in key_modifier_list
      ]
      key_list_new = {}
      for key, val in state_dict.items():
        key_new = key
        for key_modifier in key_modifier_list:
          key_new = key_modifier(key_new)
        if key_new != key:
          logging.warning('load_state_dict_pre_hook: from <%s> to <%s>', key, key_new)
          key_list_new[key] = key_new
      if key_list_new:
        for key, key_new in key_list_new.items():
          state_dict[key_new] = state_dict[key]
          del state_dict[key]

    if hasattr(model, 'register_load_state_dict_pre_hook'):
      register_hook = model.register_load_state_dict_pre_hook
    else:
      register_hook = model._register_load_state_dict_pre_hook
    return register_hook(_load_state_dict_pre_hook)

  def wrap(self, **kwargs):
    model = AlphaFold2(**kwargs)

    if self.xpu.is_available():
      model.to(self.xpu.device)

      logging.info('wrap model with nn.parallel.DistributedDataParallel class')
      model = nn.parallel.DistributedDataParallel(
          model,
          device_ids=[self.xpu.device],
          output_device=self.xpu.device,
          find_unused_parameters=False
      )
      model._set_static_graph()  # pylint: disable=protected-access

    self.hook(model)

    return model

  def load(self, f, map_location='cpu'):
    if version_cmp(torch.__version__, '2.0.0') >= 0:  # disable=warning
      checkpoint = torch.load(f, map_location=map_location, weights_only=False)
    else:
      checkpoint = torch.load(f, map_location=map_location)
    kwargs = dict(
        dim=checkpoint['dim'],
        evoformer_depth=checkpoint['evoformer_depth'],
        evoformer_head_num=checkpoint['evoformer_head_num'],
        evoformer_head_dim=checkpoint['evoformer_head_dim'],
        accept_msa_attn=checkpoint.get('evoformer_accept_msa_attn', True),
        accept_frame_attn=checkpoint.get('evoformer_accept_frame_attn', False),
        accept_frame_update=checkpoint.get('evoformer_accept_frame_update', False),
        headers=checkpoint['headers']
    )

    # optional args.
    for key in ('num_tokens', 'recycling_single_repr', 'recycling_pos'):
      if key in checkpoint:
        kwargs[key] = checkpoint[key]

    model = AlphaFold2(**kwargs)

    self.hook(model)

    model.load_state_dict(checkpoint['model'])
    if self.xpu.is_available():
      model = model.to(device=self.xpu.device)
    model.eval()

    return checkpoint['feats'], model


class WorkerFunction(object):
  """Wrap the distibuted function
  """
  def __init__(self, work_fn, log_queue=None):
    self.work_fn = work_fn
    self.log_queue = log_queue

  def __call__(self, rank, args):  # pylint: disable=redefined-outer-name
    xpu = WorkerXPU(rank, args)

    # logger for Tensors
    record_factory = _WorkerLogRecordFactory(logging.getLogRecordFactory())
    logging.setLogRecordFactory(record_factory)

    # logging
    if self.log_queue:
      root = logging.getLogger()
      ctx_handler = QueueHandler(self.log_queue)
      if xpu.is_available():
        ctx_filter = _WorkerLogFilter(xpu.rank)
        ctx_handler.addFilter(ctx_filter)
      root.addHandler(ctx_handler)

      level = logging.DEBUG if args.verbose else logging.INFO
      root.setLevel(level)

    #--------------
    # setup distributed env if needed
    #--------------
    xpu.init_process_group()

    # Starting in PyTorch 1.7, there is a new flag called allow_tf32.
    # This flag defaults to True in PyTorch 1.7 to PyTorch 1.11, and
    # False in PyTorch 1.12 and later. This flag controls whether
    # PyTorch is allowed to use the TensorFloat32 (TF32) tensor cores,
    # available on new NVIDIA GPUs since Ampere, internally to compute
    # matmul (matrix multiplies and batched matrix multiplies) and
    # convolutions.
    with torch_allow_tf32(allow=True):
      self.work_fn(xpu, args)

    #--------------
    # cleanup
    #--------------
    xpu.destroy_process_group()


def main(args, fn):  # pylint: disable=redefined-outer-name
  if exists(args.nnodes):
    mp.set_start_method('spawn', force=True)

  #--------------
  # setup logging
  #--------------
  work_log = _WorkerLogging(fn, args)
  work_log.start()

  logging.info('-----------------')
  logging.info('Arguments: %s', args)
  logging.info('-----------------')

  #--------------
  # run fn with args
  #--------------
  if hasattr(fn, 'preprocess'):
    fn.preprocess(args)

  work_fn = WorkerFunction(fn, work_log.queue)
  if exists(args.nnodes):
    #mp.set_start_method('spawn', force=True)
    mp.spawn(
        work_fn,
        args=(args, ),
        nprocs=WorkerXPU.device_count() if WorkerXPU.device_count() > 0 else 1,
        join=True
    )
  else:
    work_fn(args.local_rank, args)

  if hasattr(fn, 'postprocess'):
    fn.postprocess(args)

  logging.info('-----------------')
  logging.info('Resources(myself): %s', resource.getrusage(resource.RUSAGE_SELF))
  logging.info('Resources(children): %s', resource.getrusage(resource.RUSAGE_CHILDREN))
  logging.info('-----------------')

  #--------------
  # cleanup logging
  #--------------
  work_log.stop()
