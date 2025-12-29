"""Wrap GPU, MLU etc
"""
from datetime import timedelta
import functools
import logging
from typing import Optional

import torch

from profold2.utils import default, env


def device_count():
  return torch.cuda.device_count()

def device_type():
  if torch.cuda.is_available():
    return 'cuda'
  return 'cpu'

def world_size(nnodes: Optional[int] = None):
  return env('WORLD_SIZE', defval=device_count() * default(nnodes, 1), dtype=int)

autocast = functools.partial(torch.amp.autocast, device_type())

class XPU(object):
  """Wrap distibuted XPU(GPU,MLU etc)
  """
  def __init__(self, rank: int, nnodes: Optional[int] = None, node_rank: int = 0):
    self.local_rank = rank
    self.nnodes = nnodes
    self.node_rank = node_rank

  def is_available(self):
    if self.local_rank != -1:
      return torch.cuda.is_available()
    return False

  def is_master(self):
    if self.is_available():
      return self.node_rank == 0 and self.local_rank == 0
    return True

  @property
  def device(self):
    if self.is_available():
      return self.local_rank
    return torch.device('cpu')

  @property
  def rank(self):
    return env(
        'RANK',
        defval=device_count() * self.node_rank + self.local_rank,
        dtype=int
    )

  def memory_summary(self):
    if self.is_available() and hasattr(torch.cuda, 'memory_summary'):
      return torch.cuda.memory_summary()
    return 'only cuda supported.'

  def init_process_group(self, init_method: str):
    if self.is_available():
      timeout = env('NCCL_TIMEOUT', defval=1800, dtype=int)
      logging.info(
          'distributed.init_process_group: rank=%s@%s, world_size=%s@%s, '
          'init_method=%s, timeout=%s(s)',
          self.device, device_count(), self.rank,
          world_size(self.nnodes), init_method, timeout
      )
      torch.distributed.init_process_group(
          backend='nccl',
          init_method=init_method,
          timeout=timedelta(seconds=timeout),
          rank=self.rank,
          world_size=world_size(self.nnodes)
      )
      torch.cuda.set_device(self.local_rank)

  def destroy_process_group(self):
    if self.is_available():
      torch.distributed.destroy_process_group()
