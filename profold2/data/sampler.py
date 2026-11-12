"""Distributed data samplers.
"""
import math
from typing import Optional

import torch

from profold2.utils import default


class DistributedSamplerWrapper(torch.utils.data.Sampler):
  """Partition a sampler's draws across ranks.

  Wraps a Sampler and re-seeds it from (seed, epoch) so every rank draws an
  identical global stream, then takes its own stride slice. An epoch therefore
  covers the same total number of draws as single-process sampling, with each
  rank handling 1/num_replicas of them.

  Whether an item can appear on more than one rank depends on the wrapped
  sampler: a permutation sampler yields disjoint item sets, while sampling with
  replacement may draw the same item on several ranks. Either way the slices
  partition one common stream, so no bias is introduced.
  """

  def __init__(
      self,
      sampler: torch.utils.data.Sampler,
      num_replicas: Optional[int] = None,
      rank: Optional[int] = None,
      seed: int = 42,
      drop_last: bool = False,
  ) -> None:
    super().__init__()
    # Guard against passing a Dataset: list(dataset) would materialize every
    # item instead of yielding indices.
    assert isinstance(sampler, torch.utils.data.Sampler), type(sampler)
    # SequentialSampler is deterministic, so every rank already draws the same
    # stream. Anything else must expose a controllable seed, or it would draw a
    # different stream per rank and silently break the partition.
    assert (
        isinstance(sampler, torch.utils.data.SequentialSampler) or
        hasattr(sampler, 'generator') or hasattr(sampler, 'seed')
    ), type(sampler)

    self.sampler = sampler
    self.num_replicas = default(num_replicas, 1)
    self.rank = default(rank, 0)
    assert 0 <= self.rank < self.num_replicas
    self.seed = seed
    self.epoch = 0
    self.drop_last = drop_last

    # If the sampler length is evenly divisible by the number of replicas there
    # is nothing to drop, since the stream splits equally.
    if self.drop_last and len(self.sampler) % self.num_replicas != 0:
      self.num_samples = math.ceil(
          (len(self.sampler) - self.num_replicas) / self.num_replicas
      )
    else:
      self.num_samples = math.ceil(len(self.sampler) / self.num_replicas)
    self.total_size = self.num_samples * self.num_replicas

  def __iter__(self):
    # A CPU generator keeps the draws identical across ranks regardless of
    # device, and avoids perturbing the global torch RNG that model init and
    # dropout depend on.
    seed = self.seed + self.epoch
    if hasattr(self.sampler, 'generator'):
      g = torch.Generator()
      g.manual_seed(seed)
      self.sampler.generator = g
    elif hasattr(self.sampler, 'seed'):
      self.sampler.seed = seed
    # else: deterministic sampler (e.g. SequentialSampler), nothing to seed.

    indices = list(self.sampler)
    if self.drop_last:
      indices = indices[:self.total_size]
    else:
      padding = self.total_size - len(indices)
      if padding > 0:
        if padding <= len(indices):
          indices += indices[:padding]
        else:
          indices += (indices * math.ceil(padding / len(indices)))[:padding]
    assert len(indices) == self.total_size, (len(indices), self.total_size)

    indices = indices[self.rank:self.total_size:self.num_replicas]
    assert len(indices) == self.num_samples, (len(indices), self.num_samples)
    return iter(indices)

  def __len__(self):
    return self.num_samples

  def set_epoch(self, epoch):
    self.epoch = epoch
