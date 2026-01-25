"""lr_scheduler wrapper
"""
from enum import Enum
import functools
import math
from typing import Optional, Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from profold2.utils import default, exists


class SchedulerType(Enum):
  CONSTANT = 'constant'
  COSINE = 'cosine'
  LINEAR = 'linear'


def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    eta_min: float = 0.0,
    last_global_step: int = 0,
) -> LambdaLR:
  name = SchedulerType(name)

  if name == SchedulerType.CONSTANT:

    def lr_lambda(
        current_step: int, num_warmup_steps: Optional[int] = None
    ) -> float:
      current_step = current_step + last_global_step
      if exists(num_warmup_steps) and current_step < num_warmup_steps:
        return current_step / max(1.0, num_warmup_steps)
      return 1.0
  elif name == SchedulerType.COSINE:

    def lr_lambda(
        current_step: int, num_warmup_steps: Optional[int] = None
    ) -> float:
      current_step = current_step + last_global_step
      if exists(num_warmup_steps) and current_step < num_warmup_steps:
        return current_step / max(1.0, num_warmup_steps)
      elif current_step > num_training_steps:
        return eta_min
      num_warmup_steps = default(num_warmup_steps, 0)
      progress = (
          (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
      )
      return 0.5  * (1.0 - eta_min) * (1.0 + math.cos(math.pi * progress)) + eta_min
  elif name == SchedulerType.LINEAR:

    def lr_lambda(
        current_step: int, num_warmup_steps: Optional[int] = None
    ) -> float:
      current_step = current_step + last_global_step
      if exists(num_warmup_steps) and current_step < num_warmup_steps:
        return current_step / max(1.0, num_warmup_steps)
      elif current_step > num_training_steps:
        return eta_min

      num_warmup_steps = default(num_warmup_steps, 0)
      progress = (
          (num_training_steps - current_step) / (num_training_steps - num_warmup_steps)
      )
      return (1.0 - eta_min) * progress + eta_min

  return LambdaLR(
      optimizer, functools.partial(lr_lambda, num_warmup_steps=num_warmup_steps)
  )
