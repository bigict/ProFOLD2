"""Wrapper for pytorch profiler
  """
import contextlib
from contextlib import suppress as nullcontext

import torch
from torch.profiler import (ProfilerActivity, tensorboard_trace_handler)  # pylint: disable=unused-import


_enabled = False


@contextlib.contextmanager
def record_function(name):
  if _enabled:
    with torch.profiler.record_function(name):
      yield
  else:
    yield


@contextlib.contextmanager
def profile(enabled, *args, **kwargs):
  global _enabled
  _enabled = enabled

  if _enabled:
    prof = torch.profiler.profile(*args, **kwargs)
    prof.start()
    yield prof
    prof.stop()
  else:
    yield nullcontext()
