"""Wrapper for pytorch meory snapshots
  """
import contextlib
from datetime import datetime
import socket
import logging

import torch

from profold2.utils import default, exists, version_cmp

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def memory_snapshot(enabled, filename=None, max_entries=None, device=None):
  if enabled and not torch.cuda.is_available():
    logger.warning('Only works for debugging CUDA memory use. enabled=False')
    enabled = False
  if enabled and version_cmp(torch.__version__, '2.1') < 0:
    logger.warning('Only available with pytorch >= 2.1, enabled=False')
    enabled = False

  if enabled:
    max_entries = default(max_entries, 1000000)

    # Start recording memory snapshot history, initialized with a buffer
    # capacity of 1,000,000 memory events, via the `max_entries` field.
    logger.info('Starting snapshot (device=%s) record_memory_history', device)
    torch.cuda.memory._record_memory_history(max_entries=max_entries, device=device)

    yield

    try:
      if not exists(filename):
        host_name = socket.gethostname()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{host_name}_{timestamp}'
        if exists(device):
          if isinstance(device, torch.device):
            filename = f'{filename}_{device.type}{device.index}'
          else:
            filename = f'{filename}_{device}'
        filename = f'{filename}.pkl'
      logger.debug('Saving snapshot (device=%s) to local file: %s', device, filename)

      torch.cuda.memory._dump_snapshot(filename)
    except Exception as e:
      logger.error('Failed to capture memory snapshot %s', e)

    # Stop recording memory snapshot history.
    logger.debug('Stopping snapshot (device=%s) record_memory_history', device)
    torch.cuda.memory._record_memory_history(enabled=None, device=device)
  else:
    yield
