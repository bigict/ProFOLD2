"""Utils for profold2
"""
import os
import contextlib
from inspect import isfunction
import time
import uuid


# helpers
def exists(val):
  return val is not None


def default(val, d):
  if exists(val):
    return val
  return d() if isfunction(d) else d


def env(*keys, defval=None, type=None):
  for key in keys:
    value = os.getenv(key)
    if exists(value):
      return type(value) if exists(type) else value
  return defval


def version_cmp(x, y):
  for a, b in zip(x.split('.'), y.split('.')):
    if int(a) > int(b):
      return 1
    elif int(a) < int(b):
      return -1
  return 0


def package_dir():
  cwd = os.path.dirname(__file__)
  return os.path.dirname(cwd)


def unique_id():
  """Generate a unique ID as specified in RFC 4122."""
  # See https://docs.python.org/3/library/uuid.html
  return str(uuid.uuid4())


@contextlib.contextmanager
def timing(msg, print_fn, prefix='', callback_fn=None):
  print_fn(f'{prefix}Started {msg}')
  tic = time.time()
  yield
  toc = time.time()
  if exists(callback_fn):
    callback_fn(tic, toc)
  print_fn(f'{prefix}Finished {msg} in {(toc-tic):>.3f} seconds')
