"""Utils for profold2
"""
import os
import contextlib
import functools
from inspect import isfunction
import tempfile
import time
import urllib
import uuid

from tqdm.auto import tqdm


# helpers
def exists(val):
  return val is not None


def default(val, d):
  if exists(val):
    return val
  return d() if isfunction(d) else d


def env(*keys, defval=None, dtype=None):
  for key in keys:
    value = os.getenv(key)
    if exists(value):
      return dtype(value) if exists(dtype) else value
  return defval


def version_cmp(x, y):
  for a, b in zip(x.split('.'), y.split('.')):
    if int(a) > int(b):
      return 1
    elif int(a) < int(b):
      return -1
  return 0


def compose(*funcs):
  return functools.reduce(lambda g, f: lambda x: f(g(x)), funcs)


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


def wget(url, filename=None, chunk_size=(1 << 20)):
  with urllib.request.urlopen(url) as f:
    kwargs = {
        'total': int(f.headers.get('Content-Length', 0)),
        'desc': os.path.basename(filename) if exists(filename) else None,
    }
    with tqdm.wrapattr(f, 'read', **kwargs) as r:
      if exists(filename):
        writer = functools.partial(open, filename, mode='wb')
      else:
        writer = functools.partial(tempfile.NamedTemporaryFile, delete=False)
      with writer() as w:
        while True:
          chunk = r.read(chunk_size)
          if not chunk:
            break
          w.write(chunk)
        return w.name
