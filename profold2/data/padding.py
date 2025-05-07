import torch

from profold2.utils import default, exists


def _pad_item(item, shape, dim=0, padval=0, dtype=None):
  if exists(item):
    s = list(item.shape)
    s[dim] = shape[dim] - s[dim]
    z = torch.full(s, padval, dtype=dtype)
    c = torch.cat((item, z), dim=dim)
  else:
    c = torch.full(shape, padval, dtype=dtype)
  assert c.shape[dim] == shape[dim]
  return c


def _pad_dtype_detect(items):
  for item in items:
    if exists(item):
      return item.dtype
  return None


def _pad_shape_detect(items):
  for item in items:
    if exists(item):
      return item.shape
  return None


def pad_sequential(items, batch_length, padval=0, dtype=None):
  dtype = default(dtype, _pad_dtype_detect(items))
  shape = default(_pad_shape_detect(items), (batch_length, ))
  batch = [
      _pad_item(item, (batch_length, *shape[1:]), padval=padval, dtype=dtype)
      for item in items
  ]
  return torch.stack(batch, dim=0)


def pad_rectangle(items, batch_length, padval=0, dtype=None):
  dtype = default(dtype, _pad_dtype_detect(items))

  batch = []

  depth = max(item.shape[0] for item in items if exists(items))
  for item in items:
    c = item

    # Append columns
    c = _pad_item(c, (depth, batch_length), dim=1, padval=padval, dtype=dtype)
    # Append rows
    c = _pad_item(c, (depth, batch_length), dim=0, padval=padval, dtype=dtype)

    batch.append(c)

  return torch.stack(batch, dim=0)
