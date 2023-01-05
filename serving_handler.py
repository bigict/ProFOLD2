"""
Module for ProFOLD default handler
"""

import base64
import io
import json
import logging

import torch
from ts.torch_handler.base_handler import BaseHandler

from profold2.data.dataset import ProteinSequenceDataset
from profold2.model import ReturnValues
from profold2.data.parsers import parse_fasta
from profold2.data.utils import pdb_from_prediction

logger = logging.getLogger(__file__)

class FastaHandler(BaseHandler):  # pylint: disable=missing-class-docstring
  def initialize(self, context):
    super().initialize(context)

    assert hasattr(self.model, 'features')
    self.model.features.to(self.device)  # HACK :(

  def preprocess(self, data):
    def _data_to_text(data):
      # If the data is sent as bytesarray
      if isinstance(data, (bytearray, bytes)):
        f = io.TextIOWrapper(io.BytesIO(data))
        return f.read()
      return data

    sequences, descriptions, msa, kwargs = [], [], [], []

    for row in data:
      logger.info('%s', row)
      row = json.loads(_data_to_text(row.get('data') or row.get('body')))
      fasta = row['sequence']
      params = {'num_recycle': 2, 'shard_size': None}
      if 'num_recycle' in row:
        params['num_recycle'] = int(row['num_recycle'])
      if 'shard_size' in row and row['shard_size']:
        params['shard_size'] = int(row['shard_size'])

      fmt = row['fmt'] if 'fmt' in row else 'single'
      assert fmt in ('single', 'a3m', 'a4m')
      if fmt == 'a4m':
        s = fasta.splitlines()
        d = [None] * len(s)
      else:
        s, d = parse_fasta(fasta)
      if fmt == 'single':
        sequences += s
        descriptions += d
        msa += [None] * len(s)
        kwargs += [params] * len(s)
      else:
        sequences += s[:1]
        descriptions += d[:1]
        msa += [s]
        kwargs += [params]

    dataset = ProteinSequenceDataset(sequences, descriptions, msa=msa)
    loader = torch.utils.data.DataLoader(dataset,
        collate_fn=dataset.collate_fn,
        batch_size=1,
        shuffle=False)
    return [(batch, params) for batch, params in zip(iter(loader), kwargs)]  # pylint: disable=unnecessary-comprehension

  def inference(self, data, *args, **kwargs):
    del args
    del kwargs

    results = []

    with torch.no_grad():
      for i, (batch, kwargs) in enumerate(iter(data)):
        logger.info('inference: i=%d, batch=%s, kwargs=%s', i, batch, kwargs)
        r =  self.model(batch, **kwargs)
        logger.info(r)
        results.append((i, batch, ReturnValues(**r)))

    return results

  def postprocess(self, data):
    def to_base64(headers):
      with io.BytesIO() as f:
        torch.save(headers, f)
        t = io.TextIOWrapper(io.BytesIO(
            base64.b64encode(f.getvalue())))
        return t.read()
    def to_dict(result):
      _, b, r = result
      return pdb_from_prediction(b, r.headers, idx=0), to_base64(r.headers)

    results = []
    for r in data:
      pdb, headers = to_dict(r)
      results.append({'pdb': pdb, 'headers': headers})
    return results
