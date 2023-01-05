"""
Module for ProFOLD workflow handler
"""

import json
import logging

logger = logging.getLogger(__file__)

def preprocess(data, context):
  del context
  logger.info('preprocess: %s', data)
  if data:
    return [row.get('data') or row.get('body') for row in data]
  return data

def postprocess(data, context):
  """Aggregate outputs from all models
  """
  del context
  if data:
    logger.info('postprocess: %s', type(data))
    for i, row in enumerate(data):
      logger.info('postprocess: (%d) %s', i, row.keys())
      for model_runner in row:
        x = json.loads(row[model_runner])
        row[model_runner] = x
      data[i] = row
  return data
