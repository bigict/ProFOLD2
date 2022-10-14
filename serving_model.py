"""Model wrapper for ProFOLD
"""
import logging

import torch
from torch import nn

from profold2.model import Alphafold2, FeatureBuilder

logger = logging.getLogger(__file__)

class ProteinFolding(nn.Module):  # pylint: disable=missing-class-docstring
  def __init__(self):
    super().__init__()
    logger.debug('init.')

  def forward(self, batch, **kwargs):
    logger.debug('forward.')
    assert hasattr(self, 'features') and hasattr(self, 'impl')
    batch = self.features(batch, is_training=False)
    return self.impl.forward(batch, **kwargs)

  def load_state_dict(self, state_dict, strict=True):
    self.impl = Alphafold2(dim=state_dict['dim'],
        depth=state_dict['evoformer_depth'],
        heads=state_dict['evoformer_head_num'],
        dim_head=state_dict['evoformer_head_dim'],
        embedd_dim=state_dict['mlm_dim'],
        headers=state_dict['headers'])
    self.add_module('impl', self.impl)
    self.features = FeatureBuilder(state_dict['feats'])

    return self.impl.load_state_dict(state_dict['model'], strict=strict)

if __name__ == '__main__':
  import sys

  print(sys.argv)
  m = ProteinFolding()
  if sys.argv[1] == 'save':
    torch.save(m.impl.state_dict(), 'test_model.pth')
  else:
    x = torch.load(sys.argv[1], map_location='cpu')
    print(x.keys())
    m.load_state_dict(x)
