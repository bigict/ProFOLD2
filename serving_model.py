"""Model wrapper for ProFOLD
"""
import logging

import torch
from torch import nn

from profold2.model import Alphafold2
from profold2.utils import default

logger = logging.getLogger(__file__)

class ProteinFolding(nn.Module):  # pylint: disable=missing-class-docstring
  def __init__(self, args):  # pylint: disable=redefined-outer-name
    super().__init__()
    logger.info('init.')
    args = default(args, {})

    self.impl = Alphafold2(dim=args.get('dim', 256),
        depth=args.get('evoformer_depth', 1),
        heads=args.get('evoformer_head_num', 8),
        dim_head=args.get('evoformer_head_dim', 64),
        embedd_dim=args.get('mlm_dim', 1280),
        headers=args.get('headers', {}))

  def forward(self, batch, **kwargs):
    logger.info('forward.')
    return self.impl.forward(batch, **kwargs)

  def load_state_dict(self, state_dict, strict=True):
    return self.impl.load_state_dict(state_dict, strict=strict)

if __name__ == '__main__':
  import sys
  import json

  print(sys.argv)
  with open(sys.argv[2], 'rb') as f:
    args = json.load(f)
  m = ProteinFolding(args)
  if sys.argv[1] == 'save':
    torch.save(m.impl.state_dict(), 'test_model.pth')
  else:
    x = torch.load('test_model.pth', map_location='cpu')
    print(x.keys())
    print(m.impl.state_dict())
    m.load_state_dict(x)
