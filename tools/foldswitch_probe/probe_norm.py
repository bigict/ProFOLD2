#!/usr/bin/env python3
"""Test whether the additive mode embedding survives the trunk's LayerNorm.

Injects a mode vector of varying magnitude at the trunk input (current probe
injection point) and measures the relative change in the final single repr:
    ||s(mode=v) - s(mode=0)|| / ||s(mode=0)||
If LayerNorm attenuates the additive bias, this ratio stays tiny even for large v.
"""
import os
os.environ.setdefault('profold2_restype_list', '[1,2,3]')
os.environ.setdefault('profold2_atom14_type_num', '23')
os.environ.setdefault('profold2_chi_angles_num', '11')
os.environ.setdefault('profold2_relative_pos_s_max', '2')
os.environ.setdefault('AxialAttention_accept_edge_norm', '0')
import sys
import numpy as np
import torch
import torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from probe_repr import build_model, seq_to_batch
DATA = os.path.join(HERE, 'data')

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = build_model().to(dev)
model.impl.headers = []
model.eval()

pair = np.load(os.path.join(DATA, 'xcl1_pair.npz'))
batch = seq_to_batch(str(pair['seq']))
batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}

mode_vec = {'v': torch.zeros(256, device=dev)}
def hook(module, inp, out):
  x, m, xm, mm = out
  m[..., 0, :, :] = m[..., 0, :, :] + mode_vec['v']
  return (x, m, xm, mm)
model.impl.embedder.register_forward_hook(hook)

cap = {}
def sh(m, i, o):
  cap['s'] = o.detach()
model.impl.to_single_repr.register_forward_hook(sh)

torch.manual_seed(0)
unit = torch.randn(256, device=dev); unit = unit / unit.norm()

def repr_for(v):
  mode_vec['v'] = v
  with torch.no_grad():
    model.impl(batch)
  return cap['s'][0]

s0 = repr_for(torch.zeros(256, device=dev))
base_norm = s0.norm().item()
print('base single repr norm: %.2f, per-residue std %.2f' %
      (base_norm, s0.std().item()))
print('scale | ||ds||/||s0|| | max per-residue cos change')
for scale in [0.1, 1.0, 10.0, 100.0, 1000.0]:
  s1 = repr_for(scale * unit)
  ds = (s1 - s0).norm().item()
  cos = torch.nn.functional.cosine_similarity(s1, s0, dim=-1).mean().item()
  print('%6.1f | %.4f | %.5f' % (scale, ds / base_norm, 1 - cos))