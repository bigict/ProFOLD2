#!/usr/bin/env python3
"""Minimal trunk-representability probe (Phase 0.5) — hello-world step.

Loads the pretrained seq-only ProFOLD2 trunk from a weights-only checkpoint
(config reconstructed from the training script), runs it on a fold-switcher
sequence, and captures the frozen single representation via a forward hook on
`impl.to_single_repr`. Verifies the loading + repr-extraction infrastructure
before building the minimal-decoder representability test.

Run:
    PYTHONPATH=. python tools/foldswitch_probe/probe_repr.py
"""
import os
import sys
import json

# biomolecule alphabet + relative-pos config must match the checkpoint
os.environ.setdefault('profold2_restype_list', '[1,2,3]')
os.environ.setdefault('profold2_atom14_type_num', '23')
os.environ.setdefault('profold2_chi_angles_num', '11')
os.environ.setdefault('profold2_relative_pos_s_max', '2')
# match training-time attention config (run_train_2022111102_37_o.sh):
# accept_edge_norm=0 -> Identity@0 + Linear(bias)@1 in edges_to_attn_bias
os.environ.setdefault('AxialAttention_accept_edge_norm', '0')

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CKPT = os.path.join(REPO, 'examples', '599_dna_rna.ckpt')
HEADERS = os.path.join(REPO, 'examples', 'model_headers_2022111102_37_o.json')
PAIR = '/tmp/fsprobe/xcl1_pair.npz'

from profold2.common import residue_constants as rc
from profold2.model import AlphaFold2


def build_model():
  with open(HEADERS) as f:
    headers = json.load(f)
  config = dict(
      dim=(384, 256, 128),
      evoformer_depth=64,
      evoformer_head_num=8,
      evoformer_head_dim=32,
      evoformer_accept_msa_attn=True,
      recycling_pos=True,
      headers=headers,
  )
  model = AlphaFold2.from_config(config)
  ck = torch.load(CKPT, map_location='cpu', weights_only=False)['model']
  # checkpoint saved from DDP-wrapped model: strip leading "module."
  sd = {k[len('module.'):]: v for k, v in ck.items() if k.startswith('module.')}
  missing, unexpected = model.load_state_dict(sd, strict=False)
  print('load_state_dict: missing=%d unexpected=%d' % (len(missing), len(unexpected)))
  if missing:
    print('  missing sample:', missing[:5])
  if unexpected:
    print('  unexpected sample:', unexpected[:5])
  model.eval()
  return model


def seq_to_batch(seq_str):
  aatype = [rc.restype_order.get((aa, rc.PROT), rc.unk_restype_index) for aa in seq_str]
  seq = torch.tensor(aatype, dtype=torch.long)[None]
  n = seq.shape[1]
  mask = torch.ones(1, n, dtype=torch.bool)
  seq_index = torch.arange(n)[None]
  ones = torch.ones(1, n, dtype=torch.long)
  return dict(seq=seq, mask=mask, seq_index=seq_index, msa_enabled=False,
              seq_color=ones, seq_entity=ones, seq_sym=ones, token_index=seq_index)


def main():
  model = build_model()
  data = np.load(PAIR)
  seq_str = str(data['seq'])
  print('XCL1 seq (%d res): %s' % (len(seq_str), seq_str))
  batch = seq_to_batch(seq_str)

  captured = {}
  def hook(module, inp, out):
    captured['single'] = out.detach()
  h = model.impl.to_single_repr.register_forward_hook(hook)
  model.impl.headers = []  # trunk only; skip prediction heads (need MSA features)
  with torch.no_grad():
    model.impl(batch)
  h.remove()
  s = captured['single']
  print('captured single repr shape:', tuple(s.shape))
  print('repr stats: mean=%.4f std=%.4f norm=%.4f' %
        (s.mean().item(), s.std().item(), s.norm(dim=-1).mean().item()))
  np.save('/tmp/fsprobe/xcl1_repr.npy', s[0].numpy())
  print('saved /tmp/fsprobe/xcl1_repr.npy')


if __name__ == '__main__':
  main()