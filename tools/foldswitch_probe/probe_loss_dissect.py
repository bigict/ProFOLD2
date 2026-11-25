#!/usr/bin/env python3
"""Dissect folding loss components to locate the ~2A memorization floor.

Enables DEBUG logging on profold2.model.head (which logs backbone_fape_loss,
sidechain_fape_loss, norm_angle_loss, pred_angle_loss per traj step) and runs a
forward pass on fold0/fold1, reporting each component + total loss + CA RMSD.

Run:
    PYTHONPATH=. python tools/foldswitch_probe/probe_loss_dissect.py [--target fold1]
"""
import os
import sys
import argparse
import logging

os.environ.setdefault('profold2_restype_list', '[1,2,3]')
os.environ.setdefault('profold2_atom14_type_num', '23')
os.environ.setdefault('profold2_chi_angles_num', '11')
os.environ.setdefault('profold2_relative_pos_s_max', '2')
os.environ.setdefault('AxialAttention_accept_edge_norm', '0')

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
sys.path.insert(0, HERE)
from probe_repr import build_model, seq_to_batch  # noqa: E402
from profold2.common import residue_constants as rc  # noqa: E402
from profold2.model.features import (  # noqa: E402
    make_backbone_affine, make_affine, make_torsion_angles, make_coord_alt)


def backbone_to_coord(bb, n_atoms):
  n = len(bb)
  coord = np.zeros((n, n_atoms, 3), dtype=np.float32)
  mask = np.zeros((n, n_atoms), dtype=np.float32)
  for j in range(4):
    coord[:, j] = bb[:, j]; mask[:, j] = 1.0
  return coord, mask


def kabsch_rmsd(P, Q):
  mP, mQ = P.mean(0), Q.mean(0)
  Pc, Qc = P - mP, Q - mQ
  H = Pc.T @ Qc; U, S, Vt = np.linalg.svd(H)
  d = np.sign(np.linalg.det(Vt.T @ U.T))
  R = Vt.T @ np.diag([1, 1, d]) @ U.T
  return np.sqrt(((Pc @ R.T - Qc) ** 2).sum() / len(P))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--device', default='cuda:0')
  ap.add_argument('--pair', default=os.path.join(DATA, 'xcl1_pair.npz'))
  ap.add_argument('--target', default='fold1')
  args = ap.parse_args()
  dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')

  # capture head debug logs (per-component loss values)
  records = []
  class Cap(logging.Handler):
    def emit(self, record):
      msg = record.getMessage()
      if 'FoldingHead.loss' in msg:
        records.append(msg)
  hlogger = logging.getLogger('profold2.model.head')
  hlogger.setLevel(logging.DEBUG)
  hlogger.addHandler(Cap())

  model = build_model().to(dev)
  model.impl.headers = [h for h in model.impl.headers if h[0] == 'folding']
  pair = np.load(args.pair)
  seq_str = str(pair['seq'])
  c, m = backbone_to_coord(pair[args.target], rc.atom14_type_num)
  base = seq_to_batch(seq_str)
  base = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in base.items()}
  b = dict(base)
  b['coord'] = torch.from_numpy(c).to(dev)[None]
  b['coord_mask'] = torch.from_numpy(m).to(dev)[None]
  b['compute_loss'] = True
  for fn in (make_backbone_affine, make_affine, make_torsion_angles, make_coord_alt):
    b = fn(is_training=True)(b)

  print('=== GT feature masks (%s) ===' % args.target)
  print('  coord_mask sum/res (backbone atoms):', b['coord_mask'][0].sum(-1).unique(return_counts=True))
  print('  atom_affine_exists sum:', int(b['atom_affine_exists'].sum()) if 'atom_affine_exists' in b else 'N/A')
  print('  torsion_angles_mask sum:', int(b['torsion_angles_mask'].sum()) if 'torsion_angles_mask' in b else 'N/A')
  print('  backbone_affine_mask sum:', int(b['backbone_affine_mask'].sum()) if 'backbone_affine_mask' in b else 'N/A')

  model.eval()
  records.clear()
  with torch.no_grad():
    out = model.impl(b)
  loss = out['loss'].item()
  ca_idx = rc.atom_order['CA']
  pred_ca = out['headers']['folding']['coords'][0].cpu().numpy()[:, ca_idx]
  rmsd = kabsch_rmsd(pred_ca, pair[args.target][:, 1])
  print('\n=== %s: total loss %.4f  CA RMSD %.3f A ===' % (args.target, loss, rmsd))
  print('--- per-component debug (last traj step) ---')
  # show last occurrence of each component type
  seen = {}
  for msg in records:
    key = msg.split('(')[1].split(':')[0] if '(' in msg else msg
    seen[key] = msg
  for k, msg in seen.items():
    print('  ' + msg)


if __name__ == '__main__':
  main()