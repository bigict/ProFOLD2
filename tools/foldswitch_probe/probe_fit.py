#!/usr/bin/env python3
"""Capacity diagnostic — fit fold0 and fold1 SEPARATELY, no conditioning.

Decisive test of the skepticism "148M params should trivially fit 1 pair":
fine-tune the pretrained model on EACH target alone (no mode, no switching field),
from fresh pretrained weights, and report the RMSD each reaches.

  - If fold1 fits to ~0  -> model CAN represent fold1; probe failures were
    conditioning/setup issues, NOT representability limits.
  - If fold1 cannot fit  -> fundamental problem (architecture/loss/data).

Run:
    PYTHONPATH=. python tools/foldswitch_probe/probe_fit.py [--steps 800] [--lr 1e-4]
"""
import os
import sys
import argparse

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


def fit_target(target_key, pair, args, dev):
  print('\n===== fit %s ALONE (no conditioning) =====' % target_key, flush=True)
  model = build_model().to(dev)  # fresh pretrained
  model.impl.headers = [h for h in model.impl.headers if h[0] == 'folding']
  if args.clamp_ratio is not None:
    head = model.impl.headers[0][1]
    print('override fape_backbone_clamp_ratio: %s -> %s' %
          (head.fape_backbone_clamp_ratio, args.clamp_ratio))
    head.fape_backbone_clamp_ratio = args.clamp_ratio
  if args.clamp_distance is not None:
    head = model.impl.headers[0][1]
    print('override fape_max (clamp_distance): %s -> %s' % (head.fape_max, args.clamp_distance))
    head.fape_max = args.clamp_distance
  seq_str = str(pair['seq'])
  n_atoms = rc.atom14_type_num
  c, m = backbone_to_coord(pair[target_key], n_atoms)
  base = seq_to_batch(seq_str)
  base = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in base.items()}
  b = dict(base)
  b['coord'] = torch.from_numpy(c).to(dev)[None]
  b['coord_mask'] = torch.from_numpy(m).to(dev)[None]
  b['compute_loss'] = True
  for fn in (make_backbone_affine, make_affine, make_torsion_angles, make_coord_alt):
    b = fn(is_training=True)(b)
  if args.mask_sidechain:
    # backbone-only GT: sidechain atoms absent -> mask sidechain FAPE by coord presence,
    # not residue type (atom_affine_exists is residue-type based -> spurious loss).
    b['atom_affine_exists'] = torch.zeros_like(b['atom_affine_exists'])

  if args.freeze_trunk:
    for p in model.impl.evoformer.parameters():
      p.requires_grad = False
    for p in model.impl.embedder.parameters():
      p.requires_grad = False
  params = [p for p in model.parameters() if p.requires_grad]
  opt = torch.optim.Adam(params, lr=args.lr)
  ca_idx = rc.atom_order['CA']
  gt_ca = pair[target_key][:, 1]

  model.train()
  for step in range(args.steps):
    opt.zero_grad()
    out = model.impl(b)
    loss = out['loss']
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    if step % 100 == 0 or step == args.steps - 1:
      model.eval()
      with torch.no_grad():
        bb = dict(base); bb['compute_loss'] = False
        o = model.impl(bb)
        pred_ca = o['headers']['folding']['coords'][0].cpu().numpy()[:, ca_idx]
        rmsd = kabsch_rmsd(pred_ca, gt_ca)
      model.train()
      print('  %s step %4d loss %.4f  CA RMSD %.3f A' %
            (target_key, step, loss.item(), rmsd), flush=True)
  model.eval()
  with torch.no_grad():
    bb = dict(base); bb['compute_loss'] = False
    o = model.impl(bb)
    pred_ca = o['headers']['folding']['coords'][0].cpu().numpy()[:, ca_idx]
    rmsd = kabsch_rmsd(pred_ca, gt_ca)
  print('  FINAL %s CA RMSD %.3f A' % (target_key, rmsd), flush=True)
  return rmsd


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--steps', type=int, default=800)
  ap.add_argument('--lr', type=float, default=1e-4)
  ap.add_argument('--device', default='cuda:0')
  ap.add_argument('--pair', default=os.path.join(DATA, 'xcl1_pair.npz'))
  ap.add_argument('--freeze-trunk', action='store_true')
  ap.add_argument('--mask-sidechain', action='store_true',
                  help='zero atom_affine_exists (backbone-only GT: remove spurious sidechain FAPE)')
  ap.add_argument('--clamp-ratio', type=float, default=None,
                  help='override fape_backbone_clamp_ratio (default 1.0 full-clamp zeroes grad for dij>15)')
  ap.add_argument('--clamp-distance', type=float, default=None,
                  help='override fape_max clamp_distance (ProFOLD2 15, AF2 10)')
  ap.add_argument('--recycles', type=int, default=0,
                  help='fixed num_recycle (0 removes train-mode random recycle noise)')
  ap.add_argument('--targets', default='fold0,fold1')
  args = ap.parse_args()
  dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
  pair = np.load(args.pair)
  print('capacity diagnostic: freeze_trunk=%s lr=%g steps=%d' %
        (args.freeze_trunk, args.lr, args.steps))
  results = {}
  for t in args.targets.split(','):
    results[t] = fit_target(t.strip(), pair, args, dev)
  print('\n===== SUMMARY =====')
  for t, r in results.items():
    print('  %s alone -> %.3f A' % (t, r))


if __name__ == '__main__':
  main()