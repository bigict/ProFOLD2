#!/usr/bin/env python3
"""Minimal trunk-representability probe v1 — mode-conditioned finetuning.

Direct test of route-C's core question: CAN the pretrained seq-only trunk be
conditioned (via a learned mode embedding injected into the trunk input) to
produce the alternative fold (fold1) as well as the dominant fold (fold0)?

Finetunes the pretrained model on a fold-switcher pair:
    (seq, mode_rank=0) -> fold0 backbone
    (seq, mode_rank=1) -> fold1 backbone
using the model's own FAPE loss, then evaluates mode-conditioned predictions.

Readout:
  mode=1 recovers fold1 (and mode=0 fold0) -> trunk CAN represent alt fold (capacity)
  mode=1 cannot recover fold1 even after finetuning -> representability limited

This is a CAPACITY probe (memorization acceptable); generalization is separate.

Run:
    PYTHONPATH=. python tools/foldswitch_probe/probe_v1.py [--steps 300] [--device cuda:0]
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
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
sys.path.insert(0, HERE)
from probe_repr import build_model, seq_to_batch  # noqa: E402
from profold2.common import residue_constants as rc  # noqa: E402
from profold2.model import functional  # noqa: E402
from profold2.model.features import (  # noqa: E402
    make_backbone_affine, make_affine, make_torsion_angles, make_coord_alt)


def backbone_to_coord(bb, n_atoms):
  """bb (n,4,3)=N,CA,C,O -> atom14 coord (n,n_atoms,3) + mask."""
  n = len(bb)
  coord = np.zeros((n, n_atoms, 3), dtype=np.float32)
  mask = np.zeros((n, n_atoms), dtype=np.float32)
  for j in range(4):  # N,CA,C,O at atom14 indices 0..3
    coord[:, j] = bb[:, j]
    mask[:, j] = 1.0
  return coord, mask


def kabsch_rmsd(P, Q):
  Pc = P - P.mean(0); Qc = Q - Q.mean(0)
  H = Pc.T @ Qc; U, S, Vt = np.linalg.svd(H)
  d = np.sign(np.linalg.det(Vt.T @ U.T))
  R = Vt.T @ np.diag([1, 1, d]) @ U.T
  return np.sqrt(((Pc @ R.T - Qc) ** 2).sum() / len(P))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--steps', type=int, default=300)
  ap.add_argument('--lr', type=float, default=1e-4)
  ap.add_argument('--device', default='cuda:0')
  ap.add_argument('--pair', default=os.path.join(DATA, 'xcl1_pair.npz'))
  ap.add_argument('--freeze-trunk', action='store_true',
                  help='freeze trunk+embedder; train only mode_emb + folding head')
  args = ap.parse_args()
  dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')

  model = build_model()
  model.to(dev)
  # keep only the folding head (coords + FAPE loss); drop heads needing MSA features
  model.impl.headers = [h for h in model.impl.headers if h[0] == 'folding']
  dim_msa = 256
  mode_emb = nn.Embedding(2, dim_msa).to(dev)
  nn.init.zeros_(mode_emb.weight)  # start neutral (preserve pretrained behavior)

  cur_mode = {'m': 0}
  def embed_hook(module, inp, out):
    x, m, x_mask, msa_mask = out
    m[..., 0, :, :] = m[..., 0, :, :] + mode_emb(
        torch.tensor([cur_mode['m']], device=dev))
    return (x, m, x_mask, msa_mask)
  model.impl.embedder.register_forward_hook(embed_hook)

  pair = np.load(args.pair)
  seq_str = str(pair['seq'])
  n_atoms = rc.atom14_type_num
  folds = []
  for k in ('fold0', 'fold1'):
    c, m = backbone_to_coord(pair[k], n_atoms)
    folds.append((torch.from_numpy(c).to(dev), torch.from_numpy(m).to(dev)))
  base = seq_to_batch(seq_str)
  base = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in base.items()}

  def make_batch(mode):
    b = dict(base)
    b['coord'] = folds[mode][0][None]
    b['coord_mask'] = folds[mode][1][None]
    b['compute_loss'] = True
    for fn in (make_backbone_affine, make_affine, make_torsion_angles, make_coord_alt):
      b = fn(is_training=True)(b)
    return b

  # precompute GT batches once (avoid per-step feature recomputation)
  gt_batches = [make_batch(0), make_batch(1)]

  if args.freeze_trunk:
    for p in model.impl.evoformer.parameters():
      p.requires_grad = False
    for p in model.impl.embedder.parameters():
      p.requires_grad = False
  params = [p for p in model.parameters() if p.requires_grad] + list(mode_emb.parameters())
  opt = torch.optim.Adam(params, lr=args.lr)

  model.train()
  for step in range(args.steps):
    mode = step % 2
    cur_mode['m'] = mode
    opt.zero_grad()
    out = model.impl(gt_batches[mode])
    loss = out['loss']
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    if step % 25 == 0 or step == args.steps - 1:
      print('step %3d mode %d loss %.4f' % (step, mode, loss.item()), flush=True)

  # eval: predict mode 0 / 1, compare CA to fold0 / fold1
  model.eval()
  ca_idx = rc.atom_order['CA']
  with torch.no_grad():
    for mode in (0, 1):
      cur_mode['m'] = mode
      b = dict(base); b['compute_loss'] = False
      out = model.impl(b)
      pred = out['headers']['folding']['coords'][0].cpu().numpy()
      pred_ca = pred[:, ca_idx]
      gt_ca = pair['fold%d' % mode][:, 1]
      rmsd = kabsch_rmsd(pred_ca, gt_ca)
      print('EVAL mode %d -> fold%d: CA RMSD %.2f A' % (mode, mode, rmsd))
  # cross: does mode 1 look like fold0 (collapsed) or fold1?
  with torch.no_grad():
    cur_mode['m'] = 1
    b = dict(base); b['compute_loss'] = False
    out = model.impl(b)
    pred_ca = out['headers']['folding']['coords'][0].cpu().numpy()[:, ca_idx]
    print('EVAL mode1 vs fold0 CA RMSD %.2f A (collapse check)' %
          kabsch_rmsd(pred_ca, pair['fold0'][:, 1]))


if __name__ == '__main__':
  main()