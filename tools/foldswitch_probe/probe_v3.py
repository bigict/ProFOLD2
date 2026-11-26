#!/usr/bin/env python3
"""Minimal trunk-representability probe v3 — conditional LayerNorm in the folding head.

v1/v2 failed because the mode signal was injected BEFORE a LayerNorm (trunk input
or to_single_repr output) and washed out. ProFOLD2 has many LayerNorms; the folding
head's single path passes through single_repr_norm + ipa_block.attn_norm + ff_norm
(the IPA block is weight-tied, iterated 8x). v3 applies mode-conditioned gamma/beta
to ALL folding-head LayerNorms (post-norm modulation baked into each norm), so the
mode signal survives the whole decoder. Full trunk finetune (low lr) lets the trunk
learn a repr supporting both folds; decoder mode-conditioning selects which to decode.

Run:
    PYTHONPATH=. python tools/foldswitch_probe/probe_v3.py [--steps 400] [--lr 1e-5]
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
  Pc = P - P.mean(0); Qc = Q - Q.mean(0)
  H = Pc.T @ Qc; U, S, Vt = np.linalg.svd(H)
  d = np.sign(np.linalg.det(Vt.T @ U.T))
  R = Vt.T @ np.diag([1, 1, d]) @ U.T
  return np.sqrt(((Pc @ R.T - Qc) ** 2).sum() / len(P))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--steps', type=int, default=400)
  ap.add_argument('--lr', type=float, default=1e-5)
  ap.add_argument('--device', default='cuda:0')
  ap.add_argument('--pair', default=os.path.join(DATA, 'xcl1_pair.npz'))
  ap.add_argument('--freeze-trunk', action='store_true')
  args = ap.parse_args()
  dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')

  model = build_model().to(dev)
  model.impl.headers = [h for h in model.impl.headers if h[0] == 'folding']
  fold = model.impl.headers[0][1]

  # conditional LayerNorm on ALL folding-head norms (mode-dependent gamma/beta)
  norms = [(name, mod) for name, mod in fold.named_modules()
           if isinstance(mod, nn.LayerNorm)]
  total = sum(2 * mod.normalized_shape[0] for _, mod in norms)
  mode_film = nn.Embedding(2, total).to(dev)
  nn.init.zeros_(mode_film.weight)
  cur_mode = {'m': 0}
  print('conditional LayerNorm on %d folding-head norms (%s), full-strength post-norm' %
        (len(norms), [tuple(m.normalized_shape) for _, m in norms]))
  off = 0
  for name, mod in norms:
    d = mod.normalized_shape[0]
    o = off
    def make_hook(offset, dim):
      def hook(module, inp, out):
        gb = mode_film(torch.tensor([cur_mode['m']], device=dev))[0]
        gamma = gb[offset:offset + dim]
        beta = gb[offset + dim:offset + 2 * dim]
        return out * (1 + gamma) + beta
      return hook
    mod.register_forward_hook(make_hook(o, d))
    off += 2 * d

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
  gt_batches = [make_batch(0), make_batch(1)]

  if args.freeze_trunk:
    for p in model.impl.evoformer.parameters():
      p.requires_grad = False
    for p in model.impl.embedder.parameters():
      p.requires_grad = False
  params = [p for p in model.parameters() if p.requires_grad] + list(mode_film.parameters())
  print('trainable params: %.2f M (freeze_trunk=%s)' %
        (sum(p.numel() for p in params) / 1e6, args.freeze_trunk))
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

  model.eval()
  ca_idx = rc.atom_order['CA']
  with torch.no_grad():
    for mode in (0, 1):
      cur_mode['m'] = mode
      b = dict(base); b['compute_loss'] = False
      out = model.impl(b)
      pred_ca = out['headers']['folding']['coords'][0].cpu().numpy()[:, ca_idx]
      print('EVAL mode %d -> fold%d: CA RMSD %.2f A' %
            (mode, mode, kabsch_rmsd(pred_ca, pair['fold%d' % mode][:, 1])), flush=True)
    cur_mode['m'] = 1
    b = dict(base); b['compute_loss'] = False
    out = model.impl(b)
    pred_ca = out['headers']['folding']['coords'][0].cpu().numpy()[:, ca_idx]
    print('EVAL mode1 vs fold0 CA RMSD %.2f A (collapse check)' %
          kabsch_rmsd(pred_ca, pair['fold0'][:, 1]), flush=True)


if __name__ == '__main__':
  main()