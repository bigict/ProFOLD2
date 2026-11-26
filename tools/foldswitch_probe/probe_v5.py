#!/usr/bin/env python3
"""Switching-field probe v5 — oracle field + TARGET-DIRECTION conditioning.

v4 negative lesson: the displacement-magnitude field encodes WHICH/HOW-MUCH but
NOT the target-conformation DIRECTION. v5 adds direction:

  c_i = s_i * [sin/cos of target-fold backbone torsions (omega,phi,psi)]_i  (6-dim)

  mode0 -> c=0         -> fold0 (pretrained prior, no modulation)
  mode1 -> c=GT(fold1) -> fold1 (oracle local direction at switching residues)

Injection: per-residue cond-LN on folding-head single-path norms,
  gamma_i = W_gamma c_i, beta_i = W_beta c_i  (zero-init projections)
  x_i = x_i * (1 + gamma_i) + beta_i          (post-norm, survives LayerNorm)

Tests the user principle "局部的变化导致整体结构发生变化": can LOCAL target-direction
info at the few switching residues drive the GLOBAL fold change through IPA?

Success: mode1->fold1 ~2-2.5A (fit-alone level) with clear separation from
mode1->fold0, while mode0->fold0 preserved. ~2A FAPE floor is expected/acceptable.

Run:
    PYTHONPATH=. python tools/foldswitch_probe/probe_v5.py [--steps 1200] [--lr 1e-5] [--freeze-trunk]
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
from probe_v4 import switching_field, backbone_to_coord, kabsch_rmsd  # noqa: E402
from profold2.common import residue_constants as rc  # noqa: E402
from profold2.model.features import (  # noqa: E402
    make_backbone_affine, make_affine, make_torsion_angles, make_coord_alt)

BB_TORSIONS = 3  # slots 0,1,2 = omega,phi,psi (protein)
COND_DIM = BB_TORSIONS * 2  # sin/cos


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--steps', type=int, default=1200)
  ap.add_argument('--lr', type=float, default=1e-5)
  ap.add_argument('--eval-every', type=int, default=200)
  ap.add_argument('--device', default='cuda:0')
  ap.add_argument('--pair', default=os.path.join(DATA, 'xcl1_pair.npz'))
  ap.add_argument('--freeze-trunk', action='store_true')
  args = ap.parse_args()
  dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')

  pair = np.load(args.pair)
  f0, f1 = pair['fold0'], pair['fold1']
  n = len(f0)
  field, core, disp = switching_field(f0, f1)
  sw_idx = np.where(field > 0.5)[0]
  print('switching field: core=%d res, field>0.5: %d res at %s' %
        (core.sum(), len(sw_idx), sw_idx.tolist()), flush=True)

  model = build_model().to(dev)
  model.impl.headers = [h for h in model.impl.headers if h[0] == 'folding']
  fold = model.impl.headers[0][1]

  seq_str = str(pair['seq'])
  folds = []
  for k in ('fold0', 'fold1'):
    c, m = backbone_to_coord(pair[k], rc.atom14_type_num)
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

  # oracle conditioning feature from fold1 GT backbone torsions, gated by field
  ta = gt_batches[1]['torsion_angles'][0, :, :BB_TORSIONS, :]  # (n,3,2)
  tam = gt_batches[1]['torsion_angles_mask'][0, :, :BB_TORSIONS]  # (n,3)
  sincos = torch.nan_to_num(ta * tam[..., None]).reshape(n, COND_DIM)
  field_t = torch.from_numpy(field.astype(np.float32)).to(dev)
  cond1 = field_t[:, None] * sincos
  cond0 = torch.zeros_like(cond1)
  conds = [cond0, cond1]
  print('cond1 nonzero residues: %d (field-gated target torsions)' %
        int((cond1.abs().sum(-1) > 0).sum()), flush=True)

  # per-residue cond-LN: gamma_i/beta_i = zero-init linear proj of c_i
  single_norms = [(name, mod) for name, mod in fold.named_modules()
                  if isinstance(mod, nn.LayerNorm) and tuple(mod.normalized_shape) == (384,)]
  gamma_proj = nn.Linear(COND_DIM, 384, bias=False).to(dev)
  beta_proj = nn.Linear(COND_DIM, 384, bias=False).to(dev)
  nn.init.zeros_(gamma_proj.weight)
  nn.init.zeros_(beta_proj.weight)
  fold.cond_gamma_proj = gamma_proj
  fold.cond_beta_proj = beta_proj
  cur_cond = {'c': cond0}
  print('per-residue cond-LN on %d single norms: %s' %
        (len(single_norms), [nm for nm, _ in single_norms]), flush=True)
  for name, mod in single_norms:
    dim = mod.normalized_shape[0]
    def make_hook(m):
      def hook(module, inp, out):
        c = cur_cond['c']  # (n, COND_DIM)
        g = fold.cond_gamma_proj(c)  # (n, dim)
        b = fold.cond_beta_proj(c)
        shape = [1] * (out.dim() - 2) + [n, dim]
        return out * (1 + g.view(*shape)) + b.view(*shape)
      return hook
    mod.register_forward_hook(make_hook(mod))

  if args.freeze_trunk:
    for p in model.impl.evoformer.parameters():
      p.requires_grad = False
    for p in model.impl.embedder.parameters():
      p.requires_grad = False
  params = [p for p in model.parameters() if p.requires_grad]
  print('trainable params: %.2f M (freeze_trunk=%s)' %
        (sum(p.numel() for p in params) / 1e6, args.freeze_trunk), flush=True)
  opt = torch.optim.Adam(params, lr=args.lr)

  ca_idx = rc.atom_order['CA']
  gt_ca = [f0[:, 1], f1[:, 1]]

  def evaluate(tag):
    model.eval()
    preds = []
    with torch.no_grad():
      for mode in (0, 1):
        cur_cond['c'] = conds[mode]
        b = dict(base); b['compute_loss'] = False
        out = model.impl(b)
        preds.append(out['headers']['folding']['coords'][0].cpu().numpy()[:, ca_idx])
    print('EVAL[%s] mode0->fold0 %.2f | mode0->fold1 %.2f | '
          'mode1->fold0 %.2f | mode1->fold1 %.2f (A)' %
          (tag, kabsch_rmsd(preds[0], gt_ca[0]), kabsch_rmsd(preds[0], gt_ca[1]),
           kabsch_rmsd(preds[1], gt_ca[0]), kabsch_rmsd(preds[1], gt_ca[1])), flush=True)
    model.train()

  evaluate('baseline')
  for step in range(args.steps):
    mode = step % 2
    cur_cond['c'] = conds[mode]
    opt.zero_grad()
    out = model.impl(gt_batches[mode])
    loss = out['loss']
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    if step % 100 == 0:
      print('step %4d mode %d loss %.4f' % (step, mode, loss.item()), flush=True)
    if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
      evaluate('step %d' % (step + 1))
  print('done.', flush=True)


if __name__ == '__main__':
  main()