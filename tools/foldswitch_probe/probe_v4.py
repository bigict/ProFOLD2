#!/usr/bin/env python3
"""Switching-field probe v4 — Stage A (oracle field) + per-residue conditional LayerNorm.

Core hypothesis (design doc §4.1/§4.4): the model must represent WHICH residues drive
the fold change. Global uniform conditioning failed (v1-v3). v4 tests targeted
per-residue conditioning:

  GT switching field s_i = per-residue CA displacement fold0->fold1 after CORE
  superposition (align on stable core, not global Kabsch), normalized to [0,1].

  Stage A (oracle): condition the folding head with the GT field directly:
    mode0 -> field=0   -> predict fold0 (no modulation, pretrained behavior)
    mode1 -> field=GT  -> predict fold1 (targeted modulation of switching residues)
  Per-residue conditional LayerNorm on folding-head single-path norms:
    x_i = x_i * (1 + s_i * gamma) + s_i * beta   (gamma/beta learned per-feature, zero-init)
  so modulation is per-residue heterogeneous (via s_i), post-norm (not washed out).

  Isolates: given the CORRECT switching residues, does targeted conditioning drive
  the fold change? (Stage B later adds field prediction from seq+mode.)

Run:
    PYTHONPATH=. python tools/foldswitch_probe/probe_v4.py [--steps 600] [--lr 1e-5]
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


def kabsch(P, Q):
  mP, mQ = P.mean(0), Q.mean(0)
  Pc, Qc = P - mP, Q - mQ
  H = Pc.T @ Qc; U, S, Vt = np.linalg.svd(H)
  d = np.sign(np.linalg.det(Vt.T @ U.T))
  R = Vt.T @ np.diag([1, 1, d]) @ U.T
  return R, mP, mQ


def switching_field(f0, f1, core_thr=1.5):
  """Per-residue CA displacement after CORE superposition, normalized [0,1]."""
  ca0, ca1 = f0[:, 1], f1[:, 1]
  R, m0, m1 = kabsch(ca0, ca1)
  disp_global = np.linalg.norm((ca0 - m0) @ R.T - (ca1 - m1), axis=1)
  core = disp_global < core_thr  # stable core residues
  if core.sum() < 5:
    core = disp_global < np.percentile(disp_global, 40)
  Rc, mc0, mc1 = kabsch(ca0[core], ca1[core])
  disp = np.linalg.norm((ca0 - mc0) @ Rc.T - (ca1 - mc1), axis=1)
  field = disp / (disp.max() + 1e-8)
  return field, core, disp


def backbone_to_coord(bb, n_atoms):
  n = len(bb)
  coord = np.zeros((n, n_atoms, 3), dtype=np.float32)
  mask = np.zeros((n, n_atoms), dtype=np.float32)
  for j in range(4):
    coord[:, j] = bb[:, j]; mask[:, j] = 1.0
  return coord, mask


def kabsch_rmsd(P, Q):
  R, mP, mQ = kabsch(P, Q)
  return np.sqrt((((P - mP) @ R.T - (Q - mQ)) ** 2).sum() / len(P))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--steps', type=int, default=600)
  ap.add_argument('--lr', type=float, default=1e-5)
  ap.add_argument('--device', default='cuda:0')
  ap.add_argument('--pair', default=os.path.join(DATA, 'xcl1_pair.npz'))
  ap.add_argument('--freeze-trunk', action='store_true')
  args = ap.parse_args()
  dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')

  pair = np.load(args.pair)
  f0, f1 = pair['fold0'], pair['fold1']
  field, core, disp = switching_field(f0, f1)
  print('switching field: core=%d res, disp mean=%.2f max=%.2f, field>0.5: %d res' %
        (core.sum(), disp.mean(), disp.max(), (field > 0.5).sum()))
  field_t = torch.from_numpy(field.astype(np.float32)).to(dev)

  model = build_model().to(dev)
  model.impl.headers = [h for h in model.impl.headers if h[0] == 'folding']
  fold = model.impl.headers[0][1]

  # per-residue conditional LayerNorm on folding-head SINGLE-path norms (n, dim)
  single_norms = [(name, mod) for name, mod in fold.named_modules()
                  if isinstance(mod, nn.LayerNorm) and tuple(mod.normalized_shape) == (384,)]
  cur_field = {'s': torch.zeros(len(f0), device=dev)}
  print('per-residue cond-LN on %d single norms: %s' %
        (len(single_norms), [n for n, _ in single_norms]))
  for name, mod in single_norms:
    dim = mod.normalized_shape[0]
    gamma = nn.Parameter(torch.zeros(dim, device=dev))
    beta = nn.Parameter(torch.zeros(dim, device=dev))
    mod.register_parameter('sf_gamma', gamma)
    mod.register_parameter('sf_beta', beta)
    def make_hook(m):
      def hook(module, inp, out):
        s = cur_field['s']  # (n,)
        # out shape (..., n, dim); broadcast s over leading dims
        sb = s.view(*([1] * (out.dim() - 2)), len(s), 1)
        return out * (1 + sb * m.sf_gamma) + sb * m.sf_beta
      return hook
    mod.register_forward_hook(make_hook(mod))

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
  params = [p for p in model.parameters() if p.requires_grad]
  print('trainable params: %.2f M (freeze_trunk=%s)' %
        (sum(p.numel() for p in params) / 1e6, args.freeze_trunk))
  opt = torch.optim.Adam(params, lr=args.lr)

  model.train()
  for step in range(args.steps):
    mode = step % 2
    cur_field['s'] = torch.zeros(len(f0), device=dev) if mode == 0 else field_t
    opt.zero_grad()
    out = model.impl(gt_batches[mode])
    loss = out['loss']
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    if step % 50 == 0 or step == args.steps - 1:
      print('step %3d mode %d loss %.4f' % (step, mode, loss.item()), flush=True)

  model.eval()
  ca_idx = rc.atom_order['CA']
  with torch.no_grad():
    for mode in (0, 1):
      cur_field['s'] = torch.zeros(len(f0), device=dev) if mode == 0 else field_t
      b = dict(base); b['compute_loss'] = False
      out = model.impl(b)
      pred_ca = out['headers']['folding']['coords'][0].cpu().numpy()[:, ca_idx]
      print('EVAL mode %d (field=%s) -> fold%d: CA RMSD %.2f A' %
            (mode, 'GT' if mode else '0', mode,
             kabsch_rmsd(pred_ca, pair['fold%d' % mode][:, 1])), flush=True)
    cur_field['s'] = field_t
    b = dict(base); b['compute_loss'] = False
    out = model.impl(b)
    pred_ca = out['headers']['folding']['coords'][0].cpu().numpy()[:, ca_idx]
    print('EVAL mode1(GT field) vs fold0 CA RMSD %.2f A (collapse check)' %
          kabsch_rmsd(pred_ca, f0[:, 1]), flush=True)


if __name__ == '__main__':
  main()