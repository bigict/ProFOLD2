#!/usr/bin/env python3
"""Gradient-flow diagnostic — rule out vanishing gradients as the ~2A floor cause.

Fine-tune fold1 alone (same setup as probe_fit) and log PRE-CLIP gradient norms
per module group after each backward:
  embedder / evoformer(trunk) / to_single_repr / folding head
  head sub-groups: ipa_block (tied x8), to_affine_update, to_angles, norms

If gradients are healthy (non-zero, sensible magnitude) in head AND trunk while
RMSD floors at ~2A, the floor is loss geometry (FAPE local-frame), NOT a
vanishing-gradient bug.

Run:
    PYTHONPATH=. python tools/foldswitch_probe/probe_grad.py [--steps 150] [--lr 1e-5]
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
from probe_fit import backbone_to_coord, kabsch_rmsd  # noqa: E402
from profold2.common import residue_constants as rc  # noqa: E402
from profold2.model.features import (  # noqa: E402
    make_backbone_affine, make_affine, make_torsion_angles, make_coord_alt)


def grad_stats(params):
  sq, n_zero, n_none, n_tot, gmax = 0.0, 0, 0, 0, 0.0
  for p in params:
    n_tot += 1
    if p.grad is None:
      n_none += 1
      continue
    g = p.grad.detach()
    m = g.abs().max().item()
    gmax = max(gmax, m)
    if m == 0.0:
      n_zero += 1
    sq += g.norm().item() ** 2
  return sq ** 0.5, n_zero, n_none, n_tot, gmax


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--steps', type=int, default=150)
  ap.add_argument('--lr', type=float, default=1e-5)
  ap.add_argument('--device', default='cuda:0')
  ap.add_argument('--target', default='fold1')
  ap.add_argument('--pair', default=os.path.join(DATA, 'xcl1_pair.npz'))
  args = ap.parse_args()
  dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')

  model = build_model().to(dev)
  model.impl.headers = [h for h in model.impl.headers if h[0] == 'folding']
  head = model.impl.headers[0][1]
  sm = head.struct_module

  groups = {
      'embedder': list(model.impl.embedder.parameters()),
      'evoformer(trunk)': list(model.impl.evoformer.parameters()),
      'to_single_repr': list(model.impl.to_single_repr.parameters()),
      'head(total)': list(head.parameters()),
      'head.ipa_block': list(sm.ipa_block.parameters()),
      'head.to_affine_update': list(sm.to_affine_update.parameters()),
      'head.to_angles': list(sm.to_angles.parameters()),
      'head.norms': list(sm.single_repr_norm.parameters()) +
                    list(sm.pairwise_repr_norm.parameters()),
  }

  pair = np.load(args.pair)
  c, m = backbone_to_coord(pair[args.target], rc.atom14_type_num)
  base = seq_to_batch(str(pair['seq']))
  base = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in base.items()}
  b = dict(base)
  b['coord'] = torch.from_numpy(c).to(dev)[None]
  b['coord_mask'] = torch.from_numpy(m).to(dev)[None]
  b['compute_loss'] = True
  for fn in (make_backbone_affine, make_affine, make_torsion_angles, make_coord_alt):
    b = fn(is_training=True)(b)

  params = [p for p in model.parameters() if p.requires_grad]
  opt = torch.optim.Adam(params, lr=args.lr)
  ca_idx = rc.atom_order['CA']
  gt_ca = pair[args.target][:, 1]

  print('gradient-flow diagnostic: target=%s lr=%g steps=%d' %
        (args.target, args.lr, args.steps), flush=True)
  model.train()
  for step in range(args.steps):
    opt.zero_grad()
    out = model.impl(b)
    loss = out['loss']
    loss.backward()
    if step % 25 == 0 or step == args.steps - 1:
      total, nz, nn_, nt, gmax = grad_stats(params)
      print('\nstep %4d  loss %.4f  TOTAL grad-norm %.4e  max|g| %.3e  '
            'zero-grad %d/%d  none-grad %d' %
            (step, loss.item(), total, gmax, nz, nt, nn_), flush=True)
      for name, ps in groups.items():
        gn, z, n, t, gm = grad_stats(ps)
        print('    %-22s norm %.4e  max|g| %.3e  zero %d/%d  none %d' %
              (name, gn, gm, z, t, n), flush=True)
      with torch.no_grad():
        pred_ca = out['headers']['folding']['coords'][0].cpu().numpy()[:, ca_idx]
        rmsd = kabsch_rmsd(pred_ca, gt_ca)
      print('    train-mode CA RMSD %.3f A' % rmsd, flush=True)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
  print('\ndone.', flush=True)


if __name__ == '__main__':
  main()