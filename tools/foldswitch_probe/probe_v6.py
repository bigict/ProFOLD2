#!/usr/bin/env python3
"""Switching-field probe v6 — trunk+head field-gated adaLN-Zero + collapse monitor.

Design (user 2026-11-22): feed condition INTO the trunk (not head-only), with
mode0 EXACT identity construction:
    x_out = norm(x) * (1 + s_i*gamma_i) + s_i*beta_i
    gamma_i/beta_i = per-norm head( MLP(c_i) ),  zero-init heads
    c_i = s_i * [target-fold backbone torsion sin/cos]_i  (6-dim)
    mode0: s_i=0 -> x_out = norm(x) BITWISE pretrained (fold0 preserved by construction)
    mode1: s_i=GT -> modulation appears, grows from zero-init.

Injection: ALL 64 Evoformer blocks' single-path norms (dim_msa=256:
msa_attn.row_attn.norm, msa_attn.col_attn.norm, msa_ff.norm) + 3 folding-head
single norms (dim_single=384). Post-norm modulation survives LayerNorm (NOT
additive input embedding, which v1 showed is washed <6%).

Condition network: shared MLP(6->128->128, SiLU) + per-norm independent zero-init
gamma/beta heads (DiT adaLN-Zero style). ~13M new params.

Dynamics default: FREEZE trunk base + embedder, train only condition net + head
(lr1e-4). Removes v5 trunk-collapse confound; mode0 path = frozen pretrained.
--finetune-trunk to also train base (realistic arm).

Monitor (eval, no_grad => hooks fire once): per-layer mode divergence
    D_l = ||h_l(mode1) - h_l(mode0)||_F / ||h_l(mode0)||_F
at trunk depths [0,16,32,48,63], to_single_repr, head single_repr_norm, and each
of 8 IPA iterations; plus condition strength. Tells "never learned" (D~0 everywhere)
vs "learned then washed by IPA" (D decays across iterations) vs "healthy growth".

Run:
    PYTHONPATH=. python tools/foldswitch_probe/probe_v6.py [--steps 1200] [--lr 1e-4] [--finetune-trunk]
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
CKPT_DIR = os.path.join(HERE, 'ckpt_v6')
sys.path.insert(0, HERE)
from probe_repr import build_model, seq_to_batch  # noqa: E402
from probe_v4 import switching_field, backbone_to_coord, kabsch_rmsd  # noqa: E402
from profold2.common import residue_constants as rc  # noqa: E402
from profold2.model.features import (  # noqa: E402
    make_backbone_affine, make_affine, make_torsion_angles, make_coord_alt)

BB_TORSIONS = 3
COND_DIM = BB_TORSIONS * 2
HIDDEN = 128
TRUNK_MONITOR_DEPTHS = [0, 16, 32, 48, 63]


class CondNet(nn.Module):
  def __init__(self, norm_dims):
    super().__init__()
    self.mlp = nn.Sequential(
        nn.Linear(COND_DIM, HIDDEN), nn.SiLU(),
        nn.Linear(HIDDEN, HIDDEN), nn.SiLU())
    self.gamma_heads = nn.ModuleList(
        [nn.Linear(HIDDEN, d, bias=False) for d in norm_dims])
    self.beta_heads = nn.ModuleList(
        [nn.Linear(HIDDEN, d, bias=False) for d in norm_dims])
    for head in list(self.gamma_heads) + list(self.beta_heads):
      nn.init.zeros_(head.weight)

  def modulate(self, idx, c, s, out):
    h = self.mlp(c)                       # (n, HIDDEN)
    g = self.gamma_heads[idx](h)          # (n, dim)
    b = self.beta_heads[idx](h)
    lead = [1] * (out.dim() - 2)
    n, dim = g.shape
    sb = s.view(*lead, n, 1)
    gb = g.view(*lead, n, dim)
    bb = b.view(*lead, n, dim)
    return out * (1 + sb * gb) + sb * bb


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--steps', type=int, default=1200)
  ap.add_argument('--lr', type=float, default=1e-4)
  ap.add_argument('--eval-every', type=int, default=200)
  ap.add_argument('--ckpt-every', type=int, default=400)
  ap.add_argument('--device', default='cuda:0')
  ap.add_argument('--pair', default=os.path.join(DATA, 'xcl1_pair.npz'))
  ap.add_argument('--finetune-trunk', action='store_true',
                  help='also train trunk base (default: freeze trunk+embedder, train cond+head)')
  ap.add_argument('--trunk-lr', type=float, default=1e-5,
                  help='lr for trunk base+embedder when --finetune-trunk (cond+head use --lr)')
  ap.add_argument('--no-monitor', action='store_true')
  ap.add_argument('--tag', default='v6', help='checkpoint filename prefix (distinct per protein)')
  args = ap.parse_args()
  dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
  os.makedirs(CKPT_DIR, exist_ok=True)

  pair = np.load(args.pair)
  f0, f1 = pair['fold0'], pair['fold1']
  n = len(f0)
  field, core, disp = switching_field(f0, f1)
  sw_idx = np.where(field > 0.5)[0]
  print('switching field: core=%d, field>0.5: %d res at %s' %
        (core.sum(), len(sw_idx), sw_idx.tolist()), flush=True)

  model = build_model().to(dev)
  model.impl.headers = [h for h in model.impl.headers if h[0] == 'folding']
  fold = model.impl.headers[0][1]
  sm = fold.struct_module
  dim_single, dim_msa, dim_pair = model.impl.dim
  print('dims: single=%d msa=%d pair=%d' % (dim_single, dim_msa, dim_pair), flush=True)

  # ---- collect injection norms: trunk single-path (dim_msa) + head single (dim_single)
  trunk_norms = []   # (name, module)
  for bi, block in enumerate(model.impl.evoformer.layers):
    for sub_name in ('msa_attn', 'msa_ff'):
      sub = getattr(block, sub_name, None)
      if sub is None:
        continue
      for nm, mod in sub.named_modules():
        if isinstance(mod, nn.LayerNorm) and tuple(mod.normalized_shape) == (dim_msa,):
          trunk_norms.append(('blk%d.%s.%s' % (bi, sub_name, nm), mod))
  head_norms = [(nm, mod) for nm, mod in fold.named_modules()
                if isinstance(mod, nn.LayerNorm) and tuple(mod.normalized_shape) == (dim_single,)]
  all_norms = trunk_norms + head_norms
  norm_dims = [tuple(m.normalized_shape)[0] for _, m in all_norms]
  print('injection norms: %d trunk (dim %d) + %d head (dim %d) = %d total' %
        (len(trunk_norms), dim_msa, len(head_norms), dim_single, len(all_norms)), flush=True)

  cond_net = CondNet(norm_dims).to(dev)
  model.impl.cond_net = cond_net  # register as submodule -> in model.parameters()
  n_cond = sum(p.numel() for p in cond_net.parameters())
  print('condition net params: %.2f M' % (n_cond / 1e6), flush=True)

  # ---- conditioning feature (oracle): field-gated fold1 backbone torsions
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

  ta = gt_batches[1]['torsion_angles'][0, :, :BB_TORSIONS, :]
  tam = gt_batches[1]['torsion_angles_mask'][0, :, :BB_TORSIONS]
  sincos = torch.nan_to_num(ta * tam[..., None]).reshape(n, COND_DIM)
  field_t = torch.from_numpy(field.astype(np.float32)).to(dev)
  cond1 = field_t[:, None] * sincos
  conds = [torch.zeros_like(cond1), cond1]
  fields = [torch.zeros(n, device=dev), field_t]
  cur = {'c': conds[0], 's': fields[0]}

  # ---- register field-gated adaLN-Zero hooks (output gate s_i => mode0 exact identity)
  for idx, (name, mod) in enumerate(all_norms):
    def make_hook(i):
      def hook(module, inp, out):
        return cond_net.modulate(i, cur['c'], cur['s'], out)
      return hook
    mod.register_forward_hook(make_hook(idx))

  # ---- collapse monitor: capture repr at trunk depths + head points (eval only)
  capture = {}
  ipa_counter = {'k': 0}
  if not args.no_monitor:
    for bi in TRUNK_MONITOR_DEPTHS:
      block = model.impl.evoformer.layers[bi]
      def make_blk_hook(depth):
        def hook(module, inp, out):
          # out = (x_pair, m_msa, t_frames); single path = m[..., 0, :, :]
          capture['trunk%d' % depth] = out[1][..., 0, :, :].detach()
        return hook
      block.register_forward_hook(make_blk_hook(bi))
    def single_hook(module, inp, out):
      capture['to_single'] = out.detach()
    model.impl.to_single_repr.register_forward_hook(single_hook)
    def head_norm_hook(module, inp, out):
      capture['head_norm'] = out.detach()
    sm.single_repr_norm.register_forward_hook(head_norm_hook)
    def ipa_hook(module, inp, out):
      capture['ipa%d' % ipa_counter['k']] = out.detach()
      ipa_counter['k'] += 1
    sm.ipa_block.register_forward_hook(ipa_hook)

  def run_monitor():
    reps = []
    for mode in (0, 1):
      cur['c'], cur['s'] = conds[mode], fields[mode]
      capture.clear()
      ipa_counter['k'] = 0
      b = dict(base); b['compute_loss'] = False
      with torch.no_grad():
        model.impl(b)
      reps.append(dict(capture))
    keys = ['trunk%d' % d for d in TRUNK_MONITOR_DEPTHS] + ['to_single', 'head_norm'] + \
           ['ipa%d' % k for k in range(8)]
    ds = []
    for k in keys:
      if k in reps[0] and k in reps[1]:
        h0, h1 = reps[0][k], reps[1][k]
        d = (h1 - h0).norm().item() / (h0.norm().item() + 1e-8)
        ds.append('%s=%.3f' % (k, d))
    print('  D_l(mode1||mode0): ' + ' '.join(ds), flush=True)

  # ---- freeze / param groups
  if not args.finetune_trunk:
    for p in model.impl.evoformer.parameters():
      p.requires_grad = False
    for p in model.impl.embedder.parameters():
      p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    print('trainable params: %.2f M (finetune_trunk=False, cond+head lr=%g)' %
          (sum(p.numel() for p in params) / 1e6, args.lr), flush=True)
    opt = torch.optim.Adam(params, lr=args.lr)
  else:
    trunk_params = list(model.impl.evoformer.parameters()) + \
                   list(model.impl.embedder.parameters())
    trunk_ids = set(id(p) for p in trunk_params)
    other_params = [p for p in model.parameters()
                    if p.requires_grad and id(p) not in trunk_ids]
    params = trunk_params + other_params
    print('trainable params: %.2f M (finetune_trunk=True: trunk+emb %.2fM lr=%g, cond+head %.2fM lr=%g)' %
          (sum(p.numel() for p in params) / 1e6,
           sum(p.numel() for p in trunk_params) / 1e6, args.trunk_lr,
           sum(p.numel() for p in other_params) / 1e6, args.lr), flush=True)
    opt = torch.optim.Adam([
        {'params': trunk_params, 'lr': args.trunk_lr},
        {'params': other_params, 'lr': args.lr}])

  ca_idx = rc.atom_order['CA']
  gt_ca = [f0[:, 1], f1[:, 1]]

  def evaluate(tag):
    model.eval()
    preds = []
    with torch.no_grad():
      for mode in (0, 1):
        cur['c'], cur['s'] = conds[mode], fields[mode]
        b = dict(base); b['compute_loss'] = False
        out = model.impl(b)
        preds.append(out['headers']['folding']['coords'][0].cpu().numpy()[:, ca_idx])
    print('EVAL[%s] m0->f0 %.2f | m0->f1 %.2f | m1->f0 %.2f | m1->f1 %.2f (A)' %
          (tag, kabsch_rmsd(preds[0], gt_ca[0]), kabsch_rmsd(preds[0], gt_ca[1]),
           kabsch_rmsd(preds[1], gt_ca[0]), kabsch_rmsd(preds[1], gt_ca[1])), flush=True)
    if not args.no_monitor:
      run_monitor()
    # condition strength: mean gamma-head weight norm
    gnorm = np.mean([h.weight.norm().item() for h in cond_net.gamma_heads])
    print('  cond strength: mean ||W_gamma||=%.4e' % gnorm, flush=True)
    model.train()

  def save_ckpt(tag):
    prefix = args.tag + ('ft' if args.finetune_trunk else '')
    path = os.path.join(CKPT_DIR, '%s_%s.pt' % (prefix, tag))
    state = {'cond_net': cond_net.state_dict(), 'head': fold.state_dict(), 'step': tag}
    if args.finetune_trunk:
      state['evoformer'] = model.impl.evoformer.state_dict()
      state['embedder'] = model.impl.embedder.state_dict()
    torch.save(state, path)
    print('  saved %s' % path, flush=True)

  evaluate('baseline')
  for step in range(args.steps):
    mode = step % 2
    cur['c'], cur['s'] = conds[mode], fields[mode]
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
    if (step + 1) % args.ckpt_every == 0:
      save_ckpt('step%d' % (step + 1))
  save_ckpt('final')
  print('done.', flush=True)


if __name__ == '__main__':
  main()