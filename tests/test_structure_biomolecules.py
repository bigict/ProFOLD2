#!/usr/bin/env python3
"""Correctness tests for the ProFOLD2 biomolecule structure pipeline.

The model predicts protein / DNA / RNA all-atom coordinates with an
AlphaFold2-style StructureModule: a per-residue backbone frame plus a set of
torsion angles (sin, cos) are expanded into rigid-group frames
(``rigids_from_angles``) and then into atom positions
(``rigids_to_positions``).  Training supervises this with the inverse maps
``rigids_from_positions`` / ``angles_from_positions``.  These tests verify that
the forward and inverse maps are mutually consistent and that the
StructureModule is SE(3)-equivariant, for every residue type of all three
molecule classes.

Run (biomolecule mode is the default):
    python tests/test_structure_biomolecules.py
Protein-only mode (7 torsions / 14 atoms, the original AlphaFold2 setting):
    profold2_restype_list='[1]' profold2_atom14_type_num=14 \
    profold2_chi_angles_num=7 python tests/test_structure_biomolecules.py

The environment variables below MUST be set before ``residue_constants`` is
imported; they select the residue alphabet, the per-residue atom slots and the
number of torsion angles.  Defaults here enable protein + DNA + RNA.
"""

import os
import sys
import math
import traceback

os.environ.setdefault('profold2_restype_list', '[1,2,3]')
os.environ.setdefault('profold2_atom14_type_num', '23')
os.environ.setdefault('profold2_chi_angles_num', '11')

import numpy as np
import torch

from profold2.common import residue_constants as rc
from profold2.model import functional as F
from profold2.model import folding
from profold2.model.features import make_backbone_affine

torch.manual_seed(0)
np.random.seed(0)

PROT, DNA, RNA = rc.PROT, rc.DNA, rc.RNA
K = rc.chi_angles_num                 # torsion slots (11 for biomolecules)
G = rc.restype_rigid_group_num        # rigid groups = K + 1
NA = rc.atom14_type_num               # atom slots per residue (23)

# Torsion slots whose rotation actually moves atoms (others are dummy / empty
# groups): protein psi + chi1..4 ; nucleic alpha..nu0.  Slot 0 (pre-omega),
# slot 1 (phi / epsilon) and, for nucleics, slot 2 (zeta) carry no atoms.
COORD_SLOTS = {
    PROT: list(range(2, 7)),          # psi, chi1..chi4  (slots 2..6)
    DNA: list(range(3, K)),           # alpha..nu0      (slots 3..10)
    RNA: list(range(3, K)),
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def is_na_idx(idx):
  return rc.moltype(int(idx)) in (DNA, RNA)


def resname_of(idx):
  idx = int(idx)
  if idx == rc.unk_restype_index:
    return rc.unk_restype
  letter = rc.restypes[idx]
  return rc.restype_1to3[(letter, rc.moltype(idx))]


def rand_frames(*shape):
  return F.rotations_from_randn(*shape), torch.randn(*shape, 3)


def rand_angles(b, n):
  a = F.l2_norm(torch.randn(b, n, K, 2))
  a[..., 0, :] = torch.tensor([0., 1.])   # pre-omega is a dummy identity
  return a


def synth_coords(seq, frames, angles):
  return F.rigids_to_positions(F.rigids_from_angles(seq, frames, angles), seq)


def global_transform(frames, points, g):
  rg, tg = g
  r, t = frames
  r2 = torch.einsum('hw,...wi->...hi', rg, r)
  t2 = torch.einsum('hw,...w->...h', rg, t) + tg
  p2 = torch.einsum('hw,...w->...h', rg, points) + tg
  return (r2, t2), p2


def atom_name(idx, slot):
  return rc.restype_name_to_atom14_names[resname_of(idx)][slot]


def renaming_perm(idx):
  """Permutation of atom slots for the ambiguous-naming swap (or identity)."""
  perm = np.arange(NA)
  name = resname_of(idx)
  swap = rc.residue_atom_renaming_swaps.get(name)
  if swap:
    names = rc.restype_name_to_atom14_names[name]
    for a, b in swap.items():
      i, j = names.index(a), names.index(b)
      perm[i], perm[j] = j, i
  return perm


def dihedral(p0, p1, p2, p3):
  """Standard signed dihedral (radians) of four points, independent of model."""
  b0 = p0 - p1
  b1 = p2 - p1
  b2 = p3 - p2
  b1 = b1 / np.linalg.norm(b1)
  v = b0 - np.dot(b0, b1) * b1
  w = b2 - np.dot(b2, b1) * b1
  x = np.dot(v, w)
  y = np.dot(np.cross(b1, v), w)
  return math.atan2(y, x)


# --------------------------------------------------------------------------- #
# 1. rigid-body / quaternion math
# --------------------------------------------------------------------------- #
def test_quaternion_to_matrix_is_rotation():
  q = torch.randn(50, 4)
  q = q / q.norm(dim=-1, keepdim=True)
  r = F.quaternion_to_matrix(q)
  eye = torch.eye(3).expand_as(r)
  assert torch.allclose(r @ r.transpose(-1, -2), eye, atol=1e-5)
  assert torch.allclose(torch.linalg.det(r), torch.ones(50), atol=1e-5)
  # non-unit quaternions must give the same rotation (two_s normalisation)
  r2 = F.quaternion_to_matrix(q * 3.7)
  assert torch.allclose(r, r2, atol=1e-5)
  return 'R R^T = I, det = +1 (unit & scaled quaternions)'


def test_quaternion_matrix_roundtrip():
  q = F.l2_norm(torch.randn(50, 4))
  r = F.quaternion_to_matrix(q)
  q2 = F.matrix_to_quaternion(r)
  r2 = F.quaternion_to_matrix(q2)
  assert torch.allclose(r, r2, atol=1e-5)
  return 'matrix -> quaternion -> matrix closes'


def test_quaternion_multiply_matches_matrix():
  a, b = F.l2_norm(torch.randn(50, 4)), F.l2_norm(torch.randn(50, 4))
  ra, rb = F.quaternion_to_matrix(a), F.quaternion_to_matrix(b)
  rab = F.quaternion_to_matrix(F.quaternion_multiply(a, b))
  assert torch.allclose(rab, ra @ rb, atol=1e-5)
  return 'quat_multiply == matrix composition'


def test_rigids_compose():
  a, b = rand_frames(4, 6), rand_frames(4, 6)
  x = torch.randn(4, 6, 3)
  ab = F.rigids_multiply(a, b)
  lhs = F.rigids_apply(ab, x)
  rhs = F.rigids_apply(a, F.rigids_apply(b, x))
  assert torch.allclose(lhs, rhs, atol=1e-4)
  m = F.rigids_to_4x4(a)
  r2, t2 = F.rigids_from_4x4(m)
  assert torch.allclose(r2, a[0], atol=1e-6) and torch.allclose(t2, a[1], atol=1e-6)
  return '(A*B)(x) == A(B(x)); 4x4 roundtrip'


def test_rotations_from_vecs_orthonormal():
  r = F.rotations_from_vecs(torch.randn(100, 3), torch.randn(100, 3))
  eye = torch.eye(3).expand_as(r)
  assert torch.allclose(r @ r.transpose(-1, -2), eye, atol=1e-5)
  assert torch.allclose(torch.linalg.det(r), torch.ones(100), atol=1e-5)
  return 'Gram-Schmidt frames are proper rotations'


# --------------------------------------------------------------------------- #
# 2. residue-constant self-consistency
# --------------------------------------------------------------------------- #
def test_default_frames_valid():
  bad = []
  for idx in range(rc.restype_num):
    for g in range(G):
      if not (rc.restype_rigid_group_mask[idx, g] or g == 0):
        continue
      r = rc.restype_rigid_group_default_frame[idx, g, :3, :3]
      if not np.allclose(r @ r.T, np.eye(3), atol=1e-4) or \
         abs(np.linalg.det(r) - 1) > 1e-4:
        bad.append((resname_of(idx), g))
  assert not bad, f'non-rigid default frames: {bad}'
  return f'all existing rigid-group default frames are proper rotations ({rc.restype_num} restypes)'


def test_depend_is_dag():
  bad = []
  for idx in range(rc.restype_num):
    for g in range(1, G):
      if not rc.restype_rigid_group_mask[idx, g]:
        continue                      # empty groups carry no atoms
      dep = int(rc.restype_rigid_group_depend[idx, g])
      if dep >= g:
        bad.append((resname_of(idx), g, dep))
      if dep != rc.torsion_angles_depend[idx][g - 1]:
        bad.append((resname_of(idx), g, 'mismatch table'))
  assert not bad, f'depend must reference an earlier group: {bad}'
  return 'rigid-group dependency is a DAG consistent with torsion_angles_depend'


def test_atom14_coverage():
  bad = []
  for idx in range(rc.restype_num):
    name = resname_of(idx)
    listed = {a for a, g, _ in rc.rigid_group_atom_positions[name] if g <= K}
    masked = {atom_name(idx, s) for s in range(NA) if rc.restype_atom14_mask[idx, s]}
    if listed != masked:
      bad.append((name, sorted(listed ^ masked)))
    # NB: group-origin atoms (protein CA, nucleic P) legitimately have local
    # position (0, 0, 0), so a zero position is not itself an error.
  assert not bad, f'atom14 mask / rigid-group coverage mismatch: {bad}'
  return 'every masked atom has a rigid group and a local position (and vice versa)'


def test_torsion_defining_atom_in_rotated_group():
  """The 4th atom of torsion s must belong to rigid group s+1.

  Otherwise the extracted angle measures a dihedral that the corresponding
  forward rotation does not control, so the angle cannot be recovered from
  coordinates (forward/inverse inconsistency).
  """
  bad = []
  for idx in range(rc.restype_num):
    name = resname_of(idx)
    groups = {a: g for a, g, _ in rc.rigid_group_atom_positions[name]}
    for s in COORD_SLOTS[rc.moltype(idx)]:
      if not rc.torsion_angles_mask[idx][s]:
        continue
      atoms = rc.torsion_angles_atoms[name][s]
      if len(atoms) < 4:
        continue
      defining = atoms[3]
      grp = groups.get(defining)
      if grp != s + 1:
        bad.append(f'{name} slot {s} ({atoms}): defining atom {defining} '
                   f'in group {grp}, expected {s+1}')
  assert not bad, 'torsion defining atom not in its rotated group:\n  ' + '\n  '.join(bad)
  return 'each torsion\'s defining atom lies in the group it rotates'


def test_torsion_defining_atom_positive_y():
  """At angle 0 the defining atom must sit at +y in its group's local frame.

  ``angles_from_positions`` reads (sin, cos) = (z, y); a negative y offsets the
  recovered angle by pi, mirroring the atom on reconstruction.
  """
  bad = []
  for idx in range(rc.restype_num):
    name = resname_of(idx)
    for s in COORD_SLOTS[rc.moltype(idx)]:
      if not rc.torsion_angles_mask[idx][s]:
        continue
      atoms = rc.torsion_angles_atoms[name][s]
      if len(atoms) < 4:
        continue
      aidx = rc.restype_name_to_atom14_names[name].index(atoms[3])
      y = float(rc.restype_atom14_rigid_group_positions[idx, aidx, 1])
      if y <= 0:
        bad.append(f'{name} slot {s}: defining atom {atoms[3]} local y={y:.4f} <= 0')
  assert not bad, 'defining atom has non-positive local y (pi angle offset):\n  ' + '\n  '.join(bad)
  return 'every torsion defining atom has +y local coordinate (no pi offset)'


def test_chi_exists_matches_mask_table():
  bad = []
  for idx in range(rc.restype_num):
    for s in range(K):
      exists = bool(rc.chi_angles_atom14_exists[idx, s])
      table = bool(rc.torsion_angles_mask[idx][s])
      if exists != table:
        bad.append(f'{resname_of(idx)} slot {s}: exists={exists} but mask_table={table}')
  assert not bad, ('chi_angles_atom14_exists disagrees with torsion_angles_mask '
                   '(angles_from_positions uses exists, so undefined torsions '
                   'leak into the loss):\n  ' + '\n  '.join(bad))
  return 'chi_angles_atom14_exists consistent with torsion_angles_mask'


# --------------------------------------------------------------------------- #
# 3. backbone-frame convention: feature pipeline vs ground-truth frames
# --------------------------------------------------------------------------- #
def test_backbone_affine_consistency():
  worst = 0.0
  for idx in range(rc.restype_num):
    seq = torch.full((1, 4), idx, dtype=torch.long)
    frames = rand_frames(1, 4)
    coords = synth_coords(seq, frames, rand_angles(1, 4))
    mask = torch.from_numpy(rc.restype_atom14_mask[idx]).bool().expand(1, 4, NA)
    batch = make_backbone_affine()(dict(seq=seq, coord=coords, coord_mask=mask.float()))
    gt = F.rigids_from_positions(seq, coords, mask)
    gt_r, gt_t = gt['atom_affine']
    r0, t0 = gt_r[..., 0, :, :], gt_t[..., 0, :]
    rb, tb = batch['backbone_affine']
    worst = max(worst, (rb - r0).abs().max().item(), (tb - t0).abs().max().item())
  assert worst < 1e-4, f'make_backbone_affine != rigids_from_positions group0 (max {worst:.2e})'
  return f'make_backbone_affine matches GT backbone frame for all restypes (max err {worst:.2e})'


# --------------------------------------------------------------------------- #
# 4. forward -> inverse -> forward round trip (the core correctness check)
# --------------------------------------------------------------------------- #
def _roundtrip(seq, verbose=False):
  b, n = seq.shape
  frames = rand_frames(b, n)
  angles = rand_angles(b, n)
  coords = synth_coords(seq, frames, angles)
  mask = torch.from_numpy(
      rc.restype_atom14_mask[seq.numpy()]).bool().to(seq.device)

  gt = F.rigids_from_positions(seq, coords, mask)
  ang = F.angles_from_positions(seq, coords, mask)

  gt_r, gt_t = gt['atom_affine']
  bb_r = gt_r[..., 0, :, :]
  bb_t = gt_t[..., 0, :]
  bb_err = max((bb_r - frames[0]).abs().max().item(),
               (bb_t - frames[1]).abs().max().item())

  angles2 = F.l2_norm(ang['torsion_angles'])
  coords2 = synth_coords(seq, (bb_r, bb_t), angles2)

  # per-atom reconstruction error (ambiguity-aware)
  diff = (coords2 - coords).norm(dim=-1)              # (b, n, NA)
  perm_diff = torch.zeros_like(diff)
  for i in range(n):
    perm = torch.from_numpy(renaming_perm(seq[0, i].item())).long()
    perm_diff[0, i] = diff[0, i, perm]
  atom_err = torch.minimum(diff, perm_diff)
  atom_err = atom_err * mask.float()

  # per-slot angle recovery error for slots that move atoms
  slot_err = {}
  for s in range(K):
    d = (angles2[..., s, :] - angles[..., s, :]).norm(dim=-1)
    if 'torsion_angles_alt' in ang:
      d_alt = (F.l2_norm(ang['torsion_angles_alt'])[..., s, :] - angles[..., s, :]).norm(dim=-1)
      d = torch.minimum(d, d_alt)
    slot_err[s] = d.max().item()

  if verbose:
    idx0 = int(seq[0, 0].item())
    name = resname_of(idx0)
    print(f'    [{name}] backbone frame err={bb_err:.2e}')
    for s in COORD_SLOTS[rc.moltype(idx0)]:
      if rc.torsion_angles_mask[idx0][s]:
        print(f'      slot {s:2d} angle recovery err={slot_err[s]:.3f}')
    worst_atoms = []
    for i in range(n):
      for a in range(NA):
        if mask[0, i, a] and atom_err[0, i, a] > 1e-3:
          worst_atoms.append((atom_name(seq[0, i].item(), a),
                              int(rc.restype_atom14_to_rigid_group[seq[0, i].item(), a]),
                              atom_err[0, i, a].item()))
    worst_atoms.sort(key=lambda x: -x[2])
    for a, g, e in worst_atoms[:8]:
      print(f'      atom {a:4s} (group {g:2d}) recon err={e:.3f} A')
  return bb_err, atom_err, slot_err, mask


def test_roundtrip_per_restype():
  failures = []
  for idx in range(rc.restype_num):
    seq = torch.full((1, 4), idx, dtype=torch.long)
    bb_err, atom_err, slot_err, mask = _roundtrip(seq)
    mol = rc.moltype(idx)
    max_atom = atom_err.max().item()
    bad_slots = [s for s in COORD_SLOTS[mol]
                 if rc.torsion_angles_mask[idx][s] and slot_err[s] > 1e-2]
    if bb_err > 1e-3 or max_atom > 2e-3 or bad_slots:
      failures.append((idx, bb_err, max_atom, bad_slots))
      print(f'  FAIL restype {idx} {resname_of(idx)}: bb_err={bb_err:.2e} '
            f'max_atom_err={max_atom:.3f} A, bad_slots={bad_slots}')
      _roundtrip(seq, verbose=True)
  assert not failures, (f'{len(failures)} restypes fail the forward/inverse round trip '
                        f'(see per-atom diagnostics above)')
  return f'all {rc.restype_num} restypes reconstruct coordinates to < 2e-3 A'


def test_roundtrip_mixed_sequence():
  seq = torch.randint(0, rc.restype_num, (1, 24))
  bb_err, atom_err, slot_err, mask = _roundtrip(seq)
  max_atom = atom_err.max().item()
  if max_atom > 2e-3:
    i, a = np.unravel_index(atom_err[0].argmax().item(), atom_err[0].shape)
    print(f'  worst: residue {i} {resname_of(seq[0, i].item())} '
          f'atom {atom_name(seq[0, i].item(), a)} err={max_atom:.3f} A')
  assert bb_err < 1e-3 and max_atom < 2e-3, \
      f'mixed-chain round trip fails (bb={bb_err:.2e}, atom={max_atom:.3f} A)'
  return f'mixed protein/DNA/RNA chain reconstructs to < 2e-3 A (max {max_atom:.2e})'


# --------------------------------------------------------------------------- #
# 5. independent dihedral check of angles_from_positions
# --------------------------------------------------------------------------- #
def test_torsion_extraction_matches_dihedral():
  seq = torch.randint(0, rc.restype_num, (1, 12))
  frames = rand_frames(1, 12)
  coords = synth_coords(seq, frames, rand_angles(1, 12)).numpy()[0]
  mask = rc.restype_atom14_mask[seq.numpy()[0]]
  ang = F.angles_from_positions(
      seq, torch.from_numpy(coords).unsqueeze(0),
      torch.from_numpy(mask).bool().unsqueeze(0))
  sc = ang['torsion_angles'].numpy()[0]              # (n, K, 2)

  def pos(i, name):
    names = rc.restype_name_to_atom14_names[resname_of(seq[0, i].item())]
    return coords[i, names.index(name)]

  # angles_from_positions reads (sin, cos) = (z, y) and then multiplies slot 2
  # (psi / zeta) by -1.  Comparing the (sin, cos) VECTOR with that mirror applied
  # is exact; comparing angles up to a sign is not, because negating the vector
  # shifts the angle by pi rather than negating it.
  mirror = np.array([1., 1., -1.] + [1.] * (K - 3))
  worst, worst_where = 0.0, None
  checked = 0
  for i in range(1, seq.shape[1]):
    idx = int(seq[0, i].item())
    name = resname_of(idx)
    na = is_na_idx(idx)
    for s in range(K):
      if s >= len(rc.torsion_angles_atoms[name]):
        continue                      # protein tables define only 7 torsions
      atoms = rc.torsion_angles_atoms[name][s]
      if len(atoms) < 4 or not rc.torsion_angles_mask[idx][s]:
        continue
      off = [0, 0, 0, 0]
      if s == 0:                       # pre-omega / omega: first 2 from prev
        off[0] = off[1] = -1
      elif s == 1:                     # phi (prot): 1 from prev; epsilon (na): 3
        off[0] = -1
        if na:
          off[1] = off[2] = -1
      elif s == 2 and na:              # zeta (na): first 2 from prev
        off[0] = off[1] = -1
      if i + min(off) < 0:
        continue
      if min(off) < 0 and rc.moltype(int(seq[0, i - 1])) != rc.moltype(idx):
        continue                      # cross-molecule junction: undefined
      try:
        pts = [pos(i + o, a) for o, a in zip(off, atoms)]
      except ValueError:
        continue
      d_ref = dihedral(*pts)
      ref = mirror[s] * np.array([math.sin(d_ref), math.cos(d_ref)])
      e = float(np.linalg.norm(sc[i, s] - ref))
      checked += 1
      if e > worst:
        worst, worst_where = e, (name, s)
  assert checked > 0, 'no torsions checked'
  assert worst < 1e-3, \
      f'extracted (sin,cos) != documented dihedral (max err {worst:.2e} at {worst_where})'
  return f'extraction matches independent dihedral on {checked} torsions (max err {worst:.2e})'


# --------------------------------------------------------------------------- #
# 6. StructureModule: shapes, SE(3) equivariance, gradients
# --------------------------------------------------------------------------- #
def _make_module():
  return folding.StructureModule(
      dim=(64, 32), structure_module_depth=2, structure_module_heads=4,
      point_value_dim=4, dropout=0.0)


def rand_quat_frames(*shape):
  # StructureModule consumes frames as (quaternions, translations)
  return F.l2_norm(torch.randn(*shape, 4)), torch.randn(*shape, 3)


def test_structure_module_shapes():
  mod = _make_module().eval()
  b, n = 2, 8
  seq = torch.randint(0, rc.restype_num, (b, n))
  batch = dict(seq=seq, mask=torch.ones(b, n, dtype=torch.bool))
  reps = dict(single=torch.randn(b, n, 64), pair=torch.randn(b, n, n, 32),
              frames=rand_quat_frames(b, n))
  with torch.no_grad():
    out = mod(reps, batch)
  assert len(out) == 1, 'eval mode should emit only the last layer'
  o = out[-1]
  assert o['frames'][0].shape == (b, n, 3, 3)
  assert o['frames'][1].shape == (b, n, 3)
  assert o['atoms']['coords'].shape == (b, n, NA, 3)
  assert o['atoms']['angles'].shape == (b, n, K, 2)
  assert o['atoms']['frames'][0].shape == (b, n, G, 3, 3)
  assert torch.isfinite(o['atoms']['coords']).all()
  return f'eval output shapes correct (coords {tuple(o["atoms"]["coords"].shape)})'


def test_structure_module_equivariance():
  mod = _make_module().eval()
  b, n = 2, 8
  seq = torch.randint(0, rc.restype_num, (b, n))
  batch = dict(seq=seq, mask=torch.ones(b, n, dtype=torch.bool))
  single, pair = torch.randn(b, n, 64), torch.randn(b, n, n, 32)
  q, t = rand_quat_frames(b, n)
  gq = F.l2_norm(torch.randn(4))
  gr, gt = F.quaternion_to_matrix(gq), torch.randn(3)
  q_g = F.quaternion_multiply(gq.expand(b, n, 4), q)
  t_g = torch.einsum('hw,...w->...h', gr, t) + gt
  with torch.no_grad():
    out = mod(dict(single=single, pair=pair, frames=(q, t)), batch)[-1]
    out_g = mod(dict(single=single, pair=pair, frames=(q_g, t_g)), batch)[-1]
  (r, t_out), coords = out['frames'], out['atoms']['coords']
  (rg, tg), coords_g = global_transform((r, t_out), coords, (gr, gt))
  err_r = (out_g['frames'][0] - rg).abs().max().item()
  err_t = (out_g['frames'][1] - tg).abs().max().item()
  # non-existing atoms are zeroed by the model; a global translation would move
  # those zeros, so compare only real atoms.
  amask = torch.from_numpy(rc.restype_atom14_mask[seq.numpy()]).bool()
  err_c = ((out_g['atoms']['coords'] - coords_g).norm(dim=-1) * amask).max().item()
  assert max(err_r, err_t, err_c) < 2e-3, \
      f'StructureModule not SE(3)-equivariant (R={err_r:.2e}, t={err_t:.2e}, coords={err_c:.2e})'
  return f'equivariant under global SE(3) (max coord err {err_c:.2e} A)'


def test_structure_module_blackhole_and_train():
  mod = _make_module()
  b, n = 1, 6
  seq = torch.randint(0, rc.restype_num, (b, n))
  batch = dict(seq=seq, mask=torch.ones(b, n, dtype=torch.bool))
  reps = dict(single=torch.randn(b, n, 64), pair=torch.randn(b, n, n, 32))
  mod.eval()
  with torch.no_grad():
    out = mod(reps, batch)                      # no frames -> black-hole init
  assert torch.isfinite(out[-1]['atoms']['coords']).all()
  mod.train()
  out = mod(reps, batch)
  assert len(out) == 2, 'train mode should emit every layer'
  loss = out[-1]['atoms']['coords'].sum()
  loss.backward()
  grad = mod.to_angles.to_groups.weight.grad
  assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
  return 'black-hole init finite; train emits depth layers; angles receive gradient'


# --------------------------------------------------------------------------- #
# 7. FAPE invariance
# --------------------------------------------------------------------------- #
def test_fape_invariance():
  seq = torch.randint(0, rc.restype_num, (1, 6))
  frames = rand_frames(1, 6)
  coords = synth_coords(seq, frames, rand_angles(1, 6))
  points = coords.reshape(1, -1, 3)
  pmask = torch.ones(1, points.shape[1])
  fmask = torch.ones(1, 6)
  pred = (frames[0] + 0.05 * torch.randn_like(frames[0]),
          frames[1] + 0.05 * torch.randn_like(frames[1]))
  ppoints = points + 0.05 * torch.randn_like(points)
  base = F.fape(pred, frames, fmask, ppoints, points, pmask, epsilon=0).item()
  g = (F.rotations_from_randn(), torch.randn(3))
  pred_g, ppoints_g = global_transform(pred, ppoints, g)
  frames_g, points_g = global_transform(frames, points, g)
  moved = F.fape(pred_g, frames_g, fmask, ppoints_g, points_g, pmask, epsilon=0).item()
  same = F.fape(frames, frames, fmask, points, points, pmask, epsilon=0).item()
  assert abs(base - moved) < 1e-4, f'FAPE not SE(3)-invariant ({base} vs {moved})'
  assert same < 1e-6, f'FAPE(x, x) != 0 ({same})'
  return f'FAPE SE(3)-invariant (base={base:.4f}, moved={moved:.4f}) and zero on identity'


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
TESTS = [
    ('quat_to_matrix_is_rotation', test_quaternion_to_matrix_is_rotation),
    ('quat_matrix_roundtrip', test_quaternion_matrix_roundtrip),
    ('quat_multiply_matches_matrix', test_quaternion_multiply_matches_matrix),
    ('rigids_compose', test_rigids_compose),
    ('rotations_from_vecs_orthonormal', test_rotations_from_vecs_orthonormal),
    ('default_frames_valid', test_default_frames_valid),
    ('depend_is_dag', test_depend_is_dag),
    ('atom14_coverage', test_atom14_coverage),
    ('torsion_defining_atom_in_rotated_group', test_torsion_defining_atom_in_rotated_group),
    ('torsion_defining_atom_positive_y', test_torsion_defining_atom_positive_y),
    ('chi_exists_matches_mask_table', test_chi_exists_matches_mask_table),
    ('backbone_affine_consistency', test_backbone_affine_consistency),
    ('roundtrip_per_restype', test_roundtrip_per_restype),
    ('roundtrip_mixed_sequence', test_roundtrip_mixed_sequence),
    ('torsion_extraction_matches_dihedral', test_torsion_extraction_matches_dihedral),
    ('structure_module_shapes', test_structure_module_shapes),
    ('structure_module_equivariance', test_structure_module_equivariance),
    ('structure_module_blackhole_and_train', test_structure_module_blackhole_and_train),
    ('fape_invariance', test_fape_invariance),
]


def main():
  only = sys.argv[1:] or None
  print(f'config: restype_list={sorted(rc.restype_list)} atom14_type_num={NA} '
        f'chi_angles_num={K} restype_num={rc.restype_num}')
  print(f'protein idx {rc.prot_from_idx}-{rc.prot_to_idx}, '
        f'DNA idx {rc.dna_from_idx}-{rc.dna_to_idx}, '
        f'RNA idx {rc.rna_from_idx}-{rc.rna_to_idx}\n')
  passed, failed = [], []
  for name, fn in TESTS:
    if only and not any(o in name for o in only):
      continue
    try:
      detail = fn()
      passed.append(name)
      print(f'[PASS] {name}: {detail}')
    except Exception as e:  # noqa: BLE001
      failed.append(name)
      print(f'[FAIL] {name}: {e}')
      if not isinstance(e, AssertionError):
        traceback.print_exc()
  print(f'\n==== {len(passed)} passed, {len(failed)} failed ====')
  if failed:
    print('failed: ' + ', '.join(failed))
  return 1 if failed else 0


if __name__ == '__main__':
  sys.exit(main())