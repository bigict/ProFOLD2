#!/usr/bin/env python3
"""Real-structure validation of the ProFOLD2 nucleic-acid geometry pipeline.

Complements tests/test_structure_biomolecules.py (synthetic self-consistency)
by checking the forward/inverse maps against *experimental* DNA/RNA coordinates:

  * 1BNA  – Dickerson B-DNA dodecamer d(CGCGAATTCGCG)2, chain A (12 nt)
  * 1L2X  – RNA hairpin, chain A (27 nt)

Fixtures live in tests/data/na/ (public-domain PDB data).  For each chain we run
the training-time inverse maps (rigids_from_positions / angles_from_positions)
on the real coordinates and rebuild with the forward maps (rigids_from_angles /
rigids_to_positions), then check:

  1. round-trip RMSD is at the idealisation level (~0.5 A), not mirroring (>2 A)
  2. extracted torsions equal independent dihedrals of the documented atoms
  3. key bond lengths are physical
  4. sugar chiral-volume signs are preserved (no enantiomer / mirror flip)
  5. delta / chi fall in the canonical B-DNA / A-RNA ranges
  6. RNA O2' is recoverable (regression for the nu0 defining-atom bug)

Run:
    PYTHONPATH=. python tests/test_structure_real_na.py
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

from profold2.common import residue_constants as rc, protein
from profold2.model import functional as F

torch.manual_seed(0)
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'na')
K = rc.chi_angles_num
NA = rc.atom14_type_num

# literature bond lengths [A] (tolerance applied in the test)
BONDS = {
    'P-OP1': 1.485, 'P-OP2': 1.485, 'P-O5\'': 1.60, 'O5\'-C5\'': 1.44,
    'C5\'-C4\'': 1.51, 'C4\'-C3\'': 1.53, 'C3\'-O3\'': 1.43, 'C4\'-O4\'': 1.45,
    'O4\'-C1\'': 1.41, 'C1\'-C2\'': 1.53, 'C2\'-C3\'': 1.53, 'C2\'-O2\'': 1.42,
    'C1\'-N9': 1.47, 'C1\'-N1': 1.47,
}


def resname(idx):
  idx = int(idx)
  if idx == rc.unk_restype_index:
    return rc.unk_restype
  return rc.restype_1to3[(rc.restypes[idx], rc.moltype(idx))]


def atom_slot(idx, name):
  return rc.restype_name_to_atom14_names[resname(idx)].index(name)


def load_chain(pdb_id, chain):
  # ATOM-only: drop waters / ions so the chain is a contiguous NA polymer
  text = ''.join(l for l in open(os.path.join(DATA, f'{pdb_id}.pdb'))
                 if l.startswith(('ATOM', 'TER', 'END')))
  prot = protein.from_pdb_string(text, chain_id=chain)
  seq = torch.from_numpy(prot.aatype).long()[None]
  coords = torch.from_numpy(prot.atom_positions).float()[None]
  mask = torch.from_numpy(prot.atom_mask).bool()[None]
  assert all(rc.moltype(int(x)) in (rc.DNA, rc.RNA) for x in prot.aatype), \
      'non-NA residue parsed'
  return seq, coords, mask


def reconstruct(seq, coords, mask):
  gt = F.rigids_from_positions(seq, coords, mask)
  ang = F.angles_from_positions(seq, coords, mask)
  gtr, gtt = gt['atom_affine']
  bb = (gtr[..., 0, :, :], gtt[..., 0, :])
  rec = F.rigids_to_positions(
      F.rigids_from_angles(seq, bb, F.l2_norm(ang['torsion_angles'])), seq)
  return rec, ang, gt


def backbone_complete(seq, mask):
  n0 = resname(seq[0, 0])
  bi = [rc.restype_name_to_atom14_names[n0].index(x)
        for x in ('P', 'OP1', 'OP2', 'O5\'')]
  return mask[0][:, bi].all(dim=-1)


def dihedral(p0, p1, p2, p3):
  b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
  b1 = b1 / np.linalg.norm(b1)
  v = b0 - np.dot(b0, b1) * b1
  w = b2 - np.dot(b2, b1) * b1
  return math.atan2(np.dot(np.cross(b1, v), w), np.dot(v, w))


def get_xyz(coords, mask, seq, i, name):
  names = rc.restype_name_to_atom14_names[resname(int(seq[0, i]))]
  if name not in names:
    return None                       # atom not part of this residue type
  s = names.index(name)
  if not mask[0, i, s]:
    return None
  return coords[0, i, s].numpy().astype(np.float64)


# --------------------------------------------------------------------------- #
def test_roundtrip_real(seq, coords, mask, label, max_rmsd):
  rec, _, _ = reconstruct(seq, coords, mask)
  err = (rec - coords).norm(dim=-1)[0]
  ok = backbone_complete(seq, mask)
  m = mask[0]
  sel = ok
  rmsd = err[sel][m[sel]].pow(2).mean().sqrt().item()
  mx = err[sel][m[sel]].max().item()
  n_skip = int((~ok).sum())
  assert rmsd < max_rmsd, \
      f'{label} round-trip RMSD {rmsd:.3f} A >= {max_rmsd} (mirroring / bug?)'
  return (f'{label}: bb-complete RMSD={rmsd:.3f} A, max={mx:.3f} A '
          f'({n_skip} terminal residue(s) skipped)')


def test_torsion_extraction_real(seq, coords, mask, label):
  _, ang, _ = reconstruct(seq, coords, mask)
  sc = ang['torsion_angles'].numpy()[0]
  mirror = np.array([1., 1., -1.] + [1.] * (K - 3))
  worst, checked = 0.0, 0
  n = seq.shape[1]
  for i in range(1, n):
    idx = int(seq[0, i])
    name = resname(idx)
    for s in range(K):
      if s >= len(rc.torsion_angles_atoms[name]):
        continue
      atoms = rc.torsion_angles_atoms[name][s]
      if len(atoms) < 4 or not rc.torsion_angles_mask[idx][s]:
        continue
      off = [0, 0, 0, 0]
      if s == 0:
        off[0] = off[1] = -1
      elif s == 1:
        off[0] = -1
        if rc.moltype(idx) in (rc.DNA, rc.RNA):
          off[1] = off[2] = -1
      elif s == 2 and rc.moltype(idx) in (rc.DNA, rc.RNA):
        off[0] = off[1] = -1
      if i + min(off) < 0:
        continue
      pts = [get_xyz(coords, mask, seq, i + o, a) for o, a in zip(off, atoms)]
      if any(p is None for p in pts):
        continue
      d = dihedral(*pts)
      ref = mirror[s] * np.array([math.sin(d), math.cos(d)])
      worst = max(worst, float(np.linalg.norm(sc[i, s] - ref)))
      checked += 1
  assert checked > 0 and worst < 1e-3, \
      f'{label} extracted torsion != independent dihedral (err {worst:.2e}, n={checked})'
  return f'{label}: {checked} torsions match independent dihedral (max err {worst:.2e})'


PURINES = ('DA', 'DG', 'A', 'G')


def test_bond_lengths_real(seq, coords, mask, label):
  bad, checked = [], 0
  for i in range(seq.shape[1]):
    purine = resname(seq[0, i]) in PURINES
    for bond, lit in BONDS.items():
      # glycosidic bond is C1'-N9 (purine) or C1'-N1 (pyrimidine); the other
      # nitrogen exists in purines but is a non-bonded base atom (~5 A away).
      if bond == 'C1\'-N9' and not purine:
        continue
      if bond == 'C1\'-N1' and purine:
        continue
      a, b = bond.split('-')
      pa, pb = get_xyz(coords, mask, seq, i, a), get_xyz(coords, mask, seq, i, b)
      if pa is None or pb is None:
        continue
      d = float(np.linalg.norm(pa - pb))
      checked += 1
      if abs(d - lit) > 0.12:
        bad.append(f'{resname(seq[0,i])}{i} {bond}={d:.3f} (lit {lit})')
  assert checked > 0 and not bad, \
      f'{label} unphysical bond lengths: {bad[:8]} ({len(bad)}/{checked})'
  return f'{label}: {checked} bond lengths within 0.12 A of literature'


def _chiral_sign(coords, mask, seq, i, center, subs):
  c = get_xyz(coords, mask, seq, i, center)
  vs = [get_xyz(coords, mask, seq, i, s) for s in subs]
  if c is None or any(v is None for v in vs):
    return None
  mat = np.stack([v - c for v in vs])
  return float(np.sign(np.linalg.det(mat)))


def test_sugar_chirality_real(seq, coords, mask, label):
  rec, _, _ = reconstruct(seq, coords, mask)
  centers = [('C1\'', ('O4\'', 'C2\'', 'N9')), ('C1\'', ('O4\'', 'C2\'', 'N1')),
             ('C3\'', ('C4\'', 'C2\'', 'O3\'')), ('C4\'', ('C5\'', 'C3\'', 'O4\''))]
  if rc.moltype(int(seq[0, 0])) == rc.RNA:
    centers.append(('C2\'', ('C1\'', 'C3\'', 'O2\'')))
  mism, checked = [], 0
  ok = backbone_complete(seq, mask)   # skip termini with a degenerate frame
  for i in range(seq.shape[1]):
    if not ok[i]:
      continue
    for center, subs in centers:
      sr = _chiral_sign(coords, mask, seq, i, center, subs)
      sf = _chiral_sign(rec, mask, seq, i, center, subs)
      if sr is None or sf is None or sr == 0:
        continue
      checked += 1
      if sr != sf:
        mism.append(f'{resname(seq[0,i])}{i} {center}')
  assert checked > 0 and not mism, \
      f'{label} sugar chirality flipped (mirror image!) at {mism[:8]}'
  return f'{label}: {checked} sugar chiral centers preserve sign (no enantiomer)'


def test_delta_chi_ranges(seq, coords, mask, label):
  # delta = C5'-C4'-C3'-O3' ; chi = O4'-C1'-N9-C4 (purine) / N1-C2 (pyrimidine)
  deltas, chis = [], []
  for i in range(seq.shape[1]):
    p = [get_xyz(coords, mask, seq, i, x) for x in ('C5\'', 'C4\'', 'C3\'', 'O3\'')]
    if all(x is not None for x in p):
      deltas.append(math.degrees(dihedral(*p)))
    name = resname(seq[0, i])
    base = ('N9', 'C4') if name in ('DA', 'DG', 'A', 'G') else ('N1', 'C2')
    p = [get_xyz(coords, mask, seq, i, x) for x in ('O4\'', 'C1\'') + base]
    if all(x is not None for x in p):
      chis.append(math.degrees(dihedral(*p)))
  dm, cm = float(np.mean(deltas)), float(np.mean(chis))
  mol = rc.moltype(int(seq[0, 0]))
  if mol == rc.DNA:                     # B-form: delta ~123, chi anti ~-108
    assert 80 < dm < 170, f'{label} delta mean {dm:.1f} outside B-DNA range'
  else:                                 # A-form RNA: delta ~82
    assert 50 < dm < 115, f'{label} delta mean {dm:.1f} outside A-RNA range'
  assert -180 < cm < -60, f'{label} chi mean {cm:.1f} not anti'
  return f'{label}: delta mean={dm:.1f} deg, chi mean={cm:.1f} deg (canonical)'


def test_rna_o2prime_recoverable(seq, coords, mask, label):
  if rc.moltype(int(seq[0, 0])) != rc.RNA:
    return f'{label}: n/a (DNA has no O2\')'
  rec, _, _ = reconstruct(seq, coords, mask)
  s = atom_slot(seq[0, 0], 'O2\'')
  errs = []
  for i in range(seq.shape[1]):
    if mask[0, i, s]:
      errs.append((rec[0, i, s] - coords[0, i, s]).norm().item())
  mean, mx = float(np.mean(errs)), float(np.max(errs))
  assert mx < 1.5, f'{label} O2\' unrecoverable (max err {mx:.3f} A) — nu0 bug?'
  return f'{label}: O2\' recovered on {len(errs)} residues (mean {mean:.3f}, max {mx:.3f} A)'


# --------------------------------------------------------------------------- #
CASES = [
    ('1BNA', 'A', 'B-DNA', 1.0),
    ('1L2X', 'A', 'RNA', 1.0),
]


def main():
  only = sys.argv[1:] or None
  passed, failed = [], []
  for pdb_id, chain, label, max_rmsd in CASES:
    seq, coords, mask = load_chain(pdb_id, chain)
    print(f'\n== {label} {pdb_id} chain {chain}: {seq.shape[1]} residues ==')
    checks = [
        ('roundtrip', lambda: test_roundtrip_real(seq, coords, mask, label, max_rmsd)),
        ('torsion_extraction', lambda: test_torsion_extraction_real(seq, coords, mask, label)),
        ('bond_lengths', lambda: test_bond_lengths_real(seq, coords, mask, label)),
        ('sugar_chirality', lambda: test_sugar_chirality_real(seq, coords, mask, label)),
        ('delta_chi_ranges', lambda: test_delta_chi_ranges(seq, coords, mask, label)),
        ('o2prime', lambda: test_rna_o2prime_recoverable(seq, coords, mask, label)),
    ]
    for name, fn in checks:
      full = f'{label}.{name}'
      if only and not any(o in full for o in only):
        continue
      try:
        detail = fn()
        passed.append(full)
        print(f'[PASS] {full}: {detail}')
      except Exception as e:  # noqa: BLE001
        failed.append(full)
        print(f'[FAIL] {full}: {e}')
        if not isinstance(e, AssertionError):
          traceback.print_exc()
  print(f'\n==== {len(passed)} passed, {len(failed)} failed ====')
  if failed:
    print('failed: ' + ', '.join(failed))
  return 1 if failed else 0


if __name__ == '__main__':
  sys.exit(main())