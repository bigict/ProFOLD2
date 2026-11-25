#!/usr/bin/env python3
"""Stage 2 fold-switch metrics: fixed-alignment TM-score + SS change + rigid-body tag.

Key: use FIXED sequence-correspondence TM-score (Kabsch superposition + TM formula),
NOT TMalign's optimized alignment — TMalign aligns only the conserved core and hides
the switched region (gave 0.58-0.74 for known switchers). Fixed alignment penalizes
the full conformational change (known switchers -> ~0.06-0.27, < 0.5).

SS assignment from backbone phi/psi (no DSSP binary available); SS-change fraction
distinguishes true topology switch (helix<->strand) from pure rigid-body domain motion
(SS preserved). Rigid-body displacement tag from core-aligned switching field.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from probe_v4 import switching_field, kabsch_rmsd  # noqa: E402


def kabsch_rmsd_full(P, Q):
  """Kabsch superposition; return (R, t, rmsd, per-residue distances)."""
  mP, mQ = P.mean(0), Q.mean(0)
  Pc, Qc = P - mP, Q - mQ
  H = Pc.T @ Qc
  U, S, Vt = np.linalg.svd(H)
  d = np.sign(np.linalg.det(Vt.T @ U.T))
  R = Vt.T @ np.diag([1, 1, d]) @ U.T
  Pr = Pc @ R.T
  diff = Pr - Qc
  dist = np.sqrt((diff ** 2).sum(-1))
  rmsd = np.sqrt((dist ** 2).mean())
  return R, mP, mQ, rmsd, dist


def fixed_tm_score(bb0, bb1):
  """TM-score with fixed 1:1 correspondence (Kabsch), normalized by len(bb0)."""
  ca0, ca1 = bb0[:, 1], bb1[:, 1]
  L = len(ca0)
  _, _, _, rmsd, dist = kabsch_rmsd_full(ca0, ca1)
  d0 = 1.24 * (L - 15) ** (1.0 / 3) - 1.8 if L > 15 else 0.5
  d0 = max(d0, 0.5)
  tm = (1.0 / (1 + (dist / d0) ** 2)).mean()
  return tm, rmsd, dist


def _dihedral(p0, p1, p2, p3):
  b0 = p0 - p1
  b1 = p2 - p1
  b2 = p3 - p2
  b1n = b1 / (np.linalg.norm(b1) + 1e-8)
  v = b0 - np.dot(b0, b1n) * b1n
  w = b2 - np.dot(b2, b1n) * b1n
  x = np.dot(v, w)
  y = np.dot(np.cross(b1n, v), w)
  return np.degrees(np.arctan2(y, x))


def _dihedral_vec(p0, p1, p2, p3):
  """Vectorized dihedral angles for arrays of shape (n, 3)."""
  b0 = p0 - p1
  b1 = p2 - p1
  b2 = p3 - p2
  b1n = b1 / (np.linalg.norm(b1, axis=-1, keepdims=True) + 1e-8)
  v = b0 - np.sum(b0 * b1n, axis=-1, keepdims=True) * b1n
  w = b2 - np.sum(b2 * b1n, axis=-1, keepdims=True) * b1n
  x = np.sum(v * w, axis=-1)
  y = np.sum(np.cross(b1n, v) * w, axis=-1)
  return np.degrees(np.arctan2(y, x))


def assign_ss(bb):
  """Per-residue SS class from phi/psi (vectorized): 'H' helix, 'E' strand, 'C' coil."""
  n = len(bb)
  ss = np.full(n, 'C')
  if n < 3:
    return list(ss)
  N, CA, C = bb[:, 0], bb[:, 1], bb[:, 2]
  phi = _dihedral_vec(C[:-1], N[1:], CA[1:], C[1:])      # residues 1..n-1
  psi = _dihedral_vec(N[:-1], CA[:-1], C[:-1], N[1:])    # residues 0..n-2
  phi_full = np.full(n, np.nan); phi_full[1:] = phi
  psi_full = np.full(n, np.nan); psi_full[:-1] = psi
  H = (phi_full >= -100) & (phi_full <= -30) & (psi_full >= -75) & (psi_full <= -5)
  E = (phi_full >= -160) & (phi_full <= -80) & ((psi_full >= 90) | (psi_full <= -150))
  ss[H] = 'H'
  ss[E] = 'E'
  return list(ss)


def ss_change(bb0, bb1):
  """Fraction of residues changing SS class; and helix<->strand fraction."""
  s0, s1 = assign_ss(bb0), assign_ss(bb1)
  n = len(s0)
  changed = sum(a != b for a, b in zip(s0, s1))
  he = sum({a, b} == {'H', 'E'} for a, b in zip(s0, s1))
  return changed / n, he / n, s0, s1


def switch_rigidity(bb0, bb1, field):
  """Internal RMSD of switching residues after their OWN superposition.

  Low (<~2A) => switching region moves as a rigid body (loop/domain motion, reject).
  High => internal deformation (fold-switch candidate). NOTE: does NOT separate
  calmodulin-like hinge domain motion (hinge deforms -> high) from true switch;
  that ambiguity needs topology analysis / curation.
  """
  sw = np.where(field > 0.5)[0]
  if len(sw) < 3:
    return None
  return float(kabsch_rmsd(bb0[sw, 1], bb1[sw, 1]))


def classify_pair(bb0, bb1, swss_thresh=0.30, maxd_thresh=12.0, tm_thresh=0.6):
  """Return metrics + label: fold_switch / domain_motion / fluctuation.

  PRIMARY criterion = swSS (SS-change fraction AMONG switching-field residues,
  field>0.5) AND max_disp >= maxd_thresh. swSS catches LOCALIZED switches diluted
  in full-region TM (CLIC1 switch region 8 res, full TM 0.71 but swSS 0.62) AND
  large switches, while rejecting rigid domain motion (calmodulin swSS 0.13).
  max_disp gate rejects tiny fluctuations where a few residues cross SS boundaries
  (swSS spuriously high but max_disp < 12A, TM ~0.99). Cross-validated on
  Porter/Chakravarty 65 available known pairs: swSS>=0.30 & maxd>=12 gives 70.8%
  recall (46/65), calmodulin + tiny fluctuations excluded; 200-seq sample 298->22.
  disp_tag from probe mechanism boundary: max_disp<20 local, 20-40 medium, >40 large.
  """
  tm, rmsd, dist = fixed_tm_score(bb0, bb1)
  ss_frac, he_frac, s0, s1 = ss_change(bb0, bb1)
  field, core, disp = switching_field(bb0, bb1)
  max_disp = float(disp.max())
  mean_disp = float(disp.mean())
  sw = np.where(field > 0.5)[0]
  n_switch = len(sw)
  if n_switch >= 3:
    swss = sum(s0[i] != s1[i] for i in sw) / n_switch
    swhe = sum({s0[i], s1[i]} == {'H', 'E'} for i in sw) / n_switch
  else:
    swss = swhe = 0.0
  rigidity = switch_rigidity(bb0, bb1, field)
  if n_switch >= 3 and swss >= swss_thresh and max_disp >= maxd_thresh:
    label = 'fold_switch'
  elif tm < tm_thresh:
    label = 'domain_motion'   # large change but switching region SS preserved
  else:
    label = 'fluctuation'
  disp_tag = 'local' if max_disp < 20 else ('medium' if max_disp < 40 else 'large')
  return dict(tm=float(tm), rmsd=float(rmsd), ss_change=float(ss_frac),
              he_change=float(he_frac), swss=round(float(swss), 3),
              swhe=round(float(swhe), 3), max_disp=max_disp, mean_disp=mean_disp,
              n_switch=int(n_switch),
              rigidity=(round(rigidity, 2) if rigidity is not None else None),
              label=label, disp_tag=disp_tag)


if __name__ == '__main__':
  for name in ['xcl1', 'rfah', 'mad2']:
    p = np.load(os.path.join(HERE, 'data', '%s_pair.npz' % name))
    m = classify_pair(p['fold0'], p['fold1'])
    print('%-6s TM=%.3f RMSD=%.2f SS=%.2f HE=%.2f max_disp=%.1f n_switch=%d '
          '-> %s [%s]' %
          (name, m['tm'], m['rmsd'], m['ss_change'], m['he_change'],
           m['max_disp'], m['n_switch'], m['label'], m['disp_tag']))