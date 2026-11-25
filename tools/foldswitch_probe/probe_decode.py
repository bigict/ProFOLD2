#!/usr/bin/env python3
"""Minimal trunk-representability probe — decoder test (Phase 0.5).

Tests whether the frozen pretrained trunk single repr encodes alternative-fold
(fold1) information, by comparing how well a CONSTRAINED (ridge) decoder extracts
fold0 vs fold1 backbone dihedrals from the SAME repr, with cross-validation across
residues to limit memorization.

Hypothesis readout:
  fold0 CV error << fold1 CV error  -> repr is fold0-biased (alt-fold not accessible)
  fold0 CV error ~= fold1 CV error  -> repr encodes both (representability supported)

Run after probe_repr.py (needs data/xcl1_pair.npz + data/xcl1_repr.npy):
    PYTHONPATH=. python tools/foldswitch_probe/probe_decode.py
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')


def dihedral(p0, p1, p2, p3):
  b0 = p0 - p1; b1 = p2 - p1; b2 = p3 - p2
  b1 = b1 / np.linalg.norm(b1)
  v = b0 - np.dot(b0, b1) * b1
  w = b2 - np.dot(b2, b1) * b1
  return np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))


def backbone_dihedrals(bb):
  # bb: (n, 4, 3) = N, CA, C, O
  n = len(bb)
  N, CA, C = bb[:, 0], bb[:, 1], bb[:, 2]
  phi = np.full(n, np.nan); psi = np.full(n, np.nan)
  for i in range(1, n):
    phi[i] = dihedral(C[i - 1], N[i], CA[i], C[i])
  for i in range(n - 1):
    psi[i] = dihedral(N[i], CA[i], C[i], N[i + 1])
  return phi, psi


def sincos(a):
  return np.stack([np.sin(a), np.cos(a)], axis=-1)


def ridge_cv(X, Y, alphas=(1.0, 10.0, 100.0, 1000.0), k=5, seed=0):
  """K-fold CV across residues; return best-alpha mean angular error (deg)."""
  n = len(X)
  rng = np.random.default_rng(seed)
  folds = np.array_split(rng.permutation(n), k)
  best = None
  for alpha in alphas:
    errs = []
    for f in folds:
      tr = np.setdiff1d(np.arange(n), f)
      Xm, Xs = X[tr].mean(0), X[tr].std(0) + 1e-8
      Xtr = (X[tr] - Xm) / Xs; Xte = (X[f] - Xm) / Xs
      Ym = Y[tr].mean(0)
      Ytr = Y[tr] - Ym
      A = Xtr.T @ Xtr + alpha * np.eye(X.shape[1])
      W = np.linalg.solve(A, Xtr.T @ Ytr)
      pred = Xte @ W + Ym
      # angular error per dihedral from sin/cos
      ang_pred = np.arctan2(pred[:, 0::2], pred[:, 1::2])
      ang_true = np.arctan2(Y[f][:, 0::2], Y[f][:, 1::2])
      d = np.abs(np.arctan2(np.sin(ang_pred - ang_true), np.cos(ang_pred - ang_true)))
      errs.append(np.nanmean(d))
    me = np.mean(errs)
    if best is None or me < best[1]:
      best = (alpha, me)
  return best  # (alpha, mean angular error rad)


def main():
  pair = np.load(os.path.join(DATA, 'xcl1_pair.npz'))
  repr_ = np.load(os.path.join(DATA, 'xcl1_repr.npy'))  # (n, 384)
  f0, f1 = pair['fold0'], pair['fold1']
  print('XCL1: %d res, repr dim %d, fold0/fold1 CA RMSD %.2f A' %
        (len(f0), repr_.shape[1], pair['ca_rmsd']))

  phi0, psi0 = backbone_dihedrals(f0)
  phi1, psi1 = backbone_dihedrals(f1)
  # valid residues (phi/psi defined): interior 1..n-2
  valid = ~np.isnan(phi0) & ~np.isnan(psi0) & ~np.isnan(phi1) & ~np.isnan(psi1)
  X = repr_[valid]
  Y0 = np.concatenate([sincos(phi0[valid]), sincos(psi0[valid])], axis=-1)
  Y1 = np.concatenate([sincos(phi1[valid]), sincos(psi1[valid])], axis=-1)
  print('valid residues for CV:', valid.sum())

  a0, e0 = ridge_cv(X, Y0)
  a1, e1 = ridge_cv(X, Y1)
  # baseline: predict per-dihedral mean (no repr)
  b0 = np.nanmean(np.abs(np.arctan2(np.sin(Y0 - Y0.mean(0)), np.cos(Y0 - Y0.mean(0)))))
  b1 = np.nanmean(np.abs(np.arctan2(np.sin(Y1 - Y1.mean(0)), np.cos(Y1 - Y1.mean(0)))))
  print('fold0 ridge CV angular err: %.1f deg (alpha=%g) | mean-baseline %.1f deg' %
        (np.degrees(e0), a0, np.degrees(b0)))
  print('fold1 ridge CV angular err: %.1f deg (alpha=%g) | mean-baseline %.1f deg' %
        (np.degrees(e1), a1, np.degrees(b1)))
  print('fold1/fold0 error ratio: %.2f' % (e1 / e0))
  print('NOTE: single switcher => heuristic; needs multiple switchers + controls.')


if __name__ == '__main__':
  main()