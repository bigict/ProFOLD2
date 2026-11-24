#!/usr/bin/env python3
"""Reproduce the AF3 interface-PAE baseline AUC on the two held-out sets.

No structure prediction is involved: the shipped triad tables already carry the
AF3 confidence features, so this is table arithmetic on CPU.

Evaluation unit follows the paper: one ROC per antigen (p:MHC), then the median
across antigens. Targets quoted from the preprint:
    class I  (post-cutoff PDB)   median AUC 0.92, 14 antigens
    class II (CRESTA)            median AUC 0.81,  8 antigens

The paper does not name the exact feature in the text we could reach, so several
interface-PAE candidates are scored and reported side by side. PAE-like features
are negated (lower error = better); score-like features are used as-is.

--iedb additionally scores the IEDB training/dev pool. That is NOT a reproduction:
the paper stratifies IEDB by assay type and groups by antigen-study, neither of
which the shipped tables support. See the warning printed with those results.

Run with the isolated venv:
    tcr_bench/.venv/bin/python tcr_bench/baseline_auc.py
    tcr_bench/.venv/bin/python tcr_bench/baseline_auc.py --iedb
"""
import argparse
import collections
import os
import statistics
import sys

import pyarrow.parquet as pq

HERE = os.path.dirname(__file__)
SETS = [
    ('class I / PDB post-cutoff',
     'data/pdb/triad/pdb_validation_triad.conf_af3.parquet', 0.92, 14),
    ('class II / CRESTA',
     'data/cresta/triad/cresta_triad.conf_af3.parquet', 0.81, 8),
]
# No paper target: the published IEDB numbers are assay-stratified and grouped by
# antigen-study, so nothing here is like-for-like. Scored for reference only.
IEDB_SETS = [
    ('IEDB class I / training-dev pool',
     'data/iedb_I/triad/iedb_I_triad.conf_af3.parquet', None, None),
    ('IEDB class II / training-dev pool',
     'data/iedb_II/triad/iedb_II_triad.conf_af3.parquet', None, None),
]
# name -> sign; PAE and error-like features are negated.
FEATURES = [
    ('min_tcr_p_pae', -1), ('min_p_tcr_pae', -1),
    ('mean_tcr_p_interface_pae', -1), ('mean_p_tcr_interface_pae', -1),
    ('mean_tcr_pmhc_interface_pae', -1), ('mean_pmhc_tcr_interface_pae', -1),
    ('mean_tcr_p_pae', -1), ('mean_p_tcr_pae', -1),
    ('min_tcr_mhc_pae', -1), ('mean_tcr_mhc_interface_pae', -1),
    ('mean_tcr_p_interface_contact_prob', +1),
    ('mean_tcr_pmhc_interface_contact_prob', +1),
    ('iptm', +1), ('ranking_score', +1),
]


def auc(scores, labels):
  """Mann-Whitney AUC with average ranks for ties. None if one class is absent."""
  pos = sum(labels)
  neg = len(labels) - pos
  if pos == 0 or neg == 0:
    return None
  order = sorted(range(len(scores)), key=lambda i: scores[i])
  ranks = [0.0] * len(scores)
  i = 0
  while i < len(order):
    j = i
    while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
      j += 1
    avg = (i + j) / 2.0 + 1.0
    for k in range(i, j + 1):
      ranks[order[k]] = avg
    i = j + 1
  rpos = sum(ranks[i] for i in range(len(labels)) if labels[i])
  return (rpos - pos * (pos + 1) / 2.0) / (pos * neg)


def antigen_key(r):
  return (r['peptide'], r['mhc_1_name'], r['mhc_2_name'])


def score(title, rel, target, n_target):
  path = os.path.join(HERE, rel)
  if not os.path.exists(path):
    print(f'{title}: MISSING {path}')
    return
  cols = ['cognate', 'peptide', 'mhc_1_name', 'mhc_2_name']
  schema = pq.read_schema(path).names
  feats = [(f, s) for f, s in FEATURES if f in schema]
  rows = pq.read_table(path, columns=cols + [f for f, _ in feats]).to_pylist()

  groups = collections.defaultdict(list)
  for r in rows:
    groups[antigen_key(r)].append(r)
  usable = {k: v for k, v in groups.items()
            if 0 < sum(x['cognate'] for x in v) < len(v)}

  print(f'\n=== {title}')
  tail = (f'   paper says {n_target} antigens' if n_target is not None
          else '   NON-PAPER PROTOCOL (see note below)')
  print(f'  rows {len(rows)}  antigens(p:MHC) {len(groups)}  '
        f'usable(both classes) {len(usable)}{tail}')
  head = f'  vs paper {target}' if target is not None else ''
  print(f'  {"feature":42s} {"median":>7s} {"min":>6s} {"max":>6s}{head}')
  best = None
  for f, sign in feats:
    aucs = []
    for v in usable.values():
      vals = [x[f] for x in v]
      if any(x is None for x in vals):
        continue
      a = auc([sign * float(x) for x in vals], [x['cognate'] for x in v])
      if a is not None:
        aucs.append(a)
    if not aucs:
      continue
    med = statistics.median(aucs)
    closer = (target is not None
              and (best is None or abs(med - target) < abs(best[1] - target)))
    print(f'  {f:42s} {med:7.3f} {min(aucs):6.3f} {max(aucs):6.3f}'
          f'{"  <-- closest" if closer else ""}')
    if closer:
      best = (f, med)
  if best:
    print(f'  closest to the paper: {best[0]}  median {best[1]:.3f} '
          f'(target {target}, delta {best[1]-target:+.3f})')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--iedb', action='store_true',
                  help='also score the IEDB training/dev pool (NOT a reproduction)')
  args = ap.parse_args()

  for title, rel, target, n_target in SETS:
    score(title, rel, target, n_target)
  print('\nNOTE: the paper does not name the exact feature in the reachable text,')
  print('so "closest" is a match by value, not a confirmed identification.')

  if args.iedb:
    for title, rel, target, n_target in IEDB_SETS:
      score(title, rel, target, n_target)
    print('\nIEDB WARNING: these are NOT a reproduction of the paper.')
    print('  - grouped by epitope_id (p:MHC); the paper uses antigen-study')
    print('    (1,024 class I / 248 class II), which the shipped references column')
    print('    cannot rebuild: it is a placeholder on permuted negatives.')
    print('  - the paper reports IEDB stratified by assay type (binding 0.93/0.94,')
    print('    SPR 0.85, multimer 0.61/0.52); these tables carry no assay column.')
    print('  - rows are pre-homology-exclusion: positives exceed the paper by')
    print('    394 (I) / 43 (II); negatives match exactly.')
    print('  - IEDB is the paper\'s training/dev pool, so their model is in-sample.')
  return 0


if __name__ == '__main__':
  sys.exit(main())
