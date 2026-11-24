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

Run with the isolated venv:
    tcr_bench/.venv/bin/python tcr_bench/baseline_auc.py
"""
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


def main():
  for title, rel, target, n_target in SETS:
    path = os.path.join(HERE, rel)
    if not os.path.exists(path):
      print(f'{title}: MISSING {path}')
      continue
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
    print(f'  rows {len(rows)}  antigens(p:MHC) {len(groups)}  '
          f'usable(both classes) {len(usable)}   paper says {n_target} antigens')
    print(f'  {"feature":42s} {"median":>7s} {"min":>6s} {"max":>6s}  vs paper {target}')
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
      print(f'  {f:42s} {med:7.3f} {min(aucs):6.3f} {max(aucs):6.3f}'
            f'  {"  <-- closest" if best is None or abs(med-target) < abs(best[1]-target) else ""}')
      if best is None or abs(med - target) < abs(best[1] - target):
        best = (f, med)
    if best:
      print(f'  closest to the paper: {best[0]}  median {best[1]:.3f} '
            f'(target {target}, delta {best[1]-target:+.3f})')
  print('\nNOTE: the paper does not name the exact feature in the reachable text,')
  print('so "closest" is a match by value, not a confirmed identification.')
  return 0


if __name__ == '__main__':
  sys.exit(main())
