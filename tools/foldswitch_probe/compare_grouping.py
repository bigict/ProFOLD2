#!/usr/bin/env python3
"""Compare APSP vs MMseqs2 grouping for fold-switcher mining + union complementarity.

Loads foldswitch_apsp.json + foldswitch_mmseqs.json (same criterion swSS>=0.30 +
max_disp>=12, reps_per_seq=4, max_reps=30, group/cluster size 2-20) and the known
switcher list (TableS1). Reports candidate stats, known exact-PDB-pair recall,
Mad2/RfaH/KaiB recovery, and union complementarity (APSP-only / MMseqs-only / shared).

Run:
    PYTHONPATH=. python tools/foldswitch_probe/compare_grouping.py
"""
import os
import json
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'mining')


def load(name):
  p = os.path.join(OUT, name)
  return json.load(open(p)) if os.path.exists(p) else None


def pkey(r):
  return tuple(sorted((r['pdb0'].lower(), r['pdb1'].lower())))


def stats(c, label):
  keys = set(pkey(r) for r in c)
  print('  %-10s pairs=%d unique-PDB-pairs=%d disp=%s' %
        (label, len(c), len(keys), dict(Counter(r['disp_tag'] for r in c))))
  return keys


def recall(keys, known):
  rec = []
  for k in known:
    pk = tuple(sorted((k['pdb1'].lower(), k['pdb2'].lower())))
    if pk in keys:
      rec.append(pk)
  return rec


def main():
  apsp = load('foldswitch_apsp.json')
  mm = load('foldswitch_mmseqs.json')
  known = load('known_switchers.json')
  if apsp is None:
    print('APSP results not ready (foldswitch_apsp.json missing)')
    return
  print('=== candidate stats ===')
  ak = stats(apsp, 'APSP')
  mk = stats(mm, 'MMseqs95')
  union = ak | mk
  print('  %-10s unique-PDB-pairs=%d' % ('UNION', len(union)))

  print('\n=== known switcher exact-PDB-pair recall (of %d) ===' % len(known))
  ar = recall(ak, known); mr = recall(mk, known); ur = recall(union, known)
  print('  APSP   : %d/%d = %.1f%%' % (len(ar), len(known), 100 * len(ar) / len(known)))
  print('  MMseqs : %d/%d = %.1f%%' % (len(mr), len(known), 100 * len(mr) / len(known)))
  print('  UNION  : %d/%d = %.1f%%' % (len(ur), len(known), 100 * len(ur) / len(known)))

  print('\n=== union complementarity (unique PDB pairs) ===')
  print('  APSP-only  : %d' % len(ak - mk))
  print('  MMseqs-only: %d' % len(mk - ak))
  print('  shared     : %d' % len(ak & mk))
  aset = set(ar); mset = set(mr)
  print('  known recovered APSP-only  : %d' % len(aset - mset))
  print('  known recovered MMseqs-only: %d' % len(mset - aset))
  print('  known recovered by UNION only (neither alone): %d' %
        (len(ur) - len(aset | mset)))

  print('\n=== landmark recovery ===')
  for tag, p1, p2 in [('Mad2 O/C', '2v64', '2vfx'), ('Mad2 O/C-alt', '2v64', '1go4'),
                      ('RfaH a/b', '2oug', '6c6s'), ('KaiB', '4kso', '5n8y')]:
    pk = tuple(sorted((p1, p2)))
    print('  %-14s APSP=%s MMseqs=%s' % (tag, pk in ak, pk in mk))

  print('\n=== known recovered by UNION ===')
  print('  ' + ', '.join('%s/%s' % p for p in sorted(ur)))


if __name__ == '__main__':
  main()