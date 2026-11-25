#!/usr/bin/env python3
"""Compare MMseqs mining v1 (size2-20/reps4/max30, no singleton) vs v2
(size2-50/reps8/max50 + singleton-multi-entry) — recall + complementarity.

Run:
    PYTHONPATH=. python tools/foldswitch_probe/compare_v1v2.py
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


def recall(keys, known):
  return [tuple(sorted((k['pdb1'].lower(), k['pdb2'].lower()))) for k in known
          if tuple(sorted((k['pdb1'].lower(), k['pdb2'].lower()))) in keys]


def main():
  v1 = load('foldswitch_mmseqs.json')
  v2 = load('foldswitch_mmseqs_v2.json')
  known = load('known_switchers.json')
  if v2 is None:
    print('v2 not ready (foldswitch_mmseqs_v2.json missing)')
    return
  k1 = set(pkey(r) for r in v1)
  k2 = set(pkey(r) for r in v2)
  print('=== candidate stats ===')
  print('  v1: pairs=%d unique=%d disp=%s' %
        (len(v1), len(k1), dict(Counter(r['disp_tag'] for r in v1))))
  print('  v2: pairs=%d unique=%d disp=%s' %
        (len(v2), len(k2), dict(Counter(r['disp_tag'] for r in v2))))

  r1 = recall(k1, known); r2 = recall(k2, known)
  print('\n=== known exact-PDB-pair recall (of %d) ===' % len(known))
  print('  v1: %d = %.1f%%' % (len(r1), 100 * len(r1) / len(known)))
  print('  v2: %d = %.1f%%' % (len(r2), 100 * len(r2) / len(known)))
  s1, s2 = set(r1), set(r2)
  print('  v2 gained (new knowns): %d' % len(s2 - s1))
  print('  v2 lost: %d' % len(s1 - s2))
  print('  gained knowns: ' + ', '.join('%s/%s' % p for p in sorted(s2 - s1)))

  print('\n=== landmark recovery ===')
  for tag, p1, p2 in [('Mad2 O/C', '2v64', '2vfx'), ('RfaH a/b', '2oug', '6c6s'),
                      ('KaiB', '4kso', '5n8y'), ('CLIC1', '1k0n', '1rk4')]:
    pk = tuple(sorted((p1, p2)))
    print('  %-10s v1=%s v2=%s' % (tag, pk in k1, pk in k2))

  print('\n=== v2-only pairs (new candidates) ===')
  print('  v2-only unique: %d | v1-only: %d | shared: %d' %
        (len(k2 - k1), len(k1 - k2), len(k1 & k2)))


if __name__ == '__main__':
  main()