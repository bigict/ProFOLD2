#!/usr/bin/env python3
"""Cross-validate the mining metrics against the authoritative known fold-switcher
list (Chakravarty 2024 TableS1 = Porter & Looger ~96 pairs: Fold1/Fold2 PDB_chain +
switching-region sequence). For each known pair, extract backbone from the local
archive, compare on common modeled residues, classify_pair -> recall tally.

Run:
    PYTHONPATH=. python tools/foldswitch_probe/cross_validate.py
"""
import os
import sys
import json
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stage2_mine import extract_backbone, common_backbone, is_antibody  # noqa: E402
from stage2_metrics import classify_pair  # noqa: E402

TABLE = '/tmp/fsprobe/TableS1.xlsx'
OUT = os.path.join(HERE, 'mining')
NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'


def parse_table(path):
  z = zipfile.ZipFile(path)
  ss = []
  if 'xl/sharedStrings.xml' in z.namelist():
    root = ET.fromstring(z.read('xl/sharedStrings.xml'))
    for si in root.findall(NS + 'si'):
      ss.append(''.join(t.text or '' for t in si.iter(NS + 't')))
  root = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
  rows = []
  for row in root.iter(NS + 'row'):
    cells = []
    for c in row.findall(NS + 'c'):
      v = c.find(NS + 'v')
      if v is None:
        cells.append('')
      elif c.get('t') == 's':
        cells.append(ss[int(v.text)])
      else:
        cells.append(v.text)
    rows.append(cells)
  pairs = []
  for r in rows[1:]:
    if len(r) < 2 or not r[0] or not r[1]:
      continue
    f1, f2 = r[0].strip(), r[1].strip()
    seq = r[2].strip() if len(r) > 2 else ''
    pairs.append(dict(fold1=f1, fold2=f2, pdb1=f1[:-1], chain1=f1[-1],
                      pdb2=f2[:-1], chain2=f2[-1], region=seq))
  return pairs


def main():
  pairs = parse_table(TABLE)
  print('known fold-switcher pairs (TableS1): %d' % len(pairs))
  with open(os.path.join(OUT, 'known_switchers.json'), 'w') as f:
    json.dump(pairs, f, indent=1)

  results = []
  for p in pairs:
    bb1 = extract_backbone(p['pdb1'], p['chain1'])
    bb2 = extract_backbone(p['pdb2'], p['chain2'])
    if bb1 is None or bb2 is None:
      results.append(dict(**p, status='unavailable',
                          miss=(p['pdb1'] if bb1 is None else '') + '/' +
                               (p['pdb2'] if bb2 is None else '')))
      continue
    cb = common_backbone(bb1, bb2)
    if cb is None:
      results.append(dict(**p, status='no_common'))
      continue
    bb0, bb1c, res = cb
    m = classify_pair(bb0, bb1c)
    ab = is_antibody(p['pdb1']) or is_antibody(p['pdb2'])
    results.append(dict(**p, status='ok', n_common=len(res), antibody=ab, **m))

  with open(os.path.join(OUT, 'cross_validation.json'), 'w') as f:
    json.dump(results, f, indent=1)

  ok = [r for r in results if r['status'] == 'ok']
  labels = Counter(r['label'] for r in ok)
  print('available pairs: %d / %d (unavailable %d, no_common %d)' %
        (len(ok), len(pairs),
         sum(r['status'] == 'unavailable' for r in results),
         sum(r['status'] == 'no_common' for r in results)))
  print('label distribution on known switchers:', dict(labels))
  rec = labels.get('fold_switch', 0)
  print('RECALL (fold_switch): %d / %d available = %.1f%%' %
        (rec, len(ok), 100 * rec / max(len(ok), 1)))
  print('\n--- MISSED known switchers (not fold_switch) ---')
  for r in ok:
    if r['label'] != 'fold_switch':
      print('  %s/%s n=%d TM=%.3f SS=%.2f HE=%.2f rigid=%s disp=%.1f -> %s%s' %
            (r['fold1'], r['fold2'], r['n_common'], r['tm'], r['ss_change'],
             r['he_change'], r['rigidity'], r['max_disp'], r['label'],
             ' [AB]' if r.get('antibody') else ''))
  print('\n--- RECOVERED (fold_switch) disp_tag distribution ---')
  print(dict(Counter(r['disp_tag'] for r in ok if r['label'] == 'fold_switch')))


if __name__ == '__main__':
  main()