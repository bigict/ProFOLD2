#!/usr/bin/env python3
"""Extract fold0/fold1 template structures from mined switchers for AFDB search.

For each switcher in hybrid_switchers.json, take a representative structure per
conformation (fold) and write its backbone as a PDB query file for Foldseek
search against AFDB. Outputs a template dir + manifest mapping each template to
(switcher seq_rep, fold label, source pdb/chain).

Run:
    PYTHONPATH=. python tools/foldswitch_probe/extract_templates.py \
        --switchers tools/foldswitch_probe/mining/hybrid_switchers.json \
        --out /tmp/fsprobe/afdb_templates [--max-switchers N]
"""
import os
import sys
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fast_extract import extract_backbone_fast as extract_backbone  # noqa: E402

ARCHIVE = os.path.expanduser('~/shared_hosts/pdb_train.zip')
BB = ('N', 'CA', 'C', 'O')


def write_pdb(bb, path, name):
  with open(path, 'w') as f:
    ai = 1
    for ri, res in enumerate(bb):
      for a, xyz in zip(BB, res):
        f.write('ATOM  %5d  %-3s ALA A%4d    %8.3f%8.3f%8.3f  1.00  0.00           %s\n'
                % (ai, a, ri + 1, xyz[0], xyz[1], xyz[2], a[0]))
        ai += 1
    f.write('TER\nEND\n')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--switchers', default=os.path.join(HERE, 'mining', 'hybrid_switchers.json'))
  ap.add_argument('--out', default='/tmp/fsprobe/afdb_templates')
  ap.add_argument('--max-switchers', type=int, default=0)
  ap.add_argument('--max-folds', type=int, default=2, help='only fold0..foldN-1 templates')
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)

  switchers = json.load(open(args.switchers))
  if args.max_switchers:
    switchers = switchers[:args.max_switchers]
  manifest = []
  n_ok = n_skip = 0
  for si, sw in enumerate(switchers):
    for conf in sw['conformations'][:args.max_folds]:
      fold = conf['fold']
      # representative = first structure of this conformation
      rep = conf['structures'][0]
      bb = extract_backbone(rep['pdb'], rep['chain'], ARCHIVE)
      if bb is None or len(bb) < 30:
        n_skip += 1
        continue
      resids = sorted(bb)
      backbone = np.array([bb[r][1] for r in resids])
      tname = 'sw%d_fold%d' % (sw['seq_rep'], fold)
      write_pdb(backbone, os.path.join(args.out, tname + '.pdb'), tname)
      manifest.append(dict(template=tname, seq_rep=sw['seq_rep'], fold=fold,
                           type=sw['type'], src_pdb=rep['pdb'], src_chain=rep['chain'],
                           n_res=len(resids)))
      n_ok += 1
    if (si + 1) % 1000 == 0:
      print('  processed %d/%d switchers, templates %d' % (si + 1, len(switchers), n_ok), flush=True)
  with open(os.path.join(args.out, 'manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=1)
  print('templates written: %d (skip %d) -> %s' % (n_ok, n_skip, args.out))
  print('manifest: %s/manifest.json' % args.out)


if __name__ == '__main__':
  main()