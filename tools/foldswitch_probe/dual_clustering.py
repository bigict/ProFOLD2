#!/usr/bin/env python3
"""Dual-clustering fold-switch detection (user framework, 2026-11-24).

Sequence clusters (MMseqs2 95%) x conformation clusters (swSS-based).
A sequence cluster spanning >=2 conformation clusters IS a fold-switcher.
Conformation clustering: greedy, two structures are SAME conformation unless
they form a switch pair (swSS>=0.30 AND max_disp>=12). This splits Mad2 O/C
(high swSS) but keeps calmodulin open/closed together (low swSS, rigid motion)
— validated superior to Foldseek fold-level (which misses Mad2 same-fold).

fold0 = majority conformation cluster, fold1 = minority; full retention of every
member sequence + PDB entry per switcher cluster (random draw at training use).

Run:
    PYTHONPATH=. python tools/foldswitch_probe/dual_clustering.py \
        --cluster-tsv /tmp/fsprobe/mmseqs_clu95_cluster.tsv --max-structs 60
"""
import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stage2_mine import (load_data, parse_ids, extract_backbone,  # noqa: E402
                         common_backbone, align_common)
from stage2_metrics import classify_pair  # noqa: E402

OUT = os.path.join(HERE, 'mining')


def is_switch_pair(m):
  return m['label'] == 'fold_switch'


def cluster_conformations(structs, max_compare=60):
  """Greedy conformation clustering. structs: list of dict(idx,pdb,chain,bb,mseq).

  Same conformation unless switch-pair (swSS>=.30 & maxd>=12) vs cluster rep.
  Returns list of clusters, each {rep, members:[structs]}.
  """
  clusters = []
  for s in structs:
    placed = False
    for cl in clusters:
      r = cl['rep']
      if s['idx'] == r['idx']:
        cb = common_backbone(s['bb'], r['bb'])
      else:
        cb = align_common(s['mseq'], s['mbb'], r['mseq'], r['mbb'])
      if cb is None:
        continue
      m = classify_pair(cb[0], cb[1])
      if not is_switch_pair(m):       # similar -> same conformation
        cl['members'].append(s)
        placed = True
        break
    if not placed:
      clusters.append(dict(rep=s, members=[s]))
  return clusters


def gather_structs(ci, ids, max_structs):
  structs = []
  for idx in ci:
    for pdb, chain in parse_ids(ids[idx]):
      bb = extract_backbone(pdb, chain)
      if bb and len(bb) >= 30:
        resids = sorted(bb)
        mseq = ''.join(bb[r][0] for r in resids)
        mbb = np.array([bb[r][1] for r in resids])
        structs.append(dict(idx=int(idx), pdb=pdb, chain=chain,
                            bb=bb, mseq=mseq, mbb=mbb))
        if len(structs) >= max_structs:
          return structs
  return structs


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--cluster-tsv', default='/tmp/fsprobe/mmseqs_clu95_cluster.tsv')
  ap.add_argument('--max-cluster-size', type=int, default=50)
  ap.add_argument('--max-structs', type=int, default=60)
  ap.add_argument('--limit', type=int, default=0)
  ap.add_argument('--out', default='dual_switchers.json')
  args = ap.parse_args()

  seqs, ids = load_data()
  clu = defaultdict(list)
  for line in open(args.cluster_tsv):
    rep, mem = line.strip().split('\t')
    clu[rep].append(int(mem))
  clusters = []
  for m in clu.values():
    if len(m) == 1:
      if len(parse_ids(ids[m[0]])) >= 2:
        clusters.append(m)
    elif 2 <= len(m) <= args.max_cluster_size:
      clusters.append(m)
  print('sequence clusters to process: %d' % len(clusters))
  if args.limit:
    clusters = clusters[:args.limit]

  switchers = []
  n_conf_dist = defaultdict(int)
  for n, ci in enumerate(clusters):
    structs = gather_structs(ci, ids, args.max_structs)
    if len(structs) < 2:
      continue
    confs = cluster_conformations(structs)
    n_conf_dist[len(confs)] += 1
    if len(confs) >= 2:
      # fold0 = majority conformation cluster
      confs_sorted = sorted(confs, key=lambda c: -len(c['members']))
      conf_out = []
      for fold, cl in enumerate(confs_sorted):
        conf_out.append(dict(fold=fold, n=len(cl['members']),
                             structures=[dict(idx=s['idx'], pdb=s['pdb'], chain=s['chain'])
                                         for s in cl['members']]))
      switchers.append(dict(
          members=[dict(idx=int(idx), entries=parse_ids(ids[idx])) for idx in ci],
          n_conformations=len(confs), conformations=conf_out))
    if (n + 1) % 500 == 0:
      print('  processed %d/%d seq-clusters, switchers: %d' %
            (n + 1, len(clusters), len(switchers)), flush=True)

  with open(os.path.join(OUT, args.out), 'w') as f:
    json.dump(switchers, f, indent=1)
  print('conformation-count distribution:', dict(sorted(n_conf_dist.items())))
  print('SWITCHERS (>=2 conformations): %d -> %s/%s' % (len(switchers), OUT, args.out))
  total_entries = sum(len(m['entries']) for s in switchers for m in s['members'])
  print('full-retention entries in switchers: %d' % total_entries)


if __name__ == '__main__':
  main()