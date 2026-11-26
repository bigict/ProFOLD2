#!/usr/bin/env python3
"""Hybrid fold-switch detection: Foldseek fold-level + within-fold swSS supplement.

User framework (2026-11-24): MMseqs sequence clusters x Foldseek structure clusters.
- A sequence cluster spanning >=2 Foldseek fold clusters = fold-level switcher
  (RfaH alpha/beta, KaiB gs/fs, XCL1) — detected by Foldseek, no swSS needed.
- A sequence cluster within ONE fold cluster but with swSS-split conformations
  = same-fold topology switcher (Mad2 O/C) — caught by within-cluster swSS.
fold0 = majority conformation/fold cluster, fold1 = minority; full retention.

Foldseek (C++) does the bulk fold clustering fast; swSS only supplements the
same-fold topology cases. Faster than all-pairs Python greedy.

Run (after Foldseek easy-cluster produces foldseek_clu_cluster.tsv):
    PYTHONPATH=. python tools/foldswitch_probe/hybrid_detect.py \
        --foldseek-tsv /tmp/fsprobe/fs_global_clu_cluster.tsv
"""
import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stage2_mine import load_data, parse_ids, extract_backbone  # noqa: E402
from dual_clustering import cluster_conformations, gather_structs  # noqa: E402

OUT = os.path.join(HERE, 'mining')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--foldseek-tsv', default='/tmp/fsprobe/fs_global_clu_cluster.tsv')
  ap.add_argument('--mmseqs-tsv', default='/tmp/fsprobe/mmseqs_clu95_cluster.tsv')
  ap.add_argument('--max-cluster-size', type=int, default=50)
  ap.add_argument('--max-structs', type=int, default=60)
  ap.add_argument('--limit', type=int, default=0)
  ap.add_argument('--out', default='hybrid_switchers.json')
  args = ap.parse_args()

  seqs, ids = load_data()
  # pdb_chain -> unique-seq idx
  pc2idx = {}
  for i, s in enumerate(ids):
    for tok in s.split(','):
      if '_' in tok:
        pc2idx[tok.lower()] = i
  # idx -> MMseqs seq-cluster rep
  idx2seq = {}
  seq_clusters = defaultdict(list)
  for line in open(args.mmseqs_tsv):
    rep, mem = line.strip().split('\t')
    rep, mem = int(rep), int(mem)
    idx2seq[mem] = rep
    seq_clusters[rep].append(mem)
  # Foldseek structure_id (pdb_chain) -> fold-cluster rep
  pc2fold = {}
  for line in open(args.foldseek_tsv):
    rep, mem = line.strip().split('\t')
    pc2fold[mem.lower()] = rep.lower()
  print('seq clusters: %d | foldseek structure->fold mappings: %d' %
        (len(seq_clusters), len(pc2fold)))

  # restrict to processable seq clusters (multi-member or singleton-multi-entry)
  proc = []
  for rep, members in seq_clusters.items():
    if len(members) == 1:
      if len(parse_ids(ids[members[0]])) >= 2:
        proc.append((rep, members))
    elif 2 <= len(members) <= args.max_cluster_size:
      proc.append((rep, members))
  print('processable seq clusters: %d' % len(proc))
  if args.limit:
    proc = proc[:args.limit]

  switchers = []
  n_foldlevel = n_topology = 0
  for n, (rep, members) in enumerate(proc):
    # gather this seq cluster's structure tokens + their fold clusters
    tokens = []
    for idx in members:
      for pdb, chain in parse_ids(ids[idx]):
        tokens.append((idx, pdb, chain))
    foldset = defaultdict(list)
    for idx, pdb, chain in tokens:
      f = pc2fold.get(('%s_%s' % (pdb, chain)).lower())
      if f is not None:
        foldset[f].append((idx, pdb, chain))
    if len(foldset) >= 2:
      # fold-level switcher: fold0 = majority fold cluster
      folds_sorted = sorted(foldset.items(), key=lambda kv: -len(kv[1]))
      conf_out = [dict(fold=f, fold_cluster=fc, n=len(sts),
                       structures=[dict(idx=i, pdb=p, chain=c) for i, p, c in sts])
                  for f, (fc, sts) in enumerate(folds_sorted)]
      switchers.append(dict(seq_rep=int(rep), type='fold_level',
                            members=[dict(idx=int(i), entries=parse_ids(ids[i])) for i in members],
                            n_conformations=len(foldset), conformations=conf_out))
      n_foldlevel += 1
    elif len(foldset) == 1 and len(tokens) >= 2:
      # single fold cluster: swSS supplement for same-fold topology switch (Mad2)
      structs = gather_structs(members, ids, args.max_structs)
      if len(structs) >= 2:
        confs = cluster_conformations(structs)
        if len(confs) >= 2:
          confs_sorted = sorted(confs, key=lambda c: -len(c['members']))
          conf_out = [dict(fold=f, n=len(c['members']),
                           structures=[dict(idx=s['idx'], pdb=s['pdb'], chain=s['chain'])
                                       for s in c['members']])
                      for f, c in enumerate(confs_sorted)]
          switchers.append(dict(seq_rep=int(rep), type='topology_swSS',
                                members=[dict(idx=int(i), entries=parse_ids(ids[i])) for i in members],
                                n_conformations=len(confs), conformations=conf_out))
          n_topology += 1
    if (n + 1) % 500 == 0:
      print('  processed %d/%d, switchers: %d (fold-level %d, topology %d)' %
            (n + 1, len(proc), len(switchers), n_foldlevel, n_topology), flush=True)

  with open(os.path.join(OUT, args.out), 'w') as f:
    json.dump(switchers, f, indent=1)
  print('SWITCHERS: %d (fold-level %d + topology-swSS %d) -> %s/%s' %
        (len(switchers), n_foldlevel, n_topology, OUT, args.out))
  total_entries = sum(len(m['entries']) for s in switchers for m in s['members'])
  print('full-retention entries: %d' % total_entries)


if __name__ == '__main__':
  main()