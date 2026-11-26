#!/usr/bin/env python3
"""Mine fold-switcher candidate pairs from the local PDB train archive.

Stage 1 (this script): extract protein SEQRES from pdb_seqres.txt, dedupe exact
sequences, group same-protein constructs by near-exact suffix-prefix overlap
(APSP seed-and-extend, apsp.py) with threshold ell + coverage filter, and report
group statistics. Downstream stages (conformation TM-score + SS-change filtering)
consume the groups.

Data: ~/shared_hosts/pdb_train.zip (ratarmount-indexed ZIP), entry
pdb_train/pdb_seqres.txt (FASTA-like: '>pdb_chain mol:<type> length:<n> ...').

Run:
    PYTHONPATH=. python tools/foldswitch_probe/mine_switchers.py --stage extract
    PYTHONPATH=. python tools/foldswitch_probe/mine_switchers.py --stage group --ell 50 --cov 0.8
"""
import os
import sys
import argparse
import zipfile
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from apsp import apsp_seed_extend  # noqa: E402

ARCHIVE = os.path.expanduser('~/shared_hosts/pdb_train.zip')
SEQRES = 'pdb_train/pdb_seqres.txt'
OUT = os.path.join(HERE, 'mining')


def parse_seqres(protein_only=True, min_len=30):
  """Yield (pdb_chain, seq) for protein chains from pdb_seqres.txt."""
  z = zipfile.ZipFile(ARCHIVE)
  with z.open(SEQRES) as f:
    header = None
    for raw in f:
      line = raw.decode('utf-8', 'replace').rstrip('\n')
      if line.startswith('>'):
        header = line[1:].split()[0]  # e.g. 100d_A
        mol = 'protein' if 'mol:protein' in line else ('na' if 'mol:na' in line else 'other')
        keep = (not protein_only) or (mol == 'protein')
      elif header is not None and keep and line:
        seq = line.strip()
        if len(seq) >= min_len and set(seq) <= set('ACDEFGHIKLMNPQRSTVWY'):
          yield header, seq
        header = None  # one sequence line per header


def extract():
  os.makedirs(OUT, exist_ok=True)
  chains = []
  by_seq = defaultdict(list)
  lengths = []
  for pdb_chain, seq in parse_seqres():
    chains.append((pdb_chain, seq))
    by_seq[seq].append(pdb_chain)
    lengths.append(len(seq))
  lengths = np.array(lengths)
  print('protein chains (>=30aa, standard AA): %d' % len(chains))
  print('unique sequences: %d' % len(by_seq))
  print('length: min %d median %d mean %.0f max %d' %
        (lengths.min(), int(np.median(lengths)), lengths.mean(), lengths.max()))
  # multi-entry sequences (same seq, >1 PDB entry)
  multi = {s: ids for s, ids in by_seq.items() if len(set(i.split('_')[0] for i in ids)) > 1}
  print('sequences with >1 PDB entry: %d' % len(multi))
  # save unique sequences + mapping
  uniq = list(by_seq.keys())
  np.savez(os.path.join(OUT, 'seqs.npz'),
           seqs=np.array(uniq, dtype=object),
           ids=np.array([','.join(by_seq[s]) for s in uniq], dtype=object))
  print('saved %s/seqs.npz (%d unique seqs)' % (OUT, len(uniq)))
  return uniq, by_seq


def known_switcher_check(uniq):
  # representative sequences of known switchers (approx, for presence check)
  known = {
      'RfaH': 'MQSWYLLYCKRGQLQVKEE',  # E.coli RfaH N-term
      'Mad2': 'MALQLSREQGITLRGSA',  # Mad2 N-term
      'KaiB': 'MSPRKTYILKLYVAGNT',  # KaiB N-term
  }
  for name, probe in known.items():
    hits = [i for i, s in enumerate(uniq) if probe in s]
    print('known %s probe %s: %d unique seqs contain it' % (name, probe, len(hits)))


def group(ell, cov):
  d = np.load(os.path.join(OUT, 'seqs.npz'), allow_pickle=True)
  uniq = list(d['seqs'])
  ids = list(d['ids'])
  print('grouping %d unique seqs, ell=%d cov=%.2f' % (len(uniq), ell, cov))
  overlaps = apsp_seed_extend(uniq, ell)
  print('raw overlaps >= ell: %d' % len(overlaps))
  # coverage filter: overlap >= cov * min(len_i, len_j)
  parent = list(range(len(uniq)))

  def find(x):
    while parent[x] != x:
      parent[x] = parent[parent[x]]
      x = parent[x]
    return x

  kept = 0
  for (i, j), L in overlaps.items():
    if L >= cov * min(len(uniq[i]), len(uniq[j])):
      kept += 1
      ri, rj = find(i), find(j)
      if ri != rj:
        parent[ri] = rj
  print('overlaps after coverage filter: %d' % kept)
  groups = defaultdict(list)
  for i in range(len(uniq)):
    groups[find(i)].append(i)
  multi = {r: g for r, g in groups.items() if len(g) > 1}
  sizes = np.array([len(g) for g in multi.values()])
  print('multi-seq groups: %d' % len(multi))
  if len(sizes):
    print('group size: median %d mean %.1f max %d' %
          (int(np.median(sizes)), sizes.mean(), sizes.max()))
  # save groups (lists of unique-seq indices) + their PDB ids
  import json
  with open(os.path.join(OUT, 'groups.json'), 'w') as f:
    json.dump({str(r): [{'idx': i, 'ids': ids[i]} for i in g] for r, g in multi.items()}, f)
  print('saved %s/groups.json' % OUT)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--stage', choices=['extract', 'group'], default='extract')
  ap.add_argument('--ell', type=int, default=50)
  ap.add_argument('--cov', type=float, default=0.8)
  args = ap.parse_args()
  if args.stage == 'extract':
    uniq, by_seq = extract()
    known_switcher_check(uniq)
  else:
    group(args.ell, args.cov)


if __name__ == '__main__':
  main()