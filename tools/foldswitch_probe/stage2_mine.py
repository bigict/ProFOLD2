#!/usr/bin/env python3
"""Stage 2 group-level fold-switch mining.

Primary source: unique sequences with >1 PDB entry (same sequence, different
conformations). For each, extract representative structures (raw mmCIF backbone),
compare pairwise on common modeled residues, classify via stage2_metrics
(fixed TM-score + SS change + rigid-body displacement tag).

Antibody/common-protein noise is handled by the metric filter (same fold -> high TM
-> fluctuation; pure domain motion -> SS preserved -> domain_motion) plus a cap on
representatives per sequence to bound compute.

Run:
    PYTHONPATH=. python tools/foldswitch_probe/stage2_mine.py --validate   # known switchers
    PYTHONPATH=. python tools/foldswitch_probe/stage2_mine.py --mine --max-reps 8 --limit 2000
"""
import os
import sys
import json
import argparse
import zipfile
import re
from collections import defaultdict

import numpy as np
import edlib
from Bio.Align import PairwiseAligner

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stage2_metrics import classify_pair  # noqa: E402
from fast_extract import extract_backbone_fast, is_antibody_fast  # noqa: E402

ARCHIVE = os.path.expanduser('~/shared_hosts/pdb_train.zip')
OUT = os.path.join(HERE, 'mining')
_BB_CACHE = {}


def extract_backbone(pdb, chain):
  """Cached fast mmCIF backbone extraction: {resid: (aa, backbone(4,3))}."""
  key = (pdb.lower(), chain)
  if key not in _BB_CACHE:
    _BB_CACHE[key] = extract_backbone_fast(pdb, chain, ARCHIVE)
  return _BB_CACHE[key]


def is_antibody(pdb):
  """Cached antibody detection (shares decompression with backbone extraction)."""
  return is_antibody_fast(pdb, ARCHIVE)


def parse_ids(ids_str):
  """'2oug_A,2oug_B,...' -> list of (pdb, chain), ALL chains kept.

  Dedup by (pdb, chain), NOT by pdb — same-PDB multi-chain fold-switch pairs
  (crystal asymmetric unit two conformations, e.g. 1mnm C/D) must retain both
  chains. Per-PDB dedup dropped them (recall-gap root cause, 2026-11-24).
  """
  seen = {}
  for tok in ids_str.split(','):
    tok = tok.strip()
    if '_' not in tok:
      continue
    pdb, chain = tok.split('_', 1)
    seen[(pdb, chain)] = True
  return list(seen.keys())


def common_backbone(bb_a, bb_b):
  """Common modeled residues (same resid + same AA), ordered; return (bb0, bb1, resids)."""
  common = sorted(set(bb_a) & set(bb_b))
  common = [r for r in common if bb_a[r][0] == bb_b[r][0]]
  if len(common) < 30:
    return None
  bb0 = np.array([bb_a[r][1] for r in common])
  bb1 = np.array([bb_b[r][1] for r in common])
  return bb0, bb1, common


def mine_sequence(idx, seqs, ids, max_reps=8):
  """Mine fold-switch pairs among PDB entries of one unique sequence."""
  entries = parse_ids(ids[idx])[:max_reps]
  backbones = []
  for pdb, chain in entries:
    bb = extract_backbone(pdb, chain)  # also caches antibody flag (one decompress)
    if bb and len(bb) >= 30 and not is_antibody(pdb):
      backbones.append((pdb, chain, bb))
  results = []
  for a in range(len(backbones)):
    for b in range(a + 1, len(backbones)):
      pa, ca, bba = backbones[a]
      pb, cb, bbb = backbones[b]
      cb_pair = common_backbone(bba, bbb)
      if cb_pair is None:
        continue
      bb0, bb1, resids = cb_pair
      m = classify_pair(bb0, bb1)
      if m['label'] == 'fold_switch':
        results.append(dict(idx=int(idx), pdb0=pa, chain0=ca, pdb1=pb, chain1=cb,
                            n_common=len(resids), **m))
  return results


def get_rep(idx, ids):
  """One representative modeled structure per unique sequence."""
  for pdb, chain in parse_ids(ids[idx]):
    bb = extract_backbone(pdb, chain)
    if bb and len(bb) >= 30 and not is_antibody(pdb):
      resids = sorted(bb)
      mseq = ''.join(bb[r][0] for r in resids)
      mbb = np.array([bb[r][1] for r in resids])
      return pdb, chain, mseq, mbb
  return None


def align_common(mseq_a, mbb_a, mseq_b, mbb_b):
  """Align modeled sequences (edlib, fast C++); return common-region backbone pair.

  edlib global (NW) alignment of near-identical cluster members; parse CIGAR for
  matched columns. ~10-100x faster than Bio PairwiseAligner (the mining bottleneck).
  """
  r = edlib.align(mseq_a, mseq_b, mode='NW', task='path')
  cigar = r.get('cigar')
  if not cigar:
    return None
  ia, ib = [], []
  pa = pb = 0
  for num, op in re.findall(r'(\d+)([MID=X])', cigar):
    num = int(num)
    if op in 'M=X':
      ia.extend(range(pa, pa + num))
      ib.extend(range(pb, pb + num))
      pa += num
      pb += num
    elif op == 'I':
      pa += num
    elif op == 'D':
      pb += num
  if len(ia) < 30:
    return None
  return mbb_a[ia], mbb_b[ib], len(ia)


def mine_group(gi, seqs, ids):
  """Cross-sequence fold-switch mining within an APSP group (construct variants)."""
  reps = []
  for idx in gi:
    r = get_rep(idx, ids)
    if r:
      reps.append((idx,) + r)
  results = []
  for a in range(len(reps)):
    for b in range(a + 1, len(reps)):
      if reps[a][0] == reps[b][0]:
        continue
      cb = align_common(reps[a][3], reps[a][4], reps[b][3], reps[b][4])
      if cb is None:
        continue
      bb0, bb1, n = cb
      m = classify_pair(bb0, bb1)
      if m['label'] == 'fold_switch':
        results.append(dict(idx0=int(reps[a][0]), idx1=int(reps[b][0]),
                            pdb0=reps[a][1], chain0=reps[a][2],
                            pdb1=reps[b][1], chain1=reps[b][2],
                            n_common=n, **m))
  return results


def mine_groups(max_group_size=30, limit=0):
  seqs, ids = load_data()
  g = json.load(open(os.path.join(OUT, 'groups.json')))
  groups = [[m['idx'] for m in members] for members in g.values()]
  groups = [gi for gi in groups if 2 <= len(gi) <= max_group_size]
  print('groups (size 2-%d): %d' % (max_group_size, len(groups)))
  if limit:
    groups = groups[:limit]
  all_res = []
  for n, gi in enumerate(groups):
    all_res.extend(mine_group(gi, seqs, ids))
    if (n + 1) % 500 == 0:
      print('  processed %d/%d groups, cross-seq fold_switch pairs: %d' %
            (n + 1, len(groups), len(all_res)), flush=True)
  with open(os.path.join(OUT, 'foldswitch_crossseq.json'), 'w') as f:
    json.dump(all_res, f, indent=1)
  print('TOTAL cross-seq fold_switch pairs: %d -> %s/foldswitch_crossseq.json' %
        (len(all_res), OUT))


def get_reps(idx, ids, k=4):
  """Up to k representative modeled structures for one unique sequence.

  k=None returns ALL entries (full retention / all-pairs detection).
  """
  reps = []
  for pdb, chain in parse_ids(ids[idx]):
    bb = extract_backbone(pdb, chain)
    if bb and len(bb) >= 30 and not is_antibody(pdb):
      resids = sorted(bb)
      mseq = ''.join(bb[r][0] for r in resids)
      mbb = np.array([bb[r][1] for r in resids])
      reps.append(dict(idx=idx, pdb=pdb, chain=chain, mseq=mseq, mbb=mbb, bb=bb))
      if k is not None and len(reps) >= k:
        break
  return reps


def cluster_entry_count(ci, ids):
  return sum(len(parse_ids(ids[idx])) for idx in ci)


def mine_cluster(ci, seqs, ids, reps_per_seq=4, max_reps=30, all_pairs_threshold=40):
  """Fold-switch mining within an MMseqs2 cluster (within-seq + cross-seq).

  Small clusters (total PDB entries <= all_pairs_threshold): ALL entries compared
  (no rep sampling) so no known pair is missed. Large clusters: capped reps for
  compute (mostly non-switcher fluctuations). Retention is handled by the caller.
  """
  total_entries = cluster_entry_count(ci, ids)
  if total_entries <= all_pairs_threshold:
    k, cap = None, None      # all-pairs, full coverage
  else:
    k, cap = reps_per_seq, max_reps
  reps = []
  for idx in ci:
    for r in get_reps(idx, ids, k=k):
      reps.append(r)
      if cap is not None and len(reps) >= cap:
        break
    if cap is not None and len(reps) >= cap:
      break
  results = []
  for a in range(len(reps)):
    for b in range(a + 1, len(reps)):
      ra, rb = reps[a], reps[b]
      if ra['idx'] == rb['idx']:
        # same unique sequence: fast common-residue intersection (no alignment)
        cb = common_backbone(ra['bb'], rb['bb'])
      else:
        cb = align_common(ra['mseq'], ra['mbb'], rb['mseq'], rb['mbb'])
      if cb is None:
        continue
      bb0, bb1, n = cb
      m = classify_pair(bb0, bb1)
      if m['label'] == 'fold_switch':
        results.append(dict(idx0=int(ra['idx']), idx1=int(rb['idx']),
                            pdb0=ra['pdb'], chain0=ra['chain'],
                            pdb1=rb['pdb'], chain1=rb['chain'],
                            n_common=n, **m))
  return results


def mine_mmseqs(cluster_tsv, max_cluster_size=50, reps_per_seq=4, max_reps=30, limit=0,
                out_name='foldswitch_mmseqs.json', all_pairs_threshold=40):
  seqs, ids = load_data()
  clu = defaultdict(list)
  for line in open(cluster_tsv):
    rep, mem = line.strip().split('\t')
    clu[rep].append(int(mem))
  clusters = []
  n_single = 0
  for m in clu.values():
    if len(m) == 1:
      # singleton cluster: include only if the seq has >=2 PDB entries
      # (within-sequence multistate, e.g. KaiB 4kso/5n8y) — fixes size>=2 gap
      if len(parse_ids(ids[m[0]])) >= 2:
        clusters.append(m)
        n_single += 1
    elif 2 <= len(m) <= max_cluster_size:
      clusters.append(m)
  print('MMseqs2 clusters: %d multi-member (size 2-%d) + %d singleton-multi-entry = %d' %
        (len(clusters) - n_single, max_cluster_size, n_single, len(clusters)))
  print('all-pairs threshold: clusters with <= %d entries compared fully (no rep sampling)' %
        all_pairs_threshold)
  if limit:
    clusters = clusters[:limit]
  all_res = []
  switcher_clusters = []   # full retention: every member seq + ALL PDB entries
  for n, ci in enumerate(clusters):
    res = mine_cluster(ci, seqs, ids, reps_per_seq, max_reps, all_pairs_threshold)
    if res:
      all_res.extend(res)
      switcher_clusters.append(dict(
          members=[dict(idx=int(idx), entries=parse_ids(ids[idx])) for idx in ci],
          n_pairs=len(res)))
    if (n + 1) % 500 == 0:
      print('  processed %d/%d clusters, fold_switch pairs: %d, switcher clusters: %d' %
            (n + 1, len(clusters), len(all_res), len(switcher_clusters)), flush=True)
  with open(os.path.join(OUT, out_name), 'w') as f:
    json.dump(all_res, f, indent=1)
  clu_name = out_name.replace('foldswitch_', 'switcher_clusters_')
  with open(os.path.join(OUT, clu_name), 'w') as f:
    json.dump(switcher_clusters, f, indent=1)
  print('TOTAL MMseqs2-cluster fold_switch pairs: %d -> %s/%s' %
        (len(all_res), OUT, out_name))
  print('switcher clusters (full retention): %d -> %s/%s' %
        (len(switcher_clusters), OUT, clu_name))
  by_tag = defaultdict(int)
  for r in all_res:
    by_tag[r['disp_tag']] += 1
  print('by disp_tag:', dict(by_tag))


def mine_apsp(max_group_size=20, reps_per_seq=4, max_reps=30, limit=0):
  """Same mining logic as mine_mmseqs but on APSP groups (for comparison)."""
  seqs, ids = load_data()
  g = json.load(open(os.path.join(OUT, 'groups.json')))
  groups = [[m['idx'] for m in members] for members in g.values()]
  groups = [gi for gi in groups if 2 <= len(gi) <= max_group_size]
  print('APSP groups (size 2-%d): %d' % (max_group_size, len(groups)))
  if limit:
    groups = groups[:limit]
  all_res = []
  for n, ci in enumerate(groups):
    all_res.extend(mine_cluster(ci, seqs, ids, reps_per_seq, max_reps))
    if (n + 1) % 200 == 0:
      print('  processed %d/%d groups, fold_switch pairs: %d' %
            (n + 1, len(groups), len(all_res)), flush=True)
  with open(os.path.join(OUT, 'foldswitch_apsp.json'), 'w') as f:
    json.dump(all_res, f, indent=1)
  print('TOTAL APSP-group fold_switch pairs: %d -> %s/foldswitch_apsp.json' %
        (len(all_res), OUT))
  by_tag = defaultdict(int)
  for r in all_res:
    by_tag[r['disp_tag']] += 1
  print('by disp_tag:', dict(by_tag))


def load_data():
  d = np.load(os.path.join(OUT, 'seqs.npz'), allow_pickle=True)
  return list(d['seqs']), list(d['ids'])


def validate():
  seqs, ids = load_data()
  # known switcher unique-seq indices: RfaH 30498 (2oug vs 6c6s), Mad2 5603, KaiB 62566
  for idx in [30498, 5603, 62566]:
    print('=== idx %d len %d ids %s ===' % (idx, len(seqs[idx]), ids[idx][:60]))
    res = mine_sequence(idx, seqs, ids, max_reps=10)
    if not res:
      print('  no fold_switch pairs found')
    for r in res:
      print('  %s_%s vs %s_%s n=%d TM=%.3f SS=%.2f max_disp=%.1f -> %s [%s]' %
            (r['pdb0'], r['chain0'], r['pdb1'], r['chain1'], r['n_common'],
             r['tm'], r['ss_change'], r['max_disp'], r['label'], r['disp_tag']))


def mine(max_reps, limit):
  seqs, ids = load_data()
  # multi-entry sequences only
  multi = [i for i in range(len(seqs)) if len(parse_ids(ids[i])) > 1]
  print('multi-entry sequences: %d' % len(multi))
  if limit:
    multi = multi[:limit]
  all_res = []
  for n, idx in enumerate(multi):
    res = mine_sequence(idx, seqs, ids, max_reps=max_reps)
    all_res.extend(res)
    if (n + 1) % 200 == 0:
      print('  processed %d/%d seqs, fold_switch pairs so far: %d' %
            (n + 1, len(multi), len(all_res)), flush=True)
  with open(os.path.join(OUT, 'foldswitch_candidates.json'), 'w') as f:
    json.dump(all_res, f, indent=1)
  print('TOTAL fold_switch pairs: %d -> %s/foldswitch_candidates.json' % (len(all_res), OUT))
  by_tag = defaultdict(int)
  for r in all_res:
    by_tag[r['disp_tag']] += 1
  print('by disp_tag:', dict(by_tag))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--validate', action='store_true')
  ap.add_argument('--mine', action='store_true')
  ap.add_argument('--mine-groups', action='store_true')
  ap.add_argument('--mine-mmseqs', action='store_true')
  ap.add_argument('--mine-apsp', action='store_true')
  ap.add_argument('--cluster-tsv', default='/tmp/fsprobe/mmseqs_clu95_cluster.tsv')
  ap.add_argument('--out-name', default='foldswitch_mmseqs.json')
  ap.add_argument('--all-pairs-threshold', type=int, default=40)
  ap.add_argument('--max-cluster-size', type=int, default=50)
  ap.add_argument('--reps-per-seq', type=int, default=4)
  ap.add_argument('--max-reps', type=int, default=8)
  ap.add_argument('--max-group-size', type=int, default=30)
  ap.add_argument('--limit', type=int, default=0)
  args = ap.parse_args()
  if args.validate:
    validate()
  elif args.mine:
    mine(args.max_reps, args.limit)
  elif args.mine_groups:
    mine_groups(args.max_group_size, args.limit)
  elif args.mine_mmseqs:
    mine_mmseqs(args.cluster_tsv, args.max_cluster_size, args.reps_per_seq,
                args.max_reps, args.limit, args.out_name, args.all_pairs_threshold)
  elif args.mine_apsp:
    mine_apsp(args.max_cluster_size, args.reps_per_seq, args.max_reps, args.limit)


if __name__ == '__main__':
  main()