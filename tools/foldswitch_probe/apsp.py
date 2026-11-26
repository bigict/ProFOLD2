#!/usr/bin/env python3
"""Exact All-Pairs Suffix-Prefix (APSP) via k-mer seed-and-extend.

Finds all ordered pairs (i, j) where a suffix of S_i of length >= ell equals a
prefix of S_j (construct-boundary overlap / containment). Same semantics as the
GSA+LCP APSP (Louza/Ohlebusch-Gog) but simpler and trivially scalable to 200k+
strings (dict index, no separator/alphabet machinery).

Correctness: any overlap of length L >= ell contains the boundary ell-mer match
S_i[-ell:] == S_j[:ell], so seeding on prefix-ell-mers and querying suffix-ell-mers
has NO false negatives; exact extension then yields the maximal overlap length.

Near-exact (mutants/tags with mismatches) is handled separately by edlib extension
of the exact seed core (see extend_near_exact).
"""
from collections import defaultdict

import ahocorasick


def apsp_seed_extend(seqs, ell):
  """Return {(i, j): max_overlap_len} for all suffix-prefix overlaps >= ell.

  seqs: list of str. i != j; (i, j) means suffix of seqs[i] == prefix of seqs[j].

  Seed: the prefix-ell-mer S_j[:ell] must occur in S_i at position q = len_i - L
  for an overlap of length L. An Aho-Corasick automaton of all prefix-ell-mers
  finds every such (q, j) in O(total length); exact extension verifies the full
  suffix S_i[q:] == S_j[:L]. No false negatives (any L >= ell contains the seed).
  """
  k = len(seqs)
  lengths = [len(s) for s in seqs]
  # prefix-ell-mer -> list of string ids having that prefix
  prefix_ids = defaultdict(list)
  for j, s in enumerate(seqs):
    if lengths[j] >= ell:
      prefix_ids[s[:ell]].append(j)

  automaton = ahocorasick.Automaton()
  for key, ids in prefix_ids.items():
    automaton.add_word(key, ids)
  automaton.make_automaton()

  overlaps = {}
  for i, s in enumerate(seqs):
    if lengths[i] < ell:
      continue
    best = {}  # j -> max L found for this i
    # iter yields (end_index, ids); match occupies [end-ell+1, end]
    for end_index, ids in automaton.iter(s):
      q = end_index - ell + 1
      L = lengths[i] - q  # suffix of s from q to its end
      if L < ell:
        continue
      suffix = s[q:]
      for j in ids:
        if j == i or L > lengths[j]:
          continue
        if j in best:  # already found a longer overlap (smaller q scanned earlier)
          continue
        if suffix == seqs[j][:L]:
          best[j] = L
    for j, L in best.items():
      overlaps[(i, j)] = L
  return overlaps


def apsp_bruteforce(seqs, ell):
  """Reference O(k^2 * L) pairwise APSP for correctness verification."""
  overlaps = {}
  k = len(seqs)
  for i in range(k):
    si = seqs[i]
    if len(si) < ell:
      continue
    for j in range(k):
      if i == j:
        continue
      sj = seqs[j]
      maxL = min(len(si), len(sj))
      if maxL < ell:
        continue
      # find largest L in [ell, maxL] with si[-L:] == sj[:L]
      best = 0
      for L in range(ell, maxL + 1):
        if si[len(si) - L:] == sj[:L]:
          best = L
      if best:
        overlaps[(i, j)] = best
  return overlaps


def group_overlaps(seqs, ell):
  """Union-find groups of strings connected by suffix-prefix overlap >= ell."""
  overlaps = apsp_seed_extend(seqs, ell)
  parent = list(range(len(seqs)))

  def find(x):
    while parent[x] != x:
      parent[x] = parent[parent[x]]
      x = parent[x]
    return x

  for (i, j) in overlaps:
    ri, rj = find(i), find(j)
    if ri != rj:
      parent[ri] = rj
  groups = defaultdict(list)
  for i in range(len(seqs)):
    groups[find(i)].append(i)
  return overlaps, dict(groups)


if __name__ == '__main__':
  import random
  # correctness test vs brute force on random + crafted cases
  random.seed(0)
  craft = [
      'MKTLLILAVGGSQRST',     # 0 full
      'LLILAVGGSQRSTWW',      # 1 suffix-of-0 overlap + extra
      'MKTLLILAV',            # 2 prefix-of-0 (0 suffix? no: 2 is prefix of 0)
      'GGSQRSTWWYYYY',        # 3 overlaps 1
      'AVGGSQRST',            # 4 contained suffix region
      'MKTLLILAVGGSQRST',     # 5 duplicate of 0
      'ZZZZZZZZZZ',           # 6 unrelated
  ]
  for trial in range(200):
    if trial < 1:
      seqs = craft
    else:
      alpha = 'ACDEFGHIKLMNPQRSTVWY'
      base = [''.join(random.choice(alpha) for _ in range(random.randint(8, 30)))
              for _ in range(random.randint(3, 8))]
      # inject overlaps: make some strings suffix/prefix of others
      seqs = list(base)
      for _ in range(random.randint(1, 4)):
        a = random.choice(base)
        cut = random.randint(3, len(a))
        seqs.append(a[cut:] + ''.join(random.choice(alpha) for _ in range(random.randint(0, 8))))
        seqs.append(''.join(random.choice(alpha) for _ in range(random.randint(0, 8))) + a[:cut])
    ell = random.choice([3, 4, 5])
    fast = apsp_seed_extend(seqs, ell)
    slow = apsp_bruteforce(seqs, ell)
    if fast != slow:
      print('MISMATCH ell=%d' % ell)
      print(' fast-only:', {k: v for k, v in fast.items() if k not in slow})
      print(' slow-only:', {k: v for k, v in slow.items() if k not in fast})
      print(' seqs:', seqs)
      raise SystemExit(1)
  print('APSP seed-extend == brute force on 200 random+crafted trials (ell=3,4,5). OK')
  ov, gr = group_overlaps(craft, 5)
  print('craft overlaps (ell=5):', ov)
  print('craft groups:', {k: v for k, v in gr.items() if len(v) > 1})