#!/usr/bin/env python3
"""Extract a fold-switching backbone pair from two structures into probe npz format.

Matches residues by (PDB index, amino acid), requires full backbone (N/CA/C/O) in
both, saves {seq, fold0, fold1 (n,4,3), pdb0, pdb1, idx0, idx1, ca_rmsd} — same
schema as xcl1_pair.npz so probe_v4/v5/v6 consume it unchanged.

Example (RfaH alpha-CTD 2OUG vs beta-CTD 6C6S):
    PYTHONPATH=. python tools/foldswitch_probe/extract_pair.py \
        --struct0 /tmp/fsprobe/2oug.cif --chain0 A \
        --struct1 /tmp/fsprobe/6c6s.cif --chain1 D \
        --out tools/foldswitch_probe/data/rfah_pair.npz
"""
import os
import sys
import argparse

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import three_to_index, index_to_one

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from probe_v4 import switching_field, kabsch_rmsd  # noqa: E402

BB = ('N', 'CA', 'C', 'O')


def get_chain(path, chain):
  parser = MMCIFParser(QUIET=True) if path.endswith('.cif') else PDBParser(QUIET=True)
  struct = parser.get_structure('x', path)
  ch = struct[0][chain]
  seq, idx, bb = [], [], []
  for r in ch:
    if r.id[0] != ' ':
      continue
    try:
      aa = index_to_one(three_to_index(r.resname))
    except KeyError:
      continue
    if all(a in r for a in BB):
      seq.append(aa)
      idx.append(r.id[1])
      bb.append([r[a].coord for a in BB])
  return ''.join(seq), np.array(idx), np.array(bb, dtype=np.float32)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--struct0', required=True)
  ap.add_argument('--chain0', required=True)
  ap.add_argument('--struct1', required=True)
  ap.add_argument('--chain1', required=True)
  ap.add_argument('--out', required=True)
  args = ap.parse_args()

  s0, i0, b0 = get_chain(args.struct0, args.chain0)
  s1, i1, b1 = get_chain(args.struct1, args.chain1)
  print('struct0 %s chain %s: %d res (%d-%d)' %
        (args.struct0, args.chain0, len(s0), i0[0], i0[-1]))
  print('struct1 %s chain %s: %d res (%d-%d)' %
        (args.struct1, args.chain1, len(s1), i1[0], i1[-1]))

  m0 = {int(i): (a, b) for i, a, b in zip(i0, s0, b0)}
  m1 = {int(i): (a, b) for i, a, b in zip(i1, s1, b1)}
  common = sorted(set(m0) & set(m1))
  matched = [c for c in common if m0[c][0] == m1[c][0]]
  mismatch = [c for c in common if m0[c][0] != m1[c][0]]
  if mismatch:
    print('WARNING: %d AA mismatches at common indices: %s' % (len(mismatch), mismatch[:20]))
  seq = ''.join(m0[c][0] for c in matched)
  fold0 = np.array([m0[c][1] for c in matched])
  fold1 = np.array([m1[c][1] for c in matched])
  idx = np.array(matched)
  ca_rmsd = kabsch_rmsd(fold0[:, 1], fold1[:, 1])

  field, core, disp = switching_field(fold0, fold1)
  sw = idx[field > 0.5]
  print('matched %d res, global CA RMSD %.2f A' % (len(matched), ca_rmsd))
  print('switching field: core=%d res, field>0.5: %d res at PDB idx %s' %
        (core.sum(), len(sw), sw.tolist()))
  print('disp mean %.2f max %.2f A' % (disp.mean(), disp.max()))

  pdb0 = os.path.splitext(os.path.basename(args.struct0))[0].upper()
  pdb1 = os.path.splitext(os.path.basename(args.struct1))[0].upper()
  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  np.savez(args.out, seq=np.array(seq), fold0=fold0, fold1=fold1,
           pdb0=np.array(pdb0), pdb1=np.array(pdb1),
           idx0=idx, idx1=idx, ca_rmsd=np.array(ca_rmsd))
  print('saved %s' % args.out)


if __name__ == '__main__':
  main()