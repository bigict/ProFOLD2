#!/usr/bin/env python3
"""Export the PRP-TCR activation benchmark as a ProTCR-schema CSV.

Source: Wang N, Yeh H, ... Garcia KC, "Deep peptide recognition profiling decodes
TCR specificity and enables disease-associated antigen discovery", Nat Biotechnol
2026, doi 10.1038/s41587-026-03128-x. The two small tables are read straight from
the authors' public GitHub repo; their Dryad deposit (CC0) holds only peptide
enrichment counts, no receptor sequences.

TWO FIELDS ARE DELIBERATELY PARTIAL (user decision 2026-11-24, "没有/alpha没问题,
哪个字段空白字符串即可"):
  TCRA  - empty string. No alpha sequences were released, and the paper measured
          that alpha adds nothing here (median PR AUC 0.85 a+b vs 0.86 b-only,
          p=0.39, n=12).
  TCRB  - CDR3b ONLY, not a full beta chain. Neither GitHub nor Dryad ships full
          chains. `tcr_chain_scope` records this on every row so a partial chain
          is never mistaken for a whole one.
MHC_str is the HLA-B*27:05 sequence lifted from our own verified iedb_I triad
table, so it is byte-identical to the rest of the benchmark.

THE LABEL IS ACTIVATION, NOT BINDING. `cognate` is kept for schema compatibility
but `label_type` says what it means: CD69 upregulation >3% in TCR-transduced
SKW-3 cells. The paper's own models score AUC>0.95 on binding yet only ~0.64 on
this activation task, so the two must never be pooled.

Evaluation unit is the TCR (each TCR scored across the 16 peptides), not the
p:MHC antigen used by baseline_auc.py.

Requires the isolated venv (pyarrow) for the MHC lookup:
    tcr_bench/.venv/bin/python tcr_bench/prp_activation.py
"""
import argparse
import collections
import csv
import hashlib
import io
import os
import sys
import urllib.request

import pyarrow.parquet as pq

HERE = os.path.dirname(__file__)
RAW = 'https://raw.githubusercontent.com/akds/PRP-TCR-Specificity/main/data/'
ALLELE = 'B*27:05'
MHC_SOURCE = 'data/iedb_I/triad/iedb_I_triad.conf_af3.parquet'
PROTCR_COLS = ['Antigen', 'TCRB', 'TCRA', 'MHC_str', 'y', 'HLA']
OUR_COLS = ['triad_id', 'epitope_id', 'source', 'cognate', 'mhc_class',
            'mhc_1_name', 'mhc_2_name', 'job_name',
            'tcr_id', 'cdr3_b', 'tcr_chain_scope', 'label_type']
COLS = PROTCR_COLS + OUR_COLS


def short_id(*parts):
  return hashlib.sha1('|'.join(str(p) for p in parts).encode()).hexdigest()[:12]


def fetch(name):
  with urllib.request.urlopen(RAW + name, timeout=60) as r:
    return list(csv.DictReader(io.StringIO(r.read().decode())))


def mhc_sequences():
  path = os.path.join(HERE, MHC_SOURCE)
  cols = ['mhc_1_name', 'mhc_1_seq', 'mhc_2_name', 'mhc_2_seq']
  for r in pq.read_table(path, columns=cols).to_pylist():
    if r['mhc_1_name'] == ALLELE and r['mhc_1_seq']:
      return r['mhc_1_seq'], r['mhc_2_name'], r['mhc_2_seq']
  raise SystemExit(f'{ALLELE} not found in {MHC_SOURCE}; run collect.py first')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', default=os.path.join(HERE, 'csv'))
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)

  act = fetch('activation_binary.csv')
  cdr3 = {r['tcr_id']: r['CDR3_b'] for r in fetch('tcr_cdr3b.csv')}
  mhc_seq, mhc_2_name, _ = mhc_sequences()
  tcr_ids = [k for k in act[0] if k != 'Peptide']
  missing = [t for t in tcr_ids if t not in cdr3]
  if missing:
    raise SystemExit(f'no CDR3b for {missing}')

  rows = []
  for a in act:
    pep = a['Peptide']
    ep = short_id(pep, ALLELE, mhc_2_name)
    for t in tcr_ids:
      y = int(a[t])
      b = cdr3[t]
      rows.append({
          'Antigen': pep, 'TCRB': b, 'TCRA': '', 'MHC_str': mhc_seq,
          'y': float(y), 'HLA': ALLELE,
          # tcr_id is part of the key: 21 TCRs share only 16 distinct CDR3b, so
          # hashing the chain alone collides across receptors with different labels.
          'triad_id': short_id(ep, t, b), 'epitope_id': ep,
          'source': 'prp_activation', 'cognate': bool(y), 'mhc_class': 'I',
          'mhc_1_name': ALLELE, 'mhc_2_name': mhc_2_name,
          'job_name': f'prp_{t}_{pep}',
          'tcr_id': t, 'cdr3_b': b, 'tcr_chain_scope': 'cdr3b_only',
          'label_type': 'activation_cd69',
      })

  dst = os.path.join(args.out, 'triads_prp_activation.csv')
  with open(dst, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    w.writerows(rows)

  pos = sum(r['cognate'] for r in rows)
  by_tcr = collections.defaultdict(lambda: [0, 0])
  by_pep = collections.defaultdict(lambda: [0, 0])
  for r in rows:
    by_tcr[r['tcr_id']][0 if r['cognate'] else 1] += 1
    by_pep[r['Antigen']][0 if r['cognate'] else 1] += 1
  live_tcr = [k for k, (p, n) in by_tcr.items() if p and n]
  live_pep = [k for k, (p, n) in by_pep.items() if p and n]

  print(f'\ntriads_prp_activation.csv: {len(rows)} rows')
  print(f'  cognate(activating) {pos} / non {len(rows) - pos}   '
        f'TCRs {len(by_tcr)}   peptides {len(by_pep)}   allele {ALLELE}')
  print(f'  MHC_str len {len(mhc_seq)} (from {MHC_SOURCE})')
  print(f'  TCRA empty on all rows: {all(r["TCRA"] == "" for r in rows)}')
  print(f'  TCRB holds CDR3b only, lengths '
        f'{min(len(r["TCRB"]) for r in rows)}-{max(len(r["TCRB"]) for r in rows)}')
  print(f'  per-TCR AUC defined: {len(live_tcr)} / {len(by_tcr)}'
        f'   dead: {sorted(set(by_tcr) - set(live_tcr))}')
  print(f'  per-peptide AUC defined: {len(live_pep)} / {len(by_pep)}')
  print('  LABEL: activation (CD69), NOT binding - do not pool with IEDB/CRESTA.')
  print('  GROUP BY tcr_id, not epitope_id: the task is one TCR vs many peptides.')

  # A CDR3b-only input cannot separate receptors that share it. Where those
  # receptors disagree, no deterministic model on this representation can win.
  by_input = collections.defaultdict(list)
  for r in rows:
    by_input[(r['cdr3_b'], r['Antigen'])].append(int(r['y']))
  clash = sum(1 for v in by_input.values() if len(set(v)) > 1)
  ceiling = sum(max(v.count(0), v.count(1)) for v in by_input.values())
  print(f'  distinct CDR3b {len({r["cdr3_b"] for r in rows})} for {len(by_tcr)} TCRs; '
        f'duplicate triad_id {len(rows) - len({r["triad_id"] for r in rows})}')
  print(f'  CDR3b-only label conflicts: {clash} inputs carry both labels -> '
        f'accuracy ceiling {ceiling}/{len(rows)} = {ceiling / len(rows):.3f}')
  print('  the paper measured alpha-invariance on BINDING, not on this activation task.')

  ok = (len(rows) == 336 and pos == 65 and len(by_tcr) == 21
        and len(live_tcr) == 19 and all(r['MHC_str'] for r in rows)
        and len({r['triad_id'] for r in rows}) == len(rows))
  print(f'\nACCEPTANCE: {"COMPLETE" if ok else "INCOMPLETE - counts drifted from the published table"}')
  print(f'  expected 336 rows / 65 positives / 21 TCRs / 19 usable, got '
        f'{len(rows)} / {pos} / {len(by_tcr)} / {len(live_tcr)}')
  return 0 if ok else 1


if __name__ == '__main__':
  sys.exit(main())
