#!/usr/bin/env python3
"""Emit ProTCR-compatible input CSVs from the paper's held-out triad tables.

Source of truth is the *triad* parquet, not the raw CSV: those tables reproduce
the paper exactly (197 = 16+181 class I; 1433 = 205+1228 CRESTA) and carry the
AF3 confidence features, so the baseline score rides along for comparison.

Target schema is what bigict/ProTCR `main.py csv_to_fasta` actually reads:

    Antigen | Peptide   -> chain P
    MHC_str             -> chain M      (ONE chain; no class-II pair in the schema)
    TCRA | a_seq        -> chain A
    TCRB | b_seq | tcrb -> chain B
    y | label           -> float label
    HLA | Allele        -> MHC allele metadata

Every other column is ignored by that reader, so epitope_id / cognate / the AF3
baseline features ride along for our own grouping and comparison.

epitope_id uses (peptide, mhc_1_name, mhc_2_name) — the SAME key baseline_auc.py
groups by, so per-antigen AUC is computed over identical groups.

CLASS II IS AN EXPLICIT DECISION. ProTCR takes a single MHC_str; CRESTA is all
class II. --mhc2 defaults to 'skip' and reports the drop count rather than
silently emitting a concatenation the downstream align filter may reject.

Requires the isolated venv (pyarrow):
    tcr_bench/.venv/bin/python tcr_bench/normalize.py
    tcr_bench/.venv/bin/python tcr_bench/normalize.py --mhc2 concat
"""
import argparse
import collections
import csv
import hashlib
import os
import sys

import pyarrow.parquet as pq

HERE = os.path.dirname(__file__)
SETS = [
    ('pdb_validation', 'data/pdb/triad/pdb_validation_triad.conf_af3.parquet'),
    ('cresta', 'data/cresta/triad/cresta_triad.conf_af3.parquet'),
]
# IEDB is the paper's training/dev pool, not a held-out set. Exported in full and
# streamed (200k rows) rather than sampled; `references` is a LIST and becomes the
# study key, `pmhc_in_validation` is kept so the caller can filter.
IEDB_SETS = [
    ('iedb_I', 'data/iedb_I/triad/iedb_I_triad.conf_af3.parquet'),
    ('iedb_II', 'data/iedb_II/triad/iedb_II_triad.conf_af3.parquet'),
]
IEDB_EXTRA = ['references', 'receptor_id', 'pmhc_in_validation']
BASELINE_COLS = ['mean_p_tcr_interface_pae', 'min_p_tcr_pae']
PROTCR_COLS = ['Antigen', 'TCRB', 'TCRA', 'MHC_str', 'y', 'HLA']
OUR_COLS = ['triad_id', 'epitope_id', 'source', 'cognate', 'mhc_class',
            'mhc_1_name', 'mhc_2_name', 'job_name'] + ['af3_' + c for c in BASELINE_COLS]
COLS = PROTCR_COLS + OUR_COLS


def short_id(*parts):
  return hashlib.sha1('|'.join(str(p) for p in parts).encode()).hexdigest()[:12]


def mhc_string(r, mode):
  cls = (r['mhc_class'] or '').strip().upper()
  m1, m2 = r['mhc_1_seq'] or '', r['mhc_2_seq'] or ''
  n1, n2 = r['mhc_1_name'] or '', r['mhc_2_name'] or ''
  if cls == 'I':
    return m1, n1
  if mode == 'skip':
    return None
  if mode == 'concat':
    return m1 + m2, f'{n1}+{n2}'
  if mode == 'alpha':
    return m1, n1
  if mode == 'beta':
    return m2, n2
  raise ValueError(mode)


def load(name, rel, mhc2):
  path = os.path.join(HERE, rel)
  if not os.path.exists(path):
    print(f'skip {name}: {path} not present')
    return None, 0
  want = ['job_name', 'cognate', 'peptide', 'mhc_class',
          'mhc_1_name', 'mhc_2_name', 'mhc_1_seq', 'mhc_2_seq',
          'tcr_1_chain', 'tcr_2_chain', 'tcr_1_seq', 'tcr_2_seq']
  schema = pq.read_schema(path).names
  want += [c for c in BASELINE_COLS if c in schema]
  rows, skipped = [], 0
  for r in pq.read_table(path, columns=want).to_pylist():
    assert r['tcr_1_chain'] == 'alpha' and r['tcr_2_chain'] == 'beta', r['job_name']
    got = mhc_string(r, mhc2)
    if got is None:
      skipped += 1
      continue
    mhc_seq, allele = got
    a, b = r['tcr_1_seq'] or '', r['tcr_2_seq'] or ''
    ep = short_id(r['peptide'], r['mhc_1_name'], r['mhc_2_name'])
    row = {
        'Antigen': r['peptide'], 'TCRB': b, 'TCRA': a, 'MHC_str': mhc_seq,
        'y': 1.0 if r['cognate'] else 0.0, 'HLA': allele,
        'triad_id': short_id(ep, a, b), 'epitope_id': ep, 'source': name,
        'cognate': bool(r['cognate']), 'mhc_class': r['mhc_class'],
        'mhc_1_name': r['mhc_1_name'], 'mhc_2_name': r['mhc_2_name'],
        'job_name': r['job_name'],
    }
    for c in BASELINE_COLS:
      row['af3_' + c] = r.get(c, '')
    rows.append(row)
  return rows, skipped


def report(name, rows, skipped, mhc2):
  print(f'\n{name}: {len(rows)} rows emitted, {skipped} skipped (--mhc2 {mhc2})')
  if not rows:
    print('  NOTHING EMITTED - all rows are class II and --mhc2 skip is the default.')
    print('  Pass --mhc2 concat|alpha|beta explicitly to include them.')
    return
  pos = sum(r['cognate'] for r in rows)
  by_ep = collections.defaultdict(lambda: [0, 0])
  for r in rows:
    by_ep[r['epitope_id']][0 if r['cognate'] else 1] += 1
  print(f'  cognate {pos} / non-cognate {len(rows) - pos}   '
        f'antigens(p:MHC) {len(by_ep)}')
  dead = [e for e, (p, n) in by_ep.items() if p == 0 or n == 0]
  print(f'  antigens lacking one class: {len(dead)}'
        + ('  <- per-antigen AUC undefined there' if dead else '  (AUC defined everywhere)'))
  thin = [r for r in rows if not r['Antigen'] or not (r['TCRA'] or r['TCRB'])]
  print(f'  rows ProTCR would filter (no peptide / no TCR chain): {len(thin)}')
  print(f'  duplicate triad_id: {len(rows) - len({r["triad_id"] for r in rows})}')
  missing_mhc = sum(1 for r in rows if not r['MHC_str'])
  print(f'  rows with empty MHC_str: {missing_mhc}')


def export_iedb(name, rel, mhc2, out):
  """Stream the IEDB pool to CSV; only counters are held in memory."""
  path = os.path.join(HERE, rel)
  if not os.path.exists(path):
    print(f'skip {name}: {path} not present')
    return False
  base = ['job_name', 'cognate', 'peptide', 'mhc_class',
          'mhc_1_name', 'mhc_2_name', 'mhc_1_seq', 'mhc_2_seq',
          'tcr_1_chain', 'tcr_2_chain', 'tcr_1_seq', 'tcr_2_seq']
  schema = pq.read_schema(path).names
  cols = base + [c for c in BASELINE_COLS + IEDB_EXTRA if c in schema]
  dst = os.path.join(out, f'triads_{name}.csv')
  fields = COLS + [c for c in IEDB_EXTRA if c in schema] + ['study_id']
  n = pos = skipped = 0
  by_ep = collections.defaultdict(lambda: [0, 0])
  studies = set()
  with open(dst, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in pq.read_table(path, columns=cols).to_pylist():
      got = mhc_string(r, mhc2)
      if got is None:
        skipped += 1
        continue
      mhc_seq, allele = got
      a, b = r['tcr_1_seq'] or '', r['tcr_2_seq'] or ''
      refs = r.get('references')
      study = ';'.join(map(str, refs)) if isinstance(refs, list) else str(refs or '')
      ep = short_id(r['peptide'], r['mhc_1_name'], r['mhc_2_name'])
      cognate = bool(r['cognate'])
      row = {
          'Antigen': r['peptide'], 'TCRB': b, 'TCRA': a, 'MHC_str': mhc_seq,
          'y': 1.0 if cognate else 0.0, 'HLA': allele,
          'triad_id': short_id(ep, a, b), 'epitope_id': ep, 'source': name,
          'cognate': cognate, 'mhc_class': r['mhc_class'],
          'mhc_1_name': r['mhc_1_name'], 'mhc_2_name': r['mhc_2_name'],
          'job_name': r['job_name'],
          'references': study, 'receptor_id': r.get('receptor_id', ''),
          'pmhc_in_validation': r.get('pmhc_in_validation', ''),
          'study_id': short_id(ep, study),
      }
      for c in BASELINE_COLS:
        row['af3_' + c] = r.get(c, '')
      w.writerow({k: row.get(k, '') for k in fields})
      n += 1
      pos += cognate
      by_ep[ep][0 if cognate else 1] += 1
      studies.add(row['study_id'])
  print(f'\ntriads_{name}.csv: {n} rows, {skipped} skipped (--mhc2 {mhc2})')
  print(f'  cognate {pos} / non-cognate {n - pos}   p:MHC epitopes {len(by_ep)}')
  both = sum(1 for p_, n_ in by_ep.values() if p_ and n_)
  print(f'  epitopes with both classes: {both} / {len(by_ep)}'
        + ('' if both == len(by_ep) else '  <- AUC undefined on the rest'))
  # study_id is placeholder-valued on permuted negatives, so it splits every
  # group by class. Reported for provenance only; group by epitope_id.
  print(f'  distinct study_id: {len(studies)}  (provenance only, NOT a grouping key)')
  print('  ROLE: training/dev pool for the paper - NOT a held-out set.')
  return True


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', default=os.path.join(HERE, 'csv'))
  ap.add_argument('--mhc2', choices=['skip', 'concat', 'alpha', 'beta'], default='skip')
  ap.add_argument('--iedb', action='store_true',
                  help='also export the full IEDB training/dev pool (200k rows)')
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)

  wrote = 0
  if args.iedb:
    for name, rel in IEDB_SETS:
      wrote += export_iedb(name, rel, args.mhc2, args.out)
  for name, rel in SETS:
    rows, skipped = load(name, rel, args.mhc2)
    if rows is None:
      continue
    if rows:
      dst = os.path.join(args.out, f'triads_{name}.csv')
      with open(dst, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
      wrote += 1
    report(f'triads_{name}.csv', rows, skipped, args.mhc2)

  print(f'\nwrote {wrote} table(s) to {args.out}')
  return 0


if __name__ == '__main__':
  sys.exit(main())
