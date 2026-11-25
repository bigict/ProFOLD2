#!/usr/bin/env python3
"""Export TCRvdb as a ProTCR-schema CSV. LOCAL ONLY — DO NOT REDISTRIBUTE.

Source: Messemaker M, ... Schumacher TN, "Functional evaluation of TCR-pMHC pairs
at scale allows in silico TCR reactivity prediction", Immunity 2026,
doi 10.1016/j.immuni.2026.09.007. Obtained 2026-11-25 via the authors' named
request portal (tcrvdb.nkiai.nl); citation of that DOI is a condition of use.

LICENCE: "Redistribution or public sharing of the dataset, in whole or in part,
is not permitted without explicit permission from the authors." The output
therefore defaults to data/tcrvdb/, NOT csv/ — csv/ is the set that gets uploaded
to the shared compute server, and copying this table there would redistribute it.
Do not commit it, do not upload it, do not move it into csv/.

Unlike the other sources this one ships complete receptor chains, so TCRA/TCRB
are populated. They are VARIABLE DOMAINS ONLY (alpha 103-121 aa, beta 109-122),
whereas the Woods tables carry full ectodomains (187-274 / 238-320).
`tcr_chain_scope` records this; cross-dataset numbers need that caveat.

LABELS: three mutually exclusive flags. All rows are exported (user decision
2026-11-25) with `label_state` in {pos,neg,ambiguous} and y = 1.0 / 0.0 / 0.5.
**Filter label_state != 'ambiguous' before computing any AUC** — y=0.5 is a
placeholder to keep the column float-parseable for ProTCR, not a soft target.

MHC_str is read from our own verified iedb_I triad parquet. Both alleles are
class I, so ProTCR's single-MHC schema fits natively; no concat workaround.

Run with the isolated venv (pyarrow):
    tcr_bench/.venv/bin/python tcr_bench/tcrvdb.py
"""
import argparse
import collections
import csv
import hashlib
import os
import sys

import pyarrow.parquet as pq

HERE = os.path.dirname(__file__)
SRC = 'data/tcrvdb/ttd_gil_ylq_nlv_glc_01-10-2026_TCRvdb.csv'
MHC_SOURCE = 'data/iedb_I/triad/iedb_I_triad.conf_af3.parquet'
ALLELES = ('A*01:01', 'A*02:01')
PROTCR_COLS = ['Antigen', 'TCRB', 'TCRA', 'MHC_str', 'y', 'HLA']
OUR_COLS = ['triad_id', 'epitope_id', 'source', 'cognate', 'mhc_class',
            'mhc_1_name', 'mhc_2_name', 'job_name',
            'tcr_id', 'cdr3_alpha', 'cdr3_beta', 'tcr_chain_scope',
            'label_state', 'pos', 'neg', 'ambiguous',
            'data_origin', 'vdjdb_score', 'licence']
COLS = PROTCR_COLS + OUR_COLS
Y_BY_STATE = {'pos': 1.0, 'neg': 0.0, 'ambiguous': 0.5}


def short_id(*parts):
  return hashlib.sha1('|'.join(str(p) for p in parts).encode()).hexdigest()[:12]


def flag(v):
  return int(float(v or 0))


def mhc_map():
  path = os.path.join(HERE, MHC_SOURCE)
  cols = ['mhc_1_name', 'mhc_1_seq', 'mhc_2_name']
  out = {}
  for r in pq.read_table(path, columns=cols).to_pylist():
    n = r['mhc_1_name']
    if n in ALLELES and n not in out and r['mhc_1_seq']:
      out[n] = (r['mhc_1_seq'], r['mhc_2_name'])
  missing = set(ALLELES) - set(out)
  if missing:
    raise SystemExit(f'{missing} not in {MHC_SOURCE}; run collect.py first')
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', default=os.path.join(HERE, 'data', 'tcrvdb'),
                  help='LOCAL ONLY. Never point this at csv/ or a shared mount.')
  args = ap.parse_args()
  src = os.path.join(HERE, SRC)
  if not os.path.exists(src):
    raise SystemExit(f'{SRC} missing — it is request-gated, see the docstring')
  os.makedirs(args.out, exist_ok=True)

  mhcs = mhc_map()
  rows, skipped = [], collections.Counter()
  with open(src, newline='') as f:
    for r in csv.DictReader(f):
      states = [s for s in ('pos', 'neg', 'ambiguous') if flag(r[s])]
      if len(states) != 1:
        skipped['label_flags_not_exactly_one'] += 1
        continue
      state = states[0]
      allele = r['hla_long']
      if allele not in mhcs:
        skipped['unknown_allele'] += 1
        continue
      a, b = r['full_seq_alpha_aa'], r['full_seq_beta_aa']
      if not a or not b:
        skipped['missing_chain'] += 1
        continue
      mhc_seq, mhc_2_name = mhcs[allele]
      pep = r['epitope_aa']
      ep = short_id(pep, allele, mhc_2_name)
      rows.append({
          'Antigen': pep, 'TCRB': b, 'TCRA': a, 'MHC_str': mhc_seq,
          'y': Y_BY_STATE[state], 'HLA': allele,
          'triad_id': short_id(ep, r['tcr_id']), 'epitope_id': ep,
          'source': 'tcrvdb', 'cognate': state == 'pos', 'mhc_class': 'I',
          'mhc_1_name': allele, 'mhc_2_name': mhc_2_name,
          'job_name': f"tcrvdb_{r['tcr_id']}",
          'tcr_id': r['tcr_id'], 'cdr3_alpha': r['cdr3_alpha_aa'],
          'cdr3_beta': r['cdr3_beta_aa'], 'tcr_chain_scope': 'v_domain_only',
          'label_state': state, 'pos': flag(r['pos']), 'neg': flag(r['neg']),
          'ambiguous': flag(r['ambiguous']),
          'data_origin': r['data_origin'], 'vdjdb_score': r['vdjdb_score'],
          'licence': 'no-redistribution; cite 10.1016/j.immuni.2026.09.007',
      })

  dst = os.path.join(args.out, 'triads_tcrvdb.csv')
  with open(dst, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    w.writerows(rows)

  by_ep = collections.defaultdict(lambda: collections.Counter())
  for r in rows:
    by_ep[(r['Antigen'], r['HLA'])][r['label_state']] += 1
  live = [k for k, c in by_ep.items() if c['pos'] and c['neg']]

  print(f'\n{dst}: {len(rows)} rows')
  print('  label_state:', dict(collections.Counter(r['label_state'] for r in rows)))
  print(f'  skipped: {dict(skipped) or "none"}')
  print(f'  antigens {len(by_ep)}, all class I; per-epitope AUC defined {len(live)}/{len(by_ep)}')
  for (pep, hla), c in sorted(by_ep.items()):
    print(f"    {pep:11s} {hla:9s} pos={c['pos']:4d} neg={c['neg']:4d} amb={c['ambiguous']:4d}")
  print(f"  TCRA len {min(len(r['TCRA']) for r in rows)}-{max(len(r['TCRA']) for r in rows)}"
        f"  TCRB len {min(len(r['TCRB']) for r in rows)}-{max(len(r['TCRB']) for r in rows)}"
        '  (V DOMAINS, not ectodomains)')
  print(f"  MHC_str lengths {sorted({len(r['MHC_str']) for r in rows})}"
        f'  duplicate triad_id {len(rows) - len({r["triad_id"] for r in rows})}')
  print('  FILTER label_state != "ambiguous" BEFORE computing AUC; y=0.5 is a placeholder.')
  print('  LICENCE: no redistribution. Do NOT commit, upload, or copy into csv/.')

  ok = (len(rows) == 2034 and len(by_ep) == 5 and len(live) == 5
        and all(r['MHC_str'] and r['TCRA'] and r['TCRB'] for r in rows)
        and len({r['triad_id'] for r in rows}) == len(rows)
        and not skipped)
  print(f'\nACCEPTANCE: {"COMPLETE" if ok else "INCOMPLETE - inspect the counts above"}')
  return 0 if ok else 1


if __name__ == '__main__':
  sys.exit(main())
