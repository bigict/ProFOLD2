#!/usr/bin/env python3
"""Search AFDB with mined fold0/fold1 templates; assign hits per-mode (Foldseek).

Runs on the AFDB machine (192 cores / 1.5T RAM). Foldseek structure-search the
extracted templates (extract_templates.py) against AFDB, assign each AFDB hit to
the fold of its best-matching template (similarity-gated pseudo-labels, §4.2),
and emit a per-fold hit manifest for downstream backbone-only construction.

Two stages (can run separately):
  --stage search : Foldseek easy-search templates vs AFDB -> result m8
  --stage assign : parse m8 + template manifest -> per-fold AFDB hit manifest

Run (on AFDB machine):
    PYTHONPATH=. python tools/foldswitch_probe/afdb_search.py --stage search \
        --templates /tmp/fsprobe/afdb_templates --afdb /path/to/afdb \
        --foldseek /path/to/foldseek --threads 128
    PYTHONPATH=. python tools/foldswitch_probe/afdb_search.py --stage assign \
        --templates /tmp/fsprobe/afdb_templates --tm-threshold 0.5

NOTE: AFDB hits are PSEUDO-LABELS (predicted structures) — downweight + identity-gate
per §4.2; per-mode augmentation != new-switcher discovery (a switcher must match BOTH
folds). Backbone-only construction ties to §4.3.
"""
import os
import sys
import json
import argparse
import subprocess
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def stage_search(args):
  os.makedirs(args.workdir, exist_ok=True)
  result = os.path.join(args.workdir, 'afdb_search.m8')
  cmd = [args.foldseek, 'easy-search', args.templates, args.afdb, result,
         os.path.join(args.workdir, 'tmp'),
         '--format-output', 'query,target,alntmscore,qtmscore,ttmscore,coverage',
         '-c', str(args.coverage), '--cov-mode', '0',
         '--threads', str(args.threads), '-e', str(args.evalue)]
  print('running:', ' '.join(cmd), flush=True)
  subprocess.run(cmd, check=True)
  print('search result -> %s' % result)


def stage_assign(args):
  manifest = json.load(open(os.path.join(args.templates, 'manifest.json')))
  tmpl_fold = {m['template']: m['fold'] for m in manifest}
  tmpl_sw = {m['template']: m['seq_rep'] for m in manifest}
  result = os.path.join(args.workdir, 'afdb_search.m8')
  # best template match per AFDB target (highest alignment TM-score)
  best = {}
  for line in open(result):
    p = line.rstrip('\n').split('\t')
    if len(p) < 6:
      continue
    q, t, alntm, qtm, ttm, cov = p[0], p[1], float(p[2]), float(p[3]), float(p[4]), float(p[5])
    if alntm < args.tm_threshold or cov < args.coverage:
      continue
    if t not in best or alntm > best[t][1]:
      best[t] = (q, alntm, cov)
  per_fold = defaultdict(list)
  for t, (q, alntm, cov) in best.items():
    fold = tmpl_fold.get(q)
    if fold is None:
      continue
    per_fold[fold].append(dict(afdb_id=t, template=q, seq_rep=tmpl_sw[q],
                               fold=fold, alntmscore=round(alntm, 3), coverage=round(cov, 3)))
  out = os.path.join(args.workdir, 'afdb_per_fold.json')
  with open(out, 'w') as f:
    json.dump({str(k): v for k, v in per_fold.items()}, f, indent=1)
  print('AFDB hits per fold:', {k: len(v) for k, v in sorted(per_fold.items())})
  print('total assigned AFDB hits: %d -> %s' % (len(best), out))
  print('NOTE: pseudo-labels — downweight + identity-gate (§4.2); backbone-only construction next (§4.3).')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--stage', choices=['search', 'assign'], required=True)
  ap.add_argument('--templates', default='/tmp/fsprobe/afdb_templates')
  ap.add_argument('--afdb', default='/path/to/afdb')
  ap.add_argument('--foldseek', default='foldseek')
  ap.add_argument('--workdir', default='/tmp/fsprobe/afdb_search')
  ap.add_argument('--threads', type=int, default=128)
  ap.add_argument('--coverage', type=float, default=0.8)
  ap.add_argument('--evalue', type=float, default=0.01)
  ap.add_argument('--tm-threshold', type=float, default=0.5)
  args = ap.parse_args()
  if args.stage == 'search':
    stage_search(args)
  else:
    stage_assign(args)


if __name__ == '__main__':
  main()