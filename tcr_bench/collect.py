#!/usr/bin/env python3
"""Collect the TCR-pMHC evaluation data released with Woods et al. 2026 (Sci Adv).

Source repo: AltinLab/tcrtrifold-experiments  (preprint 10.64898/2026.06.02.729478)

Two kinds of asset live in that repo and they need different transport:

  * plain files (mostly .csv)  -> raw.githubusercontent.com, no auth
  * .parquet / .tsv            -> Git LFS pointers; the real bytes come from the
                                  LFS batch API, which serves this public repo
                                  anonymously. git-lfs is NOT required.

A naive `git clone` without git-lfs yields 130-byte pointer files that look like
data. This script resolves them and verifies sha256 against the pointer oid.

Usage:
    python tcr_bench/collect.py --dry-run          # plan only, no writes
    python tcr_bench/collect.py                    # fetch into tcr_bench/data
    python tcr_bench/collect.py --only cresta pdb  # subset by top-level dataset
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

OWNER = 'AltinLab'
REPO = 'tcrtrifold-experiments'
BRANCH = 'main'
API_TREE = f'https://api.github.com/repos/{OWNER}/{REPO}/git/trees/{BRANCH}?recursive=1'
RAW = f'https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}/'
LFS_BATCH = f'https://github.com/{OWNER}/{REPO}.git/info/lfs/objects/batch'
LFS_PREFIX = b'version https://git-lfs.github.com/spec/v1'
UA = {'User-Agent': 'tcr-bench-collector'}


def http(url, data=None, headers=None, timeout=120):
  req = urllib.request.Request(url, data=data, headers={**UA, **(headers or {})})
  with urllib.request.urlopen(req, timeout=timeout) as r:
    return r.read()


def list_data_tree(prefix='data/'):
  tree = json.loads(http(API_TREE, headers={'Accept': 'application/vnd.github+json'}))
  if tree.get('truncated'):
    raise RuntimeError('GitHub returned a truncated tree; refusing a partial plan')
  return [
      {'path': e['path'], 'size': e['size']}
      for e in tree['tree']
      if e['type'] == 'blob' and e['path'].startswith(prefix)
  ]


def parse_lfs_pointer(blob):
  """Return (oid, size) if blob is an LFS pointer, else None."""
  if not blob.startswith(LFS_PREFIX):
    return None
  fields = {}
  for line in blob.decode().splitlines()[1:]:
    if ' ' in line:
      k, v = line.split(' ', 1)
      fields[k] = v
  oid = fields.get('oid', '')
  if oid.startswith('sha256:'):
    return oid[len('sha256:'):], int(fields['size'])
  return None


def lfs_download_urls(objects):
  """One batch call for many objects. objects: [{'oid':..,'size':..}, ...]"""
  out = {}
  CHUNK = 100
  for i in range(0, len(objects), CHUNK):
    batch = objects[i:i + CHUNK]
    body = json.dumps({
        'operation': 'download', 'transfers': ['basic'], 'objects': batch
    }).encode()
    resp = json.loads(http(
        LFS_BATCH, data=body,
        headers={'Accept': 'application/vnd.git-lfs+json',
                 'Content-Type': 'application/vnd.git-lfs+json'}
    ))
    for o in resp.get('objects', []):
      if 'actions' in o and 'download' in o['actions']:
        out[o['oid']] = o['actions']['download']['href']
      else:
        out[o['oid']] = None  # quota exhausted or object missing
  return out


def sha256(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
      h.update(chunk)
  return h.hexdigest()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'data'))
  ap.add_argument('--only', nargs='*', default=None,
                  help='dataset dir names under data/, e.g. cresta pdb iedb_I')
  ap.add_argument('--dry-run', action='store_true')
  args = ap.parse_args()

  entries = list_data_tree()
  if args.only:
    keep = {f'data/{n}/' for n in args.only}
    entries = [e for e in entries if any(e['path'].startswith(k) for k in keep)]
  if not entries:
    print('nothing selected')
    return 1

  print(f'repo   : {OWNER}/{REPO}@{BRANCH}')
  print(f'selected: {len(entries)} files, {sum(e["size"] for e in entries):,} bytes '
        '(LFS sizes are pointer sizes, resolved below)')

  # Peek every file once to classify pointer vs plain. Peeking is a full GET for
  # plain files, so cache the bytes rather than downloading twice.
  plain, pointers, failed = {}, [], []
  for e in entries:
    try:
      blob = http(RAW + e['path'])
    except urllib.error.HTTPError as ex:
      failed.append((e['path'], f'HTTP {ex.code}'))
      continue
    p = parse_lfs_pointer(blob)
    if p:
      pointers.append({'path': e['path'], 'oid': p[0], 'size': p[1]})
    else:
      plain[e['path']] = blob

  real_total = sum(len(b) for b in plain.values()) + sum(p['size'] for p in pointers)
  print(f'plain  : {len(plain)} files, {sum(len(b) for b in plain.values()):,} bytes')
  print(f'LFS    : {len(pointers)} files, {sum(p["size"] for p in pointers):,} bytes')
  print(f'unreachable at peek: {len(failed)}')
  print(f'TOTAL real payload: {real_total:,} bytes')

  if args.dry_run:
    for p in sorted(pointers, key=lambda x: -x['size'])[:15]:
      print(f'  LFS {p["size"]:>10,}  {p["path"]}')
    print('\nDRY RUN - nothing written')
    return 0

  urls = lfs_download_urls([{k: p[k] for k in ('oid', 'size')} for p in pointers]) \
      if pointers else {}

  os.makedirs(args.out, exist_ok=True)
  manifest, ok, bad = [], 0, []

  def write(rel, blob, expect_sha=None):
    nonlocal ok
    dst = os.path.join(args.out, rel[len('data/'):])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, 'wb') as f:
      f.write(blob)
    got = hashlib.sha256(blob).hexdigest()
    if expect_sha and got != expect_sha:
      bad.append((rel, 'sha256 mismatch'))
      return
    ok += 1
    manifest.append({'path': rel, 'bytes': len(blob), 'sha256': got,
                     'lfs': bool(expect_sha)})

  for rel, blob in plain.items():
    write(rel, blob)

  for p in pointers:
    href = urls.get(p['oid'])
    if not href:
      bad.append((p['path'], 'no LFS download action (quota or missing object)'))
      continue
    try:
      blob = http(href)
    except urllib.error.HTTPError as ex:
      bad.append((p['path'], f'LFS HTTP {ex.code}'))
      continue
    write(p['path'], blob, expect_sha=p['oid'])

  bad.extend(failed)
  with open(os.path.join(args.out, 'manifest.json'), 'w') as f:
    json.dump({'repo': f'{OWNER}/{REPO}@{BRANCH}', 'fetched_at': time.time(),
               'files': manifest}, f, indent=2)

  print(f'\nwrote {ok} files to {args.out}')
  print(f'failures: {len(bad)}')
  for rel, why in bad[:20]:
    print(f'  FAIL {rel}: {why}')
  # The gate lives in the report, not in the reader's head.
  if bad:
    print('\nINCOMPLETE - do not treat this collection as the full benchmark.')
  else:
    print('\nCOMPLETE - every selected file fetched; LFS payloads sha256-verified.')
  return 1 if bad else 0


if __name__ == '__main__':
  sys.exit(main())
