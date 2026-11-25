#!/usr/bin/env python3
"""Extract raw mmCIF.gz structures from the PDB train archive for Foldseek.

Foldseek createdb reads mmCIF.gz directly (chain extraction mode 0). Extract the
raw/*.cif.gz entries to a directory so Foldseek can cluster all chains by fold.
"""
import os
import sys
import zipfile
import argparse

ARCHIVE = os.path.expanduser('~/shared_hosts/pdb_train.zip')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', default='/tmp/fsprobe/structs')
  ap.add_argument('--limit', type=int, default=0)
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  z = zipfile.ZipFile(ARCHIVE)
  names = [n for n in z.namelist() if n.startswith('pdb_train/raw/') and n.endswith('.cif.gz')]
  print('raw mmCIF.gz entries: %d' % len(names), flush=True)
  if args.limit:
    names = names[:args.limit]
  for i, n in enumerate(names):
    pid = os.path.basename(n)[:-len('.cif.gz')]
    dst = os.path.join(args.out, pid + '.cif.gz')
    if not os.path.exists(dst):
      with z.open(n) as src, open(dst, 'wb') as out:
        out.write(src.read())
    if (i + 1) % 10000 == 0:
      print('  extracted %d/%d' % (i + 1, len(names)), flush=True)
  print('done: %d structures -> %s' % (len(names), args.out), flush=True)


if __name__ == '__main__':
  main()