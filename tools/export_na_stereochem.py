"""Export NA stereochemical props from the PDB Chemical Component Dictionary.

Downloads the DNA (DA/DC/DG/DT) and RNA (A/C/G/U) component CIFs, computes
bond lengths and angles from the CCD ideal coordinates (filtered to the repo
atom14 heavy atoms), and appends them to stereo_chemical_props.txt in the
existing two-section format. Protein entries and the loader are untouched.

Usage:
  profold2_restype_list='[1,2,3]' profold2_atom14_type_num=23 \
      python tools/export_na_stereochem.py [--dry-run]

Honors HTTPS_PROXY for the CCD download. Idempotent: aborts if NA entries
already exist in the target file.
"""
import argparse
import os
import shlex
import tempfile
import urllib.request

import numpy as np

from profold2.common import residue_constants as rc

COMPS = ['DA', 'DC', 'DG', 'DT', 'A', 'C', 'G', 'U']
CCD_URL = 'https://files.rcsb.org/ligands/download/{}.cif'
BOND_STDDEV = 0.020
ANGLE_STDDEV_DEG = 2.0
TARGET = os.path.join(os.path.dirname(rc.__file__), 'stereo_chemical_props.txt')


def parse_loop(path, category):
  """Parse a CIF loop_ block for the given category into a list of dicts."""
  lines = open(path).read().splitlines()
  n = len(lines)
  i = 0
  while i < n:
    if lines[i].strip() == 'loop_':
      j = i + 1
      hdr = []
      while j < n and lines[j].strip().startswith('_'):
        hdr.append(lines[j].strip())
        j += 1
      if any(h.startswith(category + '.') for h in hdr):
        names = [h.split('.', 1)[1] for h in hdr]
        rows = []
        k = j
        while k < n:
          s = lines[k].strip()
          if s in ('', '#') or s.startswith(('loop_', '_')):
            break
          toks = shlex.split(s)
          if len(toks) == len(names):
            rows.append(dict(zip(names, toks)))
          k += 1
        return rows
      i = j
    else:
      i += 1
  return []


def download(comp, dest):
  proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
  handlers = [urllib.request.ProxyHandler({'https': proxy})] if proxy else []
  opener = urllib.request.build_opener(*handlers)
  with opener.open(CCD_URL.format(comp), timeout=60) as r:
    data = r.read()
  with open(dest, 'wb') as f:
    f.write(data)


def export(resname, path):
  valid = set(a for a in rc.restype_name_to_atom14_names[resname] if a)
  coord = {}
  for a in parse_loop(path, '_chem_comp_atom'):
    if a['atom_id'] in valid:
      coord[a['atom_id']] = np.array([
          float(a['pdbx_model_Cartn_x_ideal']),
          float(a['pdbx_model_Cartn_y_ideal']),
          float(a['pdbx_model_Cartn_z_ideal']),
      ])
  adj = {a: set() for a in coord}
  bond_lines, seen = [], set()
  for b in parse_loop(path, '_chem_comp_bond'):
    a1, a2 = b['atom_id_1'], b['atom_id_2']
    if a1 in coord and a2 in coord:
      adj[a1].add(a2)
      adj[a2].add(a1)
      key = tuple(sorted([a1, a2]))
      if key not in seen:
        seen.add(key)
        length = np.linalg.norm(coord[a1] - coord[a2])
        bond_lines.append(
            f'{a1}-{a2}\t\t\t{resname}\t\t{length:.3f}\t\t{BOND_STDDEV:.3f}'
        )
  angle_lines, seen_ang = [], set()
  for a2 in coord:
    nbrs = sorted(adj[a2])
    for x in range(len(nbrs)):
      for y in range(x + 1, len(nbrs)):
        a1, a3 = nbrs[x], nbrs[y]
        key = (a1, a2, a3) if a1 < a3 else (a3, a2, a1)
        if key in seen_ang:
          continue
        seen_ang.add(key)
        v1, v2 = coord[a1] - coord[a2], coord[a3] - coord[a2]
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        angle_lines.append(
            f'{a1}-{a2}-{a3}\t\t{resname}\t\t{ang:.1f}\t\t{ANGLE_STDDEV_DEG:.1f}'
        )
  return bond_lines, angle_lines


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--dry-run', action='store_true', help='print lines, do not write')
  args = ap.parse_args()

  bond_lines, angle_lines = [], []
  with tempfile.TemporaryDirectory() as tmp:
    for comp in COMPS:
      path = os.path.join(tmp, f'{comp}.cif')
      download(comp, path)
      bl, al = export(comp, path)
      bond_lines += bl
      angle_lines += al
      print(f'{comp}: {len(bl)} bonds, {len(al)} angles')

  if args.dry_run:
    print('\n'.join(bond_lines + angle_lines))
    return

  lines = open(TARGET).read().splitlines()
  if any(l.split()[1:2] == ['DA'] for l in lines if l.strip() and not l.startswith(('Bond', 'Angle', '-'))):
    raise SystemExit('NA entries already present in target; aborting (idempotent).')
  seps = [i for i, l in enumerate(lines) if l.strip() == '-']
  assert len(seps) >= 2, 'unexpected stereo_chemical_props.txt layout'
  lines[seps[1]:seps[1]] = angle_lines
  lines[seps[0]:seps[0]] = bond_lines
  with open(TARGET, 'w') as f:
    f.write('\n'.join(lines) + '\n')
  print(f'appended {len(bond_lines)} bond + {len(angle_lines)} angle lines to {TARGET}')


if __name__ == '__main__':
  main()