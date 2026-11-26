#!/usr/bin/env python3
"""Fast mmCIF backbone extraction via direct _atom_site parsing (no biopython
Structure construction, no extra deps). ~10-50x faster than MMCIFParser for the
mining scale (tens of thousands of entries). Extracts N/CA/C/O for standard AA
residues of a given chain, keyed by author residue number.
"""
import gzip
import io
import zipfile

import numpy as np

ARCHIVE = None  # set by caller or default
_AA3 = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'}
_BB = {'N', 'CA', 'C', 'O'}
_ORDER = ['N', 'CA', 'C', 'O']
_ZIP = None
_AB_FLAG = {}
import re
# word-boundary matching: short isotypes (iga/igm/ige) must NOT substring-match
# 'ligand'/'image' etc. (the bug that flagged every entry as antibody).
_AB_RE = re.compile(
    r'\b(immunoglobulin|antibody|fab|scfv|nanobody|vhh|igg[1-4]?|ige|igm|'
    r'iga[12]?|igd|camelid|single-chain fv|single domain antibody|'
    r'light chain|heavy chain|variable domain)\b')


def _antibody_from_text(pdb, text):
  if pdb in _AB_FLAG:
    return _AB_FLAG[pdb]
  low = text.lower()
  cut = low.find('_atom_site.')
  head = low[:cut] if cut > 0 else low[:20000]
  flag = bool(_AB_RE.search(head))
  _AB_FLAG[pdb] = flag
  return flag


def is_antibody_fast(pdb, archive):
  pdb = pdb.lower()
  if pdb in _AB_FLAG:
    return _AB_FLAG[pdb]
  try:
    raw = _zip(archive).read('pdb_train/raw/%s.cif.gz' % pdb)
  except KeyError:
    _AB_FLAG[pdb] = False
    return False
  text = gzip.decompress(raw).decode('utf-8', 'replace')
  return _antibody_from_text(pdb, text)


def _zip(archive):
  global _ZIP
  if _ZIP is None:
    _ZIP = zipfile.ZipFile(archive)
  return _ZIP


def extract_backbone_fast(pdb, chain, archive):
  """Return {auth_resid: (aa, backbone(4,3))} or None."""
  pdb = pdb.lower()
  try:
    raw = _zip(archive).read('pdb_train/raw/%s.cif.gz' % pdb)
  except KeyError:
    return None
  text = gzip.decompress(raw).decode('utf-8', 'replace')
  _antibody_from_text(pdb, text)  # cache antibody flag during this decompression

  # locate _atom_site loop and its column headers
  lines = text.split('\n')
  col_idx = {}
  in_loop = False
  header_done = False
  cols = []
  res = {}
  aa3_to_1 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
              'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
              'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
              'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
  iatom = icomp = ichain = iseq = ix = iy = iz = -1
  for line in lines:
    if not header_done:
      s = line.strip()
      if s.startswith('_atom_site.'):
        in_loop = True
        cols.append(s.split('.')[1].split()[0])
        continue
      if in_loop and (s == '' or s.startswith('#') or (s and not s.startswith('_'))):
        if s == '' or s.startswith('#'):
          continue
        # first data row -> finalize column indices
        idx = {name: k for k, name in enumerate(cols)}
        iatom = idx.get('auth_atom_id', idx.get('label_atom_id', -1))
        icomp = idx.get('auth_comp_id', idx.get('label_comp_id', -1))
        ichain = idx.get('auth_asym_id', idx.get('label_asym_id', -1))
        iseq = idx.get('auth_seq_id', idx.get('label_seq_id', -1))
        ix = idx.get('Cartn_x', -1)
        iy = idx.get('Cartn_y', -1)
        iz = idx.get('Cartn_z', -1)
        header_done = True
        if min(iatom, icomp, ichain, iseq, ix, iy, iz) < 0:
          return None
        # fall through to process this data row
      else:
        continue
    if not header_done:
      continue
    s = line.strip()
    if not s or s.startswith('#') or s.startswith('_') or s.startswith('loop_'):
      if s.startswith('_') or s.startswith('loop_') or s.startswith('#'):
        break  # end of _atom_site loop
      continue
    tok = s.split()
    if len(tok) <= max(iatom, icomp, ichain, iseq, ix, iy, iz):
      continue
    if tok[ichain] != chain:
      continue
    atom = tok[iatom]
    if atom not in _BB:
      continue
    comp = tok[icomp]
    if comp not in _AA3:
      continue
    try:
      rid = int(tok[iseq])
      x = float(tok[ix]); y = float(tok[iy]); z = float(tok[iz])
    except ValueError:
      continue
    if rid not in res:
      res[rid] = [aa3_to_1[comp], {}]
    res[rid][1][atom] = (x, y, z)

  out = {}
  for rid, (aa, atoms) in res.items():
    if all(a in atoms for a in _ORDER):
      bb = np.array([atoms[a] for a in _ORDER], dtype=np.float32)
      out[rid] = (aa, bb)
  return out if out else None


if __name__ == '__main__':
  import os
  import time
  import sys
  HERE = os.path.dirname(os.path.abspath(__file__))
  sys.path.insert(0, HERE)
  archive = os.path.expanduser('~/shared_hosts/pdb_train.zip')
  from stage2_mine import extract_backbone as slow_extract
  tests = [('2oug', 'A'), ('6c6s', 'D'), ('1go4', 'A'), ('4kso', 'C'), ('1y6e', 'A')]
  for pdb, chain in tests:
    t0 = time.time(); fast = extract_backbone_fast(pdb, chain, archive); t_fast = time.time() - t0
    t0 = time.time(); slow = slow_extract(pdb, chain); t_slow = time.time() - t0
    # compare
    match = (fast is not None and slow is not None and set(fast) == set(slow))
    if match:
      maxerr = max(np.abs(fast[r][1] - slow[r][1]).max() for r in fast)
    else:
      maxerr = -1
    print('%s_%s: fast %.3fs (%d res) slow %.3fs (%d res) match=%s maxerr=%.4f' %
          (pdb, chain, t_fast, len(fast or {}), t_slow, len(slow or {}), match, maxerr))