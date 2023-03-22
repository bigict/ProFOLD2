"""Tools for convert pdb to mmcif, run
     ```bash
     $python pdb2mmcif.py -h
     ```
     for further help.
"""
import gzip
import pathlib
import logging

from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.PDBParser import PDBParser

logger = logging.getLogger(__file__)

def mmcif_get_basename(filename):
  if filename.suffix == '.gz':
    filename = filename.stem
  return pathlib.Path(filename).name

def pdb_get_structure(filename):
  o = PDBParser(QUIET=True)
  if filename.suffix == '.gz':
    with gzip.open(filename, 'rt', encoding='utf-8') as f:
      return o.get_structure('1n2c', f)
  return o.get_structure('1n2c', filename)

def main(args):  # pylint: disable=redefined-outer-name
  output = pathlib.Path(args.output)
  output.mkdir(parents=True, exist_ok=True)

  for p in args.pdbfiles:
    for filename in pathlib.Path().glob(p):
      io = MMCIFIO()
      io.set_structure(pdb_get_structure(filename))

      mmcif_file = output / mmcif_get_basename(filename)
      if args.gzip:
        mmcif_file = mmcif_file.with_suffix('.cif.gz')
        with gzip.open(mmcif_file, 'wt') as f:
          io.save(f)
      else:
        mmcif_file = mmcif_file.with_suffix('.cif')
        with open(mmcif_file, 'w') as f:
          io.save(f)

      logger.info(mmcif_file)

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()

  parser.add_argument('-o', '--output', type=str, default='.',
      help='output dir, default=\'.\'')
  parser.add_argument('--gzip', action='store_true', help='verbose')
  parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
  parser.add_argument('pdbfiles', type=str, nargs='+',
      help='list of pdb files')
  args = parser.parse_args()

  logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

  main(args)
