"""Tools for convert mmcif to pdb, run
     ```bash
     $python mmcif2pdb.py -h
     ```
     for further help.
"""
import functools
import gzip
import pathlib
import multiprocessing as mp
import logging

from Bio.PDB.PDBIO import PDBIO, Select
from Bio.PDB.MMCIFParser import MMCIFParser

logger = logging.getLogger(__file__)

class CustomSelect(Select):
  def accept_chain(self, chain):
    del chain
    return True

  def accept_atom(self, atom):
    del atom
    return True

def mmcif_rename_chains(structure):
  next_chain = 0

  # single-letters stay the same
  chainmap = {c.id: c.id for c in structure.get_chains() if len(c.id) == 1}
  for chain in filter(lambda c: len(c.id) != 1, structure.get_chains()):
    while True:
      new_chain_id = f'{next_chain}'
      next_chain += 1
      if new_chain_id not in chainmap:
        break
    chainmap[chain.id] = new_chain_id
    chain.id = new_chain_id
  return structure, dict(filter(lambda c: len(c[0]) != 1, chainmap.items()))

def mmcif_get_basename(filename):
  if filename.suffix == '.gz':
    filename = filename.stem
  return pathlib.Path(filename).name

def mmcif_get_structure(filename):
  o = MMCIFParser(QUIET=True)
  if filename.suffix == '.gz':
    with gzip.open(filename, 'rt', encoding='utf-8') as f:
      structure = o.get_structure('1n2c', f)
  else:
    structure = o.get_structure('1n2c', filename)
  return structure

def mmcif2pdb(filename, args=None):  # pylint: disable=redefined-outer-name
  output = pathlib.Path(args.output)
  try:
    io = PDBIO()
    structure = mmcif_get_structure(filename)
    io.set_structure(structure)

    mmcif_file = output / mmcif_get_basename(filename)
    if args.gzip:
      mmcif_file = mmcif_file.with_suffix('.pdb.gz')
      with gzip.open(mmcif_file, 'wt') as f:
        io.save(f, select=CustomSelect())
    else:
      mmcif_file = mmcif_file.with_suffix('.pdb')
      with open(mmcif_file, 'w') as f:
        io.save(f, select=CustomSelect())
    return mmcif_file
  except Exception as e:
    logger.error('%s, %s', filename, str(e))
  return filename

def main(args):  # pylint: disable=redefined-outer-name
  output = pathlib.Path(args.output)
  output.mkdir(parents=True, exist_ok=True)

  for p in args.pdbfiles:
    with mp.Pool() as pool:
      for filename in pool.imap(functools.partial(mmcif2pdb, args=args),
                                pathlib.Path().glob(p)):
        logger.info(filename)

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
