"""Tools for convert pdb and mmcif files, run
     ```bash
     $python pdb_convert.py -h
     ```
     for further help.
"""
import sys
import functools
import gzip
import multiprocessing as mp
import pathlib
import logging

from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBIO import PDBIO, Select
from Bio.PDB.PDBParser import PDBParser

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

def output_get_basename(filename):
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

def pdb_get_structure(filename):
  o = PDBParser(QUIET=True)
  if filename.suffix == '.gz':
    with gzip.open(filename, 'rt', encoding='utf-8') as f:
      return o.get_structure('1n2c', f)
  return o.get_structure('1n2c', filename)

def cif2pdb(mmcif_file, pdb_file):
  try:
    io = PDBIO()
    structure = mmcif_get_structure(mmcif_file)
    io.set_structure(structure)

    if pdb_file.suffix == '.gz':
      with gzip.open(pdb_file, 'wt') as f:
        io.save(f, select=CustomSelect())
    else:
      with open(pdb_file, 'w') as f:
        io.save(f, select=CustomSelect())

  except Exception as e:
    logger.error('%s, %s', mmcif_file, str(e))

  logger.info(mmcif_file)
  return mmcif_file

def pdb2cif(pdb_file, mmcif_file):
  io = MMCIFIO()
  try:
    io.set_structure(pdb_get_structure(pdb_file))

    if mmcif_file.suffix == '.gz':
      with gzip.open(mmcif_file, 'wt') as f:
        io.save(f)
    else:
      with open(mmcif_file, 'w') as f:
        io.save(f)

    logger.info(mmcif_file)
  except Exception as e:
    logger.error(mmcif_file)
  return pdb_file

def read_pairwise_list(f, output):
  for line in filter(lambda x: x, map(lambda x: x.strip(), f)):
    input_file, output_file = line.split('\t')
    input_file = pathlib.Path(input_file)
    output_file = pathlib.Path(output_file)
    if not output_file.is_absolute():
      output_file = output / output_file
    yield input_file, output_file

def work_fn_wrap(item, work_fn=None):
  input_file, output_file = item
  return work_fn(input_file, output_file)

def main(args, work_fn, fmt):  # pylint: disable=redefined-outer-name
  logger.debug(args)

  output = pathlib.Path(args.output)
  output.mkdir(parents=True, exist_ok=True)

  # read (input, output)
  pairwise_list = []
  if args.pairwise_list:
    if args.pairwise_list == '-':
      pairwise_list = [
          (input_file, output_file)
          for input_file, output_file in read_pairwise_list(sys.stdin, output)
      ]
    else:
      with open(args.pairwise_list, 'r') as f:
        pairwise_list = [
            (input_file, output_file)
            for input_file, output_file in read_pairwise_list(f, output)
        ]

  for p in args.files:
    for input_file in pathlib.Path().glob(p):
      output_file = output / output_get_basename(input_file)
      if args.gzip:
        output_file = output_file.with_suffix(f'{fmt}.gz')
      else:
        output_file = output_file.with_suffix(fmt)
      pairwise_list.append((input_file, output_file))

  with mp.Pool() as p:
    for _ in p.imap(functools.partial(work_fn_wrap, work_fn=work_fn),
                    pairwise_list):
      pass


if __name__ == '__main__':
  import argparse

  commands = {
    'pdb2cif': (pdb2cif, '.cif'),
    'cif2pdb': (cif2pdb, '.pdb'),
  }

  parser = argparse.ArgumentParser()
  sub_parsers = parser.add_subparsers(dest='command', required=True)
  for cmd in commands:
    cmd_parser = sub_parsers.add_parser(cmd)
    cmd_parser.add_argument('-o', '--output', type=str, default='.',
                            help='output dir, default=\'.\'')
    cmd_parser.add_argument('--gzip', action='store_true', help='verbose')
    cmd_parser.add_argument('-v', '--verbose', action='store_true',
                            help='verbose')
    cmd_parser.add_argument('-l', '--pairwise_list', default=None,
                            help='read pdb file from list.')
    cmd_parser.add_argument('files', type=str, nargs='*',
                            help='list of pdb/mmcif files')
  args = parser.parse_args()

  logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

  main(args, *commands[args.command])
