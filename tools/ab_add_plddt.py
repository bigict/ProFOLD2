"""Tools for split complex to chain, run
     ```bash
     $python pdb_split.py -h
     ```
     for further help.
"""
import os
from collections import defaultdict
import functools
import glob
import gzip
import multiprocessing as mp
import logging

from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select
from Bio.PDB.mmcifio import MMCIFIO

import numpy as np

logger = logging.getLogger(__file__)

def get_antibody_regions(chain, chain_id, schema='imgt'):
  assert chain_id in 'HL'
  assert schema in ['chothia', 'imgt']
  cdr_def_chothia = {
      'H': {
          'fr1': (1, 25),
          'cdr1': (26, 32),
          'fr2': (33, 51),
          'cdr2': (52, 56),
          'fr3': (57, 94),
          'cdr3': (95, 102),
          'fr4': (103, 113),
      },
      'L': {
          'fr1': (1, 23),
          'cdr1': (24, 34),
          'fr2': (35, 49),
          'cdr2': (50, 56),
          'fr3': (57, 88),
          'cdr3': (89, 97),
          'fr4': (98, 109),
      }
  }

  cdr_def_imgt = {
      'H': {
          'fr1': (1, 26),
          'cdr1': (27, 38),
          'fr2': (39, 55),
          'cdr2': (56, 65),
          'fr3': (66, 104),
          'cdr3': (105, 117),
          'fr4': (118, 128),
      },
      'L': {
          'fr1': (1, 26),
          'cdr1': (27, 38),
          'fr2': (39, 55),
          'cdr2': (56, 65),
          'fr3': (66, 104),
          'cdr3': (105, 117),
          'fr4': (118, 128),
      },
  }

  cdr_def = cdr_def_imgt if schema == 'imgt' else cdr_def_chothia

  range_dict = cdr_def[chain_id]

  _schema = {
      'fr1': 1.0,
      'cdr1': 1.5,
      'fr2': 1.0,
      'cdr2': 1.5,
      'fr3': 1.0,
      'cdr3': 2.0,
      'fr4': 1.0
  }

  def _get_region(i):
    r = None
    for k, v in range_dict.items():
      if i >= v[0] and i <= v[1]:
        r = k
        break
    if r is None:
      return 1.0
    return _schema[r]

  N = 0
  for res in chain.get_residues():
    hetflag, resseq, icode = res.id
    if hetflag == ' ' and icode == ' ':
      N += 1
  region_def = np.full((N,), 1.0, dtype=np.float32)

  i = 0
  for res in chain.get_residues():
    hetflag, resseq, icode = res.id
    if hetflag == ' ' and icode == ' ':
      region_def[i] = _get_region(int(resseq))
      i += 1

  return region_def


def process(input_file, chain_dict=None):  # pylint: disable=redefined-outer-name
  logger.info('process: %s', input_file)

  input_pid = os.path.basename(input_file)
  if input_file.endswith('.gz'):
    input_pid, _ = os.path.splitext(input_pid)
  input_pid, input_type = os.path.splitext(input_pid)
  assert input_type in ('.cif', '.pdb')
  pdb_parser = PDBParser(
      QUIET=True) if input_type in ('.pdb',) else MMCIFParser(QUIET=True)
  try:
    if input_file.endswith('.gz'):
      with gzip.open(input_file, 'rt') as f:
        protein_structure = pdb_parser.get_structure('none', f)
    else:
      protein_structure = pdb_parser.get_structure('none', input_file)
    model = list(protein_structure.get_models())[0]
    chains = list(model.get_chains())

    chain_list = {}
    for heavy, light in chain_dict[input_pid]:
      chain_list[heavy] = 'H'
      chain_list[light] = 'L'

    for chain in chains:
      if chain.id in chain_list:
        region_def = get_antibody_regions(chain, chain_list[chain.id])
        region_def = np.repeat(np.reshape(region_def, (-1, 1)), 14, axis=-1)
        # print(region_def, region_def.shape, chain.id, chain_list[chain.id])
        npz_file = os.path.join(args.output, f'{input_pid}_{chain.id}.npz')
        if os.path.exists(npz_file):
          obj = np.load(npz_file, allow_pickle=True)
          assert obj['coord_mask'].shape == region_def.shape, (chain.id, obj['coord_mask'].shape, region_def.shape)
          np.savez(npz_file, coord=obj['coord'], coord_mask=obj['coord_mask'], bfactor=region_def)
  except Exception as e:
    logger.error('error: %s (%s)', input_file, str(e))

  return input_file

def read_summary_file(f):
  def _yield_item(f):
    for line in filter(lambda x: len(x) > 0, map(lambda x: x.strip(), f)):
      pid, heavy, light = line.split()
      yield pid, (heavy, light)

  chain_dict = defaultdict(list)
  for pid, item in _yield_item(f):
    chain_dict[pid].append(item)
  return chain_dict

def main(args):  # pylint: disable=redefined-outer-name
  logger.info('args - %s', args)

  with open(args.summary_file, 'r') as f:
    chain_dict = read_summary_file(f)

  os.makedirs(args.output, exist_ok=True)

  input_files = functools.reduce(
      lambda x, y: x + y,
      [glob.glob(input_file) for input_file in args.input_files])

  with mp.Pool() as p:
    for _ in p.imap(functools.partial(process, chain_dict=chain_dict), input_files):
      pass


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('input_files',
                      metavar='file',
                      type=str,
                      nargs='+',
                      help='input files')
  parser.add_argument('-o', '--output', type=str, default='.',
                      help='output dir, default=\'.\'')
  parser.add_argument('--summary_file',
                      type=str,
                      default=None,
                      help='type of output format, default=\'pdb\'')
  parser.add_argument('--cdr3_plddt',
                      type=float,
                      default=1.0,
                      help='type of output format, default=\'pdb\'')
  args = parser.parse_args()

  main(args)
