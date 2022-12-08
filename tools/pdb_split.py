#!/Users/zhangyingchen/opt/anaconda3/envs/pytorch/bin/python3
import os
import functools
import glob
import gzip
import multiprocessing as mp
import logging

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select
from Bio.PDB.mmcifio import MMCIFIO
from Bio import SeqIO

logger = logging.getLogger(__file__)

restype_1to3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
}

restype_3to1 = {v: k for k, v in restype_1to3.items()}

class CustomSelect(Select):

  def __init__(self,
      model=None, chain_id=None, backbone_only=False, skip_het=False
  ):
    super().__init__()
    self.model = model
    self.backbone_only = backbone_only
    self.chain_id = chain_id
    self.skip_het = skip_het

  def accept_model(self, model):
    if self.model is None:
      return True
    return model == self.model

  def accept_chain(self, chain):
    if self.chain_id:
      return chain.id in self.chain_id
    else:
      return True

  def accept_atom(self, atom):
    if self.backbone_only:
      return atom.name in ['N', 'C', 'CA', 'O']
    else:
      return True

  def accept_residue(self, residue):
    if self.skip_het:
      return residue.id[0] == ' '
    else:
      return True

def process(input_file, args=None):  # pylint: disable=redefined-outer-name
  logger.info('process: %s', input_file)

  input_pid = os.path.basename(input_file)
  if input_file.endswith('.gz'):
    input_pid, _ = os.path.splitext(input_pid)
  input_pid, input_type = os.path.splitext(input_pid)
  assert input_type in ('.cif', '.pdb')
  parser = PDBParser(QUIET=True) if input_type in ('.pdb',) else MMCIFParser(QUIET=True)
  try:
    if input_file.endswith('.gz'):
      with gzip.open(input_file, 'rt') as f:
        protein_structure = parser.get_structure('none', f)
    else:
      protein_structure = parser.get_structure('none', input_file)
    model = list(protein_structure.get_models())[0]
    chains = model.get_chains()
    io = PDBIO() if args.pdb_fmt == 'pdb' else MMCIFIO()
    io.set_structure(protein_structure)
    for chain in chains:
      print(chain.get_full_id())
      io.save(os.path.join(args.prefix, f'{input_pid}_{chain.id}.{args.pdb_fmt}'),
              select=CustomSelect(
                  model,
                  chain.id, args.backbone_only, args.skip_het))
  except Exception as e:
    logger.error('error: %s', input_file)

  return input_file

def main(args):  # pylint: disable=redefined-outer-name
  logger.info('args - %s', args)

  os.makedirs(args.prefix, exist_ok=True)
  input_files = functools.reduce(lambda x,y: x+y,
      [glob.glob(input_file) for input_file in args.input_files])

  with mp.Pool() as p:
    for _ in p.imap(
        functools.partial(process, args=args), input_files):
      pass


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('input_files', metavar='file', type=str, nargs='+',
      help='input files')
  parser.add_argument('-o', '--prefix')
  parser.add_argument('--pdb_fmt', type=str, default='pdb',
      choices=['pdb', 'cif'],
      help='type of output format, default=\'pdb\'')
  parser.add_argument('-c', '--chain', nargs='*')
  parser.add_argument('-M', '--backbone_only', action='store_true')
  parser.add_argument('-s', '--skip_het', action='store_true')
  args = parser.parse_args()

  main(args)
