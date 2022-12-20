"""Tools for relax, run
     ```bash
     $python relaxer.py -h
     ```
     for further help.
"""
import glob
import os
import time
import logging

from profold2.common import protein
from profold2.relax import relax

# Internal import (7716).

def main(args):  # pylint: disable=redefined-outer-name
  amber_relaxer = relax.AmberRelaxation(
      max_iterations=relax.RELAX_MAX_ITERATIONS,
      tolerance=relax.RELAX_ENERGY_TOLERANCE,
      stiffness=relax.RELAX_STIFFNESS,
      exclude_residues=relax.RELAX_EXCLUDE_RESIDUES,
      max_outer_iterations=relax.RELAX_MAX_OUTER_ITERATIONS,
      use_gpu=args.use_gpu_relax)

  for pdb_files in args.pdb_files:
    for pdb_file in glob.glob(pdb_files):
      print(f'{pdb_files} -> {pdb_file}')
      # Relax the prediction.
      t_0 = time.time()
      with open(pdb_file, 'r', encoding='utf-8') as f:
        unrelaxed_protein = protein.from_pdb_string(f.read())
      relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
      relax_timings = time.time() - t_0
      print(f'{pdb_files} -> {pdb_file} Cost: {relax_timings}')

      os.makedirs(args.output, exist_ok=True)

      pid = os.path.basename(pdb_file)
      with open(os.path.join(args.output, f'{args.prefix}{pid}'), 'w',
          encoding='utf-8') as f:
        f.write(relaxed_pdb_str)

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('pdb_files', metavar='file', type=str, nargs='+',
      help='pdb files')
  parser.add_argument('-o', '--output', type=str, default='.',
      help='output directory default=\'.\'')
  parser.add_argument('--prefix', type=str, default='',
      help='prefix of relaxed protein')
  parser.add_argument('-g', '--use_gpu_relax', action='store_true',
      help='run relax on gpu')
  parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
  args = parser.parse_args()

  logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

  main(args)
