import os
import csv
import logging

import numpy as np
import torch

from profold2.common import protein, residue_constants
from profold2.data.dataset import ProteinStructureDataset
from profold2.utils import exists

logger = logging.getLogger(__file__)


def mutation_parse(cigar):
  def cigar_pase(t):
    assert t[0] == "S"
    return t[0], t[1], int(t[2: -1]), t[-1]
  yield from map(cigar_pase, cigar.split(":"))

def mutation_apply(seq, cigar, coord_mask=None, aa_offset=0):
  seq = torch.clone(seq)
  if exists(coord_mask):
    coord_mask = torch.clone(coord_mask)

  n = seq.shape[0]
  for op, aa_from, aa_idx, aa_to in mutation_parse(cigar):
    assert op == "S"
    idx = aa_idx + aa_offset
    assert 0 < idx <= n
    assert aa_from == residue_constants.restypes_with_x[seq[idx - 1]]
    if aa_to in residue_constants.restype_order_with_x:
      seq[idx - 1] = residue_constants.restype_order_with_x[aa_to]
    else:
      seq[idx - 1] = residue_constants.restype_order_with_x["X"]
      coord_mask[idx - 1, :] = False

  if exists(coord_mask):
    return seq, coord_mask
  return seq

def main(args):
  os.makedirs(args.output_dir, exist_ok=True)

  data = ProteinStructureDataset(data_dir=args.data_dir)
  feat = data.get_monomer(args.wildtype_pdb_id)

  cb_idx = residue_constants.atom_order["CB"]
  feat["coord_mask"][:, cb_idx:] = False
  if args.coord_pad:
    feat["coord_mask"][:, :cb_idx] = True
  for mutation_file in args.mutation_file:
    with open(mutation_file, "r") as f:
      reader = csv.DictReader(f)
      for i, row in enumerate(reader):
        pid = row["pid"]
        seq, coord_mask = mutation_apply(feat["seq"], row["cigar"],
                                         coord_mask=feat["coord_mask"],
                                         aa_offset=args.wildtype_aa_offset)
        prot = protein.Protein(aatype=np.array(seq),
                               atom_positions=np.array(feat["coord"]),
                               atom_mask=np.array(coord_mask),
                               residue_index=np.array(feat["seq_index"]) + 1,
                               chain_index=np.array(feat["seq_color"]) - 1,
                               b_factors=np.zeros_like(coord_mask))
        with open(os.path.join(args.output_dir, f"{pid}.pdb"), "w") as f:
          if args.verbose:
            logger.debug("write pdb: %s", pid)
          f.write(protein.to_pdb(prot))


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("mutation_file", type=str, nargs="+",
      help="mutation file.")
  parser.add_argument("-d", "--data_dir", type=str,
      help="fasta format")
  parser.add_argument("-o", "--output_dir", type=str,
      help="")
  parser.add_argument("-w", "--wildtype_pdb_id", type=str,
      help="")
  parser.add_argument("--wildtype_aa_offset", type=int, default=0,
      help="")
  parser.add_argument(
      "--coord_pad", action="store_true", help="coord padded")
  parser.add_argument(
      "-v", "--verbose", action="store_true", help="verbose")
  args = parser.parse_args()

  logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

  main(args)
