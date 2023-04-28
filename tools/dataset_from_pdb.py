"""Tools for build dataset from cif, run
     ```bash
     $python dataset_from_pdb.py -h
     ```
     for further help.
  """
import os
import functools
import glob
import gzip
import multiprocessing as mp
import logging

import numpy as np

from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from profold2.common import residue_constants
from profold2.utils import exists

logger = logging.getLogger(__file__)

def output_get_basename(filename):
  filename = os.path.basename(filename)
  if filename.endswith('.gz'):
    filename, _ = os.path.splitext(filename)
  name, _ = os.path.splitext(filename)
  return name

def mmcif_parse(filename):
  if filename.endswith('.gz'):
    with gzip.open(filename, 'rt') as f:
      return MMCIF2Dict(f)
  return MMCIF2Dict(filename)

def mmcif_yield_chain(mmcif_dict, args):
  # two special chars as placeholders in the mmCIF format
  # for item values that cannot be explicitly assigned
  # see: pdbx/mmcif syntax web page
  _unassigned = {'.', '?'}  # pylint: disable=invalid-name

  atom_id_list = mmcif_dict['_atom_site.label_atom_id']
  residue_id_list = mmcif_dict['_atom_site.label_comp_id']
  label_seq_id_list = mmcif_dict['_atom_site.auth_seq_id']
  chain_id_list = mmcif_dict['_atom_site.auth_asym_id']
  x_list = [float(x) for x in mmcif_dict['_atom_site.Cartn_x']]
  y_list = [float(x) for x in mmcif_dict['_atom_site.Cartn_y']]
  z_list = [float(x) for x in mmcif_dict['_atom_site.Cartn_z']]
  icode_list = mmcif_dict['_atom_site.pdbx_PDB_ins_code']
  b_factor_list = mmcif_dict['_atom_site.B_iso_or_equiv']
  fieldname_list = mmcif_dict['_atom_site.group_PDB']

  def _make_npz(coord_list, coord_mask_list, bfactor_list):
    npz = dict(cood=np.concatenate(coord_list), axis=0,
               cood_mask=np.concatenate(coord_mask_list, axis=0))
    if args.add_plddt:
      npz['bfactor'] = np.concatenate(bfactor_list, axis=0)
    return npz

  chain_id, seq, domains = None, [], []
  int_resseq_start, int_resseq_end = None, None
  coord_list, coord_mask_list, bfactor_list = [], [], []
  for i, atom_id in enumerate(atom_id_list):
    if fieldname_list[i] != 'ATOM':
      continue

    icode = icode_list[i]
    if icode in _unassigned:
      icode = ' '

    if chain_id_list[i] != chain_id:
      if exists(chain_id):
        npz = _make_npz(coord_list, coord_mask_list, bfactor_list)
        yield chain_id, seq, domains, npz
      chain_id, seq, domains = chain_id_list[i], [], []
      int_resseq_start, int_resseq_end = None, None

    int_resseq = int(label_seq_id_list[i])

    if not exists(int_resseq_start):
      int_resseq_start = int_resseq
    if exists(int_resseq_end) and int_resseq - int_resseq_end > 1:
      domains += [f'{int_resseq_start}-{int_resseq_end}']
      int_resseq_start = int_resseq
    if not exists(int_resseq_end) or int_resseq != int_resseq_end:
      resname = residue_constants.restype_3to1.get(
          residue_id_list[i], residue_constants.unk_restype)
      seq.append(resname)
    int_resseq_end = int_resseq


    if residue_id_list[i] in residue_constants.restype_name_to_atom14_names:
      res_atom14_list = residue_constants.restype_name_to_atom14_names[residue_id_list[i]]  # pylint: disable=line-too-long
    else:
      res_atom14_list = residue_constants.restype_name_to_atom14_names[residue_constants.unk_restype]  # pylint: disable=line-too-long
    labels = np.zeros((14, 3), dtype=np.float32)
    label_mask = np.zeros((14,), dtype=np.bool_)
    bfactors = np.zeros((14,), dtype=np.float32)

    try:
      atom14idx = res_atom14_list.index(atom_id)
      coord = np.asarray((x_list[i], y_list[i], z_list[i]))
      if np.any(np.isnan(coord)):
        continue
      if np.any(coord != 0):
        # occupancy & B factor
        tempfactor = 0.0
        try:
          tempfactor = float(b_factor_list[i])
        except ValueError as e:
          raise PDBConstructionException('Invalid or missing B factor') from e
        bfactors[atom14idx] = tempfactor
        label_mask[atom14idx] = True
    except ValueError as e:
      logger.debug(e)

    coord_list.append(labels)
    coord_mask_list.append(label_mask)
    bfactor_list.append(bfactors)
  if exists(chain_id):
    domains += [f'{int_resseq_start}-{int_resseq_end}']
    npz = _make_npz(coord_list, coord_mask_list, bfactor_list)
    yield chain_id, seq, domains, npz

def process(input_file, args=None):
  mmcif_dict = mmcif_parse(input_file)
  assert mmcif_dict is not None

  revision = mmcif_dict.get('_pdbx_audit_revision_history.revision_date')
  if revision:
    revision = min(revision)

  results = []
  for chain, seq, domains, npz in mmcif_yield_chain(mmcif_dict, args):
    npz['revision'] = revision
    results.append((chain, seq, domains, npz))
  return input_file, results

def main(args):  # pylint: disable=redefined-outer-name
  logger.info('args - %s', args)

  os.makedirs(os.path.join(args.output, 'fasta'), exist_ok=True)
  os.makedirs(os.path.join(args.output, 'npz'), exist_ok=True)

  input_files = functools.reduce(
      lambda x, y: x + y,
      [glob.glob(input_file) for input_file in args.input_files])

  mapping_dict = {}
  if args.mapping_idx:
    with open(args.mapping_idx, 'r') as f:
      for line in filter(lambda x: len(x) > 0, map(lambda x: x.strip(), f)):
        items = line.split()
        v, pk, sk_list = items[0], items[1], items[2:]
        if v in mapping_dict:
          assert pk == mapping_dict[v][0]
          mapping_dict[v] = (pk, mapping_dict[v][1] | set(sk_list))
        else:
          mapping_dict[v] = (pk, set(sk_list))

  with mp.Pool() as p:
    for input_file, results in p.imap(functools.partial(process, args=args),
                                      input_files):
      pid = output_get_basename(input_file).lower()
      for chain, seq, domains, npz in results:
        seq, domains = ''.join(seq), ','.join(domains)
        fid = f'{pid}_{chain}'

        if seq in mapping_dict:
          pk, sk_list = mapping_dict[seq]
          mapping_dict[seq] = (pk, sk_list | set([fid]))
        else:
          mapping_dict[seq] = (fid, set([fid]))

        with open(os.path.join(args.output, 'fasta', f'{fid}.fasta'),
                  'w') as f:
          f.write(f'>{pid}_{chain} {domains}\n')
          f.write(seq)

        np.savez(os.path.join(args.output, 'npz', f'{fid}.npz'), **npz)

  with open(os.path.join(args.output, 'mapping.idx'), 'w') as f:
    for v, (pk, sk_list) in mapping_dict.items():
      for sk in sk_list:
        f.write(f'{v}\t{pk}\t{sk}')


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()

  parser.add_argument('-o', '--output', type=str, default='.',
                      help='output dir, default=\'.\'')
  parser.add_argument('--mapping_idx', type=str, default=None,
                      help='bc-out.100')
  parser.add_argument('--add_plddt', action='store_true', help='add plddt')
  parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
  parser.add_argument('input_files', metavar='file', type=str, nargs='+',
                      help='input files')
  args = parser.parse_args()

  logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

  main(args)
