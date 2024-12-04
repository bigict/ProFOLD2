"""Tools for build dataset from cif, run
     ```bash
     $python dataset_from_pdb.py -h
     ```
     for further help.
  """
import os
import collections
import functools
import glob
import gzip
import multiprocessing as mp
import logging

import numpy as np

from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from profold2.common import residue_constants
from profold2.utils import exists, timing

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
  label_seq_id_list = mmcif_dict['_atom_site.label_seq_id']
  chain_id_list = mmcif_dict['_atom_site.auth_asym_id']
  x_list = [float(x) for x in mmcif_dict['_atom_site.Cartn_x']]
  y_list = [float(x) for x in mmcif_dict['_atom_site.Cartn_y']]
  z_list = [float(x) for x in mmcif_dict['_atom_site.Cartn_z']]
  icode_list = mmcif_dict['_atom_site.pdbx_PDB_ins_code']
  b_factor_list = mmcif_dict['_atom_site.B_iso_or_equiv']
  fieldname_list = mmcif_dict['_atom_site.group_PDB']
  model_list = mmcif_dict['_atom_site.pdbx_PDB_model_num']
  mod_residue_id_list = collections.defaultdict(list)
  if '_pdbx_struct_mod_residue.id' in mmcif_dict:
    for asym_id, seq_id, comp_id, parent_comp_id in zip(
        mmcif_dict['_pdbx_struct_mod_residue.auth_asym_id'],
        mmcif_dict['_pdbx_struct_mod_residue.label_seq_id'],
        mmcif_dict['_pdbx_struct_mod_residue.label_comp_id'],
        mmcif_dict['_pdbx_struct_mod_residue.parent_comp_id']):
      mod_residue_id_list[(asym_id, seq_id, comp_id)].append(parent_comp_id)
  elif '_struct_ref_seq_dif.mon_id' in mmcif_dict:  # FIX: 1xmz
    for asym_id, seq_id, comp_id, parent_comp_id in zip(
        mmcif_dict['_struct_ref_seq_dif.pdbx_pdb_strand_id'],
        mmcif_dict['_struct_ref_seq_dif.seq_num'],
        mmcif_dict['_struct_ref_seq_dif.mon_id'],
        mmcif_dict['_struct_ref_seq_dif.db_mon_id']):
      if parent_comp_id in residue_constants.restypes_with_x:
        mod_residue_id_list[(asym_id, seq_id, comp_id)].append(parent_comp_id)

  def _get_residue(i):
    if residue_id_list[i] == 'MSE':
      return 'MET'
    return residue_id_list[i]
  def _make_npz(coord_list, coord_mask_list, bfactor_list):
    npz = dict(coord=np.stack(coord_list, axis=0),
               coord_mask=np.stack(coord_mask_list, axis=0))
    if args.add_plddt:
      npz['bfactor'] = np.stack(bfactor_list, axis=0)
    return npz
  def _make_domain(start, end, delta=0):
    return (start + delta, end + delta)

  chain_id, seq, domains = None, [], []
  int_resseq_start, int_resseq_end = None, None
  int_resseq_delta, int_resseq_offset = 0, 0
  coord_list, coord_mask_list, bfactor_list = [], [], []
  labels, label_mask, bfactors = None, None, None
  for i, atom_id in enumerate(atom_id_list):
    if model_list[i] != model_list[0]:  # the 1st model only: 5w0s
      continue

    icode = icode_list[i]
    if icode in _unassigned:
      icode = ' '
    if icode and icode != ' ':
      continue

    if chain_id_list[i] != chain_id:
      if exists(chain_id) and seq:  # FIX: 5wj3
        domains += [
            _make_domain(int_resseq_start, int_resseq_end + int_resseq_offset, int_resseq_delta)]
        if exists(labels):
          coord_list.append(labels)
          coord_mask_list.append(label_mask)
          bfactor_list.append(bfactors)
        npz = _make_npz(coord_list, coord_mask_list, bfactor_list)
        yield chain_id, seq, domains, npz
      chain_id, seq, domains = chain_id_list[i], [], []
      int_resseq_start, int_resseq_end = None, None
      int_resseq_delta, int_resseq_offset = 0, 0
      coord_list, coord_mask_list, bfactor_list = [], [], []
      labels, label_mask, bfactors = None, None, None

    int_resseq = label_seq_id_list[i]
    residue_id = residue_id_list[i]
    if fieldname_list[i] == 'HETATM' and (chain_id, int_resseq, residue_id) in mod_residue_id_list:
      residue_id_arr = mod_residue_id_list[(chain_id, int_resseq, residue_id)]
    elif fieldname_list[i] == 'ATOM':
      residue_id_arr = [residue_id]
    else:
      continue

    int_resseq = int(int_resseq)
    if not exists(int_resseq_start):
      int_resseq_start = int_resseq
      if int_resseq_start < 0:
        int_resseq_delta = -int_resseq_start

    if exists(int_resseq_end) and int_resseq - int_resseq_end > 1:
      domains += [
          _make_domain(int_resseq_start, int_resseq_end + int_resseq_offset, int_resseq_delta)]
      int_resseq_start = int_resseq
      int_resseq_offset = 0

    if not exists(int_resseq_end) or int_resseq != int_resseq_end:
      for j, residue_id in enumerate(residue_id_arr):
        if residue_id == 'MSE':
          residue_id = 'MET'

        int_resseq_offset += (1 if j > 0 else 0)
        resname = residue_constants.restype_3to1.get(
            residue_id, residue_constants.restypes_with_x[-1])
        seq.append(resname)

        if exists(labels):
          coord_list.append(labels)
          coord_mask_list.append(label_mask)
          bfactor_list.append(bfactors)
        labels = np.zeros((14, 3), dtype=np.float32)
        label_mask = np.zeros((14,), dtype=np.bool_)
        bfactors = np.zeros((14,), dtype=np.float32)

    int_resseq_end = int_resseq


    if len(residue_id_arr) == 1:
      if residue_id in residue_constants.restype_name_to_atom14_names:
        res_atom14_list = residue_constants.restype_name_to_atom14_names[residue_id]  # pylint: disable=line-too-long
      else:
        res_atom14_list = residue_constants.restype_name_to_atom14_names[residue_constants.unk_restype]  # pylint: disable=line-too-long
      try:
        atom14idx = res_atom14_list.index(atom_id)
        coord = np.asarray((x_list[i], y_list[i], z_list[i]))
        if np.any(np.isnan(coord)):
          continue
        labels[atom14idx] = coord
        if np.any(coord != 0):
          # occupancy & B factor
          tempfactor = 0.0
          try:
            tempfactor = float(b_factor_list[i]) / 100.
          except ValueError as e:
            raise PDBConstructionException('Invalid or missing B factor') from e
          bfactors[atom14idx] = tempfactor
          label_mask[atom14idx] = True
      except ValueError as e:
        logger.debug(e)

  if exists(chain_id) and seq:
    domains += [
        _make_domain(int_resseq_start, int_resseq_end + int_resseq_offset, int_resseq_delta)]
    if exists(labels):
      coord_list.append(labels)
      coord_mask_list.append(label_mask)
      bfactor_list.append(bfactors)
    npz = _make_npz(coord_list, coord_mask_list, bfactor_list)
    yield chain_id, seq, domains, npz

def process(input_file, args=None):
  with timing(f'processing {input_file}', logger.info):
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

  with timing('processing', logger.info):
    with mp.Pool() as p:
      for input_file, results in p.imap(functools.partial(process, args=args),
                                        input_files):
        pid = output_get_basename(input_file)
        for chain, seq, domains, npz in results:
          seq = ''.join(seq)
          if chain and chain != '.' and not args.ignore_chain:
            fid = f'{pid}_{chain}'
          else:
            fid = pid

          if seq in mapping_dict:
            pk, sk_list = mapping_dict[seq]
            mapping_dict[seq] = (pk, sk_list | set([fid]))
          else:
            mapping_dict[seq] = (fid, set([fid]))
            with open(os.path.join(args.output, 'fasta', f'{fid}.fasta'),
                      'w') as f:
              if args.ignore_domain_parser:
                f.write(f'>{fid}\n')
              else:
                l = sum(map(lambda x: x[1] - x[0] + 1, domains))
                domain_str = ','.join(f'{i}-{j}' for i, j in domains)
                f.write(f'>{fid} domains:{domain_str} length={l}\n')
              f.write(seq)

          np.savez(os.path.join(args.output, 'npz', f'{fid}.npz'), **npz)

  with timing('writiing mapping.dict', logger.info):
    with open(os.path.join(args.output, 'mapping.dict'), 'w') as f:
      for v, (pk, sk_list) in mapping_dict.items():
        for sk in sk_list:
          f.write(f'{v}\t{pk}\t{sk}\n')

  with timing('writiing mapping.idx', logger.info):
    with open(os.path.join(args.output, 'mapping.idx'), 'w') as f:
      for _, (pk, sk_list) in mapping_dict.items():
        for sk in sk_list:
          f.write(f'{pk}\t{sk}\n')

  with timing('writiing name.idx', logger.info):
    with open(os.path.join(args.output, 'name.idx'), 'w') as f:
      for _, (_, sk_list) in mapping_dict.items():
        sk_list = ' '.join(sk_list)
        f.write(f'{sk_list}\n')

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()

  parser.add_argument('-o', '--output', type=str, default='.',
                      help='output dir, default=\'.\'')
  parser.add_argument('--mapping_idx', type=str, default=None,
                      help='bc-out.100')
  parser.add_argument('--add_plddt', action='store_true', help='add plddt')
  parser.add_argument('--ignore_domain_parser', action='store_true',
                      help='ignore domain parser')
  parser.add_argument('--ignore_chain', action='store_true',
                      help='ignore chain')
  parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
  parser.add_argument('input_files', metavar='file', type=str, nargs='+',
                      help='input files')
  args = parser.parse_args()

  logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

  main(args)
