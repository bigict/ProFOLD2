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


def mmcif_yield_chain_type(mmcif_dict):
  for chain_type, chain_list in zip(
      mmcif_dict['_entity_poly.type'], mmcif_dict['_entity_poly.pdbx_strand_id']
  ):
    # NOTE: 'polydeoxyribonucleotide/polyribonucleotide hybrid'
    if chain_type.startswith('polypeptide'):
      chain_type = 'mol:protein'
    elif chain_type.find('polydeoxyribonucleotide') != -1:
      chain_type = 'mol:dna'
    elif chain_type.find('polyribonucleotide') != -1:
      chain_type = 'mol:rna'
    else:  # FIX: 3ok2, chain_type=other
      continue
    for chain in chain_list.split(','):
      yield chain, chain_type


def mmcif_yield_chain(mmcif_dict, _):  # pylint: disable=redefined-outer-name
  # two special chars as placeholders in the mmCIF format
  # for item values that cannot be explicitly assigned
  # see: pdbx/mmcif syntax web page
  _unassigned = {'.', '?'}  # pylint: disable=invalid-name

  atom_id_list = mmcif_dict['_atom_site.label_atom_id']
  residue_id_list = mmcif_dict['_atom_site.label_comp_id']
  auth_comp_id_list = mmcif_dict['_atom_site.auth_comp_id']
  # label_seq_id_list = mmcif_dict['_atom_site.auth_seq_id']
  label_seq_id_list = mmcif_dict['_atom_site.label_seq_id']
  auth_seq_id_list = mmcif_dict['_atom_site.auth_seq_id']
  chain_id_list = mmcif_dict['_atom_site.auth_asym_id']
  icode_list = mmcif_dict['_atom_site.pdbx_PDB_ins_code']
  fieldname_list = mmcif_dict['_atom_site.group_PDB']
  model_list = mmcif_dict['_atom_site.pdbx_PDB_model_num']
  mod_residue_id_list = collections.defaultdict(list)
  if '_pdbx_struct_mod_residue.id' in mmcif_dict:
    for asym_id, seq_id, comp_id, parent_comp_id in zip(
        mmcif_dict['_pdbx_struct_mod_residue.auth_asym_id'],
        mmcif_dict['_pdbx_struct_mod_residue.label_seq_id'],
        mmcif_dict['_pdbx_struct_mod_residue.label_comp_id'],
        mmcif_dict['_pdbx_struct_mod_residue.parent_comp_id']
    ):
      mod_residue_id_list[(asym_id, seq_id, comp_id)].append(parent_comp_id)
  elif '_struct_ref_seq_dif.mon_id' in mmcif_dict:  # FIX: 1xmz
    for asym_id, seq_id, comp_id, parent_comp_id in zip(
        mmcif_dict['_struct_ref_seq_dif.pdbx_pdb_strand_id'],
        mmcif_dict['_struct_ref_seq_dif.seq_num'],
        mmcif_dict['_struct_ref_seq_dif.mon_id'],
        mmcif_dict['_struct_ref_seq_dif.db_mon_id']
    ):
      if parent_comp_id in residue_constants.restypes_with_x:
        mod_residue_id_list[(asym_id, seq_id, comp_id)].append(parent_comp_id)

  chain_type_dict = dict(mmcif_yield_chain_type(mmcif_dict))

  def _get_residue_id(residue_id, chain_type):
    del chain_type
    while len(residue_id) < 3:
      residue_id = f' {residue_id}'
    if residue_id == 'MSE':
      return 'MET'
    return residue_id

  def _get_unktype(chain_type):
    if chain_type == 'mol:dna':
      return residue_constants.unk_dnatype
    elif chain_type == 'mol:rna':
      return residue_constants.unk_rnatype
    return residue_constants.unk_restype

  def _get_residue_letter(residue_id, chain_type):
    if residue_id not in residue_constants.restype_3to1:
      unktype = _get_unktype(chain_type)
      if unktype not in residue_constants.restype_3to1:
        return residue_constants.restypes_with_x[-1]
      letter = residue_constants.restype_3to1[unktype]
    else:
      letter = residue_constants.restype_3to1[residue_id]
    letter, _ = letter
    if chain_type in ('mol:dna', 'mol:rna'):
      letter = letter.upper()
    return letter

  chain_id, seq = None, []
  int_resseq_start, int_resseq_end = None, None
  for i, _ in enumerate(atom_id_list):
    if model_list[i] != model_list[0]:  # the 1st model only: 5w0s
      continue

    icode = icode_list[i]
    if icode in _unassigned:
      icode = ' '
    if icode and icode != ' ':
      continue

    if chain_id_list[i] != chain_id:
      if exists(chain_id) and seq:  # FIX: 5wj3
        yield chain_id, chain_type_dict[chain_id], seq
      chain_id, seq = chain_id_list[i], []
      # assert chain_id in chain_type_dict
      int_resseq_start, int_resseq_end = None, None

    chain_type = chain_type_dict.get(chain_id)
    if not exists(chain_type):  # FIX: 146d
      continue

    int_resseq = label_seq_id_list[i]
    residue_id = residue_id_list[i]
    if fieldname_list[i] == 'HETATM' and (
        chain_id, int_resseq, residue_id
    ) in mod_residue_id_list:
      residue_id_arr = mod_residue_id_list[(chain_id, int_resseq, residue_id)]
    elif fieldname_list[i] == 'ATOM':
      residue_id_arr = [residue_id]
    else:
      continue

    residue_id_arr = [
        _get_residue_id(residue_id, chain_type) for residue_id in residue_id_arr
    ]

    int_resseq = int(int_resseq)
    if not exists(int_resseq_start):
      int_resseq_start = int_resseq

    if exists(int_resseq_end) and int_resseq - int_resseq_end > 1:
      int_resseq_start = int_resseq

    if not exists(int_resseq_end) or int_resseq != int_resseq_end:
      if len(residue_id_arr) == 1:
        label_comp_id = _get_residue_letter(residue_id_arr[0], chain_type)
        auth_comp_id = _get_residue_letter(
            _get_residue_id(auth_comp_id_list[i], chain_type), chain_type
        )
        seq.append(
            (label_seq_id_list[i], label_comp_id, auth_seq_id_list[i], auth_comp_id)
        )

    int_resseq_end = int_resseq

  if exists(chain_id) and seq:
    yield chain_id, chain_type_dict[chain_id], seq


_exptl_method_dict = {
    'SOLUTION NMR': 'NMR',
    'SOLID-STATE NMR': 'NMR',
    'ELECTRON MICROSCOPY': 'EM',
    'X-RAY DIFFRACTION': 'diffraction',
    'FIBER DIFFRACTION': 'diffraction',
    'NEUTRON DIFFRACTION': 'diffraction',
    'FLUORESCENCE TRANSFER': 'NMR',
}


def exptl_method_fn(method):
  return _exptl_method_dict.get(method)


def process(input_file, args):  # pylint: disable=redefined-outer-name
  with timing(f'processing {input_file}', logger.info):
    mmcif_dict = mmcif_parse(input_file)
    assert mmcif_dict is not None

    revision = mmcif_dict.get('_pdbx_audit_revision_history.revision_date')
    if revision:
      revision = min(revision)

    method = mmcif_dict.get('_exptl.method')
    if method:
      # assert method[0] in _exptl_method_dict, (
      #     method,
      #     input_file,
      # )
      method = exptl_method_fn(method[0])

    results = []

    if not args.exptl_method or method in args.exptl_method:
      for chain, typ, seq in mmcif_yield_chain(mmcif_dict, args):
        results.append((chain, typ, seq))
  return input_file, results


def main(args):  # pylint: disable=redefined-outer-name
  logger.info('args - %s', args)

  input_files = functools.reduce(
      lambda x, y: x + y, [glob.glob(input_file) for input_file in args.input_files]
  )

  with timing('processing', logger.info):
    with open(args.output_file, 'w') as f:
      with mp.Pool() as p:
        for input_file, results in p.imap(
            functools.partial(process, args=args), input_files
        ):
          pid = output_get_basename(input_file)
          for chain, typ, seq in results:
            if chain and chain != '.' and not args.ignore_chain:
              fid = f'{pid}_{chain}'
            else:
              fid = pid

            for label_seq_id, label_comp_id, auth_seq_id, auth_comp_id in seq:
              mapping = f'{label_seq_id} {label_comp_id} {auth_seq_id} {auth_comp_id}'
              f.write(f'{fid} {typ} {mapping}\n')


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()

  parser.add_argument(
      '-o', '--output_file', type=str, default=None, help='output file, default=None'
  )
  parser.add_argument('--ignore_chain', action='store_true', help='ignore chain')
  parser.add_argument(
      '--exptl_method',
      type=str,
      default=None,
      nargs='*',
      choices=['NMR', 'EM', 'diffraction'],
      help='exptl method filter'
  )
  parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
  parser.add_argument(
      'input_files', metavar='file', type=str, nargs='+', help='input files'
  )
  args = parser.parse_args()

  logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

  main(args)
