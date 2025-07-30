import functools
import logging

from biotite import structure
from rdkit import Chem
import numpy as np

from profold2.common import residue_constants
from profold2.utils import exists

logger = logging.getLogger(__name__)

# NOTE: https://www.wwpdb.org/data/ccd

elem_type_num = 128
name_char_channel = 4
name_char_num = 64


@functools.cache
def element_idx(elem: str) -> int:
  t = Chem.GetPeriodicTable()
  # GetAtomicNumber() -> [1, ...]
  return min(t.GetAtomicNumber(elem), elem_type_num) - 1


@functools.cache
def atom_name_idx(atom_name: str) -> list[int]:
  name_chars = [0] * name_char_channel 
  for i, c in enumerate(atom_name[:name_char_channel]):
    name_chars[i] = min(ord(c) - 32, name_char_num - 1)
  return name_chars


@functools.cache
def residue_atom_array(
    residue_id: str, keep_leaving_atoms: bool = False, keep_hydrogens: bool = False
) -> structure.AtomArray:
  atom_array = structure.info.residue(residue_id)
  atom_category = structure.info.get_from_ccd('chem_comp_atom', residue_id)

  atom_array.set_annotation('mask', np.ones_like(atom_array, dtype=np.bool_))
  atom_array.set_annotation('charge', atom_category['charge'].as_array())
  for atom_id in ['alt_atom_id', 'pdbx_component_atom_id']:
    atom_array.set_annotation(atom_id, atom_category[atom_id].as_array())
  leaving_atom_flag = atom_category['pdbx_leaving_atom_flag'].as_array()
  atom_array.set_annotation('leaving_atom_flag', leaving_atom_flag == 'Y')

  if not keep_leaving_atoms:
    atom_array = atom_array[~atom_array.leaving_atom_flag]
  if not keep_hydrogens:
    atom_array = atom_array[~np.isin(atom_array.element, ['H', 'D'])]

  atom_list = residue_constants.restype_name_to_atom14_names[residue_id]
  atom_array = atom_array[np.isin(atom_array.atom_name, atom_list)]

  atom_array.set_annotation(
      'atom_within_token_idx',
      np.array(
          [atom_list.index(atom_name) for atom_name in atom_array.atom_name],
          dtype=np.int32
      )
  )
  atom_array.set_annotation(
      'element_idx', np.array([element_idx(elem) for elem in atom_array.element])
  )
  atom_array.set_annotation(
      'atom_name_chars',
      np.array([atom_name_idx(atom_name) for atom_name in atom_array.atom_name])
  )

  return atom_array


def polymer_atom_array(seq, seq_index) -> structure.AtomArray:
  atom_array = None  # structure.AtomArray(0)

  for token_idx, (restype, res_id) in enumerate(zip(seq, seq_index)):
    mol_type = residue_constants.moltype(restype)
    residue_id = residue_constants.restype_1to3[
        (residue_constants.restypes_with_x[restype], mol_type)
    ]
    residue = residue_atom_array(residue_id, keep_leaving_atoms=True)
    residue.res_id[:] = res_id
    residue.set_annotation(
        'atom_to_token_idx', np.full_like(residue, token_idx, dtype=np.int32)
    )
    residue.set_annotation('space_uid', np.full_like(residue, res_id, dtype=np.int32))
    if exists(atom_array):
      atom_array += residue
    else:
      atom_array = residue

  bonds = structure.connect_via_residue_names(atom_array)
  if exists(atom_array.bonds):
    atom_array.bonds = atom_array.bonds.merge(bonds)
  else:
    atom_array.bonds = bonds

  return atom_array
