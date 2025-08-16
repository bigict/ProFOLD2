import functools
import logging

from biotite import structure
import numpy as np

from profold2.common import residue_constants
from profold2.utils import exists

logger = logging.getLogger(__name__)

# NOTE: https://www.wwpdb.org/data/ccd


@functools.cache
def residue_atom_array(residue_id, keep_leaving_atoms=False, keep_hydrogens=False):
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

  return atom_array


def polymer_atom_array(seq, seq_index):
  atom_array = None  # structure.AtomArray(0)

  for token_idx, (restype, res_id) in enumerate(zip(seq, seq_index)):
    residue = residue_atom_array(
        residue_constants.restype_1to3[residue_constants.restypes[restype]],
        keep_leaving_atoms=True,
    )
    residue.res_id[:] = res_id
    residue.set_annotation('atom_to_token_idx', np.full_like(residue, token_idx))
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


if __name__ == '__main__':
  x = residue_atom_array('ARG')

  seq = np.asarray([1, 2, 4, 6])
  seq_index = np.asarray([0, 1, 2, 3])

  x = polymer_atom_array(seq, seq_index)
  print(x.bonds)
  print(x.coord)
  print(dir(x))
  print(x.element)
  print(x.atom_name)
  print(x.charge)
  print(x.mask)
  print(x.atom_to_token_idx)
