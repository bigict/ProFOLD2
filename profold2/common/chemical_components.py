import concurrent.futures
import functools
import logging

from biotite import structure
from biotite.interface.rdkit import from_mol as FromMol
from rdkit.Chem import AllChem, GetPeriodicTable
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
  t = GetPeriodicTable()
  # GetAtomicNumber() -> [1, ...]
  return min(t.GetAtomicNumber(elem), elem_type_num) - 1


@functools.cache
def atom_name_idx(atom_name: str) -> list[int]:
  name_chars = [0] * name_char_channel
  for i, c in enumerate(atom_name[:name_char_channel]):
    name_chars[i] = min(ord(c) - 32, name_char_num - 1)
  return name_chars


@functools.cache
def has_residue(residue_id: str) -> bool:
  try:
    return exists(structure.info.residue(residue_id))
  except KeyError:
    return False


@functools.cache
def atom14_type_num(residue_id: str) -> int:
  if residue_id in residue_constants.restype_3to1:
    mol_idx = residue_constants.restype_order_with_x[
        residue_constants.restype_3to1[residue_id]
    ]
    mol_type = residue_constants.moltype(mol_idx)
    if mol_type == residue_constants.DNA:
      return residue_constants.dna_atom_type_num
    elif mol_type == residue_constants.RNA:
      return residue_constants.rna_atom_type_num
    return residue_constants.prot_atom_type_num
  return -1  # UNK, without padding


@functools.cache
def pad_virtual_atom_num(residue_id: str):
  _table = {
      'ASP': 3,
      'LEU': 6,
      'GLU': 5,
      'SER': 8,
      'VAL': 7,
        'U': 1,
  }
  return _table.get(residue_id, 0)


@functools.cache
def residue_atom_array(
    residue_id: str,
    keep_leaving_atoms: bool = True,
    pad_with_virtual_atoms: bool = False
) -> structure.AtomArray:
  assert has_residue(residue_id), (
      f'CCD: No atom information found for residue {residue_id}'
  )
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
  # remove hydrogens
  atom_array = atom_array[~np.isin(atom_array.element, ['H', 'D'])]

  atom_list = residue_constants.restype_name_to_atom14_names[residue_id]
  atom_array = atom_array[np.isin(atom_array.atom_name, atom_list)]
  atom_array.set_annotation(
      'atom_within_token_mask', np.ones_like(atom_array, dtype=np.bool_)
  )
  atom_array.set_annotation(
      'atom_padding_token_idx',
      np.array(
          [atom_list.index(atom_name) for atom_name in atom_array.atom_name],
          dtype=np.int32
      )
  )
  atom_array.set_annotation(
      'atom_repr_token_mask',
      np.array(
          [atom_name in ('CA', 'P') for atom_name in atom_array.atom_name],
          dtype=np.bool_
      )
  )

  def _pad_atoms(atom_array, atom_names, pad_length):
    if pad_length <= 0:
      return atom_array
    pad_atoms = atom_array[np.isin(atom_array.atom_name, atom_names)]
    pad_atoms.set_annotation(
        'atom_within_token_mask', np.zeros_like(pad_atoms.atom_within_token_mask)
    )
    pad_atoms.set_annotation(
        'atom_repr_token_mask', np.zeros_like(pad_atoms.atom_within_token_mask)
    )
    pad_atoms = structure.repeat(pad_atoms, np.stack([pad_atoms.coord] * pad_length))
    pad_atoms.set_annotation(
        'atom_padding_token_idx', np.arange(pad_length) + atom_array.array_length()
    )
    return structure.concatenate([atom_array, pad_atoms])

  if pad_with_virtual_atoms:
    atom_array = _pad_atoms(
        atom_array, ['O', 'O5\''],
        min(
            pad_virtual_atom_num(residue_id),
            atom14_type_num(residue_id) - atom_array.array_length()
        )
    )
    atom_array = _pad_atoms(
        atom_array, ['CA', 'P'], atom14_type_num(residue_id) - atom_array.array_length()
    )

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


def smiles_atom_array(ligand_string: str) -> structure.AtomArray:
  mol = AllChem.MolFromSmiles(ligand_string)
  # RDKit uses implicit hydrogen atoms by default, but Biotite requires explicit ones
  mol = AllChem.AddHs(mol)
  # create a 3D conformer
  with concurrent.futures.ThreadPoolExecutor() as executor:
    future = executor.submit(AllChem.EmbedMolecule, mol)
    try:
      conformer_id = future.result(timeout=90)
    except TimeoutError as e:
      raise TimeoutError('RDKit conformer generation timed out.') from e
  if conformer_id != 0:
    # retry with random coords
    conformer_id = AllChem.EmbedMolecule(mol, useRandomCoords=True)
  assert conformer_id == 0, (
      f'RDKit conformer generation failed for input SMILES: {ligand_string}'
  )
  AllChem.UFFOptimizeMolecule(mol)
  atom_array = FromMol(mol, conformer_id)

  # remove hydrogens
  atom_array = atom_array[~np.isin(atom_array.element, ['H', 'D'])]

  return atom_array


def polymer_atom_array(
    seq, seq_index, pad_with_virtual_atoms: bool = False
) -> structure.AtomArray:
  atom_array = None  # structure.AtomArray(0)

  for token_idx, (restype, res_id) in enumerate(zip(seq, seq_index)):
    mol_type = residue_constants.moltype(restype)
    residue_id = residue_constants.restype_1to3[
        (residue_constants.restypes_with_x[restype], mol_type)
    ]
    residue = residue_atom_array(
        residue_id, pad_with_virtual_atoms=pad_with_virtual_atoms
    )
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
