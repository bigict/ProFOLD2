import os
import functools
import io

import numpy as np
import torch
from torch import nn
from einops import rearrange

from profold2.common import residue_constants
from profold2.model import commons
from profold2.model import evoformer
from profold2.model import functional
from profold2.model import folding
from profold2.model import head
from profold2.model import alphafold2

def make_atom14_positions(protein, along_axis=1, prefix=''):
  restype_atom14_to_atom37 = []  # mapping (restype, atom14) --> atom37
  restype_atom37_to_atom14 = []  # mapping (restype, atom37) --> atom14
  restype_atom14_mask = []

  for rt in residue_constants.restypes_with_x:
    atom_names = residue_constants.restype_name_to_atom14_names[
        residue_constants.restype_1to3.get(rt, 'UNK')]

    restype_atom14_to_atom37.append([
        (residue_constants.atom_order[name] if name else 0)
        for name in atom_names
    ])

    atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
    restype_atom37_to_atom14.append([
        (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
        for name in residue_constants.atom_types
    ])

    restype_atom14_mask.append([(1. if name else 0.) for name in atom_names])

  # Add dummy mapping for restype 'UNK'.
  restype_atom14_to_atom37.append([0] * 14)
  restype_atom37_to_atom14.append([0] * 37)
  restype_atom14_mask.append([0.] * 14)

  restype_atom14_to_atom37 = np.array(restype_atom14_to_atom37, dtype=np.int32)
  restype_atom37_to_atom14 = np.array(restype_atom37_to_atom14, dtype=np.int32)
  restype_atom14_mask = np.array(restype_atom14_mask, dtype=np.float32)

  # Create the mapping for (residx, atom14) --> atom37, i.e. an array
  # with shape (num_res, 14) containing the atom37 indices for this protein.
  residx_atom14_to_atom37 = restype_atom14_to_atom37[protein[f'{prefix}aatype_index']]
  residx_atom14_mask = restype_atom14_mask[protein[f'{prefix}aatype_index']]

  # Create a mask for known ground truth positions.
  residx_atom14_gt_mask = residx_atom14_mask * np.take_along_axis(
      protein[f"{prefix}all_atom_mask"], residx_atom14_to_atom37, axis=along_axis).astype(np.float32)

  # Gather the ground truth positions.
  residx_atom14_gt_positions = residx_atom14_gt_mask[..., None] * (
      np.take_along_axis(protein[f"{prefix}all_atom_positions"],
                         residx_atom14_to_atom37[..., None],
                         axis=along_axis))

  protein[f"{prefix}atom14_atom_exists"] = residx_atom14_mask
  protein[f"{prefix}atom14_gt_exists"] = residx_atom14_gt_mask
  protein[f"{prefix}atom14_gt_positions"] = residx_atom14_gt_positions

  protein[f"{prefix}residx_atom14_to_atom37"] = residx_atom14_to_atom37

  # Create the gather indices for mapping back.
  residx_atom37_to_atom14 = restype_atom37_to_atom14[protein[f'{prefix}aatype_index']]
  protein[f"{prefix}residx_atom37_to_atom14"] = residx_atom37_to_atom14

  # Create the corresponding mask.
  restype_atom37_mask = np.zeros([21, 37], dtype=np.float32)
  for restype, restype_letter in enumerate(residue_constants.restypes):
    restype_name = residue_constants.restype_1to3[restype_letter]
    atom_names = residue_constants.residue_atoms[restype_name]
    for atom_name in atom_names:
      atom_type = residue_constants.atom_order[atom_name]
      restype_atom37_mask[restype, atom_type] = 1

  residx_atom37_mask = restype_atom37_mask[protein[f'{prefix}aatype_index']]
  protein[f"{prefix}atom37_atom_exists"] = residx_atom37_mask

  # As the atom naming is ambiguous for 7 of the 20 amino acids, provide
  # alternative ground truth coordinates where the naming is swapped
  restype_3 = [
      residue_constants.restype_1to3.get(res, 'UNK') for res in residue_constants.restypes_with_x
  ]
  restype_3 += ["UNK"]

  # Matrices for renaming ambiguous atoms.
  all_matrices = {res: np.eye(14, dtype=np.float32) for res in restype_3}
  for resname, swap in residue_constants.residue_atom_renaming_swaps.items():
    correspondences = np.arange(14)
    for source_atom_swap, target_atom_swap in swap.items():
      source_index = residue_constants.restype_name_to_atom14_names[
          resname].index(source_atom_swap)
      target_index = residue_constants.restype_name_to_atom14_names[
          resname].index(target_atom_swap)
      correspondences[source_index] = target_index
      correspondences[target_index] = source_index
      renaming_matrix = np.zeros((14, 14), dtype=np.float32)
      for index, correspondence in enumerate(correspondences):
        renaming_matrix[index, correspondence] = 1.
    all_matrices[resname] = renaming_matrix.astype(np.float32)
  renaming_matrices = np.stack([all_matrices[restype] for restype in restype_3])

  # Pick the transformation matrices for the given residue sequence
  # shape (num_res, 14, 14).
  renaming_transform = renaming_matrices[protein[f'{prefix}aatype_index']]

  # Apply it to the ground truth positions. shape (num_res, 14, 3).
  alternative_gt_positions = np.einsum("...rac,...rab->...rbc",
                                       residx_atom14_gt_positions,
                                       renaming_transform)
  protein[f"{prefix}atom14_alt_gt_positions"] = alternative_gt_positions

  # Create the mask for the alternative ground truth (differs from the
  # ground truth mask, if only one of the atoms in an ambiguous pair has a
  # ground truth position).
  alternative_gt_mask = np.einsum("...ra,...rab->...rb",
                                  residx_atom14_gt_mask,
                                  renaming_transform)

  protein[f"{prefix}atom14_alt_gt_exists"] = alternative_gt_mask

  # Create an ambiguous atoms mask.  shape: (21, 14).
  restype_atom14_is_ambiguous = np.zeros((21, 14), dtype=np.float32)
  for resname, swap in residue_constants.residue_atom_renaming_swaps.items():
    for atom_name1, atom_name2 in swap.items():
      restype = residue_constants.restype_order[
          residue_constants.restype_3to1[resname]]
      atom_idx1 = residue_constants.restype_name_to_atom14_names[resname].index(
          atom_name1)
      atom_idx2 = residue_constants.restype_name_to_atom14_names[resname].index(
          atom_name2)
      restype_atom14_is_ambiguous[restype, atom_idx1] = 1
      restype_atom14_is_ambiguous[restype, atom_idx2] = 1

  # From this create an ambiguous_mask for the given sequence.
  protein[f"{prefix}atom14_atom_is_ambiguous"] = (
      restype_atom14_is_ambiguous[protein[f'{prefix}aatype_index']])

  return protein

def batch_from_file(module_name, args):
  with open(os.path.join(args.output, f'{module_name}.npz'), 'rb') as f:
    data = np.load(io.BytesIO(f.read()), allow_pickle=True)

  return data

def fix_atom_order(batch):
  cb_idx = residue_constants.atom_order['CB']
  o_idx = residue_constants.atom_order['O']

  k = 'template_all_atom_positions'
  t = np.copy(batch[k][...,o_idx,:])
  batch[k][...,o_idx,:] = batch[k][...,cb_idx,:]
  batch[k][...,cb_idx,:] = t

  k = 'template_all_atom_masks'
  t = np.copy(batch[k][...,o_idx])
  batch[k][...,o_idx] = batch[k][...,cb_idx]
  batch[k][...,cb_idx] = t

  return batch

def functions_chain(x, fn_list=None):
  for fn in fn_list:
    x = fn(x)
  return x

def npz_to_tensor(x):
  return torch.as_tensor(x)

def weights_to_weights(x):
  return npz_to_tensor(rearrange(x, 'i o -> o i'))

def linear_to_embedding(weights, bias):
  return npz_to_tensor(weights + bias)

def chunk_weights_to_one(x, fn=weights_to_weights):
  if fn:
    x = [fn(c)for c in x]
  return torch.cat(x, dim=0)

def chunk_bias_to_one(x, fn=npz_to_tensor):
  if fn:
    x = [fn(c)for c in x]
  return torch.cat(x, dim=0)

def weights_to_chunk(x, from_idx, to_idx):
  return weights_to_weights(x[:, from_idx:to_idx])

def npz_to_chunk(x, from_idx, to_idx):
  return npz_to_tensor(x[from_idx:to_idx])

def layer_stack_to_module_list(x, i=0, fn=npz_to_tensor):
  if isinstance(x, list):
    x = [a[i] for a in x]
  else:
    x = x[i]
  return fn(x)

def params_get(params, v):
  if isinstance(v, list):
    return [params_get(params, i) for i in v]
  return params[v]

def state_dict_get(params, scope_list):
  state_dict = {}
  for key, (cb, args) in scope_list.items():
    args = dict([(k, params_get(params, v)) for k, v in args.items()])
    state_dict[key] = cb(**args) 
    if key == 'template_single_embedder.0.weight':
      print(key, state_dict[key])
      print(key, args)
  return state_dict

def input_embedder(args):
  pass

def make_features(args):
  if args.output:
    with open(os.path.join(args.output, f'make_features.npz'), 'rb') as f:
      data = np.load(io.BytesIO(f.read()), allow_pickle=True)

    feature_dict, result = data['feature_dict'], data['batch']
    feature_dict = feature_dict.tolist()
    result = result.tolist()

    protein = {}
    protein['pid'] = feature_dict['domain_name'].astype(str)[0]
    protein['seq'] = np.argmax(feature_dict['aatype'], axis=-1)
    protein['seq_index'] = feature_dict['residue_index']
    protein['str_seq'] = feature_dict['sequence'].astype(str)[0]

    new_order = np.asarray(residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE)
    protein['msa'] = new_order[feature_dict['msa']]
    protein['deletion_matrix'] = feature_dict['deletion_matrix_int'].astype(np.float32)

    protein['template_aatype'] = new_order[np.argmax(feature_dict['template_aatype'], axis=-1)]
    for k, v in protein.items():
      if k.startswith('template_'):
        protein[k] = v[:4,...]
    assert np.all(protein['template_aatype'] == result['template_aatype'])

    print(protein)
    print(result)

def distogram_from_positions(args):
  if args.output:
    with open(os.path.join(args.output, f'distogram_from_positions.npz'), 'rb') as f:
      data = np.load(io.BytesIO(f.read()), allow_pickle=True)

    coords, result = data['coords'], data['result']
    coords = rearrange(torch.from_numpy(coords), 'i d -> () i d')

    breaks = torch.linspace(float(data['min_bin']),
            float(data['max_bin']), steps=int(data['num_bins']))
    r = functional.distogram_from_positions(coords, breaks)
    print(r.shape)
    print(result.shape)
    print(torch.argmax(r, dim=-1))
    print(np.argmax(result, axis=-1))

def angles_from_positions(args):
  if args.output:
    with open(os.path.join(args.output, f'angles_from_positions.npz'), 'rb') as f:
      data = np.load(io.BytesIO(f.read()), allow_pickle=True)

    batch, result = data['batch'], data['result']
    batch = batch.tolist()
    batch = fix_atom_order(batch)
    template_batch = {k: batch[k] for k in batch if k.startswith('template_')}  # all template

    template_batch['template_aatype_index'] = np.clip(template_batch['template_aatype'], 0, 20)
    template_batch['template_all_atom_mask'] = template_batch['template_all_atom_masks']
    print(template_batch.keys())
    template_batch = make_atom14_positions(template_batch, along_axis=2, prefix='template_')
    print(template_batch.keys())

    for k in template_batch:
      if (isinstance(template_batch[k], np.ndarray)):
        template_batch[k] = torch.from_numpy(template_batch[k])
        print(k, template_batch[k].shape)

    template_angles = functional.angles_from_positions(template_batch['template_aatype'], template_batch['template_atom14_gt_positions'], template_batch['template_atom14_gt_exists'], placeholder_for_undefined=False)
    print(template_angles)
    print(result)

def template_pair_stack(args):
  def attention_with_head(x, fn=None):
    if isinstance(x, list):
      x = list(map(lambda a: rearrange(a, '... h d -> ... (h d)'), x))
    else:
      x = rearrange(x, '... h d -> ... (h d)')
    return fn(x)

  def proj_out_weight(x):
    x = rearrange(x, 'h d ... -> (h d) ...')
    return weights_to_weights(x)

  def proj_out_rearrange(x):
    x = rearrange(x, 'c d f -> (c d) f')
    return weights_to_weights(x)
  proj_out_wrap = functools.partial(layer_stack_to_module_list, fn=proj_out_rearrange)

  npz_to_tensor_fn = functools.partial(attention_with_head, fn=npz_to_tensor)
  weights_to_weights_fn = functools.partial(attention_with_head, fn=weights_to_weights)
  chunk_weights_to_one_fn = functools.partial(attention_with_head, fn=chunk_weights_to_one)
  chunk_bias_to_one_fn = functools.partial(attention_with_head, fn=chunk_bias_to_one)

  npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, fn=npz_to_tensor)
  weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, fn=weights_to_weights)
  chunk_weights_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_weights_to_one)
  chunk_bias_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_bias_to_one)

  attn_npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, fn=npz_to_tensor_fn)
  attn_weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, fn=weights_to_weights_fn)
  attn_chunk_weights_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_weights_to_one_fn)
  attn_chunk_bias_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_bias_to_one_fn)
  attn_proj_out_wrap = functools.partial(layer_stack_to_module_list, fn=proj_out_weight)



  scope_iteration = {
     'layer.0.triangle_multiply_outgoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/layer_norm_input//scale')),
     'layer.0.triangle_multiply_outgoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/layer_norm_input//offset')),
     'layer.0.triangle_multiply_outgoing.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/left_projection//weights')),
     'layer.0.triangle_multiply_outgoing.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/left_projection//bias')),
     'layer.0.triangle_multiply_outgoing.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/right_projection//weights')),
     'layer.0.triangle_multiply_outgoing.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/right_projection//bias')),
     'layer.0.triangle_multiply_outgoing.left_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/left_gate//weights')),
     'layer.0.triangle_multiply_outgoing.left_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/left_gate//bias')),
     'layer.0.triangle_multiply_outgoing.right_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/right_gate//weights')),
     'layer.0.triangle_multiply_outgoing.right_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/right_gate//bias')),
     'layer.0.triangle_multiply_outgoing.out_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/gating_linear//weights')),
     'layer.0.triangle_multiply_outgoing.out_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/gating_linear//bias')),
     'layer.0.triangle_multiply_outgoing.to_out_norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/center_layer_norm//scale')),
     'layer.0.triangle_multiply_outgoing.to_out_norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/center_layer_norm//offset')),
     'layer.0.triangle_multiply_outgoing.to_out.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/output_projection//weights')),
     'layer.0.triangle_multiply_outgoing.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/output_projection//bias')),
     'layer.0.triangle_multiply_ingoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/layer_norm_input//scale')),
     'layer.0.triangle_multiply_ingoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/layer_norm_input//offset')),
     'layer.0.triangle_multiply_ingoing.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/left_projection//weights')),
     'layer.0.triangle_multiply_ingoing.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/left_projection//bias')),
     'layer.0.triangle_multiply_ingoing.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/right_projection//weights')),
     'layer.0.triangle_multiply_ingoing.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/right_projection//bias')),
     'layer.0.triangle_multiply_ingoing.left_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/left_gate//weights')),
     'layer.0.triangle_multiply_ingoing.left_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/left_gate//bias')),
     'layer.0.triangle_multiply_ingoing.right_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/right_gate//weights')),
     'layer.0.triangle_multiply_ingoing.right_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/right_gate//bias')),
     'layer.0.triangle_multiply_ingoing.out_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/gating_linear//weights')),
     'layer.0.triangle_multiply_ingoing.out_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/gating_linear//bias')),
     'layer.0.triangle_multiply_ingoing.to_out_norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/center_layer_norm//scale')),
     'layer.0.triangle_multiply_ingoing.to_out_norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/center_layer_norm//offset')),
     'layer.0.triangle_multiply_ingoing.to_out.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/output_projection//weights')),
     'layer.0.triangle_multiply_ingoing.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/output_projection//bias')),
     'layer.0.triangle_attention_outgoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/query_norm//scale')),
     'layer.0.triangle_attention_outgoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/query_norm//offset')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/query_norm//scale')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/query_norm//offset')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node//feat_2d_weights')),
     'layer.0.triangle_attention_outgoing.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//query_w')),
     'layer.0.triangle_attention_outgoing.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//key_w', 'alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//value_w'])),
     'layer.0.triangle_attention_outgoing.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//gating_w')),
     'layer.0.triangle_attention_outgoing.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//gating_b')),
     'layer.0.triangle_attention_outgoing.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//output_w')),
     'layer.0.triangle_attention_outgoing.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//output_b')),
     'layer.0.triangle_attention_ingoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/query_norm//scale')),
     'layer.0.triangle_attention_ingoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/query_norm//offset')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/query_norm//scale')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/query_norm//offset')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node//feat_2d_weights')),
     'layer.0.triangle_attention_ingoing.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//query_w')),
     'layer.0.triangle_attention_ingoing.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//key_w', 'alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//value_w'])),
     'layer.0.triangle_attention_ingoing.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//gating_w')),
     'layer.0.triangle_attention_ingoing.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//gating_b')),
     'layer.0.triangle_attention_ingoing.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//output_w')),
     'layer.0.triangle_attention_ingoing.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//output_b')),
     'layer.1.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/input_layer_norm//scale')),
     'layer.1.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/input_layer_norm//offset')),
     'layer.1.net.0.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/transition1//weights')),
     'layer.1.net.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/transition1//bias')),
     'layer.1.net.3.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/transition2//weights')),
     'layer.1.net.3.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/transition2//bias')),
  }
  layers = 2

  scope_list = {}
  for i in range(layers):
    for k, (fn, kwargs) in scope_iteration.items():
      scope_list[f'layers.{i}.{k}'] = (functools.partial(fn, i=i), kwargs)

  with open(os.path.join(args.data_dir, 'params', f'params_{args.model_name}.npz'), 'rb') as f:
    params = np.load(io.BytesIO(f.read()), allow_pickle=False)
  state_dict = state_dict_get(params, scope_list)

  template_pair_stack = evoformer.TemplatePairStack(depth=layers,
                                                    dim=64,
                                                    heads=4,
                                                    dim_head=16,
                                                    attn_dropout=0.25,
                                                    ff_dropout=0)
  print(template_pair_stack)
  template_pair_stack.load_state_dict(state_dict)
  if args.output:
    with open(os.path.join(args.output, f'template_pair_stack.npz'), 'rb') as f:
      data = np.load(io.BytesIO(f.read()), allow_pickle=True)

    pair_mask, pair_act, result = data['pair_mask'], data['pair_act'], data['result']
    pair_act = rearrange(torch.from_numpy(pair_act), 'i j d -> () i j d')
    pair_mask = rearrange(torch.from_numpy(pair_mask), 'i j -> () i j')

    template_pair_stack.eval()
    with torch.no_grad():
      print(pair_act.shape)
      r = template_pair_stack(pair_act.float(), pair_mask.float())
    print(r)
    print(result)

def single_template_embedding(args):
  def attention_with_head(x, fn=None):
    if isinstance(x, list):
      x = list(map(lambda a: rearrange(a, '... h d -> ... (h d)'), x))
    else:
      x = rearrange(x, '... h d -> ... (h d)')
    return fn(x)

  def proj_out_weight(x):
    x = rearrange(x, 'h d ... -> (h d) ...')
    return weights_to_weights(x)

  def proj_out_rearrange(x):
    x = rearrange(x, 'c d f -> (c d) f')
    return weights_to_weights(x)
  proj_out_wrap = functools.partial(layer_stack_to_module_list, fn=proj_out_rearrange)

  npz_to_tensor_fn = functools.partial(attention_with_head, fn=npz_to_tensor)
  weights_to_weights_fn = functools.partial(attention_with_head, fn=weights_to_weights)
  chunk_weights_to_one_fn = functools.partial(attention_with_head, fn=chunk_weights_to_one)
  chunk_bias_to_one_fn = functools.partial(attention_with_head, fn=chunk_bias_to_one)

  npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, fn=npz_to_tensor)
  weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, fn=weights_to_weights)
  chunk_weights_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_weights_to_one)
  chunk_bias_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_bias_to_one)

  attn_npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, fn=npz_to_tensor_fn)
  attn_weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, fn=weights_to_weights_fn)
  attn_chunk_weights_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_weights_to_one_fn)
  attn_chunk_bias_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_bias_to_one_fn)
  attn_proj_out_wrap = functools.partial(layer_stack_to_module_list, fn=proj_out_weight)



  scope_iteration = {
     'layer.0.triangle_multiply_outgoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/layer_norm_input//scale')),
     'layer.0.triangle_multiply_outgoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/layer_norm_input//offset')),
     'layer.0.triangle_multiply_outgoing.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/left_projection//weights')),
     'layer.0.triangle_multiply_outgoing.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/left_projection//bias')),
     'layer.0.triangle_multiply_outgoing.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/right_projection//weights')),
     'layer.0.triangle_multiply_outgoing.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/right_projection//bias')),
     'layer.0.triangle_multiply_outgoing.left_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/left_gate//weights')),
     'layer.0.triangle_multiply_outgoing.left_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/left_gate//bias')),
     'layer.0.triangle_multiply_outgoing.right_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/right_gate//weights')),
     'layer.0.triangle_multiply_outgoing.right_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/right_gate//bias')),
     'layer.0.triangle_multiply_outgoing.out_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/gating_linear//weights')),
     'layer.0.triangle_multiply_outgoing.out_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/gating_linear//bias')),
     'layer.0.triangle_multiply_outgoing.to_out_norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/center_layer_norm//scale')),
     'layer.0.triangle_multiply_outgoing.to_out_norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/center_layer_norm//offset')),
     'layer.0.triangle_multiply_outgoing.to_out.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/output_projection//weights')),
     'layer.0.triangle_multiply_outgoing.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/output_projection//bias')),
     'layer.0.triangle_multiply_ingoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/layer_norm_input//scale')),
     'layer.0.triangle_multiply_ingoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/layer_norm_input//offset')),
     'layer.0.triangle_multiply_ingoing.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/left_projection//weights')),
     'layer.0.triangle_multiply_ingoing.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/left_projection//bias')),
     'layer.0.triangle_multiply_ingoing.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/right_projection//weights')),
     'layer.0.triangle_multiply_ingoing.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/right_projection//bias')),
     'layer.0.triangle_multiply_ingoing.left_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/left_gate//weights')),
     'layer.0.triangle_multiply_ingoing.left_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/left_gate//bias')),
     'layer.0.triangle_multiply_ingoing.right_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/right_gate//weights')),
     'layer.0.triangle_multiply_ingoing.right_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/right_gate//bias')),
     'layer.0.triangle_multiply_ingoing.out_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/gating_linear//weights')),
     'layer.0.triangle_multiply_ingoing.out_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/gating_linear//bias')),
     'layer.0.triangle_multiply_ingoing.to_out_norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/center_layer_norm//scale')),
     'layer.0.triangle_multiply_ingoing.to_out_norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/center_layer_norm//offset')),
     'layer.0.triangle_multiply_ingoing.to_out.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/output_projection//weights')),
     'layer.0.triangle_multiply_ingoing.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/output_projection//bias')),
     'layer.0.triangle_attention_outgoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/query_norm//scale')),
     'layer.0.triangle_attention_outgoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/query_norm//offset')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/query_norm//scale')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/query_norm//offset')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node//feat_2d_weights')),
     'layer.0.triangle_attention_outgoing.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//query_w')),
     'layer.0.triangle_attention_outgoing.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//key_w', 'alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//value_w'])),
     'layer.0.triangle_attention_outgoing.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//gating_w')),
     'layer.0.triangle_attention_outgoing.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//gating_b')),
     'layer.0.triangle_attention_outgoing.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//output_w')),
     'layer.0.triangle_attention_outgoing.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//output_b')),
     'layer.0.triangle_attention_ingoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/query_norm//scale')),
     'layer.0.triangle_attention_ingoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/query_norm//offset')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/query_norm//scale')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/query_norm//offset')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node//feat_2d_weights')),
     'layer.0.triangle_attention_ingoing.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//query_w')),
     'layer.0.triangle_attention_ingoing.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//key_w', 'alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//value_w'])),
     'layer.0.triangle_attention_ingoing.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//gating_w')),
     'layer.0.triangle_attention_ingoing.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//gating_b')),
     'layer.0.triangle_attention_ingoing.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//output_w')),
     'layer.0.triangle_attention_ingoing.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//output_b')),
     'layer.1.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/input_layer_norm//scale')),
     'layer.1.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/input_layer_norm//offset')),
     'layer.1.net.0.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/transition1//weights')),
     'layer.1.net.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/transition1//bias')),
     'layer.1.net.3.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/transition2//weights')),
     'layer.1.net.3.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/transition2//bias')),
  }

  scope_list = {
      'to_pair.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/embedding2d//weights')),
      'to_pair.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/embedding2d//bias')),
      'to_out_norm.weight': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/output_layer_norm//scale')),
      'to_out_norm.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/output_layer_norm//offset')),
  }

  layers = 2
  for i in range(layers):
    for k, (fn, kwargs) in scope_iteration.items():
      scope_list[f'pair_stack.layers.{i}.{k}'] = (functools.partial(fn, i=i), kwargs)

  with open(os.path.join(args.data_dir, 'params', f'params_{args.model_name}.npz'), 'rb') as f:
    params = np.load(io.BytesIO(f.read()), allow_pickle=False)
  state_dict = state_dict_get(params, scope_list)

  template_pair_stack = evoformer.SingleTemplateEmbedding(
                                                    depth=layers,
                                                    dim=64,
                                                    heads=4,
                                                    dim_head=16,
                                                    attn_dropout=0.25,
                                                    ff_dropout=0)
  print(template_pair_stack)
  template_pair_stack.load_state_dict(state_dict)
  if args.output:
    with open(os.path.join(args.output, f'single_template_embedding.npz'), 'rb') as f:
      data = np.load(io.BytesIO(f.read()), allow_pickle=True)

    pair_act, result = data['pair_act'], data['result']
    pair_act = rearrange(torch.from_numpy(pair_act), 'i j d -> () i j d')
    # pair_mask = rearrange(torch.from_numpy(pair_mask), 'i j -> () i j')
    batch = data['batch'].tolist()
    mask_2d = batch['seq_mask'][:, None] * batch['seq_mask'][None, :]
    mask_2d = rearrange(torch.from_numpy(mask_2d), 'i j -> () i j')
    template_batch = {k: batch[k][0:1,...] for k in batch if k.startswith('template_')}  # template 0
    for k in template_batch:
      if (isinstance(template_batch[k], np.ndarray)):
        template_batch[k] = rearrange(torch.from_numpy(template_batch[k]), '... -> () ...')

    template_pair_stack.eval()
    with torch.no_grad():
      print(pair_act.shape)
      r = template_pair_stack(template_batch, mask_2d)
    print(r)
    print(result)

def template_embedding(args):
  def attention_with_head(x, fn=None):
    if isinstance(x, list):
      x = list(map(lambda a: rearrange(a, '... h d -> ... (h d)'), x))
    else:
      x = rearrange(x, '... h d -> ... (h d)')
    return fn(x)

  def proj_out_weight(x):
    x = rearrange(x, 'h d ... -> (h d) ...')
    return weights_to_weights(x)

  def proj_out_rearrange(x):
    x = rearrange(x, 'c d f -> (c d) f')
    return weights_to_weights(x)
  proj_out_wrap = functools.partial(layer_stack_to_module_list, fn=proj_out_rearrange)

  npz_to_tensor_fn = functools.partial(attention_with_head, fn=npz_to_tensor)
  weights_to_weights_fn = functools.partial(attention_with_head, fn=weights_to_weights)
  chunk_weights_to_one_fn = functools.partial(attention_with_head, fn=chunk_weights_to_one)
  chunk_bias_to_one_fn = functools.partial(attention_with_head, fn=chunk_bias_to_one)

  npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, fn=npz_to_tensor)
  weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, fn=weights_to_weights)
  chunk_weights_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_weights_to_one)
  chunk_bias_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_bias_to_one)

  attn_npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, fn=npz_to_tensor_fn)
  attn_weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, fn=weights_to_weights_fn)
  attn_chunk_weights_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_weights_to_one_fn)
  attn_chunk_bias_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_bias_to_one_fn)
  attn_proj_out_wrap = functools.partial(layer_stack_to_module_list, fn=proj_out_weight)



  scope_iteration = {
     'layer.0.triangle_multiply_outgoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/layer_norm_input//scale')),
     'layer.0.triangle_multiply_outgoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/layer_norm_input//offset')),
     'layer.0.triangle_multiply_outgoing.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/left_projection//weights')),
     'layer.0.triangle_multiply_outgoing.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/left_projection//bias')),
     'layer.0.triangle_multiply_outgoing.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/right_projection//weights')),
     'layer.0.triangle_multiply_outgoing.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/right_projection//bias')),
     'layer.0.triangle_multiply_outgoing.left_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/left_gate//weights')),
     'layer.0.triangle_multiply_outgoing.left_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/left_gate//bias')),
     'layer.0.triangle_multiply_outgoing.right_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/right_gate//weights')),
     'layer.0.triangle_multiply_outgoing.right_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/right_gate//bias')),
     'layer.0.triangle_multiply_outgoing.out_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/gating_linear//weights')),
     'layer.0.triangle_multiply_outgoing.out_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/gating_linear//bias')),
     'layer.0.triangle_multiply_outgoing.to_out_norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/center_layer_norm//scale')),
     'layer.0.triangle_multiply_outgoing.to_out_norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/center_layer_norm//offset')),
     'layer.0.triangle_multiply_outgoing.to_out.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/output_projection//weights')),
     'layer.0.triangle_multiply_outgoing.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/output_projection//bias')),
     'layer.0.triangle_multiply_ingoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/layer_norm_input//scale')),
     'layer.0.triangle_multiply_ingoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/layer_norm_input//offset')),
     'layer.0.triangle_multiply_ingoing.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/left_projection//weights')),
     'layer.0.triangle_multiply_ingoing.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/left_projection//bias')),
     'layer.0.triangle_multiply_ingoing.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/right_projection//weights')),
     'layer.0.triangle_multiply_ingoing.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/right_projection//bias')),
     'layer.0.triangle_multiply_ingoing.left_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/left_gate//weights')),
     'layer.0.triangle_multiply_ingoing.left_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/left_gate//bias')),
     'layer.0.triangle_multiply_ingoing.right_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/right_gate//weights')),
     'layer.0.triangle_multiply_ingoing.right_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/right_gate//bias')),
     'layer.0.triangle_multiply_ingoing.out_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/gating_linear//weights')),
     'layer.0.triangle_multiply_ingoing.out_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/gating_linear//bias')),
     'layer.0.triangle_multiply_ingoing.to_out_norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/center_layer_norm//scale')),
     'layer.0.triangle_multiply_ingoing.to_out_norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/center_layer_norm//offset')),
     'layer.0.triangle_multiply_ingoing.to_out.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/output_projection//weights')),
     'layer.0.triangle_multiply_ingoing.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/output_projection//bias')),
     'layer.0.triangle_attention_outgoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/query_norm//scale')),
     'layer.0.triangle_attention_outgoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/query_norm//offset')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/query_norm//scale')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/query_norm//offset')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node//feat_2d_weights')),
     'layer.0.triangle_attention_outgoing.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//query_w')),
     'layer.0.triangle_attention_outgoing.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//key_w', 'alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//value_w'])),
     'layer.0.triangle_attention_outgoing.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//gating_w')),
     'layer.0.triangle_attention_outgoing.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//gating_b')),
     'layer.0.triangle_attention_outgoing.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//output_w')),
     'layer.0.triangle_attention_outgoing.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//output_b')),
     'layer.0.triangle_attention_ingoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/query_norm//scale')),
     'layer.0.triangle_attention_ingoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/query_norm//offset')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/query_norm//scale')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/query_norm//offset')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node//feat_2d_weights')),
     'layer.0.triangle_attention_ingoing.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//query_w')),
     'layer.0.triangle_attention_ingoing.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//key_w', 'alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//value_w'])),
     'layer.0.triangle_attention_ingoing.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//gating_w')),
     'layer.0.triangle_attention_ingoing.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//gating_b')),
     'layer.0.triangle_attention_ingoing.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//output_w')),
     'layer.0.triangle_attention_ingoing.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//output_b')),
     'layer.1.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/input_layer_norm//scale')),
     'layer.1.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/input_layer_norm//offset')),
     'layer.1.net.0.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/transition1//weights')),
     'layer.1.net.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/transition1//bias')),
     'layer.1.net.3.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/transition2//weights')),
     'layer.1.net.3.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/transition2//bias')),
  }

  scope_list = {
     'template_pairwise_embedder.to_pair.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/embedding2d//weights')),
     'template_pairwise_embedder.to_pair.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/embedding2d//bias')),
     'template_pairwise_embedder.to_out_norm.weight': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/output_layer_norm//scale')),
     'template_pairwise_embedder.to_out_norm.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/output_layer_norm//offset')),
     'template_pointwise_attn.to_q.weight': (weights_to_weights_fn, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/attention//query_w')),
     'template_pointwise_attn.to_kv.weight': (chunk_weights_to_one_fn, dict(x=['alphafold/alphafold_iteration/evoformer/template_embedding/attention//key_w', 'alphafold/alphafold_iteration/evoformer/template_embedding/attention//value_w'])),
     'template_pointwise_attn.to_out.weight': (proj_out_weight, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/attention//output_w')),
     'template_pointwise_attn.to_out.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/attention//output_b')),
  }

  layers = 2
  for i in range(layers):
    for k, (fn, kwargs) in scope_iteration.items():
      scope_list[f'template_pairwise_embedder.pair_stack.layers.{i}.{k}'] = (functools.partial(fn, i=i), kwargs)

  with open(os.path.join(args.data_dir, 'params', f'params_{args.model_name}.npz'), 'rb') as f:
    params = np.load(io.BytesIO(f.read()), allow_pickle=False)
  state_dict = state_dict_get(params, scope_list)

  template_embedding = evoformer.TemplateEmbedding(dim=128,
                                                   depth=layers,
                                                   dim_templ=64,
                                                   heads=4,
                                                   dim_head=16,
                                                   attn_dropout=0.25,
                                                   ff_dropout=0)
  print(template_embedding)
  template_embedding.load_state_dict(state_dict)
  if args.output:
    with open(os.path.join(args.output, f'template_embedding.npz'), 'rb') as f:
      data = np.load(io.BytesIO(f.read()), allow_pickle=True)

    pair_act, result = data['pair_act'], data['result']
    pair_act = rearrange(torch.from_numpy(pair_act), 'i j d -> () i j d')
    # pair_mask = rearrange(torch.from_numpy(pair_mask), 'i j -> () i j')
    batch = data['batch'].tolist()
    mask_2d = batch['seq_mask'][:, None] * batch['seq_mask'][None, :]
    mask_2d = rearrange(torch.from_numpy(mask_2d), 'i j -> () i j')
    template_batch = {k: batch[k] for k in batch if k.startswith('template_')}  # all template
    for k in template_batch:
      if (isinstance(template_batch[k], np.ndarray)):
        template_batch[k] = rearrange(torch.from_numpy(template_batch[k]), '... -> () ...')
        print(k, template_batch[k].shape)

    b, n = template_batch['template_aatype'].shape[:2]
    template_embedding.eval()
    with torch.no_grad():
      print(pair_act.shape)
      r = template_embedding(pair_act.float(), mask_2d, None, None, template_batch)
    print(r)
    print(result)

def template_embedding_with_angles(args):
  def attention_with_head(x, fn=None):
    if isinstance(x, list):
      x = list(map(lambda a: rearrange(a, '... h d -> ... (h d)'), x))
    else:
      x = rearrange(x, '... h d -> ... (h d)')
    return fn(x)

  def proj_out_weight(x):
    x = rearrange(x, 'h d ... -> (h d) ...')
    return weights_to_weights(x)

  def proj_out_rearrange(x):
    x = rearrange(x, 'c d f -> (c d) f')
    return weights_to_weights(x)
  proj_out_wrap = functools.partial(layer_stack_to_module_list, fn=proj_out_rearrange)

  npz_to_tensor_fn = functools.partial(attention_with_head, fn=npz_to_tensor)
  weights_to_weights_fn = functools.partial(attention_with_head, fn=weights_to_weights)
  chunk_weights_to_one_fn = functools.partial(attention_with_head, fn=chunk_weights_to_one)
  chunk_bias_to_one_fn = functools.partial(attention_with_head, fn=chunk_bias_to_one)

  npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, fn=npz_to_tensor)
  weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, fn=weights_to_weights)
  chunk_weights_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_weights_to_one)
  chunk_bias_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_bias_to_one)

  attn_npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, fn=npz_to_tensor_fn)
  attn_weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, fn=weights_to_weights_fn)
  attn_chunk_weights_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_weights_to_one_fn)
  attn_chunk_bias_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_bias_to_one_fn)
  attn_proj_out_wrap = functools.partial(layer_stack_to_module_list, fn=proj_out_weight)



  scope_iteration = {
     'layer.0.triangle_multiply_outgoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/layer_norm_input//scale')),
     'layer.0.triangle_multiply_outgoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/layer_norm_input//offset')),
     'layer.0.triangle_multiply_outgoing.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/left_projection//weights')),
     'layer.0.triangle_multiply_outgoing.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/left_projection//bias')),
     'layer.0.triangle_multiply_outgoing.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/right_projection//weights')),
     'layer.0.triangle_multiply_outgoing.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/right_projection//bias')),
     'layer.0.triangle_multiply_outgoing.left_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/left_gate//weights')),
     'layer.0.triangle_multiply_outgoing.left_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/left_gate//bias')),
     'layer.0.triangle_multiply_outgoing.right_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/right_gate//weights')),
     'layer.0.triangle_multiply_outgoing.right_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/right_gate//bias')),
     'layer.0.triangle_multiply_outgoing.out_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/gating_linear//weights')),
     'layer.0.triangle_multiply_outgoing.out_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/gating_linear//bias')),
     'layer.0.triangle_multiply_outgoing.to_out_norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/center_layer_norm//scale')),
     'layer.0.triangle_multiply_outgoing.to_out_norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/center_layer_norm//offset')),
     'layer.0.triangle_multiply_outgoing.to_out.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/output_projection//weights')),
     'layer.0.triangle_multiply_outgoing.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/output_projection//bias')),
     'layer.0.triangle_multiply_ingoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/layer_norm_input//scale')),
     'layer.0.triangle_multiply_ingoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/layer_norm_input//offset')),
     'layer.0.triangle_multiply_ingoing.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/left_projection//weights')),
     'layer.0.triangle_multiply_ingoing.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/left_projection//bias')),
     'layer.0.triangle_multiply_ingoing.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/right_projection//weights')),
     'layer.0.triangle_multiply_ingoing.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/right_projection//bias')),
     'layer.0.triangle_multiply_ingoing.left_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/left_gate//weights')),
     'layer.0.triangle_multiply_ingoing.left_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/left_gate//bias')),
     'layer.0.triangle_multiply_ingoing.right_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/right_gate//weights')),
     'layer.0.triangle_multiply_ingoing.right_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/right_gate//bias')),
     'layer.0.triangle_multiply_ingoing.out_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/gating_linear//weights')),
     'layer.0.triangle_multiply_ingoing.out_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/gating_linear//bias')),
     'layer.0.triangle_multiply_ingoing.to_out_norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/center_layer_norm//scale')),
     'layer.0.triangle_multiply_ingoing.to_out_norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/center_layer_norm//offset')),
     'layer.0.triangle_multiply_ingoing.to_out.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/output_projection//weights')),
     'layer.0.triangle_multiply_ingoing.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/output_projection//bias')),
     'layer.0.triangle_attention_outgoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/query_norm//scale')),
     'layer.0.triangle_attention_outgoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/query_norm//offset')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/query_norm//scale')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/query_norm//offset')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node//feat_2d_weights')),
     'layer.0.triangle_attention_outgoing.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//query_w')),
     'layer.0.triangle_attention_outgoing.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//key_w', 'alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//value_w'])),
     'layer.0.triangle_attention_outgoing.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//gating_w')),
     'layer.0.triangle_attention_outgoing.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//gating_b')),
     'layer.0.triangle_attention_outgoing.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//output_w')),
     'layer.0.triangle_attention_outgoing.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//output_b')),
     'layer.0.triangle_attention_ingoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/query_norm//scale')),
     'layer.0.triangle_attention_ingoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/query_norm//offset')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/query_norm//scale')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/query_norm//offset')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node//feat_2d_weights')),
     'layer.0.triangle_attention_ingoing.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//query_w')),
     'layer.0.triangle_attention_ingoing.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//key_w', 'alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//value_w'])),
     'layer.0.triangle_attention_ingoing.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//gating_w')),
     'layer.0.triangle_attention_ingoing.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//gating_b')),
     'layer.0.triangle_attention_ingoing.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//output_w')),
     'layer.0.triangle_attention_ingoing.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//output_b')),
     'layer.1.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/input_layer_norm//scale')),
     'layer.1.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/input_layer_norm//offset')),
     'layer.1.net.0.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/transition1//weights')),
     'layer.1.net.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/transition1//bias')),
     'layer.1.net.3.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/transition2//weights')),
     'layer.1.net.3.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/transition2//bias')),
  }

  scope_list = {
     'template_pairwise_embedder.to_pair.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/embedding2d//weights')),
     'template_pairwise_embedder.to_pair.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/embedding2d//bias')),
     'template_pairwise_embedder.to_out_norm.weight': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/output_layer_norm//scale')),
     'template_pairwise_embedder.to_out_norm.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/output_layer_norm//offset')),
     'template_pointwise_attn.to_q.weight': (weights_to_weights_fn, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/attention//query_w')),
     'template_pointwise_attn.to_kv.weight': (chunk_weights_to_one_fn, dict(x=['alphafold/alphafold_iteration/evoformer/template_embedding/attention//key_w', 'alphafold/alphafold_iteration/evoformer/template_embedding/attention//value_w'])),
     'template_pointwise_attn.to_out.weight': (proj_out_weight, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/attention//output_w')),
     'template_pointwise_attn.to_out.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/attention//output_b')),
     'template_single_embedder.0.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/evoformer/template_single_embedding//weights')),
     'template_single_embedder.0.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/template_single_embedding//bias')),
     'template_single_embedder.2.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/evoformer/template_projection//weights')),
     'template_single_embedder.2.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/template_projection//bias')),
  }

  layers = 2
  for i in range(layers):
    for k, (fn, kwargs) in scope_iteration.items():
      scope_list[f'template_pairwise_embedder.pair_stack.layers.{i}.{k}'] = (functools.partial(fn, i=i), kwargs)

  with open(os.path.join(args.data_dir, 'params', f'params_{args.model_name}.npz'), 'rb') as f:
    params = np.load(io.BytesIO(f.read()), allow_pickle=False)
  state_dict = state_dict_get(params, scope_list)

  template_embedding = evoformer.TemplateEmbedding(dim=128,
                                                   depth=layers,
                                                   dim_templ=64,
                                                   heads=4,
                                                   dim_head=16,
                                                   dim_msa=256,
                                                   attn_dropout=0.25,
                                                   ff_dropout=0)
  print(template_embedding)
  template_embedding.load_state_dict(state_dict)
  if args.output:
    with open(os.path.join(args.output, f'template_embedding.npz'), 'rb') as f:
      data = np.load(io.BytesIO(f.read()), allow_pickle=True)

    pair_act, result = data['pair_act'], data['result']
    pair_act = rearrange(torch.from_numpy(pair_act), 'i j d -> () i j d')
    # pair_mask = rearrange(torch.from_numpy(pair_mask), 'i j -> () i j')
    batch = data['batch'].tolist()

    batch = fix_atom_order(batch) # FIX

    mask_2d = batch['seq_mask'][:, None] * batch['seq_mask'][None, :]
    mask_2d = rearrange(torch.from_numpy(mask_2d), 'i j -> () i j')
    template_batch = {k: batch[k] for k in batch if k.startswith('template_')}  # all template

    template_batch['template_aatype_index'] = np.clip(template_batch['template_aatype'], 0, 20)
    template_batch['template_all_atom_mask'] = template_batch['template_all_atom_masks']
    print(template_batch.keys())
    template_batch = make_atom14_positions(template_batch, along_axis=2, prefix='template_')
    print(template_batch.keys())

    for k_from, k_to in (('torsion_angles_sin_cos', 'torsion_angles'), ('alt_torsion_angles_sin_cos', 'torsion_angles_alt'), ('torsion_angles_mask', 'torsion_angles_mask')):
        #if f'template_{k_from}' in batch:
        print('xxx', k_from, k_to)
        template_batch[f'template_{k_to}'] = template_batch[f'template_{k_from}']
        print(template_batch[f'template_{k_from}'])
    for k in template_batch:
      if (isinstance(template_batch[k], np.ndarray)):
        template_batch[k] = torch.from_numpy(template_batch[k])
        print(k, template_batch[k].shape)

    # template_angles = functional.angles_from_positions(template_batch['template_aatype'], template_batch['template_atom14_gt_positions'], template_batch['template_atom14_gt_exists'])
    # for k, v in template_angles.items():
    #   template_batch[f'template_{k}'] = v
    print(template_batch.keys())
    for k in template_batch:
      if (isinstance(template_batch[k], torch.Tensor)):
        template_batch[k] = rearrange(template_batch[k], '... -> () ...')
        print(k, template_batch[k].shape)
      else:
        print('aaa', k, type(template_batch[k]))

    b, n = template_batch['template_aatype'].shape[:2]
    template_embedding.eval()
    with torch.no_grad():
      print(pair_act.shape)
      r = template_embedding(pair_act.float(), mask_2d, None, None, template_batch)
    print(r)
    print(result)


def msa_row_attention_with_pair_bias(args):
  def attention_with_head(x, fn=None):
    if isinstance(x, list):
      x = list(map(lambda a: rearrange(a, '... h d -> ... (h d)'), x))
    else:
      x = rearrange(x, '... h d -> ... (h d)')
    return fn(x)

  def proj_out_weight(x):
    x = rearrange(x, 'h d ... -> (h d) ...')
    return weights_to_weights(x)

  npz_to_tensor_fn = functools.partial(attention_with_head, fn=npz_to_tensor)
  weights_to_weights_fn = functools.partial(attention_with_head, fn=weights_to_weights)
  chunk_weights_to_one_fn = functools.partial(attention_with_head, fn=chunk_weights_to_one)
  chunk_bias_to_one_fn = functools.partial(attention_with_head, fn=chunk_bias_to_one)

  npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=npz_to_tensor)
  weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=weights_to_weights)
  chunk_weights_to_one_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=chunk_weights_to_one)
  chunk_bias_to_one_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=chunk_bias_to_one)

  attn_npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=npz_to_tensor_fn)
  attn_weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=weights_to_weights_fn)
  attn_chunk_weights_to_one_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=chunk_weights_to_one_fn)
  attn_chunk_bias_to_one_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=chunk_bias_to_one_fn)
  attn_proj_out_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=proj_out_weight)


  scope_list = {
     'norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/query_norm//scale')),
     'norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/query_norm//offset')),
     'edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/feat_2d_norm//scale')),
     'edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/feat_2d_norm//offset')),
     'edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias//feat_2d_weights')),
     'attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//query_w')),
     'attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//key_w', 'alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//value_w'])),
     'attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//gating_w')),
     'attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//gating_b')),
     'attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//output_w')),
     'attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//output_b')),
  }

  with open(os.path.join(args.data_dir, 'params', f'params_{args.model_name}.npz'), 'rb') as f:
    params = np.load(io.BytesIO(f.read()), allow_pickle=False)
  state_dict = state_dict_get(params, scope_list)

  msa_row_attention_with_pair_bias = commons.AxialAttention(dim=(384, 256, 128, 32),
                                                            heads=8,
                                                            dim_head=32,
                                                            row_attn=True,
                                                            col_attn=False,
                                                            accept_edges=True)
  msa_row_attention_with_pair_bias.load_state_dict(state_dict)
  print(msa_row_attention_with_pair_bias)
  if args.output:
    with open(os.path.join(args.output, f'msa_row_attention_with_pair_bias.npz'), 'rb') as f:
      data = np.load(io.BytesIO(f.read()), allow_pickle=True)

    msa_act, msa_mask, pair_act, result = data['msa_act'], data['msa_mask'], data['pair_act'], data['result']
    msa_act, msa_mask = rearrange(torch.from_numpy(msa_act), 'm i d -> () m i d'), rearrange(torch.from_numpy(msa_mask), 'm i -> () m i')
    pair_act = rearrange(torch.from_numpy(pair_act), 'i j d -> () i j d')

    msa_row_attention_with_pair_bias.eval()
    with torch.no_grad():
      print(msa_act.shape)
      print(msa_mask.shape)
      print(pair_act.shape)
      r = msa_row_attention_with_pair_bias(msa_act.float(), mask=msa_mask, edges=pair_act.float())
    print(r)
    print(result)

def msa_column_attention(args):
  def attention_with_head(x, fn=None):
    if isinstance(x, list):
      x = list(map(lambda a: rearrange(a, '... h d -> ... (h d)'), x))
    else:
      x = rearrange(x, '... h d -> ... (h d)')
    return fn(x)

  def proj_out_weight(x):
    x = rearrange(x, 'h d ... -> (h d) ...')
    return weights_to_weights(x)

  npz_to_tensor_fn = functools.partial(attention_with_head, fn=npz_to_tensor)
  weights_to_weights_fn = functools.partial(attention_with_head, fn=weights_to_weights)
  chunk_weights_to_one_fn = functools.partial(attention_with_head, fn=chunk_weights_to_one)
  chunk_bias_to_one_fn = functools.partial(attention_with_head, fn=chunk_bias_to_one)

  npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=npz_to_tensor)
  weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=weights_to_weights)
  chunk_weights_to_one_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=chunk_weights_to_one)
  chunk_bias_to_one_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=chunk_bias_to_one)

  attn_npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=npz_to_tensor_fn)
  attn_weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=weights_to_weights_fn)
  attn_chunk_weights_to_one_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=chunk_weights_to_one_fn)
  attn_chunk_bias_to_one_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=chunk_bias_to_one_fn)
  attn_proj_out_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=proj_out_weight)


  scope_list = {
     'norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/query_norm//scale')),
     'norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/query_norm//offset')),
     'attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//query_w')),
     'attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//key_w', 'alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//value_w'])),
     'attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//gating_w')),
     'attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//gating_b')),
     'attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//output_w')),
     'attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//output_b')),
  }

  with open(os.path.join(args.data_dir, 'params', f'params_{args.model_name}.npz'), 'rb') as f:
    params = np.load(io.BytesIO(f.read()), allow_pickle=False)
  state_dict = state_dict_get(params, scope_list)

  msa_column_attention = commons.AxialAttention(dim=(384, 256, 128, 32),
                                                            heads=8,
                                                            dim_head=32,
                                                            row_attn=False,
                                                            col_attn=True,
                                                            accept_edges=False)
  msa_column_attention.load_state_dict(state_dict)
  print(msa_column_attention)
  if args.output:
    with open(os.path.join(args.output, f'msa_column_attention.npz'), 'rb') as f:
      data = np.load(io.BytesIO(f.read()), allow_pickle=True)

    msa_act, msa_mask, result = data['msa_act'], data['msa_mask'], data['result']
    msa_act, msa_mask = rearrange(torch.from_numpy(msa_act), 'm i d -> () m i d'), rearrange(torch.from_numpy(msa_mask), 'm i -> () m i')

    msa_column_attention.eval()
    with torch.no_grad():
      print(msa_act.shape)
      print(msa_mask.shape)
      r = msa_column_attention(msa_act.float(), mask=msa_mask)
    print(r)
    print(result)

def msa_column_global_attention(args):
  def attention_with_head(x, fn=None):
    if isinstance(x, list):
      x = list(map(lambda a: rearrange(a, '... h d -> ... (h d)'), x))
    else:
      x = rearrange(x, '... h d -> ... (h d)')
    return fn(x)

  def proj_out_weight(x):
    x = rearrange(x, 'h d ... -> (h d) ...')
    return weights_to_weights(x)

  npz_to_tensor_fn = functools.partial(attention_with_head, fn=npz_to_tensor)
  weights_to_weights_fn = functools.partial(attention_with_head, fn=weights_to_weights)
  chunk_weights_to_one_fn = functools.partial(attention_with_head, fn=chunk_weights_to_one)
  chunk_bias_to_one_fn = functools.partial(attention_with_head, fn=chunk_bias_to_one)

  npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=npz_to_tensor)
  weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=weights_to_weights)
  chunk_weights_to_one_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=chunk_weights_to_one)
  chunk_bias_to_one_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=chunk_bias_to_one)

  attn_npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=npz_to_tensor_fn)
  attn_weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=weights_to_weights_fn)
  attn_chunk_weights_to_one_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=chunk_weights_to_one)
  attn_chunk_bias_to_one_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=chunk_bias_to_one)
  attn_proj_out_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=proj_out_weight)

  scope_list = {
     'norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/query_norm//scale')),
     'norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/query_norm//offset')),
     'attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/attention//query_w')),
     'attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/attention//key_w', 'alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/attention//value_w'])),
     'attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/attention//gating_w')),
     'attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/attention//gating_b')),
     'attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/attention//output_w')),
     'attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/attention//output_b')),
  }

  with open(os.path.join(args.data_dir, 'params', f'params_{args.model_name}.npz'), 'rb') as f:
    params = np.load(io.BytesIO(f.read()), allow_pickle=False)
  state_dict = state_dict_get(params, scope_list)

  msa_column_global_attention = commons.AxialAttention(dim=(384, 64, 128, 32),
                                                            heads=8,
                                                            dim_head=8,
                                                            row_attn=False,
                                                            col_attn=True,
                                                            accept_edges=False,
                                                            global_query_attn=True)
  print(msa_column_global_attention)
  msa_column_global_attention.load_state_dict(state_dict)
  if args.output:
    with open(os.path.join(args.output, f'msa_column_global_attention.npz'), 'rb') as f:
      data = np.load(io.BytesIO(f.read()), allow_pickle=True)

    msa_act, msa_mask, result = data['msa_act'], data['msa_mask'], data['result']
    print('msa_mask', msa_mask)
    msa_act, msa_mask = rearrange(torch.from_numpy(msa_act), 'm i d -> () m i d'), rearrange(torch.from_numpy(msa_mask), 'm i -> () m i')

    msa_column_global_attention.eval()
    with torch.no_grad():
      print(msa_act.shape)
      print(msa_mask.shape)
      r = msa_column_global_attention(msa_act.float(), mask=msa_mask)
    print(r)
    print(result)


def msa_transition(args):
  npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=npz_to_tensor)
  weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=weights_to_weights)

  scope_list = {
     'norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_transition/input_layer_norm//scale')),
     'norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_transition/input_layer_norm//offset')),
     'net.0.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_transition/transition1//weights')),
     'net.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_transition/transition1//bias')),
     'net.3.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_transition/transition2//weights')),
     'net.3.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_transition/transition2//bias')),
  }

  with open(os.path.join(args.data_dir, 'params', f'params_{args.model_name}.npz'), 'rb') as f:
    params = np.load(io.BytesIO(f.read()), allow_pickle=False)
  state_dict = state_dict_get(params, scope_list)

  msa_transition = commons.FeedForward(dim=256)
  msa_transition.load_state_dict(state_dict)
  print(msa_transition)
  if args.output:
    with open(os.path.join(args.output, f'msa_transition.npz'), 'rb') as f:
      data = np.load(io.BytesIO(f.read()), allow_pickle=True)

    msa_act, msa_mask, result = data['msa_act'], data['msa_mask'], data['result']
    msa_act, msa_mask = rearrange(torch.from_numpy(msa_act), 'm i d -> () m i d'), rearrange(torch.from_numpy(msa_mask), 'm i -> () m i')

    msa_transition.eval()
    with torch.no_grad():
      print(msa_act.shape)
      print(msa_mask.shape)
      r = msa_transition(msa_act.float())
    print(r)
    print(result)


def outer_product_mean(args):
  with open(os.path.join(args.data_dir, 'params', f'params_{args.model_name}.npz'), 'rb') as f:
    params = np.load(io.BytesIO(f.read()), allow_pickle=False)
  npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=npz_to_tensor)
  weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=weights_to_weights)
  def proj_out_rearrange(x):
    x = rearrange(x, 'c d f -> (c d) f')
    return weights_to_weights(x)
  proj_out_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=proj_out_rearrange)

  scope_list = {
     'norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean/layer_norm_input//scale')),
     'norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean/layer_norm_input//offset')),
     'left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean/left_projection//weights')),
     'left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean/left_projection//bias')),
     'right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean/right_projection//weights')),
     'right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean/right_projection//bias')),
     'proj_out.weight': (proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean//output_w')),
     'proj_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean//output_b')),
  }
  state_dict = state_dict_get(params, scope_list)

  outer_product_mean = commons.OuterMean(dim=(384, 256, 128, 32))
  outer_product_mean.load_state_dict(state_dict)
  if args.output:
    with open(os.path.join(args.output, f'outer_product_mean.npz'), 'rb') as f:
      data = np.load(io.BytesIO(f.read()), allow_pickle=True)
    act, mask, result = data['act'], data['mask'], data['result']
    x, mask = rearrange(torch.from_numpy(act), 'm i d -> () m i d'), rearrange(torch.from_numpy(mask), 'm i -> () m i')
    outer_product_mean.eval()
    with torch.no_grad():
      r = outer_product_mean(x.float(), mask, shard_size=None)
    print(r)
    print(result)

def triangle_multiplication_outgoing(args):
  npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=npz_to_tensor)
  weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=weights_to_weights)

  scope_list = {
     'norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/layer_norm_input//scale')),
     'norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/layer_norm_input//offset')),
     'left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/left_projection//weights')),
     'left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/left_projection//bias')),
     'right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/right_projection//weights')),
     'right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/right_projection//bias')),
     'left_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/left_gate//weights')),
     'left_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/left_gate//bias')),
     'right_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/right_gate//weights')),
     'right_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/right_gate//bias')),
     'out_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/gating_linear//weights')),
     'out_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/gating_linear//bias')),
     'to_out_norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/center_layer_norm//scale')),
     'to_out_norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/center_layer_norm//offset')),
     'to_out.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/output_projection//weights')),
     'to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/output_projection//bias')),
  }

  with open(os.path.join(args.data_dir, 'params', f'params_{args.model_name}.npz'), 'rb') as f:
    params = np.load(io.BytesIO(f.read()), allow_pickle=False)
  state_dict = state_dict_get(params, scope_list)

  triangle_multiplication_outgoing = commons.TriangleMultiplicativeModule(dim=128, mix='outgoing')
  triangle_multiplication_outgoing.load_state_dict(state_dict)
  print(triangle_multiplication_outgoing)
  if args.output:
    with open(os.path.join(args.output, f'triangle_multiplication_outgoing.npz'), 'rb') as f:
      data = np.load(io.BytesIO(f.read()), allow_pickle=True)

    pair_act, pair_mask, result = data['pair_act'], data['pair_mask'], data['result']
    pair_act, pair_mask = rearrange(torch.from_numpy(pair_act), 'i j d -> () i j d'), rearrange(torch.from_numpy(pair_mask), 'i j -> () i j')

    triangle_multiplication_outgoing.eval()
    with torch.no_grad():
      print(pair_act.shape)
      print(pair_mask.shape)
      r = triangle_multiplication_outgoing(pair_act.float())
    print(r)
    print(result)

def triangle_multiplication_incoming(args):
  npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=npz_to_tensor)
  weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=weights_to_weights)

  scope_list = {
     'norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/layer_norm_input//scale')),
     'norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/layer_norm_input//offset')),
     'left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/left_projection//weights')),
     'left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/left_projection//bias')),
     'right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/right_projection//weights')),
     'right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/right_projection//bias')),
     'left_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/left_gate//weights')),
     'left_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/left_gate//bias')),
     'right_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/right_gate//weights')),
     'right_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/right_gate//bias')),
     'out_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/gating_linear//weights')),
     'out_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/gating_linear//bias')),
     'to_out_norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/center_layer_norm//scale')),
     'to_out_norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/center_layer_norm//offset')),
     'to_out.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/output_projection//weights')),
     'to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/output_projection//bias')),
  }

  with open(os.path.join(args.data_dir, 'params', f'params_{args.model_name}.npz'), 'rb') as f:
    params = np.load(io.BytesIO(f.read()), allow_pickle=False)
  state_dict = state_dict_get(params, scope_list)

  triangle_multiplication_incoming = commons.TriangleMultiplicativeModule(dim=128, mix='ingoing')
  triangle_multiplication_incoming.load_state_dict(state_dict)
  print(triangle_multiplication_incoming)
  if args.output:
    with open(os.path.join(args.output, f'triangle_multiplication_incoming.npz'), 'rb') as f:
      data = np.load(io.BytesIO(f.read()), allow_pickle=True)

    pair_act, pair_mask, result = data['pair_act'], data['pair_mask'], data['result']
    pair_act, pair_mask = rearrange(torch.from_numpy(pair_act), 'i j d -> () i j d'), rearrange(torch.from_numpy(pair_mask), 'i j -> () i j')

    triangle_multiplication_incoming.eval()
    with torch.no_grad():
      print(pair_act.shape)
      print(pair_mask.shape)
      r = triangle_multiplication_incoming(pair_act.float())
    print(r)
    print(result)

def triangle_attention_starting_node(args):
  def attention_with_head(x, fn=None):
    if isinstance(x, list):
      x = list(map(lambda a: rearrange(a, '... h d -> ... (h d)'), x))
    else:
      x = rearrange(x, '... h d -> ... (h d)')
    return fn(x)

  def proj_out_weight(x):
    x = rearrange(x, 'h d ... -> (h d) ...')
    return weights_to_weights(x)

  npz_to_tensor_fn = functools.partial(attention_with_head, fn=npz_to_tensor)
  weights_to_weights_fn = functools.partial(attention_with_head, fn=weights_to_weights)
  chunk_weights_to_one_fn = functools.partial(attention_with_head, fn=chunk_weights_to_one)
  chunk_bias_to_one_fn = functools.partial(attention_with_head, fn=chunk_bias_to_one)

  npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=npz_to_tensor)
  weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=weights_to_weights)
  chunk_weights_to_one_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=chunk_weights_to_one)
  chunk_bias_to_one_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=chunk_bias_to_one)

  attn_npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=npz_to_tensor_fn)
  attn_weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=weights_to_weights_fn)
  attn_chunk_weights_to_one_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=chunk_weights_to_one_fn)
  attn_chunk_bias_to_one_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=chunk_bias_to_one_fn)
  attn_proj_out_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=proj_out_weight)


  scope_list = {
     'norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/query_norm//scale')),
     'norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/query_norm//offset')),
     'edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/query_norm//scale')),
     'edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/query_norm//offset')),
     'edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node//feat_2d_weights')),
     'attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//query_w')),
     'attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//key_w', 'alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//value_w'])),
     'attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//gating_w')),
     'attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//gating_b')),
     'attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//output_w')),
     'attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//output_b')),
  }

  with open(os.path.join(args.data_dir, 'params', f'params_{args.model_name}.npz'), 'rb') as f:
    params = np.load(io.BytesIO(f.read()), allow_pickle=False)
  state_dict = state_dict_get(params, scope_list)

  triangle_attention_starting_node = commons.AxialAttention(dim=128,
                                                            heads=4,
                                                            dim_head=32,
                                                            row_attn=True,
                                                            col_attn=False,
                                                            accept_edges=True)
  triangle_attention_starting_node.load_state_dict(state_dict)
  print(triangle_attention_starting_node)
  if args.output:
    with open(os.path.join(args.output, f'triangle_attention_starting_node.npz'), 'rb') as f:
      data = np.load(io.BytesIO(f.read()), allow_pickle=True)

    pair_act, pair_mask, result = data['pair_act'], data['pair_mask'], data['result']
    pair_act, pair_mask = rearrange(torch.from_numpy(pair_act).float(), 'i j d -> () i j d'), rearrange(torch.from_numpy(pair_mask), 'i j -> () i j')

    triangle_attention_starting_node.eval()
    with torch.no_grad():
      print(pair_act.shape)
      print(pair_mask.shape)
      r = triangle_attention_starting_node(pair_act, edges=pair_act, mask=pair_mask)
    print('x', result)
    print('y', r)

def triangle_attention_ending_node(args):
  def attention_with_head(x, fn=None):
    if isinstance(x, list):
      x = list(map(lambda a: rearrange(a, '... h d -> ... (h d)'), x))
    else:
      x = rearrange(x, '... h d -> ... (h d)')
    return fn(x)

  def proj_out_weight(x):
    x = rearrange(x, 'h d ... -> (h d) ...')
    return weights_to_weights(x)

  npz_to_tensor_fn = functools.partial(attention_with_head, fn=npz_to_tensor)
  weights_to_weights_fn = functools.partial(attention_with_head, fn=weights_to_weights)
  chunk_weights_to_one_fn = functools.partial(attention_with_head, fn=chunk_weights_to_one)
  chunk_bias_to_one_fn = functools.partial(attention_with_head, fn=chunk_bias_to_one)

  npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=npz_to_tensor)
  weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=weights_to_weights)
  chunk_weights_to_one_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=chunk_weights_to_one)
  chunk_bias_to_one_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=chunk_bias_to_one)

  attn_npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=npz_to_tensor_fn)
  attn_weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=weights_to_weights_fn)
  attn_chunk_weights_to_one_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=chunk_weights_to_one_fn)
  attn_chunk_bias_to_one_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=chunk_bias_to_one_fn)
  attn_proj_out_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=proj_out_weight)


  scope_list = {
     'norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/query_norm//scale')),
     'norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/query_norm//offset')),
     'edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/query_norm//scale')),
     'edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/query_norm//offset')),
     'edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node//feat_2d_weights')),
     'attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//query_w')),
     'attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//key_w', 'alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//value_w'])),
     'attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//gating_w')),
     'attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//gating_b')),
     'attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//output_w')),
     'attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//output_b')),
  }

  with open(os.path.join(args.data_dir, 'params', f'params_{args.model_name}.npz'), 'rb') as f:
    params = np.load(io.BytesIO(f.read()), allow_pickle=False)
  state_dict = state_dict_get(params, scope_list)

  triangle_attention_ending_node = commons.AxialAttention(dim=128,
                                                            heads=4,
                                                            dim_head=32,
                                                            row_attn=False,
                                                            col_attn=True,
                                                            accept_edges=True)
  triangle_attention_ending_node.load_state_dict(state_dict)
  print(triangle_attention_ending_node)
  if args.output:
    with open(os.path.join(args.output, f'triangle_attention_ending_node.npz'), 'rb') as f:
      data = np.load(io.BytesIO(f.read()), allow_pickle=True)

    pair_act, pair_mask, result = data['pair_act'], data['pair_mask'], data['result']
    pair_act, pair_mask = rearrange(torch.from_numpy(pair_act).float(), 'i j d -> () i j d'), rearrange(torch.from_numpy(pair_mask), 'i j -> () i j')

    triangle_attention_ending_node.eval()
    with torch.no_grad():
      print(pair_act.shape)
      print(pair_mask.shape)
      r = triangle_attention_ending_node(pair_act, edges=pair_act, mask=pair_mask)
    print('x', result)
    print('y', r)


def pair_transition(args):
  npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=npz_to_tensor)
  weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=weights_to_weights)

  scope_list = {
     'norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/pair_transition/input_layer_norm//scale')),
     'norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/pair_transition/input_layer_norm//offset')),
     'net.0.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/pair_transition/transition1//weights')),
     'net.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/pair_transition/transition1//bias')),
     'net.3.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/pair_transition/transition2//weights')),
     'net.3.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/pair_transition/transition2//bias')),
  }

  with open(os.path.join(args.data_dir, 'params', f'params_{args.model_name}.npz'), 'rb') as f:
    params = np.load(io.BytesIO(f.read()), allow_pickle=False)
  state_dict = state_dict_get(params, scope_list)

  pair_transition = commons.FeedForward(dim=128)
  pair_transition.load_state_dict(state_dict)
  print(pair_transition)
  if args.output:
    with open(os.path.join(args.output, f'pair_transition.npz'), 'rb') as f:
      data = np.load(io.BytesIO(f.read()), allow_pickle=True)

    pair_act, result = data['pair_act'], data['result']
    pair_act = rearrange(torch.from_numpy(pair_act), 'i j d -> () i j d')

    pair_transition.eval()
    with torch.no_grad():
      print(pair_act.shape)
      r = pair_transition(pair_act.float())
    print(r)
    print(result)

def evoformer_iteration(args):
  def attention_with_head(x, fn=None):
    if isinstance(x, list):
      x = list(map(lambda a: rearrange(a, '... h d -> ... (h d)'), x))
    else:
      x = rearrange(x, '... h d -> ... (h d)')
    return fn(x)

  def proj_out_weight(x):
    x = rearrange(x, 'h d ... -> (h d) ...')
    return weights_to_weights(x)

  def proj_out_rearrange(x):
    x = rearrange(x, 'c d f -> (c d) f')
    return weights_to_weights(x)
  proj_out_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=proj_out_rearrange)

  npz_to_tensor_fn = functools.partial(attention_with_head, fn=npz_to_tensor)
  weights_to_weights_fn = functools.partial(attention_with_head, fn=weights_to_weights)
  chunk_weights_to_one_fn = functools.partial(attention_with_head, fn=chunk_weights_to_one)
  chunk_bias_to_one_fn = functools.partial(attention_with_head, fn=chunk_bias_to_one)

  npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=npz_to_tensor)
  weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=weights_to_weights)
  chunk_weights_to_one_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=chunk_weights_to_one)
  chunk_bias_to_one_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=chunk_bias_to_one)

  attn_npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=npz_to_tensor_fn)
  attn_weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=weights_to_weights_fn)
  attn_chunk_weights_to_one_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=chunk_weights_to_one_fn)
  attn_chunk_bias_to_one_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=chunk_bias_to_one_fn)
  attn_proj_out_wrap = functools.partial(layer_stack_to_module_list, i=0, fn=proj_out_weight)


  scope_list = {
     'layer.2.row_attn.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/query_norm//scale')),
     'layer.2.row_attn.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/query_norm//offset')),
     'layer.2.row_attn.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/feat_2d_norm//scale')),
     'layer.2.row_attn.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/feat_2d_norm//offset')),
     'layer.2.row_attn.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias//feat_2d_weights')),
     'layer.2.row_attn.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//query_w')),
     'layer.2.row_attn.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//key_w', 'alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//value_w'])),
     'layer.2.row_attn.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//gating_w')),
     'layer.2.row_attn.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//gating_b')),
     'layer.2.row_attn.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//output_w')),
     'layer.2.row_attn.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//output_b')),
     'layer.2.col_attn.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/query_norm//scale')),
     'layer.2.col_attn.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/query_norm//offset')),
     'layer.2.col_attn.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//query_w')),
     'layer.2.col_attn.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//key_w', 'alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//value_w'])),
     'layer.2.col_attn.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//gating_w')),
     'layer.2.col_attn.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//gating_b')),
     'layer.2.col_attn.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//output_w')),
     'layer.2.col_attn.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//output_b')),
     'layer.3.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_transition/input_layer_norm//scale')),
     'layer.3.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_transition/input_layer_norm//offset')),
     'layer.3.net.0.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_transition/transition1//weights')),
     'layer.3.net.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_transition/transition1//bias')),
     'layer.3.net.3.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_transition/transition2//weights')),
     'layer.3.net.3.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_transition/transition2//bias')),
     'layer.0.outer_mean.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean/layer_norm_input//scale')),
     'layer.0.outer_mean.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean/layer_norm_input//offset')),
     'layer.0.outer_mean.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean/left_projection//weights')),
     'layer.0.outer_mean.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean/left_projection//bias')),
     'layer.0.outer_mean.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean/right_projection//weights')),
     'layer.0.outer_mean.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean/right_projection//bias')),
     'layer.0.outer_mean.proj_out.weight': (proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean//output_w')),
     'layer.0.outer_mean.proj_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean//output_b')),
     'layer.0.triangle_multiply_outgoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/layer_norm_input//scale')),
     'layer.0.triangle_multiply_outgoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/layer_norm_input//offset')),
     'layer.0.triangle_multiply_outgoing.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/left_projection//weights')),
     'layer.0.triangle_multiply_outgoing.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/left_projection//bias')),
     'layer.0.triangle_multiply_outgoing.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/right_projection//weights')),
     'layer.0.triangle_multiply_outgoing.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/right_projection//bias')),
     'layer.0.triangle_multiply_outgoing.left_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/left_gate//weights')),
     'layer.0.triangle_multiply_outgoing.left_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/left_gate//bias')),
     'layer.0.triangle_multiply_outgoing.right_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/right_gate//weights')),
     'layer.0.triangle_multiply_outgoing.right_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/right_gate//bias')),
     'layer.0.triangle_multiply_outgoing.out_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/gating_linear//weights')),
     'layer.0.triangle_multiply_outgoing.out_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/gating_linear//bias')),
     'layer.0.triangle_multiply_outgoing.to_out_norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/center_layer_norm//scale')),
     'layer.0.triangle_multiply_outgoing.to_out_norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/center_layer_norm//offset')),
     'layer.0.triangle_multiply_outgoing.to_out.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/output_projection//weights')),
     'layer.0.triangle_multiply_outgoing.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/output_projection//bias')),
     'layer.0.triangle_multiply_ingoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/layer_norm_input//scale')),
     'layer.0.triangle_multiply_ingoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/layer_norm_input//offset')),
     'layer.0.triangle_multiply_ingoing.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/left_projection//weights')),
     'layer.0.triangle_multiply_ingoing.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/left_projection//bias')),
     'layer.0.triangle_multiply_ingoing.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/right_projection//weights')),
     'layer.0.triangle_multiply_ingoing.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/right_projection//bias')),
     'layer.0.triangle_multiply_ingoing.left_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/left_gate//weights')),
     'layer.0.triangle_multiply_ingoing.left_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/left_gate//bias')),
     'layer.0.triangle_multiply_ingoing.right_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/right_gate//weights')),
     'layer.0.triangle_multiply_ingoing.right_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/right_gate//bias')),
     'layer.0.triangle_multiply_ingoing.out_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/gating_linear//weights')),
     'layer.0.triangle_multiply_ingoing.out_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/gating_linear//bias')),
     'layer.0.triangle_multiply_ingoing.to_out_norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/center_layer_norm//scale')),
     'layer.0.triangle_multiply_ingoing.to_out_norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/center_layer_norm//offset')),
     'layer.0.triangle_multiply_ingoing.to_out.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/output_projection//weights')),
     'layer.0.triangle_multiply_ingoing.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/output_projection//bias')),
     'layer.0.triangle_attention_outgoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/query_norm//scale')),
     'layer.0.triangle_attention_outgoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/query_norm//offset')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/query_norm//scale')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/query_norm//offset')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node//feat_2d_weights')),
     'layer.0.triangle_attention_outgoing.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//query_w')),
     'layer.0.triangle_attention_outgoing.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//key_w', 'alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//value_w'])),
     'layer.0.triangle_attention_outgoing.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//gating_w')),
     'layer.0.triangle_attention_outgoing.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//gating_b')),
     'layer.0.triangle_attention_outgoing.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//output_w')),
     'layer.0.triangle_attention_outgoing.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//output_b')),
     'layer.0.triangle_attention_ingoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/query_norm//scale')),
     'layer.0.triangle_attention_ingoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/query_norm//offset')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/query_norm//scale')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/query_norm//offset')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node//feat_2d_weights')),
     'layer.0.triangle_attention_ingoing.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//query_w')),
     'layer.0.triangle_attention_ingoing.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//key_w', 'alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//value_w'])),
     'layer.0.triangle_attention_ingoing.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//gating_w')),
     'layer.0.triangle_attention_ingoing.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//gating_b')),
     'layer.0.triangle_attention_ingoing.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//output_w')),
     'layer.0.triangle_attention_ingoing.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//output_b')),
     'layer.1.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/pair_transition/input_layer_norm//scale')),
     'layer.1.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/pair_transition/input_layer_norm//offset')),
     'layer.1.net.0.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/pair_transition/transition1//weights')),
     'layer.1.net.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/pair_transition/transition1//bias')),
     'layer.1.net.3.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/pair_transition/transition2//weights')),
     'layer.1.net.3.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/pair_transition/transition2//bias')),
  }

  with open(os.path.join(args.data_dir, 'params', f'params_{args.model_name}.npz'), 'rb') as f:
    params = np.load(io.BytesIO(f.read()), allow_pickle=False)
  state_dict = state_dict_get(params, scope_list)

  evoformer_iteration = evoformer.EvoformerBlock(dim=(384, 256, 128, 32),
                                               heads=(8, 4),
                                               dim_head=32,
                                               attn_dropout=(0.15, 0.25),
                                               ff_dropout=.0)
  evoformer_iteration.load_state_dict(state_dict)
  print(evoformer_iteration)
  if args.output:
    with open(os.path.join(args.output, f'evoformer_iteration.npz'), 'rb') as f:
      data = np.load(io.BytesIO(f.read()), allow_pickle=True)

    msa_act, msa_mask = data['msa_act'], data['msa_mask']
    pair_act, pair_mask = data['pair_act'], data['pair_mask']
    result = data['result']

    msa_act = rearrange(torch.from_numpy(msa_act), 'm i d -> () m i d')
    msa_mask = rearrange(torch.from_numpy(msa_mask), 'm i -> () m i')
    pair_act = rearrange(torch.from_numpy(pair_act), 'i j d -> () i j d')
    pair_mask = rearrange(torch.from_numpy(pair_mask), 'i j -> () i j')

    evoformer_iteration.eval()
    with torch.no_grad():
      print(pair_act.shape)
      pair_act, msa_act, *_ = evoformer_iteration([pair_act.float(), msa_act.float(), pair_mask.bool(), msa_mask.bool()])
    print('pair', pair_act)
    print('msa', msa_act)
    print('result', result)

def trunk_evoformer_stack(args):
  def attention_with_head(x, fn=None):
    if isinstance(x, list):
      x = list(map(lambda a: rearrange(a, '... h d -> ... (h d)'), x))
    else:
      x = rearrange(x, '... h d -> ... (h d)')
    return fn(x)

  def proj_out_weight(x):
    x = rearrange(x, 'h d ... -> (h d) ...')
    return weights_to_weights(x)

  def proj_out_rearrange(x):
    x = rearrange(x, 'c d f -> (c d) f')
    return weights_to_weights(x)
  proj_out_wrap = functools.partial(layer_stack_to_module_list, fn=proj_out_rearrange)

  npz_to_tensor_fn = functools.partial(attention_with_head, fn=npz_to_tensor)
  weights_to_weights_fn = functools.partial(attention_with_head, fn=weights_to_weights)
  chunk_weights_to_one_fn = functools.partial(attention_with_head, fn=chunk_weights_to_one)
  chunk_bias_to_one_fn = functools.partial(attention_with_head, fn=chunk_bias_to_one)

  npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, fn=npz_to_tensor)
  weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, fn=weights_to_weights)
  chunk_weights_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_weights_to_one)
  chunk_bias_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_bias_to_one)

  attn_npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, fn=npz_to_tensor_fn)
  attn_weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, fn=weights_to_weights_fn)
  attn_chunk_weights_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_weights_to_one_fn)
  attn_chunk_bias_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_bias_to_one_fn)
  attn_proj_out_wrap = functools.partial(layer_stack_to_module_list, fn=proj_out_weight)


  scope_iteration = {
     'layer.2.row_attn.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/query_norm//scale')),
     'layer.2.row_attn.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/query_norm//offset')),
     'layer.2.row_attn.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/feat_2d_norm//scale')),
     'layer.2.row_attn.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/feat_2d_norm//offset')),
     'layer.2.row_attn.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias//feat_2d_weights')),
     'layer.2.row_attn.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//query_w')),
     'layer.2.row_attn.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//key_w', 'alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//value_w'])),
     'layer.2.row_attn.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//gating_w')),
     'layer.2.row_attn.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//gating_b')),
     'layer.2.row_attn.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//output_w')),
     'layer.2.row_attn.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//output_b')),
     'layer.2.col_attn.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/query_norm//scale')),
     'layer.2.col_attn.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/query_norm//offset')),
     'layer.2.col_attn.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//query_w')),
     'layer.2.col_attn.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//key_w', 'alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//value_w'])),
     'layer.2.col_attn.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//gating_w')),
     'layer.2.col_attn.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//gating_b')),
     'layer.2.col_attn.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//output_w')),
     'layer.2.col_attn.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//output_b')),
     'layer.3.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_transition/input_layer_norm//scale')),
     'layer.3.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_transition/input_layer_norm//offset')),
     'layer.3.net.0.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_transition/transition1//weights')),
     'layer.3.net.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_transition/transition1//bias')),
     'layer.3.net.3.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_transition/transition2//weights')),
     'layer.3.net.3.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_transition/transition2//bias')),
     'layer.0.outer_mean.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean/layer_norm_input//scale')),
     'layer.0.outer_mean.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean/layer_norm_input//offset')),
     'layer.0.outer_mean.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean/left_projection//weights')),
     'layer.0.outer_mean.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean/left_projection//bias')),
     'layer.0.outer_mean.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean/right_projection//weights')),
     'layer.0.outer_mean.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean/right_projection//bias')),
     'layer.0.outer_mean.proj_out.weight': (proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean//output_w')),
     'layer.0.outer_mean.proj_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean//output_b')),
     'layer.0.triangle_multiply_outgoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/layer_norm_input//scale')),
     'layer.0.triangle_multiply_outgoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/layer_norm_input//offset')),
     'layer.0.triangle_multiply_outgoing.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/left_projection//weights')),
     'layer.0.triangle_multiply_outgoing.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/left_projection//bias')),
     'layer.0.triangle_multiply_outgoing.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/right_projection//weights')),
     'layer.0.triangle_multiply_outgoing.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/right_projection//bias')),
     'layer.0.triangle_multiply_outgoing.left_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/left_gate//weights')),
     'layer.0.triangle_multiply_outgoing.left_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/left_gate//bias')),
     'layer.0.triangle_multiply_outgoing.right_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/right_gate//weights')),
     'layer.0.triangle_multiply_outgoing.right_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/right_gate//bias')),
     'layer.0.triangle_multiply_outgoing.out_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/gating_linear//weights')),
     'layer.0.triangle_multiply_outgoing.out_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/gating_linear//bias')),
     'layer.0.triangle_multiply_outgoing.to_out_norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/center_layer_norm//scale')),
     'layer.0.triangle_multiply_outgoing.to_out_norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/center_layer_norm//offset')),
     'layer.0.triangle_multiply_outgoing.to_out.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/output_projection//weights')),
     'layer.0.triangle_multiply_outgoing.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/output_projection//bias')),
     'layer.0.triangle_multiply_ingoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/layer_norm_input//scale')),
     'layer.0.triangle_multiply_ingoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/layer_norm_input//offset')),
     'layer.0.triangle_multiply_ingoing.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/left_projection//weights')),
     'layer.0.triangle_multiply_ingoing.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/left_projection//bias')),
     'layer.0.triangle_multiply_ingoing.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/right_projection//weights')),
     'layer.0.triangle_multiply_ingoing.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/right_projection//bias')),
     'layer.0.triangle_multiply_ingoing.left_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/left_gate//weights')),
     'layer.0.triangle_multiply_ingoing.left_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/left_gate//bias')),
     'layer.0.triangle_multiply_ingoing.right_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/right_gate//weights')),
     'layer.0.triangle_multiply_ingoing.right_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/right_gate//bias')),
     'layer.0.triangle_multiply_ingoing.out_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/gating_linear//weights')),
     'layer.0.triangle_multiply_ingoing.out_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/gating_linear//bias')),
     'layer.0.triangle_multiply_ingoing.to_out_norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/center_layer_norm//scale')),
     'layer.0.triangle_multiply_ingoing.to_out_norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/center_layer_norm//offset')),
     'layer.0.triangle_multiply_ingoing.to_out.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/output_projection//weights')),
     'layer.0.triangle_multiply_ingoing.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/output_projection//bias')),
     'layer.0.triangle_attention_outgoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/query_norm//scale')),
     'layer.0.triangle_attention_outgoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/query_norm//offset')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/query_norm//scale')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/query_norm//offset')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node//feat_2d_weights')),
     'layer.0.triangle_attention_outgoing.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//query_w')),
     'layer.0.triangle_attention_outgoing.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//key_w', 'alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//value_w'])),
     'layer.0.triangle_attention_outgoing.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//gating_w')),
     'layer.0.triangle_attention_outgoing.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//gating_b')),
     'layer.0.triangle_attention_outgoing.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//output_w')),
     'layer.0.triangle_attention_outgoing.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//output_b')),
     'layer.0.triangle_attention_ingoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/query_norm//scale')),
     'layer.0.triangle_attention_ingoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/query_norm//offset')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/query_norm//scale')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/query_norm//offset')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node//feat_2d_weights')),
     'layer.0.triangle_attention_ingoing.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//query_w')),
     'layer.0.triangle_attention_ingoing.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//key_w', 'alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//value_w'])),
     'layer.0.triangle_attention_ingoing.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//gating_w')),
     'layer.0.triangle_attention_ingoing.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//gating_b')),
     'layer.0.triangle_attention_ingoing.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//output_w')),
     'layer.0.triangle_attention_ingoing.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//output_b')),
     'layer.1.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/pair_transition/input_layer_norm//scale')),
     'layer.1.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/pair_transition/input_layer_norm//offset')),
     'layer.1.net.0.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/pair_transition/transition1//weights')),
     'layer.1.net.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/pair_transition/transition1//bias')),
     'layer.1.net.3.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/pair_transition/transition2//weights')),
     'layer.1.net.3.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/pair_transition/transition2//bias')),
  }

  layers = 48

  scope_list = {}
  for i in range(layers):
    for k, (fn, kwargs) in scope_iteration.items():
      scope_list[f'layers.{i}.{k}'] = (functools.partial(fn, i=i), kwargs)

  with open(os.path.join(args.data_dir, 'params', f'params_{args.model_name}.npz'), 'rb') as f:
    params = np.load(io.BytesIO(f.read()), allow_pickle=False)
  state_dict = state_dict_get(params, scope_list)

  evoformer_stack = evoformer.Evoformer(depth=layers,
                                        dim=(384, 256, 128, 32),
                                        heads=(8, 4),
                                        dim_head=32,
                                        attn_dropout=(0.15, 0.25),
                                        ff_dropout=.0)
  evoformer_stack.load_state_dict(state_dict)
  print(evoformer_stack)
  if args.output:
    with open(os.path.join(args.output, f'evoformer_stack.npz'), 'rb') as f:
      data = np.load(io.BytesIO(f.read()), allow_pickle=True)

    msa_act, msa_mask = data['msa_act'], data['msa_mask']
    pair_act, pair_mask = data['pair_act'], data['pair_mask']
    result = data['result']

    msa_act = rearrange(torch.from_numpy(msa_act), 'm i d -> () m i d')
    msa_mask = rearrange(torch.from_numpy(msa_mask), 'm i -> () m i')
    pair_act = rearrange(torch.from_numpy(pair_act), 'i j d -> () i j d')
    pair_mask = rearrange(torch.from_numpy(pair_mask), 'i j -> () i j')

    evoformer_stack.eval()
    with torch.no_grad():
      print(pair_act.shape)
      pair_act, msa_act, *_ = evoformer_stack(
              pair_act.float(),
              msa_act.float(),
              mask=pair_mask.bool(),
              msa_mask=msa_mask.bool())
    print('pair', pair_act)
    print('msa', msa_act)
    print('result', result)

def extra_evoformer_stack(args):
  def attention_with_head(x, has_head=True, fn=None):
    if has_head:
      if isinstance(x, list):
        x = list(map(lambda a: rearrange(a, '... h d -> ... (h d)'), x))
      else:
        x = rearrange(x, '... h d -> ... (h d)')
    return fn(x)

  def proj_out_weight(x):
    x = rearrange(x, 'h d ... -> (h d) ...')
    return weights_to_weights(x)

  def proj_out_rearrange(x):
    x = rearrange(x, 'c d f -> (c d) f')
    return weights_to_weights(x)
  proj_out_wrap = functools.partial(layer_stack_to_module_list, fn=proj_out_rearrange)

  npz_to_tensor_fn = functools.partial(attention_with_head, fn=npz_to_tensor)
  weights_to_weights_fn = functools.partial(attention_with_head, fn=weights_to_weights)
  chunk_weights_to_one_fn = functools.partial(attention_with_head, fn=chunk_weights_to_one)
  chunk_bias_to_one_fn = functools.partial(attention_with_head, fn=chunk_bias_to_one)
  chunk_weights_to_one_fn_without_head = functools.partial(attention_with_head, has_head=False, fn=chunk_weights_to_one)
  chunk_bias_to_one_fn_without_head = functools.partial(attention_with_head, has_head=False, fn=chunk_bias_to_one)

  npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, fn=npz_to_tensor)
  weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, fn=weights_to_weights)
  chunk_weights_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_weights_to_one)
  chunk_weights_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_weights_to_one)
  chunk_bias_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_bias_to_one)

  attn_npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, fn=npz_to_tensor_fn)
  attn_weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, fn=weights_to_weights_fn)
  attn_chunk_weights_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_weights_to_one_fn)
  attn_chunk_weights_to_one_wrap_without_head = functools.partial(layer_stack_to_module_list, fn=chunk_weights_to_one_fn_without_head)
  attn_chunk_bias_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_bias_to_one_fn)
  attn_proj_out_wrap = functools.partial(layer_stack_to_module_list, fn=proj_out_weight)


  scope_iteration = {
     'layer.2.row_attn.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/query_norm//scale')),
     'layer.2.row_attn.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/query_norm//offset')),
     'layer.2.row_attn.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/feat_2d_norm//scale')),
     'layer.2.row_attn.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/feat_2d_norm//offset')),
     'layer.2.row_attn.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias//feat_2d_weights')),
     'layer.2.row_attn.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/attention//query_w')),
     'layer.2.row_attn.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/attention//key_w', 'alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/attention//value_w'])),
     'layer.2.row_attn.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/attention//gating_w')),
     'layer.2.row_attn.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/attention//gating_b')),
     'layer.2.row_attn.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/attention//output_w')),
     'layer.2.row_attn.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/attention//output_b')),
     'layer.2.col_attn.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/query_norm//scale')),
     'layer.2.col_attn.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/query_norm//offset')),
     'layer.2.col_attn.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/attention//query_w')),
     'layer.2.col_attn.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap_without_head, dict(x=['alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/attention//key_w', 'alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/attention//value_w'])),
     'layer.2.col_attn.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/attention//gating_w')),
     'layer.2.col_attn.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/attention//gating_b')),
     'layer.2.col_attn.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/attention//output_w')),
     'layer.2.col_attn.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/attention//output_b')),
     'layer.3.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_transition/input_layer_norm//scale')),
     'layer.3.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_transition/input_layer_norm//offset')),
     'layer.3.net.0.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_transition/transition1//weights')),
     'layer.3.net.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_transition/transition1//bias')),
     'layer.3.net.3.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_transition/transition2//weights')),
     'layer.3.net.3.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_transition/transition2//bias')),
     'layer.0.outer_mean.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/outer_product_mean/layer_norm_input//scale')),
     'layer.0.outer_mean.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/outer_product_mean/layer_norm_input//offset')),
     'layer.0.outer_mean.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/outer_product_mean/left_projection//weights')),
     'layer.0.outer_mean.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/outer_product_mean/left_projection//bias')),
     'layer.0.outer_mean.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/outer_product_mean/right_projection//weights')),
     'layer.0.outer_mean.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/outer_product_mean/right_projection//bias')),
     'layer.0.outer_mean.proj_out.weight': (proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/outer_product_mean//output_w')),
     'layer.0.outer_mean.proj_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/outer_product_mean//output_b')),
     'layer.0.triangle_multiply_outgoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/layer_norm_input//scale')),
     'layer.0.triangle_multiply_outgoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/layer_norm_input//offset')),
     'layer.0.triangle_multiply_outgoing.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/left_projection//weights')),
     'layer.0.triangle_multiply_outgoing.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/left_projection//bias')),
     'layer.0.triangle_multiply_outgoing.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/right_projection//weights')),
     'layer.0.triangle_multiply_outgoing.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/right_projection//bias')),
     'layer.0.triangle_multiply_outgoing.left_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/left_gate//weights')),
     'layer.0.triangle_multiply_outgoing.left_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/left_gate//bias')),
     'layer.0.triangle_multiply_outgoing.right_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/right_gate//weights')),
     'layer.0.triangle_multiply_outgoing.right_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/right_gate//bias')),
     'layer.0.triangle_multiply_outgoing.out_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/gating_linear//weights')),
     'layer.0.triangle_multiply_outgoing.out_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/gating_linear//bias')),
     'layer.0.triangle_multiply_outgoing.to_out_norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/center_layer_norm//scale')),
     'layer.0.triangle_multiply_outgoing.to_out_norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/center_layer_norm//offset')),
     'layer.0.triangle_multiply_outgoing.to_out.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/output_projection//weights')),
     'layer.0.triangle_multiply_outgoing.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/output_projection//bias')),
     'layer.0.triangle_multiply_ingoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/layer_norm_input//scale')),
     'layer.0.triangle_multiply_ingoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/layer_norm_input//offset')),
     'layer.0.triangle_multiply_ingoing.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/left_projection//weights')),
     'layer.0.triangle_multiply_ingoing.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/left_projection//bias')),
     'layer.0.triangle_multiply_ingoing.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/right_projection//weights')),
     'layer.0.triangle_multiply_ingoing.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/right_projection//bias')),
     'layer.0.triangle_multiply_ingoing.left_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/left_gate//weights')),
     'layer.0.triangle_multiply_ingoing.left_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/left_gate//bias')),
     'layer.0.triangle_multiply_ingoing.right_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/right_gate//weights')),
     'layer.0.triangle_multiply_ingoing.right_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/right_gate//bias')),
     'layer.0.triangle_multiply_ingoing.out_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/gating_linear//weights')),
     'layer.0.triangle_multiply_ingoing.out_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/gating_linear//bias')),
     'layer.0.triangle_multiply_ingoing.to_out_norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/center_layer_norm//scale')),
     'layer.0.triangle_multiply_ingoing.to_out_norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/center_layer_norm//offset')),
     'layer.0.triangle_multiply_ingoing.to_out.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/output_projection//weights')),
     'layer.0.triangle_multiply_ingoing.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/output_projection//bias')),
     'layer.0.triangle_attention_outgoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/query_norm//scale')),
     'layer.0.triangle_attention_outgoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/query_norm//offset')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/query_norm//scale')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/query_norm//offset')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node//feat_2d_weights')),
     'layer.0.triangle_attention_outgoing.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/attention//query_w')),
     'layer.0.triangle_attention_outgoing.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/attention//key_w', 'alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/attention//value_w'])),
     'layer.0.triangle_attention_outgoing.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/attention//gating_w')),
     'layer.0.triangle_attention_outgoing.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/attention//gating_b')),
     'layer.0.triangle_attention_outgoing.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/attention//output_w')),
     'layer.0.triangle_attention_outgoing.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/attention//output_b')),
     'layer.0.triangle_attention_ingoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/query_norm//scale')),
     'layer.0.triangle_attention_ingoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/query_norm//offset')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/query_norm//scale')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/query_norm//offset')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node//feat_2d_weights')),
     'layer.0.triangle_attention_ingoing.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/attention//query_w')),
     'layer.0.triangle_attention_ingoing.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/attention//key_w', 'alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/attention//value_w'])),
     'layer.0.triangle_attention_ingoing.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/attention//gating_w')),
     'layer.0.triangle_attention_ingoing.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/attention//gating_b')),
     'layer.0.triangle_attention_ingoing.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/attention//output_w')),
     'layer.0.triangle_attention_ingoing.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/attention//output_b')),
     'layer.1.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/pair_transition/input_layer_norm//scale')),
     'layer.1.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/pair_transition/input_layer_norm//offset')),
     'layer.1.net.0.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/pair_transition/transition1//weights')),
     'layer.1.net.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/pair_transition/transition1//bias')),
     'layer.1.net.3.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/pair_transition/transition2//weights')),
     'layer.1.net.3.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/pair_transition/transition2//bias')),
  }

  layers = 4

  scope_list = {}
  for i in range(layers):
    for k, (fn, kwargs) in scope_iteration.items():
      scope_list[f'layers.{i}.{k}'] = (functools.partial(fn, i=i), kwargs)

  with open(os.path.join(args.data_dir, 'params', f'params_{args.model_name}.npz'), 'rb') as f:
    params = np.load(io.BytesIO(f.read()), allow_pickle=False)
  state_dict = state_dict_get(params, scope_list)

  evoformer_stack = evoformer.Evoformer(depth=layers,
                                        dim=(384, 64, 128, 32),
                                        heads=(8, 4),
                                        dim_head=(8, 32),
                                        attn_dropout=(0.15, 0.25),
                                        ff_dropout=.0,
                                        global_column_attn=True)
  print(evoformer_stack)
  evoformer_stack.load_state_dict(state_dict)
  if args.output:
    with open(os.path.join(args.output, f'extra_evoformer_stack.npz'), 'rb') as f:
      data = np.load(io.BytesIO(f.read()), allow_pickle=True)

    msa_act, msa_mask = data['msa_act'], data['msa_mask']
    pair_act, pair_mask = data['pair_act'], data['pair_mask']
    result = data['result']

    msa_act = rearrange(torch.from_numpy(msa_act), 'm i d -> () m i d')
    msa_mask = rearrange(torch.from_numpy(msa_mask), 'm i -> () m i')
    pair_act = rearrange(torch.from_numpy(pair_act), 'i j d -> () i j d')
    pair_mask = rearrange(torch.from_numpy(pair_mask), 'i j -> () i j')

    evoformer_stack.eval()
    with torch.no_grad():
      print(pair_act.shape)
      pair_act, msa_act, *_ = evoformer_stack(
              pair_act.float(),
              msa_act.float(),
              mask=pair_mask.bool(),
              msa_mask=msa_mask.bool())
    print('pair', pair_act)
    print('msa', msa_act)
    print('result', result)


def evoformer_stack(args):
  def attention_with_head(x, has_head=True, fn=None):
    if has_head:
      if isinstance(x, list):
        x = list(map(lambda a: rearrange(a, '... h d -> ... (h d)'), x))
      else:
        x = rearrange(x, '... h d -> ... (h d)')
    return fn(x)

  def proj_out_weight(x):
    x = rearrange(x, 'h d ... -> (h d) ...')
    return weights_to_weights(x)

  def proj_out_rearrange(x):
    x = rearrange(x, 'c d f -> (c d) f')
    return weights_to_weights(x)
  proj_out_wrap = functools.partial(layer_stack_to_module_list, fn=proj_out_rearrange)

  npz_to_tensor_fn = functools.partial(attention_with_head, fn=npz_to_tensor)
  weights_to_weights_fn = functools.partial(attention_with_head, fn=weights_to_weights)
  chunk_weights_to_one_fn = functools.partial(attention_with_head, fn=chunk_weights_to_one)
  chunk_bias_to_one_fn = functools.partial(attention_with_head, fn=chunk_bias_to_one)
  chunk_weights_to_one_fn_without_head = functools.partial(attention_with_head, has_head=False, fn=chunk_weights_to_one)
  chunk_bias_to_one_fn_without_head = functools.partial(attention_with_head, has_head=False, fn=chunk_bias_to_one)

  npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, fn=npz_to_tensor)
  weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, fn=weights_to_weights)
  chunk_weights_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_weights_to_one)
  chunk_weights_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_weights_to_one)
  chunk_bias_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_bias_to_one)

  attn_npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, fn=npz_to_tensor_fn)
  attn_weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, fn=weights_to_weights_fn)
  attn_chunk_weights_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_weights_to_one_fn)
  attn_chunk_weights_to_one_wrap_without_head = functools.partial(layer_stack_to_module_list, fn=chunk_weights_to_one_fn_without_head)
  attn_chunk_bias_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_bias_to_one_fn)
  attn_proj_out_wrap = functools.partial(layer_stack_to_module_list, fn=proj_out_weight)

  scope_list = {
     'input_emb.to_single_emb.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/evoformer/preprocess_1d//weights')),
     'input_emb.to_single_emb.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/preprocess_1d//bias')),
     'input_emb.to_msa_emb.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/evoformer/preprocess_msa//weights')),
     'input_emb.to_msa_emb.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/preprocess_msa//bias')),
     'input_emb.to_pairwise_emb.to_pairwise_repr.weight': (chunk_weights_to_one, dict(x=['alphafold/alphafold_iteration/evoformer/left_single//weights', 'alphafold/alphafold_iteration/evoformer/right_single//weights'])),
     'input_emb.to_pairwise_emb.to_pairwise_repr.bias': (chunk_bias_to_one, dict(x=['alphafold/alphafold_iteration/evoformer/left_single//bias', 'alphafold/alphafold_iteration/evoformer/right_single//bias'])),
     'input_emb.to_pairwise_emb.relative_pos_emb.embedding.weight': (linear_to_embedding, dict(weights='alphafold/alphafold_iteration/evoformer/pair_activiations//weights', bias='alphafold/alphafold_iteration/evoformer/pair_activiations//bias')),
     'template_emb.template_pairwise_embedder.to_pair.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/embedding2d//weights')),
     'template_emb.template_pairwise_embedder.to_pair.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/embedding2d//bias')),
     'template_emb.template_pairwise_embedder.to_out_norm.weight': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/output_layer_norm//scale')),
     'template_emb.template_pairwise_embedder.to_out_norm.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/output_layer_norm//offset')),
     'template_emb.template_pointwise_attn.to_q.weight': (weights_to_weights_fn, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/attention//query_w')),
     'template_emb.template_pointwise_attn.to_kv.weight': (chunk_weights_to_one_fn, dict(x=['alphafold/alphafold_iteration/evoformer/template_embedding/attention//key_w', 'alphafold/alphafold_iteration/evoformer/template_embedding/attention//value_w'])),
     'template_emb.template_pointwise_attn.to_out.weight': (proj_out_weight, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/attention//output_w')),
     'template_emb.template_pointwise_attn.to_out.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/attention//output_b')),
     'template_emb.template_single_embedder.0.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/evoformer/template_single_embedding//weights')),
     'template_emb.template_single_embedder.0.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/template_single_embedding//bias')),
     'template_emb.template_single_embedder.2.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/evoformer/template_projection//weights')),
     'template_emb.template_single_embedder.2.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/template_projection//bias')),
     'msa_activations_extra.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_activations//weights')),
     'msa_activations_extra.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_activations//bias')),
     'recycling_pos_linear.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/evoformer/prev_pos_linear//weights')),
     'recycling_pos_linear.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/prev_pos_linear//bias')),
     'recycling_msa_norm.weight': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/prev_msa_first_row_norm//scale')),
     'recycling_msa_norm.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/prev_msa_first_row_norm//offset')),
     'recycling_pairwise_norm.weight': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/prev_pair_norm//scale')),
     'recycling_pairwise_norm.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/prev_pair_norm//offset')),
     'to_single_repr.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/evoformer/single_activations//weights')),
     'to_single_repr.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/single_activations//bias')),
  }
  
  template_scope_iteration = {
     'layer.0.triangle_multiply_outgoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/layer_norm_input//scale')),
     'layer.0.triangle_multiply_outgoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/layer_norm_input//offset')),
     'layer.0.triangle_multiply_outgoing.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/left_projection//weights')),
     'layer.0.triangle_multiply_outgoing.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/left_projection//bias')),
     'layer.0.triangle_multiply_outgoing.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/right_projection//weights')),
     'layer.0.triangle_multiply_outgoing.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/right_projection//bias')),
     'layer.0.triangle_multiply_outgoing.left_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/left_gate//weights')),
     'layer.0.triangle_multiply_outgoing.left_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/left_gate//bias')),
     'layer.0.triangle_multiply_outgoing.right_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/right_gate//weights')),
     'layer.0.triangle_multiply_outgoing.right_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/right_gate//bias')),
     'layer.0.triangle_multiply_outgoing.out_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/gating_linear//weights')),
     'layer.0.triangle_multiply_outgoing.out_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/gating_linear//bias')),
     'layer.0.triangle_multiply_outgoing.to_out_norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/center_layer_norm//scale')),
     'layer.0.triangle_multiply_outgoing.to_out_norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/center_layer_norm//offset')),
     'layer.0.triangle_multiply_outgoing.to_out.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/output_projection//weights')),
     'layer.0.triangle_multiply_outgoing.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/output_projection//bias')),
     'layer.0.triangle_multiply_ingoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/layer_norm_input//scale')),
     'layer.0.triangle_multiply_ingoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/layer_norm_input//offset')),
     'layer.0.triangle_multiply_ingoing.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/left_projection//weights')),
     'layer.0.triangle_multiply_ingoing.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/left_projection//bias')),
     'layer.0.triangle_multiply_ingoing.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/right_projection//weights')),
     'layer.0.triangle_multiply_ingoing.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/right_projection//bias')),
     'layer.0.triangle_multiply_ingoing.left_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/left_gate//weights')),
     'layer.0.triangle_multiply_ingoing.left_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/left_gate//bias')),
     'layer.0.triangle_multiply_ingoing.right_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/right_gate//weights')),
     'layer.0.triangle_multiply_ingoing.right_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/right_gate//bias')),
     'layer.0.triangle_multiply_ingoing.out_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/gating_linear//weights')),
     'layer.0.triangle_multiply_ingoing.out_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/gating_linear//bias')),
     'layer.0.triangle_multiply_ingoing.to_out_norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/center_layer_norm//scale')),
     'layer.0.triangle_multiply_ingoing.to_out_norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/center_layer_norm//offset')),
     'layer.0.triangle_multiply_ingoing.to_out.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/output_projection//weights')),
     'layer.0.triangle_multiply_ingoing.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/output_projection//bias')),
     'layer.0.triangle_attention_outgoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/query_norm//scale')),
     'layer.0.triangle_attention_outgoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/query_norm//offset')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/query_norm//scale')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/query_norm//offset')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node//feat_2d_weights')),
     'layer.0.triangle_attention_outgoing.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//query_w')),
     'layer.0.triangle_attention_outgoing.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//key_w', 'alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//value_w'])),
     'layer.0.triangle_attention_outgoing.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//gating_w')),
     'layer.0.triangle_attention_outgoing.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//gating_b')),
     'layer.0.triangle_attention_outgoing.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//output_w')),
     'layer.0.triangle_attention_outgoing.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//output_b')),
     'layer.0.triangle_attention_ingoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/query_norm//scale')),
     'layer.0.triangle_attention_ingoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/query_norm//offset')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/query_norm//scale')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/query_norm//offset')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node//feat_2d_weights')),
     'layer.0.triangle_attention_ingoing.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//query_w')),
     'layer.0.triangle_attention_ingoing.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//key_w', 'alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//value_w'])),
     'layer.0.triangle_attention_ingoing.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//gating_w')),
     'layer.0.triangle_attention_ingoing.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//gating_b')),
     'layer.0.triangle_attention_ingoing.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//output_w')),
     'layer.0.triangle_attention_ingoing.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//output_b')),
     'layer.1.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/input_layer_norm//scale')),
     'layer.1.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/input_layer_norm//offset')),
     'layer.1.net.0.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/transition1//weights')),
     'layer.1.net.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/transition1//bias')),
     'layer.1.net.3.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/transition2//weights')),
     'layer.1.net.3.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/transition2//bias')),
  }

  extra_evoformer_scope_iteration = {
     'layer.2.row_attn.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/query_norm//scale')),
     'layer.2.row_attn.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/query_norm//offset')),
     'layer.2.row_attn.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/feat_2d_norm//scale')),
     'layer.2.row_attn.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/feat_2d_norm//offset')),
     'layer.2.row_attn.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias//feat_2d_weights')),
     'layer.2.row_attn.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/attention//query_w')),
     'layer.2.row_attn.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/attention//key_w', 'alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/attention//value_w'])),
     'layer.2.row_attn.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/attention//gating_w')),
     'layer.2.row_attn.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/attention//gating_b')),
     'layer.2.row_attn.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/attention//output_w')),
     'layer.2.row_attn.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/attention//output_b')),
     'layer.2.col_attn.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/query_norm//scale')),
     'layer.2.col_attn.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/query_norm//offset')),
     'layer.2.col_attn.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/attention//query_w')),
     'layer.2.col_attn.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap_without_head, dict(x=['alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/attention//key_w', 'alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/attention//value_w'])),
     'layer.2.col_attn.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/attention//gating_w')),
     'layer.2.col_attn.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/attention//gating_b')),
     'layer.2.col_attn.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/attention//output_w')),
     'layer.2.col_attn.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/attention//output_b')),
     'layer.3.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_transition/input_layer_norm//scale')),
     'layer.3.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_transition/input_layer_norm//offset')),
     'layer.3.net.0.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_transition/transition1//weights')),
     'layer.3.net.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_transition/transition1//bias')),
     'layer.3.net.3.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_transition/transition2//weights')),
     'layer.3.net.3.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_transition/transition2//bias')),
     'layer.0.outer_mean.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/outer_product_mean/layer_norm_input//scale')),
     'layer.0.outer_mean.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/outer_product_mean/layer_norm_input//offset')),
     'layer.0.outer_mean.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/outer_product_mean/left_projection//weights')),
     'layer.0.outer_mean.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/outer_product_mean/left_projection//bias')),
     'layer.0.outer_mean.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/outer_product_mean/right_projection//weights')),
     'layer.0.outer_mean.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/outer_product_mean/right_projection//bias')),
     'layer.0.outer_mean.proj_out.weight': (proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/outer_product_mean//output_w')),
     'layer.0.outer_mean.proj_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/outer_product_mean//output_b')),
     'layer.0.triangle_multiply_outgoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/layer_norm_input//scale')),
     'layer.0.triangle_multiply_outgoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/layer_norm_input//offset')),
     'layer.0.triangle_multiply_outgoing.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/left_projection//weights')),
     'layer.0.triangle_multiply_outgoing.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/left_projection//bias')),
     'layer.0.triangle_multiply_outgoing.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/right_projection//weights')),
     'layer.0.triangle_multiply_outgoing.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/right_projection//bias')),
     'layer.0.triangle_multiply_outgoing.left_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/left_gate//weights')),
     'layer.0.triangle_multiply_outgoing.left_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/left_gate//bias')),
     'layer.0.triangle_multiply_outgoing.right_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/right_gate//weights')),
     'layer.0.triangle_multiply_outgoing.right_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/right_gate//bias')),
     'layer.0.triangle_multiply_outgoing.out_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/gating_linear//weights')),
     'layer.0.triangle_multiply_outgoing.out_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/gating_linear//bias')),
     'layer.0.triangle_multiply_outgoing.to_out_norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/center_layer_norm//scale')),
     'layer.0.triangle_multiply_outgoing.to_out_norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/center_layer_norm//offset')),
     'layer.0.triangle_multiply_outgoing.to_out.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/output_projection//weights')),
     'layer.0.triangle_multiply_outgoing.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/output_projection//bias')),
     'layer.0.triangle_multiply_ingoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/layer_norm_input//scale')),
     'layer.0.triangle_multiply_ingoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/layer_norm_input//offset')),
     'layer.0.triangle_multiply_ingoing.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/left_projection//weights')),
     'layer.0.triangle_multiply_ingoing.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/left_projection//bias')),
     'layer.0.triangle_multiply_ingoing.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/right_projection//weights')),
     'layer.0.triangle_multiply_ingoing.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/right_projection//bias')),
     'layer.0.triangle_multiply_ingoing.left_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/left_gate//weights')),
     'layer.0.triangle_multiply_ingoing.left_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/left_gate//bias')),
     'layer.0.triangle_multiply_ingoing.right_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/right_gate//weights')),
     'layer.0.triangle_multiply_ingoing.right_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/right_gate//bias')),
     'layer.0.triangle_multiply_ingoing.out_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/gating_linear//weights')),
     'layer.0.triangle_multiply_ingoing.out_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/gating_linear//bias')),
     'layer.0.triangle_multiply_ingoing.to_out_norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/center_layer_norm//scale')),
     'layer.0.triangle_multiply_ingoing.to_out_norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/center_layer_norm//offset')),
     'layer.0.triangle_multiply_ingoing.to_out.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/output_projection//weights')),
     'layer.0.triangle_multiply_ingoing.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/output_projection//bias')),
     'layer.0.triangle_attention_outgoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/query_norm//scale')),
     'layer.0.triangle_attention_outgoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/query_norm//offset')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/query_norm//scale')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/query_norm//offset')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node//feat_2d_weights')),
     'layer.0.triangle_attention_outgoing.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/attention//query_w')),
     'layer.0.triangle_attention_outgoing.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/attention//key_w', 'alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/attention//value_w'])),
     'layer.0.triangle_attention_outgoing.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/attention//gating_w')),
     'layer.0.triangle_attention_outgoing.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/attention//gating_b')),
     'layer.0.triangle_attention_outgoing.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/attention//output_w')),
     'layer.0.triangle_attention_outgoing.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/attention//output_b')),
     'layer.0.triangle_attention_ingoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/query_norm//scale')),
     'layer.0.triangle_attention_ingoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/query_norm//offset')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/query_norm//scale')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/query_norm//offset')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node//feat_2d_weights')),
     'layer.0.triangle_attention_ingoing.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/attention//query_w')),
     'layer.0.triangle_attention_ingoing.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/attention//key_w', 'alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/attention//value_w'])),
     'layer.0.triangle_attention_ingoing.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/attention//gating_w')),
     'layer.0.triangle_attention_ingoing.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/attention//gating_b')),
     'layer.0.triangle_attention_ingoing.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/attention//output_w')),
     'layer.0.triangle_attention_ingoing.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/attention//output_b')),
     'layer.1.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/pair_transition/input_layer_norm//scale')),
     'layer.1.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/pair_transition/input_layer_norm//offset')),
     'layer.1.net.0.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/pair_transition/transition1//weights')),
     'layer.1.net.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/pair_transition/transition1//bias')),
     'layer.1.net.3.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/pair_transition/transition2//weights')),
     'layer.1.net.3.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/pair_transition/transition2//bias')),
  }
  evoformer_scope_iteration = {
     'layer.2.row_attn.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/query_norm//scale')),
     'layer.2.row_attn.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/query_norm//offset')),
     'layer.2.row_attn.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/feat_2d_norm//scale')),
     'layer.2.row_attn.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/feat_2d_norm//offset')),
     'layer.2.row_attn.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias//feat_2d_weights')),
     'layer.2.row_attn.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//query_w')),
     'layer.2.row_attn.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//key_w', 'alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//value_w'])),
     'layer.2.row_attn.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//gating_w')),
     'layer.2.row_attn.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//gating_b')),
     'layer.2.row_attn.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//output_w')),
     'layer.2.row_attn.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//output_b')),
     'layer.2.col_attn.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/query_norm//scale')),
     'layer.2.col_attn.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/query_norm//offset')),
     'layer.2.col_attn.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//query_w')),
     'layer.2.col_attn.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//key_w', 'alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//value_w'])),
     'layer.2.col_attn.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//gating_w')),
     'layer.2.col_attn.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//gating_b')),
     'layer.2.col_attn.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//output_w')),
     'layer.2.col_attn.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//output_b')),
     'layer.3.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_transition/input_layer_norm//scale')),
     'layer.3.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_transition/input_layer_norm//offset')),
     'layer.3.net.0.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_transition/transition1//weights')),
     'layer.3.net.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_transition/transition1//bias')),
     'layer.3.net.3.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_transition/transition2//weights')),
     'layer.3.net.3.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_transition/transition2//bias')),
     'layer.0.outer_mean.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean/layer_norm_input//scale')),
     'layer.0.outer_mean.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean/layer_norm_input//offset')),
     'layer.0.outer_mean.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean/left_projection//weights')),
     'layer.0.outer_mean.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean/left_projection//bias')),
     'layer.0.outer_mean.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean/right_projection//weights')),
     'layer.0.outer_mean.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean/right_projection//bias')),
     'layer.0.outer_mean.proj_out.weight': (proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean//output_w')),
     'layer.0.outer_mean.proj_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean//output_b')),
     'layer.0.triangle_multiply_outgoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/layer_norm_input//scale')),
     'layer.0.triangle_multiply_outgoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/layer_norm_input//offset')),
     'layer.0.triangle_multiply_outgoing.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/left_projection//weights')),
     'layer.0.triangle_multiply_outgoing.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/left_projection//bias')),
     'layer.0.triangle_multiply_outgoing.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/right_projection//weights')),
     'layer.0.triangle_multiply_outgoing.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/right_projection//bias')),
     'layer.0.triangle_multiply_outgoing.left_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/left_gate//weights')),
     'layer.0.triangle_multiply_outgoing.left_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/left_gate//bias')),
     'layer.0.triangle_multiply_outgoing.right_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/right_gate//weights')),
     'layer.0.triangle_multiply_outgoing.right_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/right_gate//bias')),
     'layer.0.triangle_multiply_outgoing.out_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/gating_linear//weights')),
     'layer.0.triangle_multiply_outgoing.out_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/gating_linear//bias')),
     'layer.0.triangle_multiply_outgoing.to_out_norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/center_layer_norm//scale')),
     'layer.0.triangle_multiply_outgoing.to_out_norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/center_layer_norm//offset')),
     'layer.0.triangle_multiply_outgoing.to_out.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/output_projection//weights')),
     'layer.0.triangle_multiply_outgoing.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/output_projection//bias')),
     'layer.0.triangle_multiply_ingoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/layer_norm_input//scale')),
     'layer.0.triangle_multiply_ingoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/layer_norm_input//offset')),
     'layer.0.triangle_multiply_ingoing.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/left_projection//weights')),
     'layer.0.triangle_multiply_ingoing.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/left_projection//bias')),
     'layer.0.triangle_multiply_ingoing.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/right_projection//weights')),
     'layer.0.triangle_multiply_ingoing.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/right_projection//bias')),
     'layer.0.triangle_multiply_ingoing.left_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/left_gate//weights')),
     'layer.0.triangle_multiply_ingoing.left_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/left_gate//bias')),
     'layer.0.triangle_multiply_ingoing.right_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/right_gate//weights')),
     'layer.0.triangle_multiply_ingoing.right_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/right_gate//bias')),
     'layer.0.triangle_multiply_ingoing.out_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/gating_linear//weights')),
     'layer.0.triangle_multiply_ingoing.out_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/gating_linear//bias')),
     'layer.0.triangle_multiply_ingoing.to_out_norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/center_layer_norm//scale')),
     'layer.0.triangle_multiply_ingoing.to_out_norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/center_layer_norm//offset')),
     'layer.0.triangle_multiply_ingoing.to_out.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/output_projection//weights')),
     'layer.0.triangle_multiply_ingoing.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/output_projection//bias')),
     'layer.0.triangle_attention_outgoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/query_norm//scale')),
     'layer.0.triangle_attention_outgoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/query_norm//offset')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/query_norm//scale')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/query_norm//offset')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node//feat_2d_weights')),
     'layer.0.triangle_attention_outgoing.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//query_w')),
     'layer.0.triangle_attention_outgoing.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//key_w', 'alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//value_w'])),
     'layer.0.triangle_attention_outgoing.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//gating_w')),
     'layer.0.triangle_attention_outgoing.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//gating_b')),
     'layer.0.triangle_attention_outgoing.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//output_w')),
     'layer.0.triangle_attention_outgoing.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//output_b')),
     'layer.0.triangle_attention_ingoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/query_norm//scale')),
     'layer.0.triangle_attention_ingoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/query_norm//offset')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/query_norm//scale')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/query_norm//offset')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node//feat_2d_weights')),
     'layer.0.triangle_attention_ingoing.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//query_w')),
     'layer.0.triangle_attention_ingoing.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//key_w', 'alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//value_w'])),
     'layer.0.triangle_attention_ingoing.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//gating_w')),
     'layer.0.triangle_attention_ingoing.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//gating_b')),
     'layer.0.triangle_attention_ingoing.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//output_w')),
     'layer.0.triangle_attention_ingoing.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//output_b')),
     'layer.1.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/pair_transition/input_layer_norm//scale')),
     'layer.1.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/pair_transition/input_layer_norm//offset')),
     'layer.1.net.0.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/pair_transition/transition1//weights')),
     'layer.1.net.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/pair_transition/transition1//bias')),
     'layer.1.net.3.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/pair_transition/transition2//weights')),
     'layer.1.net.3.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/pair_transition/transition2//bias')),
  }

  for name, layers, scope_iteration in (('template_emb.template_pairwise_embedder.pair_stack', 2, template_scope_iteration), ('evoformer_extra', 4, extra_evoformer_scope_iteration), ('evoformer', 48, evoformer_scope_iteration)):
    for i in range(layers):
      for k, (fn, kwargs) in scope_iteration.items():
        scope_list[f'{name}.layers.{i}.{k}'] = (functools.partial(fn, i=i), kwargs)

  with open(os.path.join(args.data_dir, 'params', f'params_{args.model_name}.npz'), 'rb') as f:
    params = np.load(io.BytesIO(f.read()), allow_pickle=False)
  state_dict = state_dict_get(params, scope_list)

  evoformer_stack = alphafold2.Alphafold2(dim=(384, 128),
                               evoformer_depth=48,
                               template_depth=2,
                               num_tokens=22,
                               num_msa_tokens=49,
                               attn_dropout=(0.15, 0.25),
                               recycling_single_repr=False,
                               recycling_pos=True)
  print(evoformer_stack)
  evoformer_stack.load_state_dict(state_dict)
  if args.output:
    with open(os.path.join(args.output, f'evoformer.npz'), 'rb') as f:
      data = np.load(io.BytesIO(f.read()), allow_pickle=True)

    batch, result = data['batch'], data['result']
    batch = batch.tolist()
    print(batch.keys())
    batch['template_aatype_index'] = np.clip(batch['template_aatype'], 0, 20)
    batch['template_all_atom_mask'] = batch['template_all_atom_masks']
    batch = make_atom14_positions(batch, along_axis=2, prefix='template_')
    print(batch.keys())

    template_angles = functional.angles_from_positions(
        torch.from_numpy(batch['template_aatype_index']),
        torch.from_numpy(batch['template_atom14_gt_positions']),
        torch.from_numpy(batch['template_atom14_gt_exists']),
        placeholder_for_undefined=False)
    for k, v in template_angles.items():
      batch[f'template_{k}'] = v.numpy()
    print(batch.keys())

    kv_trans = [
       ('aatype', 'seq', 'i -> () i'),
       ('seq_mask', 'mask', 'i -> () i'),
       ('residue_index', 'seq_index', 'i -> () i'),
       ('target_feat', 'target_feat', 'i d -> () i d'),
       ('msa_feat', 'msa_feat', 'm i d -> () m i d'),
       ('msa_mask', 'msa_mask', 'm i -> () m i'),
       ('template_aatype', 'template_aatype', '... -> () ...'),
       ('template_mask', 'template_mask', '... -> () ...'),
       ('template_pseudo_beta_mask', 'template_pseudo_beta_mask', '... -> () ...'),
       ('template_pseudo_beta', 'template_pseudo_beta', '... -> () ...'),
       ('template_all_atom_positions', 'template_all_atom_positions', '... -> () ...'),
       ('template_all_atom_masks', 'template_all_atom_masks', '... -> () ...'),
       ('template_torsion_angles', 'template_torsion_angles', '... -> () ...'),
       ('template_torsion_angles_alt', 'template_torsion_angles_alt', '... -> () ...'),
       ('template_torsion_angles_mask', 'template_torsion_angles_mask', '... -> () ...'),
       ('extra_msa', 'extra_msa', '... -> () ...'),
       ('extra_msa_mask', 'extra_msa_mask', '... -> () ...'),
       ('extra_has_deletion', 'extra_has_deletion', '... -> () ...'),
       ('extra_deletion_value', 'extra_deletion_value', '... -> () ...'),
       ('prev_pos', 'prev_pos', '... -> () ...'),
       ('prev_msa_first_row', 'prev_msa_first_row', '... -> () ...'),
       ('prev_pair', 'prev_pair', '... -> () ...'),
    ]
    batch_new = {}
    for from_k, to_k, reshape in kv_trans:
      if reshape:
        batch_new[to_k] = rearrange(torch.from_numpy(np.array(batch[from_k])), reshape)
      else:
        batch_new[to_k] = torch.from_numpy(batch[from_k])

    recyclables = alphafold2.Recyclables(batch_new.pop('prev_msa_first_row'),
                                         batch_new.pop('prev_pair'),
                                         batch_new.pop('prev_pos'))
    batch_new['recyclables'] = recyclables
    evoformer_stack.eval()
    with torch.no_grad():
      r = evoformer_stack(batch_new, shard_size=4)
    print('result1', r)
    print('result2', result)


def invariant_point_attention(args):
  def rearrange_point(x):
    return rearrange(x, '... (c x) -> ... (x c)', c=3)

  def rearrange_out(x):
    # x = rearrange(x, '... (p q r s) -> ... p q r s', p=192, q=288, r=96, s=1536)
    p, q, r, s = 192, 288, 96, 1536
    x = list(torch.split(x, (p, q, r, s), dim=-1))
    x[1] = rearrange(x[1], '... (c x) -> ... (x c)', c=3)
    #return rearrange(x, '... p (c x) r s -> ... (p x c r s)', c=3)
    return torch.cat(x, dim=-1)

  def weights_to_chunk_with_heads(x, heads, from_idx, to_idx, fn=None):
    x = rearrange(x, 'i (h o) -> (i h) o', h=heads)
    x = rearrange(x[:,from_idx//heads:to_idx//heads], '(i h) o -> i (h o)', h=heads)
    if fn:
      x = fn(x)
    return weights_to_weights(x)

  def npz_to_chunk_with_heads(x, heads, from_idx, to_idx, fn=None):
    x = rearrange(x, '(h o) -> h o', h=heads)
    x = rearrange(x[:,from_idx//heads:to_idx//heads], 'h o -> (h o)')
    if fn:
      x = fn(x)
    return npz_to_tensor(x)

  scope_list = {
      'to_scalar_q.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/q_scalar//weights')),
      'to_scalar_q.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/q_scalar//bias')),
      'to_scalar_k.weight': (functools.partial(weights_to_chunk_with_heads, heads=12, from_idx=0, to_idx=192), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/kv_scalar//weights')),
      'to_scalar_k.bias': (functools.partial(npz_to_chunk_with_heads, heads=12, from_idx=0, to_idx=192), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/kv_scalar//bias')),
      'to_scalar_v.weight': (functools.partial(weights_to_chunk_with_heads, heads=12, from_idx=192, to_idx=384), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/kv_scalar//weights')),
      'to_scalar_v.bias': (functools.partial(npz_to_chunk_with_heads, heads=12, from_idx=192, to_idx=384), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/kv_scalar//bias')),
      'to_point_q.weight': (functools.partial(functions_chain, fn_list=[rearrange_point, weights_to_weights]), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/q_point_local//weights')),
      'to_point_q.bias': (functools.partial(functions_chain, fn_list=[rearrange_point, npz_to_tensor]), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/q_point_local//bias')),
      'to_point_k.weight': (functools.partial(weights_to_chunk_with_heads, heads=12*3, from_idx=0, to_idx=144, fn=rearrange_point), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/kv_point_local//weights')),
      'to_point_k.bias': (functools.partial(npz_to_chunk_with_heads, heads=12*3, from_idx=0, to_idx=144, fn=rearrange_point), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/kv_point_local//bias')),
      'to_point_v.weight': (functools.partial(weights_to_chunk_with_heads, heads=12*3, from_idx=144, to_idx=432, fn=rearrange_point), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/kv_point_local//weights')),
      'to_point_v.bias': (functools.partial(npz_to_chunk_with_heads, heads=12*3, from_idx=144, to_idx=432, fn=rearrange_point), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/kv_point_local//bias')),
      'point_weights': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention//trainable_point_weights')),
      'to_pairwise_attn_bias.0.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/attention_2d//weights')),
      'to_pairwise_attn_bias.0.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/attention_2d//bias')),
      'to_out.weight': (functools.partial(functions_chain, fn_list=[weights_to_weights, rearrange_out]), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/output_projection//weights')),
      'to_out.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/output_projection//bias')),
  }
  with open(os.path.join(args.data_dir, 'params', f'params_{args.model_name}.npz'), 'rb') as f:
    params = np.load(io.BytesIO(f.read()), allow_pickle=False)
  state_dict = state_dict_get(params, scope_list)

  invariant_point_attention = folding.InvariantPointAttention(
                                        dim=384,
                                        pairwise_repr_dim=128,
                                        heads=12,
                                        qkv_use_bias=True)
  invariant_point_attention.load_state_dict(state_dict)
  print(invariant_point_attention)
  if args.output:
    with open(os.path.join(args.output, f'invariant_point_attention.npz'), 'rb') as f:
      data = np.load(io.BytesIO(f.read()), allow_pickle=True)

    seq_act, seq_mask = data['seq_act'], data['seq_mask']
    pair_act = data['pair_act']
    affine = data['affine']
    result = data['result']

    quaternions = rearrange(torch.from_numpy(affine[...,:4]), 'i d -> () i d')
    translations = rearrange(torch.from_numpy(affine[...,4:]), 'i d -> () i d')
    print(affine.shape)
    print(quaternions.shape)
    print(translations.shape)
    rotations = functional.quaternion_to_matrix(quaternions)
    seq_act = rearrange(torch.from_numpy(seq_act), 'i d -> () i d')
    seq_mask = rearrange(torch.from_numpy(seq_mask), 'i -> () i')
    pair_act = rearrange(torch.from_numpy(pair_act), 'i j d -> () i j d')

    invariant_point_attention.eval()
    with torch.no_grad():
      print(pair_act.shape)
      seq_act = invariant_point_attention(
              seq_act.float(),
              pair_act.float(),
              rotations=rotations.float(),
              translations=translations.float(),
              mask=seq_mask.bool())
    print('pair', pair_act)
    print('seq', seq_act)
    print('result', result)

def structure_module(args):
  def rearrange_point(x):
    return rearrange(x, '... (c x) -> ... (x c)', c=3)

  def rearrange_out(x):
    # x = rearrange(x, '... (p q r s) -> ... p q r s', p=192, q=288, r=96, s=1536)
    p, q, r, s = 192, 288, 96, 1536
    x = list(torch.split(x, (p, q, r, s), dim=-1))
    x[1] = rearrange(x[1], '... (c x) -> ... (x c)', c=3)
    #return rearrange(x, '... p (c x) r s -> ... (p x c r s)', c=3)
    return torch.cat(x, dim=-1)

  def weights_to_chunk_with_heads(x, heads, from_idx, to_idx, fn=None):
    x = rearrange(x, 'i (h o) -> (i h) o', h=heads)
    x = rearrange(x[:,from_idx//heads:to_idx//heads], '(i h) o -> i (h o)', h=heads)
    if fn:
      x = fn(x)
    return weights_to_weights(x)

  def npz_to_chunk_with_heads(x, heads, from_idx, to_idx, fn=None):
    x = rearrange(x, '(h o) -> h o', h=heads)
    x = rearrange(x[:,from_idx//heads:to_idx//heads], 'h o -> (h o)')
    if fn:
      x = fn(x)
    return npz_to_tensor(x)

  scope_list = {
      'struct_module.single_repr_norm.weight': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/single_layer_norm//scale')),
      'struct_module.single_repr_norm.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/single_layer_norm//offset')),
      'struct_module.pairwise_repr_norm.weight': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/pair_layer_norm//scale')),
      'struct_module.pairwise_repr_norm.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/pair_layer_norm//offset')),
      'struct_module.single_repr_dim.0.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/structure_module/initial_projection//weights')),
      'struct_module.single_repr_dim.0.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/initial_projection//bias')),
      'struct_module.ipa_block.attn.to_scalar_q.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/q_scalar//weights')),
      'struct_module.ipa_block.attn.to_scalar_q.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/q_scalar//bias')),
      'struct_module.ipa_block.attn.to_scalar_k.weight': (functools.partial(weights_to_chunk_with_heads, heads=12, from_idx=0, to_idx=192), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/kv_scalar//weights')),
      'struct_module.ipa_block.attn.to_scalar_k.bias': (functools.partial(npz_to_chunk_with_heads, heads=12, from_idx=0, to_idx=192), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/kv_scalar//bias')),
      'struct_module.ipa_block.attn.to_scalar_v.weight': (functools.partial(weights_to_chunk_with_heads, heads=12, from_idx=192, to_idx=384), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/kv_scalar//weights')),
      'struct_module.ipa_block.attn.to_scalar_v.bias': (functools.partial(npz_to_chunk_with_heads, heads=12, from_idx=192, to_idx=384), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/kv_scalar//bias')),
      'struct_module.ipa_block.attn.to_point_q.weight': (functools.partial(functions_chain, fn_list=[rearrange_point, weights_to_weights]), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/q_point_local//weights')),
      'struct_module.ipa_block.attn.to_point_q.bias': (functools.partial(functions_chain, fn_list=[rearrange_point, npz_to_tensor]), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/q_point_local//bias')),
      'struct_module.ipa_block.attn.to_point_k.weight': (functools.partial(weights_to_chunk_with_heads, heads=12*3, from_idx=0, to_idx=144, fn=rearrange_point), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/kv_point_local//weights')),
      'struct_module.ipa_block.attn.to_point_k.bias': (functools.partial(npz_to_chunk_with_heads, heads=12*3, from_idx=0, to_idx=144, fn=rearrange_point), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/kv_point_local//bias')),
      'struct_module.ipa_block.attn.to_point_v.weight': (functools.partial(weights_to_chunk_with_heads, heads=12*3, from_idx=144, to_idx=432, fn=rearrange_point), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/kv_point_local//weights')),
      'struct_module.ipa_block.attn.to_point_v.bias': (functools.partial(npz_to_chunk_with_heads, heads=12*3, from_idx=144, to_idx=432, fn=rearrange_point), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/kv_point_local//bias')),
      'struct_module.ipa_block.attn.point_weights': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention//trainable_point_weights')),
      'struct_module.ipa_block.attn.to_pairwise_attn_bias.0.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/attention_2d//weights')),
      'struct_module.ipa_block.attn.to_pairwise_attn_bias.0.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/attention_2d//bias')),
      'struct_module.ipa_block.attn.to_out.weight': (functools.partial(functions_chain, fn_list=[weights_to_weights, rearrange_out]), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/output_projection//weights')),
      'struct_module.ipa_block.attn.to_out.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/output_projection//bias')),
      'struct_module.ipa_block.attn_norm.weight': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/attention_layer_norm//scale')),
      'struct_module.ipa_block.attn_norm.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/attention_layer_norm//offset')),
      'struct_module.ipa_block.ff.0.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/transition//weights')),
      'struct_module.ipa_block.ff.0.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/transition//bias')),
      'struct_module.ipa_block.ff.2.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/transition_1//weights')),
      'struct_module.ipa_block.ff.2.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/transition_1//bias')),
      'struct_module.ipa_block.ff.4.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/transition_2//weights')),
      'struct_module.ipa_block.ff.4.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/transition_2//bias')),
      'struct_module.ipa_block.ff_norm.weight': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/transition_layer_norm//scale')),
      'struct_module.ipa_block.ff_norm.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/transition_layer_norm//offset')),
      'struct_module.to_affine_update.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/affine_update//weights')),
      'struct_module.to_affine_update.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/affine_update//bias')),
      'struct_module.to_angles.projection.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/rigid_sidechain/input_projection//weights')),
      'struct_module.to_angles.projection.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/rigid_sidechain/input_projection//bias')),
      'struct_module.to_angles.projection_init.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/rigid_sidechain/input_projection_1//weights')),
      'struct_module.to_angles.projection_init.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/rigid_sidechain/input_projection_1//bias')),
      'struct_module.to_angles.blocks.0.net.1.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/rigid_sidechain/resblock1//weights')),
      'struct_module.to_angles.blocks.0.net.1.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/rigid_sidechain/resblock1//bias')),
      'struct_module.to_angles.blocks.0.net.3.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/rigid_sidechain/resblock2//weights')),
      'struct_module.to_angles.blocks.0.net.3.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/rigid_sidechain/resblock2//bias')),
      'struct_module.to_angles.blocks.1.net.1.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/rigid_sidechain/resblock1_1//weights')),
      'struct_module.to_angles.blocks.1.net.1.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/rigid_sidechain/resblock1_1//bias')),
      'struct_module.to_angles.blocks.1.net.3.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/rigid_sidechain/resblock2_1//weights')),
      'struct_module.to_angles.blocks.1.net.3.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/rigid_sidechain/resblock2_1//bias')),
      'struct_module.to_angles.to_groups.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/rigid_sidechain/unnormalized_angles//weights')),
      'struct_module.to_angles.to_groups.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/rigid_sidechain/unnormalized_angles//bias')),
  }
  with open(os.path.join(args.data_dir, 'params', f'params_{args.model_name}.npz'), 'rb') as f:
    params = np.load(io.BytesIO(f.read()), allow_pickle=False)
  state_dict = state_dict_get(params, scope_list)

  # structure_module = folding.StructureModule(
  #                                       dim=(384, 256, 128, 32),
  #                                       structure_module_depth=8,
  #                                       structure_module_heads=12,
  #                                       qkv_use_bias=True,
  #                                       position_scale=10.0)
  structure_module = head.FoldingHead(
      dim=(384, 256, 128, 32),
      structure_module_depth=8,
      structure_module_heads=12,
      qkv_use_bias=True,
      position_scale=10.0,
      fape_max=10.0,
      fape_z=10.0)
  print(structure_module)
  structure_module.load_state_dict(state_dict)
  if args.output:
    with open(os.path.join(args.output, f'structure_module.npz'), 'rb') as f:
      data = np.load(io.BytesIO(f.read()), allow_pickle=True)

    print(list(data.keys()))
    seq_act = data['seq_act']
    pair_act = data['pair_act']
    #batch = dict(data['batch'])
    result = data['result']

    seq = rearrange(torch.from_numpy(data['aatype']), 'i -> () i')
    seq = seq.long()
    seq_act = rearrange(torch.from_numpy(seq_act), 'i d -> () i d')
    seq_mask = rearrange(torch.from_numpy(data['seq_mask']), 'i -> () i')
    pair_act = rearrange(torch.from_numpy(pair_act), 'i j d -> () i j d')

    # structure_module.eval()
    with torch.no_grad():
      print(pair_act.shape)
      headers = {}
      representation = {'single': seq_act.float(), 'pair': pair_act.float()}
      batch = {'seq': seq, 'mask': seq_mask}
      for k_to, k_from in (('coord_exists', 'atom14_gt_exists'),
                           ('atom_affine_exists', 'rigidgroups_gt_exists'),
                           ('backbone_affine', 'backbone_affine_tensor'),
                           ('backbone_affine_mask', 'backbone_affine_mask'),
                           ('coord', 'atom14_gt_positions'),
                           ('coord_mask', 'atom14_gt_exists')):
        batch[k_to] = rearrange(torch.from_numpy(data[k_from]), 'i ... -> () i ...')
      for k in ('backbone_affine', 'coord'):
        batch[k] = batch[k].float()
      quaternion, t = torch.split(batch['backbone_affine'], [4, 3], dim=-1)
      R = functional.quaternion_to_matrix(quaternion)
      print('xxx', R.shape)
      print('yyy', t.shape)
      n_idx = residue_constants.atom_order['N']
      ca_idx = residue_constants.atom_order['CA']
      c_idx = residue_constants.atom_order['C']
      X, y = functional.rigids_from_3x3(batch['coord'], indices=(c_idx, ca_idx, n_idx))
      print('R', R)
      print('t', t)
      print('X', X)
      print('y', y)
      batch['backbone_affine'] = (R, t)
      r = structure_module(
              headers,
              representation,
              batch)
      loss = structure_module.loss(r, batch)
    print('r', r)
    print('loss', loss)
    print('result', result)

def plddt(args):
  scope_list = {
     'net.0.weight': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/predicted_lddt_head/input_layer_norm//scale')),
     'net.0.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/predicted_lddt_head/input_layer_norm//offset')),
     'net.1.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/predicted_lddt_head/act_0//weights')),
     'net.1.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/predicted_lddt_head/act_0//bias')),
     'net.3.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/predicted_lddt_head/act_1//weights')),
     'net.3.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/predicted_lddt_head/act_1//bias')),
     'net.5.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/predicted_lddt_head/logits//weights')),
     'net.5.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/predicted_lddt_head/logits//bias')),
  }

  with open(os.path.join(args.data_dir, 'params', f'params_{args.model_name}.npz'), 'rb') as f:
    params = np.load(io.BytesIO(f.read()), allow_pickle=False)
  state_dict = state_dict_get(params, scope_list)

  plddt = head.LDDTHead(dim=(384, 256, 128, 32), num_channels=128)
  plddt.load_state_dict(state_dict)
  print(plddt)
  if args.output:
    with open(os.path.join(args.output, f'predicted_lddt_head.npz'), 'rb') as f:
      data = np.load(io.BytesIO(f.read()), allow_pickle=True)

    seq_act, result = data['seq_act'], data['result']
    seq_act = rearrange(torch.from_numpy(seq_act), 'i d -> () i d')

    plddt.eval()
    with torch.no_grad():
      print(seq_act.shape)
      headers = {'folding': {'act': seq_act.float(), 'coords': None}}
      r = plddt(headers, {}, {})
    print(r)
    print(result)

def alphafold_iteration(args):
  pass

def save_model(args):
  def rearrange_point(x):
    return rearrange(x, '... (c x) -> ... (x c)', c=3)

  def rearrange_out(x):
    # x = rearrange(x, '... (p q r s) -> ... p q r s', p=192, q=288, r=96, s=1536)
    p, q, r, s = 192, 288, 96, 1536
    x = list(torch.split(x, (p, q, r, s), dim=-1))
    x[1] = rearrange(x[1], '... (c x) -> ... (x c)', c=3)
    #return rearrange(x, '... p (c x) r s -> ... (p x c r s)', c=3)
    return torch.cat(x, dim=-1)

  def weights_to_chunk_with_heads(x, heads, from_idx, to_idx, fn=None):
    x = rearrange(x, 'i (h o) -> (i h) o', h=heads)
    x = rearrange(x[:,from_idx//heads:to_idx//heads], '(i h) o -> i (h o)', h=heads)
    if fn:
      x = fn(x)
    return weights_to_weights(x)

  def npz_to_chunk_with_heads(x, heads, from_idx, to_idx, fn=None):
    x = rearrange(x, '(h o) -> h o', h=heads)
    x = rearrange(x[:,from_idx//heads:to_idx//heads], 'h o -> (h o)')
    if fn:
      x = fn(x)
    return npz_to_tensor(x)

  def attention_with_head(x, has_head=True, fn=None):
    if has_head:
      if isinstance(x, list):
        x = list(map(lambda a: rearrange(a, '... h d -> ... (h d)'), x))
      else:
        x = rearrange(x, '... h d -> ... (h d)')
    return fn(x)

  def proj_out_weight(x):
    x = rearrange(x, 'h d ... -> (h d) ...')
    return weights_to_weights(x)

  def proj_out_rearrange(x):
    x = rearrange(x, 'c d f -> (c d) f')
    return weights_to_weights(x)
  proj_out_wrap = functools.partial(layer_stack_to_module_list, fn=proj_out_rearrange)

  npz_to_tensor_fn = functools.partial(attention_with_head, fn=npz_to_tensor)
  weights_to_weights_fn = functools.partial(attention_with_head, fn=weights_to_weights)
  chunk_weights_to_one_fn = functools.partial(attention_with_head, fn=chunk_weights_to_one)
  chunk_bias_to_one_fn = functools.partial(attention_with_head, fn=chunk_bias_to_one)
  chunk_weights_to_one_fn_without_head = functools.partial(attention_with_head, has_head=False, fn=chunk_weights_to_one)
  chunk_bias_to_one_fn_without_head = functools.partial(attention_with_head, has_head=False, fn=chunk_bias_to_one)

  npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, fn=npz_to_tensor)
  weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, fn=weights_to_weights)
  chunk_weights_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_weights_to_one)
  chunk_weights_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_weights_to_one)
  chunk_bias_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_bias_to_one)

  attn_npz_to_tensor_wrap = functools.partial(layer_stack_to_module_list, fn=npz_to_tensor_fn)
  attn_weights_to_weights_wrap = functools.partial(layer_stack_to_module_list, fn=weights_to_weights_fn)
  attn_chunk_weights_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_weights_to_one_fn)
  attn_chunk_weights_to_one_wrap_without_head = functools.partial(layer_stack_to_module_list, fn=chunk_weights_to_one_fn_without_head)
  attn_chunk_bias_to_one_wrap = functools.partial(layer_stack_to_module_list, fn=chunk_bias_to_one_fn)
  attn_proj_out_wrap = functools.partial(layer_stack_to_module_list, fn=proj_out_weight)

  scope_list = {
     'input_emb.to_single_emb.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/evoformer/preprocess_1d//weights')),
     'input_emb.to_single_emb.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/preprocess_1d//bias')),
     'input_emb.to_msa_emb.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/evoformer/preprocess_msa//weights')),
     'input_emb.to_msa_emb.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/preprocess_msa//bias')),
     'input_emb.to_pairwise_emb.to_pairwise_repr.weight': (chunk_weights_to_one, dict(x=['alphafold/alphafold_iteration/evoformer/left_single//weights', 'alphafold/alphafold_iteration/evoformer/right_single//weights'])),
     'input_emb.to_pairwise_emb.to_pairwise_repr.bias': (chunk_bias_to_one, dict(x=['alphafold/alphafold_iteration/evoformer/left_single//bias', 'alphafold/alphafold_iteration/evoformer/right_single//bias'])),
     'input_emb.to_pairwise_emb.relative_pos_emb.embedding.weight': (linear_to_embedding, dict(weights='alphafold/alphafold_iteration/evoformer/pair_activiations//weights', bias='alphafold/alphafold_iteration/evoformer/pair_activiations//bias')),
     'template_emb.template_pairwise_embedder.to_pair.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/embedding2d//weights')),
     'template_emb.template_pairwise_embedder.to_pair.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/embedding2d//bias')),
     'template_emb.template_pairwise_embedder.to_out_norm.weight': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/output_layer_norm//scale')),
     'template_emb.template_pairwise_embedder.to_out_norm.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/output_layer_norm//offset')),
     'template_emb.template_pointwise_attn.to_q.weight': (weights_to_weights_fn, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/attention//query_w')),
     'template_emb.template_pointwise_attn.to_kv.weight': (chunk_weights_to_one_fn, dict(x=['alphafold/alphafold_iteration/evoformer/template_embedding/attention//key_w', 'alphafold/alphafold_iteration/evoformer/template_embedding/attention//value_w'])),
     'template_emb.template_pointwise_attn.to_out.weight': (proj_out_weight, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/attention//output_w')),
     'template_emb.template_pointwise_attn.to_out.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/attention//output_b')),
     'template_emb.template_single_embedder.0.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/evoformer/template_single_embedding//weights')),
     'template_emb.template_single_embedder.0.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/template_single_embedding//bias')),
     'template_emb.template_single_embedder.2.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/evoformer/template_projection//weights')),
     'template_emb.template_single_embedder.2.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/template_projection//bias')),
     'msa_activations_extra.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_activations//weights')),
     'msa_activations_extra.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_activations//bias')),
     'recycling_pos_linear.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/evoformer/prev_pos_linear//weights')),
     'recycling_pos_linear.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/prev_pos_linear//bias')),
     'recycling_msa_norm.weight': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/prev_msa_first_row_norm//scale')),
     'recycling_msa_norm.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/prev_msa_first_row_norm//offset')),
     'recycling_pairwise_norm.weight': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/prev_pair_norm//scale')),
     'recycling_pairwise_norm.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/prev_pair_norm//offset')),
     'to_single_repr.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/evoformer/single_activations//weights')),
     'to_single_repr.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/evoformer/single_activations//bias')),
     'head_lddt.net.0.weight': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/predicted_lddt_head/input_layer_norm//scale')),
     'head_lddt.net.0.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/predicted_lddt_head/input_layer_norm//offset')),
     'head_lddt.net.1.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/predicted_lddt_head/act_0//weights')),
     'head_lddt.net.1.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/predicted_lddt_head/act_0//bias')),
     'head_lddt.net.3.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/predicted_lddt_head/act_1//weights')),
     'head_lddt.net.3.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/predicted_lddt_head/act_1//bias')),
     'head_lddt.net.5.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/predicted_lddt_head/logits//weights')),
     'head_lddt.net.5.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/predicted_lddt_head/logits//bias')),
     'head_folding.struct_module.single_repr_norm.weight': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/single_layer_norm//scale')),
     'head_folding.struct_module.single_repr_norm.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/single_layer_norm//offset')),
     'head_folding.struct_module.pairwise_repr_norm.weight': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/pair_layer_norm//scale')),
     'head_folding.struct_module.pairwise_repr_norm.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/pair_layer_norm//offset')),
     'head_folding.struct_module.single_repr_dim.0.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/structure_module/initial_projection//weights')),
     'head_folding.struct_module.single_repr_dim.0.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/initial_projection//bias')),
     'head_folding.struct_module.ipa_block.attn.to_scalar_q.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/q_scalar//weights')),
     'head_folding.struct_module.ipa_block.attn.to_scalar_q.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/q_scalar//bias')),
     'head_folding.struct_module.ipa_block.attn.to_scalar_k.weight': (functools.partial(weights_to_chunk_with_heads, heads=12, from_idx=0, to_idx=192), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/kv_scalar//weights')),
     'head_folding.struct_module.ipa_block.attn.to_scalar_k.bias': (functools.partial(npz_to_chunk_with_heads, heads=12, from_idx=0, to_idx=192), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/kv_scalar//bias')),
     'head_folding.struct_module.ipa_block.attn.to_scalar_v.weight': (functools.partial(weights_to_chunk_with_heads, heads=12, from_idx=192, to_idx=384), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/kv_scalar//weights')),
     'head_folding.struct_module.ipa_block.attn.to_scalar_v.bias': (functools.partial(npz_to_chunk_with_heads, heads=12, from_idx=192, to_idx=384), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/kv_scalar//bias')),
     'head_folding.struct_module.ipa_block.attn.to_point_q.weight': (functools.partial(functions_chain, fn_list=[rearrange_point, weights_to_weights]), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/q_point_local//weights')),
     'head_folding.struct_module.ipa_block.attn.to_point_q.bias': (functools.partial(functions_chain, fn_list=[rearrange_point, npz_to_tensor]), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/q_point_local//bias')),
     'head_folding.struct_module.ipa_block.attn.to_point_k.weight': (functools.partial(weights_to_chunk_with_heads, heads=12*3, from_idx=0, to_idx=144, fn=rearrange_point), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/kv_point_local//weights')),
     'head_folding.struct_module.ipa_block.attn.to_point_k.bias': (functools.partial(npz_to_chunk_with_heads, heads=12*3, from_idx=0, to_idx=144, fn=rearrange_point), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/kv_point_local//bias')),
     'head_folding.struct_module.ipa_block.attn.to_point_v.weight': (functools.partial(weights_to_chunk_with_heads, heads=12*3, from_idx=144, to_idx=432, fn=rearrange_point), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/kv_point_local//weights')),
     'head_folding.struct_module.ipa_block.attn.to_point_v.bias': (functools.partial(npz_to_chunk_with_heads, heads=12*3, from_idx=144, to_idx=432, fn=rearrange_point), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/kv_point_local//bias')),
     'head_folding.struct_module.ipa_block.attn.point_weights': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention//trainable_point_weights')),
     'head_folding.struct_module.ipa_block.attn.to_pairwise_attn_bias.0.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/attention_2d//weights')),
     'head_folding.struct_module.ipa_block.attn.to_pairwise_attn_bias.0.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/attention_2d//bias')),
     'head_folding.struct_module.ipa_block.attn.to_out.weight': (functools.partial(functions_chain, fn_list=[weights_to_weights, rearrange_out]), dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/output_projection//weights')),
     'head_folding.struct_module.ipa_block.attn.to_out.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/output_projection//bias')),
     'head_folding.struct_module.ipa_block.attn_norm.weight': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/attention_layer_norm//scale')),
     'head_folding.struct_module.ipa_block.attn_norm.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/attention_layer_norm//offset')),
     'head_folding.struct_module.ipa_block.ff.0.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/transition//weights')),
     'head_folding.struct_module.ipa_block.ff.0.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/transition//bias')),
     'head_folding.struct_module.ipa_block.ff.2.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/transition_1//weights')),
     'head_folding.struct_module.ipa_block.ff.2.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/transition_1//bias')),
     'head_folding.struct_module.ipa_block.ff.4.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/transition_2//weights')),
     'head_folding.struct_module.ipa_block.ff.4.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/transition_2//bias')),
     'head_folding.struct_module.ipa_block.ff_norm.weight': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/transition_layer_norm//scale')),
     'head_folding.struct_module.ipa_block.ff_norm.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/transition_layer_norm//offset')),
     'head_folding.struct_module.to_affine_update.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/affine_update//weights')),
     'head_folding.struct_module.to_affine_update.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/affine_update//bias')),
     'head_folding.struct_module.to_angles.projection.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/rigid_sidechain/input_projection//weights')),
     'head_folding.struct_module.to_angles.projection.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/rigid_sidechain/input_projection//bias')),
     'head_folding.struct_module.to_angles.projection_init.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/rigid_sidechain/input_projection_1//weights')),
     'head_folding.struct_module.to_angles.projection_init.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/rigid_sidechain/input_projection_1//bias')),
     'head_folding.struct_module.to_angles.blocks.0.net.1.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/rigid_sidechain/resblock1//weights')),
     'head_folding.struct_module.to_angles.blocks.0.net.1.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/rigid_sidechain/resblock1//bias')),
     'head_folding.struct_module.to_angles.blocks.0.net.3.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/rigid_sidechain/resblock2//weights')),
     'head_folding.struct_module.to_angles.blocks.0.net.3.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/rigid_sidechain/resblock2//bias')),
     'head_folding.struct_module.to_angles.blocks.1.net.1.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/rigid_sidechain/resblock1_1//weights')),
     'head_folding.struct_module.to_angles.blocks.1.net.1.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/rigid_sidechain/resblock1_1//bias')),
     'head_folding.struct_module.to_angles.blocks.1.net.3.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/rigid_sidechain/resblock2_1//weights')),
     'head_folding.struct_module.to_angles.blocks.1.net.3.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/rigid_sidechain/resblock2_1//bias')),
     'head_folding.struct_module.to_angles.to_groups.weight': (weights_to_weights, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/rigid_sidechain/unnormalized_angles//weights')),
     'head_folding.struct_module.to_angles.to_groups.bias': (npz_to_tensor, dict(x='alphafold/alphafold_iteration/structure_module/fold_iteration/rigid_sidechain/unnormalized_angles//bias')),
  }
  
  template_scope_iteration = {
     'layer.0.triangle_multiply_outgoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/layer_norm_input//scale')),
     'layer.0.triangle_multiply_outgoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/layer_norm_input//offset')),
     'layer.0.triangle_multiply_outgoing.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/left_projection//weights')),
     'layer.0.triangle_multiply_outgoing.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/left_projection//bias')),
     'layer.0.triangle_multiply_outgoing.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/right_projection//weights')),
     'layer.0.triangle_multiply_outgoing.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/right_projection//bias')),
     'layer.0.triangle_multiply_outgoing.left_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/left_gate//weights')),
     'layer.0.triangle_multiply_outgoing.left_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/left_gate//bias')),
     'layer.0.triangle_multiply_outgoing.right_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/right_gate//weights')),
     'layer.0.triangle_multiply_outgoing.right_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/right_gate//bias')),
     'layer.0.triangle_multiply_outgoing.out_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/gating_linear//weights')),
     'layer.0.triangle_multiply_outgoing.out_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/gating_linear//bias')),
     'layer.0.triangle_multiply_outgoing.to_out_norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/center_layer_norm//scale')),
     'layer.0.triangle_multiply_outgoing.to_out_norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/center_layer_norm//offset')),
     'layer.0.triangle_multiply_outgoing.to_out.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/output_projection//weights')),
     'layer.0.triangle_multiply_outgoing.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/output_projection//bias')),
     'layer.0.triangle_multiply_ingoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/layer_norm_input//scale')),
     'layer.0.triangle_multiply_ingoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/layer_norm_input//offset')),
     'layer.0.triangle_multiply_ingoing.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/left_projection//weights')),
     'layer.0.triangle_multiply_ingoing.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/left_projection//bias')),
     'layer.0.triangle_multiply_ingoing.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/right_projection//weights')),
     'layer.0.triangle_multiply_ingoing.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/right_projection//bias')),
     'layer.0.triangle_multiply_ingoing.left_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/left_gate//weights')),
     'layer.0.triangle_multiply_ingoing.left_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/left_gate//bias')),
     'layer.0.triangle_multiply_ingoing.right_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/right_gate//weights')),
     'layer.0.triangle_multiply_ingoing.right_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/right_gate//bias')),
     'layer.0.triangle_multiply_ingoing.out_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/gating_linear//weights')),
     'layer.0.triangle_multiply_ingoing.out_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/gating_linear//bias')),
     'layer.0.triangle_multiply_ingoing.to_out_norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/center_layer_norm//scale')),
     'layer.0.triangle_multiply_ingoing.to_out_norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/center_layer_norm//offset')),
     'layer.0.triangle_multiply_ingoing.to_out.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/output_projection//weights')),
     'layer.0.triangle_multiply_ingoing.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_multiplication_incoming/output_projection//bias')),
     'layer.0.triangle_attention_outgoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/query_norm//scale')),
     'layer.0.triangle_attention_outgoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/query_norm//offset')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/query_norm//scale')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/query_norm//offset')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node//feat_2d_weights')),
     'layer.0.triangle_attention_outgoing.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//query_w')),
     'layer.0.triangle_attention_outgoing.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//key_w', 'alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//value_w'])),
     'layer.0.triangle_attention_outgoing.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//gating_w')),
     'layer.0.triangle_attention_outgoing.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//gating_b')),
     'layer.0.triangle_attention_outgoing.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//output_w')),
     'layer.0.triangle_attention_outgoing.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention//output_b')),
     'layer.0.triangle_attention_ingoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/query_norm//scale')),
     'layer.0.triangle_attention_ingoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/query_norm//offset')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/query_norm//scale')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/query_norm//offset')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node//feat_2d_weights')),
     'layer.0.triangle_attention_ingoing.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//query_w')),
     'layer.0.triangle_attention_ingoing.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//key_w', 'alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//value_w'])),
     'layer.0.triangle_attention_ingoing.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//gating_w')),
     'layer.0.triangle_attention_ingoing.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//gating_b')),
     'layer.0.triangle_attention_ingoing.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//output_w')),
     'layer.0.triangle_attention_ingoing.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/triangle_attention_ending_node/attention//output_b')),
     'layer.1.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/input_layer_norm//scale')),
     'layer.1.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/input_layer_norm//offset')),
     'layer.1.net.0.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/transition1//weights')),
     'layer.1.net.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/transition1//bias')),
     'layer.1.net.3.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/transition2//weights')),
     'layer.1.net.3.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/pair_transition/transition2//bias')),
  }

  extra_evoformer_scope_iteration = {
     'layer.2.row_attn.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/query_norm//scale')),
     'layer.2.row_attn.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/query_norm//offset')),
     'layer.2.row_attn.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/feat_2d_norm//scale')),
     'layer.2.row_attn.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/feat_2d_norm//offset')),
     'layer.2.row_attn.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias//feat_2d_weights')),
     'layer.2.row_attn.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/attention//query_w')),
     'layer.2.row_attn.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/attention//key_w', 'alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/attention//value_w'])),
     'layer.2.row_attn.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/attention//gating_w')),
     'layer.2.row_attn.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/attention//gating_b')),
     'layer.2.row_attn.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/attention//output_w')),
     'layer.2.row_attn.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_row_attention_with_pair_bias/attention//output_b')),
     'layer.2.col_attn.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/query_norm//scale')),
     'layer.2.col_attn.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/query_norm//offset')),
     'layer.2.col_attn.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/attention//query_w')),
     'layer.2.col_attn.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap_without_head, dict(x=['alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/attention//key_w', 'alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/attention//value_w'])),
     'layer.2.col_attn.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/attention//gating_w')),
     'layer.2.col_attn.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/attention//gating_b')),
     'layer.2.col_attn.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/attention//output_w')),
     'layer.2.col_attn.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_column_global_attention/attention//output_b')),
     'layer.3.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_transition/input_layer_norm//scale')),
     'layer.3.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_transition/input_layer_norm//offset')),
     'layer.3.net.0.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_transition/transition1//weights')),
     'layer.3.net.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_transition/transition1//bias')),
     'layer.3.net.3.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_transition/transition2//weights')),
     'layer.3.net.3.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/msa_transition/transition2//bias')),
     'layer.0.outer_mean.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/outer_product_mean/layer_norm_input//scale')),
     'layer.0.outer_mean.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/outer_product_mean/layer_norm_input//offset')),
     'layer.0.outer_mean.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/outer_product_mean/left_projection//weights')),
     'layer.0.outer_mean.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/outer_product_mean/left_projection//bias')),
     'layer.0.outer_mean.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/outer_product_mean/right_projection//weights')),
     'layer.0.outer_mean.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/outer_product_mean/right_projection//bias')),
     'layer.0.outer_mean.proj_out.weight': (proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/outer_product_mean//output_w')),
     'layer.0.outer_mean.proj_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/outer_product_mean//output_b')),
     'layer.0.triangle_multiply_outgoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/layer_norm_input//scale')),
     'layer.0.triangle_multiply_outgoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/layer_norm_input//offset')),
     'layer.0.triangle_multiply_outgoing.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/left_projection//weights')),
     'layer.0.triangle_multiply_outgoing.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/left_projection//bias')),
     'layer.0.triangle_multiply_outgoing.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/right_projection//weights')),
     'layer.0.triangle_multiply_outgoing.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/right_projection//bias')),
     'layer.0.triangle_multiply_outgoing.left_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/left_gate//weights')),
     'layer.0.triangle_multiply_outgoing.left_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/left_gate//bias')),
     'layer.0.triangle_multiply_outgoing.right_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/right_gate//weights')),
     'layer.0.triangle_multiply_outgoing.right_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/right_gate//bias')),
     'layer.0.triangle_multiply_outgoing.out_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/gating_linear//weights')),
     'layer.0.triangle_multiply_outgoing.out_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/gating_linear//bias')),
     'layer.0.triangle_multiply_outgoing.to_out_norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/center_layer_norm//scale')),
     'layer.0.triangle_multiply_outgoing.to_out_norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/center_layer_norm//offset')),
     'layer.0.triangle_multiply_outgoing.to_out.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/output_projection//weights')),
     'layer.0.triangle_multiply_outgoing.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_outgoing/output_projection//bias')),
     'layer.0.triangle_multiply_ingoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/layer_norm_input//scale')),
     'layer.0.triangle_multiply_ingoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/layer_norm_input//offset')),
     'layer.0.triangle_multiply_ingoing.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/left_projection//weights')),
     'layer.0.triangle_multiply_ingoing.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/left_projection//bias')),
     'layer.0.triangle_multiply_ingoing.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/right_projection//weights')),
     'layer.0.triangle_multiply_ingoing.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/right_projection//bias')),
     'layer.0.triangle_multiply_ingoing.left_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/left_gate//weights')),
     'layer.0.triangle_multiply_ingoing.left_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/left_gate//bias')),
     'layer.0.triangle_multiply_ingoing.right_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/right_gate//weights')),
     'layer.0.triangle_multiply_ingoing.right_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/right_gate//bias')),
     'layer.0.triangle_multiply_ingoing.out_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/gating_linear//weights')),
     'layer.0.triangle_multiply_ingoing.out_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/gating_linear//bias')),
     'layer.0.triangle_multiply_ingoing.to_out_norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/center_layer_norm//scale')),
     'layer.0.triangle_multiply_ingoing.to_out_norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/center_layer_norm//offset')),
     'layer.0.triangle_multiply_ingoing.to_out.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/output_projection//weights')),
     'layer.0.triangle_multiply_ingoing.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_multiplication_incoming/output_projection//bias')),
     'layer.0.triangle_attention_outgoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/query_norm//scale')),
     'layer.0.triangle_attention_outgoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/query_norm//offset')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/query_norm//scale')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/query_norm//offset')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node//feat_2d_weights')),
     'layer.0.triangle_attention_outgoing.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/attention//query_w')),
     'layer.0.triangle_attention_outgoing.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/attention//key_w', 'alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/attention//value_w'])),
     'layer.0.triangle_attention_outgoing.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/attention//gating_w')),
     'layer.0.triangle_attention_outgoing.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/attention//gating_b')),
     'layer.0.triangle_attention_outgoing.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/attention//output_w')),
     'layer.0.triangle_attention_outgoing.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_starting_node/attention//output_b')),
     'layer.0.triangle_attention_ingoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/query_norm//scale')),
     'layer.0.triangle_attention_ingoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/query_norm//offset')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/query_norm//scale')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/query_norm//offset')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node//feat_2d_weights')),
     'layer.0.triangle_attention_ingoing.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/attention//query_w')),
     'layer.0.triangle_attention_ingoing.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/attention//key_w', 'alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/attention//value_w'])),
     'layer.0.triangle_attention_ingoing.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/attention//gating_w')),
     'layer.0.triangle_attention_ingoing.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/attention//gating_b')),
     'layer.0.triangle_attention_ingoing.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/attention//output_w')),
     'layer.0.triangle_attention_ingoing.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/triangle_attention_ending_node/attention//output_b')),
     'layer.1.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/pair_transition/input_layer_norm//scale')),
     'layer.1.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/pair_transition/input_layer_norm//offset')),
     'layer.1.net.0.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/pair_transition/transition1//weights')),
     'layer.1.net.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/pair_transition/transition1//bias')),
     'layer.1.net.3.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/pair_transition/transition2//weights')),
     'layer.1.net.3.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/extra_msa_stack/pair_transition/transition2//bias')),
  }
  evoformer_scope_iteration = {
     'layer.2.row_attn.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/query_norm//scale')),
     'layer.2.row_attn.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/query_norm//offset')),
     'layer.2.row_attn.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/feat_2d_norm//scale')),
     'layer.2.row_attn.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/feat_2d_norm//offset')),
     'layer.2.row_attn.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias//feat_2d_weights')),
     'layer.2.row_attn.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//query_w')),
     'layer.2.row_attn.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//key_w', 'alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//value_w'])),
     'layer.2.row_attn.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//gating_w')),
     'layer.2.row_attn.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//gating_b')),
     'layer.2.row_attn.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//output_w')),
     'layer.2.row_attn.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias/attention//output_b')),
     'layer.2.col_attn.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/query_norm//scale')),
     'layer.2.col_attn.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/query_norm//offset')),
     'layer.2.col_attn.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//query_w')),
     'layer.2.col_attn.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//key_w', 'alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//value_w'])),
     'layer.2.col_attn.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//gating_w')),
     'layer.2.col_attn.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//gating_b')),
     'layer.2.col_attn.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//output_w')),
     'layer.2.col_attn.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_column_attention/attention//output_b')),
     'layer.3.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_transition/input_layer_norm//scale')),
     'layer.3.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_transition/input_layer_norm//offset')),
     'layer.3.net.0.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_transition/transition1//weights')),
     'layer.3.net.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_transition/transition1//bias')),
     'layer.3.net.3.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_transition/transition2//weights')),
     'layer.3.net.3.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_transition/transition2//bias')),
     'layer.0.outer_mean.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean/layer_norm_input//scale')),
     'layer.0.outer_mean.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean/layer_norm_input//offset')),
     'layer.0.outer_mean.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean/left_projection//weights')),
     'layer.0.outer_mean.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean/left_projection//bias')),
     'layer.0.outer_mean.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean/right_projection//weights')),
     'layer.0.outer_mean.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean/right_projection//bias')),
     'layer.0.outer_mean.proj_out.weight': (proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean//output_w')),
     'layer.0.outer_mean.proj_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean//output_b')),
     'layer.0.triangle_multiply_outgoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/layer_norm_input//scale')),
     'layer.0.triangle_multiply_outgoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/layer_norm_input//offset')),
     'layer.0.triangle_multiply_outgoing.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/left_projection//weights')),
     'layer.0.triangle_multiply_outgoing.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/left_projection//bias')),
     'layer.0.triangle_multiply_outgoing.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/right_projection//weights')),
     'layer.0.triangle_multiply_outgoing.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/right_projection//bias')),
     'layer.0.triangle_multiply_outgoing.left_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/left_gate//weights')),
     'layer.0.triangle_multiply_outgoing.left_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/left_gate//bias')),
     'layer.0.triangle_multiply_outgoing.right_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/right_gate//weights')),
     'layer.0.triangle_multiply_outgoing.right_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/right_gate//bias')),
     'layer.0.triangle_multiply_outgoing.out_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/gating_linear//weights')),
     'layer.0.triangle_multiply_outgoing.out_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/gating_linear//bias')),
     'layer.0.triangle_multiply_outgoing.to_out_norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/center_layer_norm//scale')),
     'layer.0.triangle_multiply_outgoing.to_out_norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/center_layer_norm//offset')),
     'layer.0.triangle_multiply_outgoing.to_out.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/output_projection//weights')),
     'layer.0.triangle_multiply_outgoing.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/output_projection//bias')),
     'layer.0.triangle_multiply_ingoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/layer_norm_input//scale')),
     'layer.0.triangle_multiply_ingoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/layer_norm_input//offset')),
     'layer.0.triangle_multiply_ingoing.left_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/left_projection//weights')),
     'layer.0.triangle_multiply_ingoing.left_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/left_projection//bias')),
     'layer.0.triangle_multiply_ingoing.right_proj.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/right_projection//weights')),
     'layer.0.triangle_multiply_ingoing.right_proj.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/right_projection//bias')),
     'layer.0.triangle_multiply_ingoing.left_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/left_gate//weights')),
     'layer.0.triangle_multiply_ingoing.left_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/left_gate//bias')),
     'layer.0.triangle_multiply_ingoing.right_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/right_gate//weights')),
     'layer.0.triangle_multiply_ingoing.right_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/right_gate//bias')),
     'layer.0.triangle_multiply_ingoing.out_gate.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/gating_linear//weights')),
     'layer.0.triangle_multiply_ingoing.out_gate.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/gating_linear//bias')),
     'layer.0.triangle_multiply_ingoing.to_out_norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/center_layer_norm//scale')),
     'layer.0.triangle_multiply_ingoing.to_out_norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/center_layer_norm//offset')),
     'layer.0.triangle_multiply_ingoing.to_out.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/output_projection//weights')),
     'layer.0.triangle_multiply_ingoing.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_incoming/output_projection//bias')),
     'layer.0.triangle_attention_outgoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/query_norm//scale')),
     'layer.0.triangle_attention_outgoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/query_norm//offset')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/query_norm//scale')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/query_norm//offset')),
     'layer.0.triangle_attention_outgoing.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node//feat_2d_weights')),
     'layer.0.triangle_attention_outgoing.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//query_w')),
     'layer.0.triangle_attention_outgoing.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//key_w', 'alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//value_w'])),
     'layer.0.triangle_attention_outgoing.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//gating_w')),
     'layer.0.triangle_attention_outgoing.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//gating_b')),
     'layer.0.triangle_attention_outgoing.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//output_w')),
     'layer.0.triangle_attention_outgoing.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_starting_node/attention//output_b')),
     'layer.0.triangle_attention_ingoing.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/query_norm//scale')),
     'layer.0.triangle_attention_ingoing.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/query_norm//offset')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.0.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/query_norm//scale')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/query_norm//offset')),
     'layer.0.triangle_attention_ingoing.edges_to_attn_bias.1.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node//feat_2d_weights')),
     'layer.0.triangle_attention_ingoing.attn.to_q.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//query_w')),
     'layer.0.triangle_attention_ingoing.attn.to_kv.weight': (attn_chunk_weights_to_one_wrap, dict(x=['alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//key_w', 'alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//value_w'])),
     'layer.0.triangle_attention_ingoing.attn.gating.weight': (attn_weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//gating_w')),
     'layer.0.triangle_attention_ingoing.attn.gating.bias': (attn_npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//gating_b')),
     'layer.0.triangle_attention_ingoing.attn.to_out.weight': (attn_proj_out_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//output_w')),
     'layer.0.triangle_attention_ingoing.attn.to_out.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_attention_ending_node/attention//output_b')),
     'layer.1.norm.weight': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/pair_transition/input_layer_norm//scale')),
     'layer.1.norm.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/pair_transition/input_layer_norm//offset')),
     'layer.1.net.0.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/pair_transition/transition1//weights')),
     'layer.1.net.0.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/pair_transition/transition1//bias')),
     'layer.1.net.3.weight': (weights_to_weights_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/pair_transition/transition2//weights')),
     'layer.1.net.3.bias': (npz_to_tensor_wrap, dict(x='alphafold/alphafold_iteration/evoformer/evoformer_iteration/pair_transition/transition2//bias')),
  }

  for name, layers, scope_iteration in (('template_emb.template_pairwise_embedder.pair_stack', 2, template_scope_iteration), ('evoformer_extra', 4, extra_evoformer_scope_iteration), ('evoformer', 48, evoformer_scope_iteration)):
    for i in range(layers):
      for k, (fn, kwargs) in scope_iteration.items():
        scope_list[f'{name}.layers.{i}.{k}'] = (functools.partial(fn, i=i), kwargs)

  with open(os.path.join(args.data_dir, 'params', f'params_{args.model_name}.npz'), 'rb') as f:
    params = np.load(io.BytesIO(f.read()), allow_pickle=False)
  state_dict = state_dict_get(params, scope_list)

  headers = [
          ('folding', {'structure_module_depth': 8,
              'structure_module_heads': 12,
              'qkv_use_bias': True,
              'position_scale': 10.0,
              'fape_max': 10.0,
              'fape_z': 10.0}, {'weight', 0.4}),
          ('lddt', {'num_channels': 128}, {'weights': 0.01}),
  ]

  model_dim=(384, 128)
  model_evoformer_depth=48
  model_evoformer_head_num=(8, 4)
  model_evoformer_head_dim=(32, 32)
  model_recycling_single_repr=False
  model_recycling_pos=True
  model_num_tokens=22
  model_num_msa_tokens=49
  model_template_depth=2
  evoformer_stack = alphafold2.Alphafold2(dim=model_dim,
                               evoformer_depth=model_evoformer_depth,
                               evoformer_head_num=model_evoformer_head_num,
                               evoformer_head_dim=model_evoformer_head_dim,
                               template_depth=model_template_depth,
                               num_tokens=model_num_tokens,
                               num_msa_tokens=model_num_msa_tokens,
                               attn_dropout=(0.15, 0.25),
                               recycling_single_repr=model_recycling_single_repr,
                               recycling_pos=model_recycling_pos,
                               headers=headers)
  print(evoformer_stack)
  evoformer_stack.load_state_dict(state_dict)
  if args.output:
    with open(os.path.join(args.output, f'evoformer.npz'), 'rb') as f:
      data = np.load(io.BytesIO(f.read()), allow_pickle=True)

    batch, result = data['batch'], data['result']
    batch = batch.tolist()
    print(batch.keys())
    batch['template_aatype_index'] = np.clip(batch['template_aatype'], 0, 20)
    batch['template_all_atom_mask'] = batch['template_all_atom_masks']
    batch = make_atom14_positions(batch, along_axis=2, prefix='template_')
    print(batch.keys())

    template_angles = functional.angles_from_positions(
        torch.from_numpy(batch['template_aatype_index']),
        torch.from_numpy(batch['template_atom14_gt_positions']),
        torch.from_numpy(batch['template_atom14_gt_exists']),
        placeholder_for_undefined=False)
    for k, v in template_angles.items():
      batch[f'template_{k}'] = v.numpy()
    print(batch.keys())

    kv_trans = [
       ('aatype', 'seq', 'i -> () i'),
       ('seq_mask', 'mask', 'i -> () i'),
       ('residue_index', 'seq_index', 'i -> () i'),
       ('target_feat', 'target_feat', 'i d -> () i d'),
       ('msa_feat', 'msa_feat', 'm i d -> () m i d'),
       ('msa_mask', 'msa_mask', 'm i -> () m i'),
       ('template_aatype', 'template_seq', '... -> () ...'),
       ('template_mask', 'template_mask', '... -> () ...'),
       ('template_pseudo_beta_mask', 'template_pseudo_beta_mask', '... -> () ...'),
       ('template_pseudo_beta', 'template_pseudo_beta', '... -> () ...'),
       ('template_atom14_gt_positions', 'template_coord', '... -> () ...'),
       ('template_atom14_gt_exists', 'template_coord_mask', '... -> () ...'),
       ('template_torsion_angles', 'template_torsion_angles', '... -> () ...'),
       ('template_torsion_angles_alt', 'template_torsion_angles_alt', '... -> () ...'),
       ('template_torsion_angles_mask', 'template_torsion_angles_mask', '... -> () ...'),
       ('extra_msa', 'extra_msa', '... -> () ...'),
       ('extra_msa_mask', 'extra_msa_mask', '... -> () ...'),
       ('extra_has_deletion', 'extra_has_deletion', '... -> () ...'),
       ('extra_deletion_value', 'extra_deletion_value', '... -> () ...'),
       ('prev_pos', 'prev_pos', '... -> () ...'),
       ('prev_msa_first_row', 'prev_msa_first_row', '... -> () ...'),
       ('prev_pair', 'prev_pair', '... -> () ...'),
    ]
    batch_new = {}
    for from_k, to_k, reshape in kv_trans:
      if reshape:
        batch_new[to_k] = rearrange(torch.from_numpy(np.array(batch[from_k])), reshape)
      else:
        batch_new[to_k] = torch.from_numpy(batch[from_k])
      print(to_k, batch_new[to_k].shape)

    recyclables = alphafold2.Recyclables(batch_new.pop('prev_msa_first_row'),
                                         batch_new.pop('prev_pair'),
                                         batch_new.pop('prev_pos'))
    batch_new['recyclables'] = recyclables
    evoformer_stack.eval()
    with torch.no_grad():
      r = evoformer_stack(batch_new, shard_size=4)
    print('result1', r)
    print('result2', result)
    torch.save(dict(dim=model_dim,
                    model_evoformer_depth=model_evoformer_depth,
                    model_evoformer_head_num=model_evoformer_head_num,
                    model_evoformer_head_dim=model_evoformer_head_dim,
                    model_template_depth=model_template_depth,
                    model_num_tokens=model_num_tokens,
                    model_num_msa_tokens=model_num_msa_tokens,
                    model_recycling_single_repr=model_recycling_single_repr,
                    model_recycling_pos=model_recycling_pos,
                    headers=headers,
                    model=evoformer_stack.state_dict()),
        os.path.join(args.output, f'args.model_name.pth'))

if __name__ == '__main__':
  import argparse

  commands = {
    'input_embedder': input_embedder,
    'make_features': make_features,
    'distogram_from_positions': distogram_from_positions,
    'angles_from_positions': angles_from_positions,
    'template_pair_stack': template_pair_stack,
    'single_template_embedding': single_template_embedding,
    'template_embedding': template_embedding,
    'template_embedding_with_angles': template_embedding_with_angles,
    'msa_row_attention_with_pair_bias': msa_row_attention_with_pair_bias,
    'msa_column_attention': msa_column_attention,
    'msa_column_global_attention': msa_column_global_attention,
    'msa_transition': msa_transition,
    'outer_product_mean': outer_product_mean,
    'triangle_multiplication_outgoing': triangle_multiplication_outgoing,
    'triangle_multiplication_incoming': triangle_multiplication_incoming,
    'triangle_attention_starting_node': triangle_attention_starting_node,
    'triangle_attention_ending_node': triangle_attention_ending_node,
    'pair_transition': pair_transition,
    'evoformer_iteration': evoformer_iteration,
    'extra_evoformer_stack': extra_evoformer_stack,
    'trunk_evoformer': trunk_evoformer_stack,
    'evoformer': evoformer_stack,
    'invariant_point_attention': invariant_point_attention,
    'structure_module': structure_module,
    'plddt': plddt,
    'alphafold_iteration': alphafold_iteration,
    'save_model': save_model,
  }

  parser = argparse.ArgumentParser()
  sub_parsers = parser.add_subparsers(dest='command', required=True)
  for cmd in commands:
    cmd_parser = sub_parsers.add_parser(cmd)
    cmd_parser.add_argument('-d', '--data_dir', type=str, default='.',
                            help='data dir, default=\'.\'')
    cmd_parser.add_argument('-m', '--model_name', type=str, default='model_1',
                            help='model name, default=\'model_1\'')
    cmd_parser.add_argument('-o', '--output', type=str, default=None,
                            help='output dir, default=\'.\'')
  args = parser.parse_args()

  commands[args.command](args)
