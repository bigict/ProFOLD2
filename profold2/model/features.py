"""feature functions
  """
import functools
from inspect import isfunction
import math

import numpy as np
import torch
from torch.nn import functional as F
from einops import rearrange

from profold2.common import residue_constants
from profold2.data.esm import ESMEmbeddingExtractor
from profold2.model import functional
from profold2.utils import default, exists

_feats_fn = {}


def take1st(fn):
  """Supply all arguments but the first."""
  @functools.wraps(fn)
  def fc(*args, **kwargs):
    return lambda x: fn(x, *args, **kwargs)

  _feats_fn[fn.__name__] = fc

  return fc


@take1st
def make_seq_mask(protein, padd_id=20, is_training=True):
  del is_training

  mask = protein['seq'] != padd_id
  protein['mask'] = mask.bool()
  return protein


@functools.cache
def _restype_atom14_mask(includes=None, excludes=None):
  if exists(includes) or exists(excludes):
    restype_atom14_mask = np.copy(residue_constants.restype_atom14_mask)
    if exists(includes):
      for i, restype in enumerate(residue_constants.restypes):
        mol_type = residue_constants.moltype(i)
        resname = residue_constants.restype_1to3[(restype, mol_type)]
        atom_list = residue_constants.restype_name_to_atom14_names[resname]
        for j in range(restype_atom14_mask.shape[1]):
          if restype_atom14_mask[i, j] > 0 and (atom_list[j], mol_type) not in includes:
            restype_atom14_mask[i, j] = 0
    if exists(excludes):
      for i, restype in enumerate(residue_constants.restypes):
        mol_type = residue_constants.moltype(i)
        resname = residue_constants.restype_1to3[(restype, mol_type)]
        atom_list = residue_constants.restype_name_to_atom14_names[resname]
        for j in range(restype_atom14_mask.shape[1]):
          if restype_atom14_mask[i, j] > 0 and (atom_list[j], mol_type) in excludes:
            restype_atom14_mask[i, j] = 0
    return restype_atom14_mask
  return residue_constants.restype_atom14_mask


@take1st
def make_coord_mask(protein, includes=None, excludes=None, is_training=True):
  del is_training

  # FIX: hashable
  if exists(includes):
    includes = frozenset(map(tuple, includes))
  if exists(excludes):
    excludes = frozenset(map(tuple, excludes))

  coord_exists = functional.batched_gather(
      _restype_atom14_mask(includes=includes, excludes=excludes), protein['seq']
  )
  protein['coord_exists'] = coord_exists
  if 'coord_mask' in protein:
    protein['coord_mask'] *= coord_exists
  return protein


@take1st
def make_coord_plddt(
    protein, threshold=0, gamma=None, use_weighted_mask=True, is_training=True
):
  if is_training and 'coord_plddt' in protein:
    ca_idx = residue_constants.atom_order['CA']
    plddt_mean = functional.masked_mean(
        value=protein['coord_plddt'][..., ca_idx], mask=protein['mask'], dim=-1
    )
    protein['plddt_mean'] = plddt_mean
    plddt_mask = (protein['coord_plddt'] >= threshold)
    if exists(gamma):
      protein['coord_plddt'] = torch.exp(gamma * (protein['coord_plddt'] - 1.0))
    protein['coord_plddt'] *= plddt_mask
    protein['coord_plddt_use_weighted_mask'] = use_weighted_mask
  return protein


@take1st
def make_loss_weight(protein, distogram_w=.5, folding_w=.0, is_training=True):
  assert distogram_w <= 1.0 and folding_w <= 1.0
  if is_training and 'msa_idx' in protein:
    mask = (protein['msa_idx'] == 0)
    if 'var_idx' in protein:
      mask = mask & (protein['var_idx'] == 0)
    protein['loss.distogram.w'] = mask * (1.0 - distogram_w) + distogram_w
    protein['loss.folding.w'] = mask * (1.0 - folding_w) + folding_w
  return protein


@take1st
def make_coord_alt(protein, is_training=True):
  if is_training:
    protein.update(
        functional.symmetric_ground_truth_create_alt(
            protein['seq'], protein.get('coord'), protein.get('coord_mask')
        )
    )
  return protein


@take1st
def make_msa_mask(protein):
  return protein


@take1st
def make_seq_profile(protein, mask=None, density=False, epsilon=1e-8, is_training=True):
  assert not is_training or 'msa' in protein
  # Shape (b, m, l, c)
  if 'msa' in protein:
    msa = protein['msa']
    p = F.one_hot(
        msa.long(), num_classes=len(residue_constants.restypes_with_x_and_gap)
    )
    # num_msa = p.shape[1]
    # Shape (b, l, c)
    if 'msa_mask' in protein:
      p = torch.einsum(
          '... m i c,... m i -> ... i c', p.float(), protein['msa_mask'].float()
      )
    else:
      p = torch.sum(p.float(), dim=-3)
  else:
    # num_msa = 1
    p = F.one_hot(
        protein['seq'].long(),
        num_classes=len(residue_constants.restypes_with_x_and_gap)
    )

  # # gap value (b, l)
  # gap_idx = residue_constants.restypes_with_x_and_gap.index('-')
  # protein['sequence_profile_gap_value'] = p[..., gap_idx] / (num_msa + epsilon)

  if exists(mask) and len(mask) > 0:
    # Shape (k, c)
    m = functional.make_mask(
        residue_constants.restypes_with_x_and_gap, mask, device=p.device
    )
    protein['sequence_profile_mask'] = m
    # Shape (c)
    p = p * rearrange(m, 'c -> () () c')
  # Shape (b, l, c)
  if density:
    p = p / (torch.sum(p, dim=-1, keepdim=True) + epsilon)
  protein['sequence_profile'] = p
  return protein


@take1st
def make_bert_mask(
    protein,
    fraction=None,
    span_mean=None,
    span_min=None,
    span_max=None,
    is_training=True
):
  if is_training and (exists(fraction) or exists(span_mean)):
    masked_shape = protein['seq'].shape
    mask = protein['mask']

    if exists(fraction):  # vanilla BERT masking
      masked_position = torch.rand(masked_shape, device=mask.device) < fraction
    elif exists(span_mean):  # SpanBERT-like masking
      b, n = masked_shape[:2]
      span_num = torch.poisson(torch.full((b, ), span_mean, device=mask.device))
      if exists(span_min) or exists(span_max):
        span_num = torch.clamp(span_num, min=span_min, max=span_max)
      span_num = torch.clamp(span_num, max=n - 1)
      span_i = torch.rand((b, ), device=mask.device) * (n - span_num)
      span_j = span_i + span_num
      masked_position = torch.zeros(masked_shape, device=mask.device)
      for k in range(b):
        i, j = int(span_i[k]), int(span_j[k])
        masked_position[k, i:j] = 1

    protein['bert_mask'] = masked_position * mask
    protein['true_seq'] = torch.clone(protein['seq'])
    protein['seq'] = protein['seq'].masked_fill(
        masked_position.bool(), residue_constants.unk_restype_index
    )
  return protein


def pseudo_beta_alphafold(aatype, all_atom_positions, all_atom_masks):
  """Create pseudo beta features (from AlphaFold)."""
  return functional.pseudo_beta_fn(
      aatype, all_atom_positions, all_atom_masks=all_atom_masks
  )


def pseudo_beta_rosetta(aatype, all_atom_positions, all_atom_masks):
  """Create pseudo beta features (from RoseTTAFold)."""

  n_idx = residue_constants.atom_order['N']
  ca_idx = residue_constants.atom_order['CA']
  c_idx = residue_constants.atom_order['C']

  b = all_atom_positions[..., ca_idx, :] - all_atom_positions[..., n_idx, :]
  c = all_atom_positions[..., c_idx, :] - all_atom_positions[..., ca_idx, :]
  a = torch.cross(b, c, dim=-1)
  pseudo_beta = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + all_atom_positions[
      ..., ca_idx, :]

  if exists(all_atom_masks):
    pseudo_beta_mask = all_atom_masks[..., n_idx] * all_atom_masks[
        ..., ca_idx] * all_atom_masks[..., c_idx]
    pseudo_beta_mask = pseudo_beta_mask.float()
    return pseudo_beta, pseudo_beta_mask

  return pseudo_beta


@take1st
def make_pseudo_beta(protein, prefix='', type='alphafold', is_training=True):
  del is_training
  assert type in ('alphafold', 'rosettafold')
  pseudo_beta_fn = pseudo_beta_alphafold if type == 'alphafold' else pseudo_beta_rosetta

  if (
      prefix + 'seq' in protein and prefix + 'coord' in protein and
      prefix + 'coord_mask' in protein
  ):
    protein[prefix + 'pseudo_beta'], protein[prefix + 'pseudo_beta_mask'] = (
        pseudo_beta_fn(
            protein[prefix + 'seq'], protein[prefix + 'coord'],
            protein[prefix + 'coord_mask']
        )
    )
  return protein


@take1st
def make_backbone_affine(protein, is_training=True):
  #assert (not is_training) or ('coord' in protein and 'coord_mask' in protein)
  if is_training and ('coord' in protein and 'coord_mask' in protein):
    n_idx = residue_constants.atom_order['N']
    ca_idx = residue_constants.atom_order['CA']
    c_idx = residue_constants.atom_order['C']

    assert protein['coord'].shape[-2] > min(n_idx, ca_idx, c_idx)
    protein['backbone_affine'] = functional.rigids_from_3x3(
        protein['coord'], indices=(c_idx, ca_idx, n_idx)
    )
    coord_mask = protein['coord_mask']
    coord_mask = torch.stack(
        (coord_mask[..., c_idx], coord_mask[..., ca_idx], coord_mask[..., n_idx]),
        dim=-1
    )
    protein['backbone_affine_mask'] = torch.all(coord_mask != 0, dim=-1)
  return protein


@take1st
def make_affine(protein, is_training=True):
  if is_training:
    feats = functional.rigids_from_positions(
        protein['seq'], protein.get('coord'), protein.get('coord_mask')
    )
    protein.update(feats)
  return protein


@take1st
def make_torsion_angles(protein, is_training=True):
  if is_training and ('coord' in protein and 'coord_mask' in protein):
    feats = functional.angles_from_positions(
        protein['seq'], protein['coord'], protein['coord_mask']
    )
    protein.update(feats)
  return protein


@take1st
def make_esm_embedd(
    protein,
    model,
    repr_layer,
    max_seq_len=None,
    device=None,
    field='embedds',
    is_training=True
):
  del is_training

  esm_extractor = ESMEmbeddingExtractor.get(*model, device=device)
  data_in = list(
      zip(protein['pid'], map(lambda x: x[:max_seq_len], protein['str_seq']))
  )
  data_out = esm_extractor.extract(data_in, repr_layer=repr_layer, device=device)

  if len(data_out.shape) == 3:
    data_out = rearrange(data_out, '... l c -> ... () l c')
  assert len(data_out.shape) == 4
  protein[field] = data_out

  return protein


@take1st
def make_to_device(protein, fields, device, is_training=True):
  del is_training

  if isfunction(device):
    device = device()

  def _to_device(tensor):
    if isinstance(tensor, torch.Tensor):
      return tensor.to(device)
    if isinstance(tensor, list):
      return [_to_device(t) for t in tensor]
    elif isinstance(tensor, dict):
      return {k: _to_device(v) for k, v in tensor.items()}
    return tensor

  for k in fields:
    if k in protein:
      protein[k] = _to_device(protein[k])
  return protein


@take1st
def make_delete_fields(protein, fields, is_training=True):
  del is_training

  for k in fields:
    if k in protein:
      del protein[k]
  return protein


@take1st
def make_selection(protein, fields, is_training=True):
  del is_training

  return {k: protein[k] for k in fields}

@take1st
def make_target_feat(protein, is_training=True):
  del is_training
  # Whether there is a domain break. Always zero for chains, but keeping
  # for compatibility with domain datasets.
  seq = protein['seq']
  has_break = torch.zeros_like(seq)

  target_feat = [
      has_break[..., None].float(),
      F.one_hot(seq.long(),
                num_classes=len(residue_constants.restypes_with_x)).float(),
  ]
  protein['target_feat'] = torch.cat(target_feat, dim=-1)

  return protein

@take1st
def make_msa_feat(protein, is_training=True):
  del is_training
  has_deletion = torch.clamp(protein['deletion_matrix'], min=0, max=1.0)
  deletion_value = torch.atan(protein['deletion_matrix'] / 3.0) * (2. / math.pi)

  msa_feat = [
      F.one_hot(protein['msa'].long(), num_classes=23),
      has_deletion[..., None],
      deletion_value[..., None],
  ]
  if 'cluster_profile' in protein:
    deletion_mean_value = torch.atan(
        protein['cluster_deletion_mean'] / 3.0) * (2. / math.pi)
    msa_feat += [protein['cluster_profile'], deletion_mean_value[..., None]]

  if 'extra_deletion_matrix' in protein:
    protein['extra_has_deletion'] = torch.clamp(
        protein['extra_deletion_matrix'], min=0, max=1.0)
    protein['extra_deletion_value'] = torch.atan(
        protein['extra_deletion_matrix'] / 3.0) * (2. / math.pi)

  protein['msa_feat'] = torch.cat(msa_feat, dim=-1)
  return protein

@take1st
def make_nn_clusters(protein, gap_agreement_weight=0., is_training=True):
  """Assign each extra MSA sequence to its nearest neighbor in sampled MSA."""

  del is_training

  # Determine how much weight we assign to each agreement.  In theory, we could
  # use a full blosum matrix here, but right now let's just down-weight gap
  # agreement because it could be spurious.
  # Never put weight on agreeing on BERT mask
  weights = torch.cat([
      torch.ones((21,)),
      gap_agreement_weight * torch.ones((1,)),
      torch.zeros((1,))], dim=0)
  weights = weights.to(protein['msa_mask'].device)

  # Make agreement score as weighted Hamming distance
  sample_one_hot = (protein['msa_mask'][..., None] *
                    F.one_hot(protein['msa'].long(), num_classes=23))
  extra_one_hot = (protein['extra_msa_mask'][..., None] *
                   F.one_hot(protein['extra_msa'].long(), num_classes=23))

  # Compute tf.einsum('mrc,nrc,c->mn', sample_one_hot, extra_one_hot, weights)
  # in an optimized fashion to avoid possible memory or computation blowup.
  agreement = torch.matmul(
      rearrange(extra_one_hot.float(), '... m r c -> ... m (r c)'),
      rearrange(sample_one_hot.float() * weights, '... n r c -> ... (r c) n'))

  # Assign each sequence in the extra sequences to the closest MSA sample
  protein['extra_cluster_assignment'] = torch.argmax(agreement, dim=-1)

  return protein

@take1st
def make_summerized_clusters(protein, is_training=True):
  """Produce profile and deletion_matrix_mean within each cluster."""

  del is_training

  num_seq = protein['msa'].shape[-2]
  def csum(x):
    return torch.einsum(
        'b m n, b m ... -> b n ...',
        F.one_hot(protein['extra_cluster_assignment'],
                  num_classes=num_seq).float(),
        x.float())

  mask = protein['extra_msa_mask']
  mask_counts = 1e-6 + protein['msa_mask'] + csum(mask)  # Include center

  msa_sum = csum(mask[..., None] *
                 F.one_hot(protein['extra_msa'].long(), num_classes=23))
  msa_sum += F.one_hot(protein['msa'].long(),
                       num_classes=23)  # Original sequence
  protein['cluster_profile'] = msa_sum / mask_counts[..., None]

  del msa_sum

  del_sum = csum(mask * protein['extra_deletion_matrix'])
  del_sum += protein['deletion_matrix']  # Original sequence
  protein['cluster_deletion_mean'] = del_sum / mask_counts
  del del_sum

  return protein

@take1st
def sample_msa(protein, max_depth, keep_extra=True, is_training=True):
  """Sample MSA randomly, remaining sequences are stored as `extra_*`.
  """
  del is_training
  msa_depth, device = protein['msa'].shape[1], protein['msa'].device
  index_order = torch.full((1,), 0, device=device)
  if msa_depth > 1:
    index_order = torch.cat(
        (index_order, torch.randperm(msa_depth - 1, device=device) + 1), dim=-1)
  max_msa_depth = min(msa_depth, max_depth)
  sel_msa, not_sel_msa = index_order[:max_msa_depth], index_order[
      max_msa_depth:]
  for k in ('msa', 'deletion_matrix', 'msa_mask'):
    if keep_extra:
      protein[f'extra_{k}'] = torch.index_select(protein[k], 1, not_sel_msa)
    protein[k] = torch.index_select(protein[k], 1, sel_msa)

  return protein

@take1st
def crop_extra_msa(protein, max_depth=None, is_training=True):
  del is_training
  if exists(max_depth):
    msa_depth, device = protein['extra_msa'].shape[1], protein[
        'extra_msa'].device
    index_order = torch.full((1,), 0, device=device)
    if msa_depth > 1:
      index_order = torch.cat(
          (index_order, torch.randperm(msa_depth - 1, device=device) + 1),
          dim=-1)
    max_msa_depth = min(msa_depth, max_depth)
    sel_idx = index_order[:max_msa_depth]
    for k in ('msa', 'deletion_matrix', 'msa_mask'):
      if f'extra_{k}' in protein:
        protein[f'extra_{k}'] = torch.index_select(protein[f'extra_{k}'], 1,
                                                   sel_idx)

  return protein

class FeatureBuilder:
  """Build features by feature functions in config
    """
  def __init__(self, config):
    self.config = config

  def to(self, device):
    if exists(self.config):
      for i, (fn, kwargs) in enumerate(self.config):
        if 'device' in kwargs:
          kwargs['device'] = device
          self.config[i] = (fn, kwargs)
    return self

  def build(self, protein, is_training=True):
    for fn, kwargs in default(self.config, []):
      f = _feats_fn[fn](is_training=is_training, **kwargs)
      protein = f(protein)
    return protein

  def __call__(self, protein, is_training=True):
    return self.build(protein, is_training=is_training)
