"""feature functions
  """
import functools
from inspect import isfunction

import numpy as np
import torch
from torch.nn import functional as F
from einops import rearrange, repeat

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


@take1st
def make_coord_mask(protein, includes=None, excludes=None, is_training=True):
  del is_training

  if exists(includes) or exists(excludes):
    restype_atom14_mask = np.copy(residue_constants.restype_atom14_mask)
    if exists(includes):
      includes = set(includes)
      for i in range(residue_constants.restype_num):
        resname = residue_constants.restype_1to3[residue_constants.restypes[i]]
        atom_list = residue_constants.restype_name_to_atom14_names[resname]
        for j in range(restype_atom14_mask.shape[1]):
          if restype_atom14_mask[i, j] > 0 and atom_list[j] not in includes:
            restype_atom14_mask[i, j] = 0
    if exists(excludes):
      excludes = set(excludes)
      for i in range(residue_constants.restype_num):
        resname = residue_constants.restype_1to3[residue_constants.restypes[i]]
        atom_list = residue_constants.restype_name_to_atom14_names[resname]
        for j in range(restype_atom14_mask.shape[1]):
          if restype_atom14_mask[i, j] > 0 and atom_list[j] in excludes:
            restype_atom14_mask[i, j] = 0
    coord_exists = functional.batched_gather(restype_atom14_mask,
                                             protein['seq'])
  else:
    coord_exists = functional.batched_gather(
        residue_constants.restype_atom14_mask, protein['seq'])
  protein['coord_exists'] = coord_exists
  if 'coord_mask' in protein:
    protein['coord_mask'] *= coord_exists
  return protein


@take1st
def make_coord_plddt(protein,
                     threshold=0,
                     gamma=None,
                     use_weighted_mask=True,
                     is_training=True):
  if is_training and 'coord_plddt' in protein:
    ca_idx = residue_constants.atom_order['CA']
    plddt_mean = functional.masked_mean(value=protein['coord_plddt'][...,
                                                                     ca_idx],
                                        mask=protein['mask'],
                                        dim=-1)
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
    protein['loss.distogram.w'] = mask * (1.0 - distogram_w) + distogram_w
    protein['loss.folding.w'] = mask * (1.0 - folding_w) + folding_w
  return protein


@take1st
def make_coord_alt(protein, is_training=True):
  if is_training:
    protein.update(
        functional.symmetric_ground_truth_create_alt(protein['seq'],
                                                     protein.get('coord'),
                                                     protein.get('coord_mask')))
  return protein


@take1st
def make_msa_mask(protein):
  return protein


@take1st
def make_seq_profile(protein,
                     mask=None,
                     density=False,
                     epsilon=1e-8,
                     is_training=True):
  assert not is_training or 'msa' in protein
  # Shape (b, m, l, c)
  if 'msa' in protein:
    msa = protein['msa']
    p = F.one_hot(msa.long(),
                  num_classes=len(residue_constants.restypes_with_x_and_gap))
    num_msa = p.shape[1]
    # Shape (b, l, c)
    if 'msa_row_mask' in protein:
      p = torch.einsum('b m i c,b m -> b i c', p.float(),
                       protein['msa_row_mask'].float())
    else:
      p = torch.sum(p.float(), dim=1)
  else:
    num_msa = 1
    p = F.one_hot(protein['seq'].long(),
                  num_classes=len(residue_constants.restypes_with_x_and_gap))

  # # gap value (b, l)
  # gap_idx = residue_constants.restypes_with_x_and_gap.index('-')
  # protein['sequence_profile_gap_value'] = p[..., gap_idx] / (num_msa + epsilon)

  if exists(mask) and len(mask) > 0:
    # Shape (k, c)
    m = functional.make_mask(residue_constants.restypes_with_x_and_gap,
                             mask,
                             device=p.device)
    protein['sequence_profile_mask'] = m
    # Shape (c)
    p = p * rearrange(m, 'c -> () () c')
  # Shape (b, l, c)
  if density:
    p = p / (torch.sum(p, dim=-1, keepdim=True) + epsilon)
  protein['sequence_profile'] = p
  return protein


@take1st
def make_seq_profile_pairwise(protein,
                              mask=None,
                              density=False,
                              epsilon=1e-8,
                              is_training=True,
                              chunk=8):
  assert 'seq' in protein
  assert not is_training or 'msa' in protein
  # Shape (b, m, l, c)
  if 'msa' in protein:
    msa = protein['msa']
    if hasattr(protein['seq'], 'device'):
      msa = msa.to(device=protein['seq'].device)
    q = F.one_hot(msa.long(),
                  num_classes=len(residue_constants.restypes_with_x_and_gap))
    del msa
  else:
    q = F.one_hot(rearrange(protein['seq'].long(), 'b ... -> b () ...'),
                  num_classes=len(residue_constants.restypes_with_x_and_gap))
  b, m, l, c = q.shape
  p = torch.zeros((b, l, l, c, c), device=q.device)
  for i in range(0, m, chunk):
    p += torch.sum(
        rearrange(q[:, i:i + chunk, ...], 'b m i c -> b m i () c ()') *
        rearrange(q[:, i:i + chunk, ...], 'b m j d -> b m () j () d'),
        dim=1)
  if exists(mask) and len(mask) > 0:
    # Shape (k, c)
    m = functional.make_mask(residue_constants.restypes_with_x_and_gap,
                             mask,
                             device=p.device)
    m = rearrange(m, 'c -> c ()') * rearrange(m, 'd -> () d')
    protein['sequence_profile_pairwise_mask'] = rearrange(m, 'c d -> (c d)')
    p = p * rearrange(m, 'c d -> () () () c d')
  # Shape (b, l, l, c, c)
  if density:
    p = p / (torch.sum(p, dim=(-2, -1), keepdim=True) + epsilon)
  # Shape (b, l, l, c^2)
  protein['sequence_profile_pairwise'] = rearrange(p, '... c d -> ... (c d)')
  return protein


@take1st
def make_bert_mask(protein,
                   fraction=None,
                   span_mean=None,
                   span_min=None,
                   span_max=None,
                   is_training=True):
  if is_training and (exists(fraction) or exists(span_mean)):
    masked_shape = protein['seq'].shape
    mask = protein['mask']

    if exists(fraction):  # vanilla BERT masking
      masked_position = torch.rand(masked_shape, device=mask.device) < fraction
    elif exists(span_mean):  # SpanBERT-like masking
      b, n = masked_shape[:2]
      span_num = torch.poisson(torch.full((b,), span_mean, device=mask.device))
      if exists(span_min) or exists(span_max):
        span_num = torch.clamp(span_num, min=span_min, max=span_max)
      span_num = torch.clamp(span_num, max=n - 1)
      span_i = torch.rand((b,), device=mask.device) * (n - span_num)
      span_j = span_i + span_num
      masked_position = torch.zeros(masked_shape, device=mask.device)
      for k in range(b):
        i, j = int(span_i[k]), int(span_j[k])
        masked_position[k, i:j] = 1

    protein['bert_mask'] = masked_position * mask
    protein['true_seq'] = torch.clone(protein['seq'])
    protein['seq'] = protein['seq'].masked_fill(
        masked_position.bool(), residue_constants.unk_restype_index)
  return protein


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
  """Create pseudo beta features."""

  is_gly = torch.eq(aatype, residue_constants.restype_order['G'])
  ca_idx = residue_constants.atom_order['CA']
  cb_idx = residue_constants.atom_order['CB']
  pseudo_beta = torch.where(
      torch.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
      all_atom_positions[..., ca_idx, :], all_atom_positions[..., cb_idx, :])

  if all_atom_masks is not None:
    pseudo_beta_mask = torch.where(is_gly, all_atom_masks[..., ca_idx],
                                   all_atom_masks[..., cb_idx])
    pseudo_beta_mask = pseudo_beta_mask.float()
    return pseudo_beta, pseudo_beta_mask

  return pseudo_beta


@take1st
def make_pseudo_beta(protein, prefix='', is_training=True):
  del is_training

  if (prefix + 'seq' in protein and prefix + 'coord' in protein and
      prefix + 'coord_mask' in protein):
    protein[prefix +
            'pseudo_beta'], protein[prefix +
                                    'pseudo_beta_mask'] = (pseudo_beta_fn(
                                        protein[prefix + 'seq'],
                                        protein[prefix + 'coord'],
                                        protein[prefix + 'coord_mask']))
  return protein


@take1st
def make_mutation(protein,
                  mutation_prob=0.0,
                  replace_fraction=0.6,
                  cutoff=8,
                  is_training=True):
  assert 0.0 <= mutation_prob <= 1.0
  if is_training and 'pseudo_beta' in protein and 'pseudo_beta_mask' in protein:
    assert 'seq_color' in protein
    assert 'sequence_profile' in protein
    seq_color = protein['seq_color']
    positions = protein['pseudo_beta']
    mask = protein['pseudo_beta_mask']

    squared_mask = rearrange(mask, '... i -> ... i ()') * rearrange(
        mask, '... j -> ... () j')
    pair_clr_mask = (rearrange(seq_color, '... i -> ... i ()') != rearrange(
        seq_color, '... j -> ... () j'))

    dist2 = functional.squared_cdist(positions, positions)
    contacts = squared_mask * pair_clr_mask * (dist2 <= cutoff**2)

    b, n = mask.shape[0], torch.amax(seq_color)

    # ppi label && mask
    ppi_label = torch.scatter_add(torch.zeros(b, n, device=contacts.device), -1,
                                  seq_color.long() - 1,
                                  torch.sum(contacts, dim=-1)) > 0
    ppi_mask = torch.scatter_add(torch.zeros(b, n, device=contacts.device), -1,
                                 seq_color.long() - 1, mask) > 0

    if np.random.random() < mutation_prob and replace_fraction > 0:
      # Select one chain randomly
      selected_clr = 1 + torch.floor(
          torch.rand(b, device=mask.device) *
          torch.amax(seq_color, dim=-1)).int()

      # masked_clr = (seq_color == selected_clr[..., None])
      masked_clr = (seq_color == selected_clr)

      # Number of contacts for each chain with selected chain except itself.
      contacts = torch.sum(contacts * masked_clr[..., None], dim=-1)
      if torch.any(contacts > 0):
        # Nuber of AAs to be mutated
        d = int(
            max(4,
                torch.max(torch.sum(contacts > 0, dim=-1)) * replace_fraction))
        # Sample muations based on contacts
        mutation_idx = torch.multinomial(contacts + 0.1, d)

        c = protein['sequence_profile'].shape[-1]
        p = torch.gather(
            protein['sequence_profile'], -2,
            repeat(mutation_idx[..., None], '... r -> ... (r c)', c=c))

        mask = 'X-'
        # Shape (k, c)
        m = functional.make_mask(residue_constants.restypes_with_x_and_gap,
                                 mask=mask,
                                 device=p.device)

        p = rearrange(m, 'c -> () () c') / (p + 1e-4)
        mutation_aa = torch.multinomial(rearrange(p, 'b i c -> (b i) c'),
                                        1).int()

        mutation_aa = rearrange(mutation_aa, '(b i) c -> b (i c)', b=b)
        wt_aa = torch.gather(protein['seq'], -1, mutation_idx)

        def yield_mutation_str(i):  # pylint: disable=unused-variable
          for idx, x, y in zip(mutation_idx[i], mutation_aa[i], wt_aa[i]):
            x, y = residue_constants.restypes[x], residue_constants.restypes[y]
            yield f'{y}{idx}{x}'

        # protein['mutation_str'] = '|'.join(
        #     ','.join(yield_mutation_str(i)) for i in range(b))
        protein['mutation_idx'] = selected_clr
        # ppi_label = torch.scatter(
        #     ppi_label, -1, selected_clr - 1,
        #     torch.zeros(b, n, device=ppi_label.device, dtype=torch.bool))
        ppi_label = ppi_label.masked_fill(
            F.one_hot(selected_clr.long() - 1, num_classes=n), False)

        protein['true_seq'] = protein['seq']
        protein['seq'] = torch.scatter(protein['seq'], -1, mutation_idx,
                                       mutation_aa)
        protein['loss.folding.w'] = torch.zeros_like(selected_clr)  # FIXME
    protein['ppi_label'], protein['ppi_mask'] = ppi_label, ppi_mask

  return protein


@take1st
def make_backbone_affine(protein, is_training=True):
  #assert (not is_training) or ('coord' in protein and 'coord_mask' in protein)
  if is_training and ('coord' in protein and 'coord_mask' in protein):
    n_idx = residue_constants.atom_order['N']
    ca_idx = residue_constants.atom_order['CA']
    c_idx = residue_constants.atom_order['C']

    assert protein['coord'].shape[-2] > min(n_idx, ca_idx, c_idx)
    protein['backbone_affine'] = functional.rigids_from_3x3(protein['coord'],
                                                            indices=(c_idx,
                                                                     ca_idx,
                                                                     n_idx))
    coord_mask = protein['coord_mask']
    coord_mask = torch.stack(
        (coord_mask[..., c_idx], coord_mask[..., ca_idx], coord_mask[...,
                                                                     n_idx]),
        dim=-1)
    protein['backbone_affine_mask'] = torch.all(coord_mask != 0, dim=-1)
  return protein


@take1st
def make_affine(protein, is_training=True):
  if is_training:
    feats = functional.rigids_from_positions(protein['seq'],
                                             protein.get('coord'),
                                             protein.get('coord_mask'))
    protein.update(feats)
  return protein


@take1st
def make_torsion_angles(protein, is_training=True):
  if is_training and ('coord' in protein and 'coord_mask' in protein):
    feats = functional.angles_from_positions(protein['seq'], protein['coord'],
                                             protein['coord_mask'])
    protein.update(feats)
  return protein


@take1st
def make_esm_embedd(protein,
                    model,
                    repr_layer,
                    max_seq_len=None,
                    device=None,
                    field='embedds',
                    is_training=True):
  del is_training

  esm_extractor = ESMEmbeddingExtractor.get(*model, device=device)
  data_in = list(
      zip(protein['pid'], map(lambda x: x[:max_seq_len], protein['str_seq'])))
  data_out = esm_extractor.extract(data_in,
                                   repr_layer=repr_layer,
                                   device=device)

  if len(data_out.shape) == 3:
    data_out = rearrange(data_out, 'b l c -> b () l c')
  assert len(data_out.shape) == 4
  protein[field] = data_out

  return protein


@take1st
def make_to_device(protein, fields, device, is_training=True):
  del is_training

  if isfunction(device):
    device = device()
  for k in fields:
    if k in protein:
      protein[k] = protein[k].to(device)
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
