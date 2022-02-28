import functools
from inspect import isfunction

import torch
from torch.nn import functional as F
from einops import rearrange

from profold2.common import residue_constants
from profold2.data.esm import ESMEmbeddingExtractor
from profold2.model.functional import rigids_from_3x3
from profold2.utils import default, exists

_feats_fn = {}


def take1st(fn):
    """Supply all arguments but the first."""
    @functools.wraps(fn)
    def fc(*args, **kwargs):
        return lambda x: fn(x, *args, **kwargs)

    global _feats_fn
    _feats_fn[fn.__name__] = fc

    return fc

@take1st
def make_seq_mask(protein, padd_id=20, is_training=True):
    mask = protein['seq'] != padd_id
    protein['mask'] = mask.bool()
    return protein

@take1st
def make_msa_mask(protein):
    return protein

@take1st
def make_seq_profile(protein, mask=None, density=False, epsilon=1e-8, is_training=True):
    assert not is_training or 'msa' in protein
    # Shape (b, m, l, c)
    if 'msa' in protein:
        p = F.one_hot(
                protein['msa'],
                num_classes=len(residue_constants.restypes_with_x_and_gap))
        # Shape (b, l, c)
        p = torch.sum(p, dim=1)
    else:
        p = F.one_hot(
                protein['seq'],
                num_classes=len(residue_constants.restypes_with_x_and_gap))
    if exists(mask) and len(mask) > 0:
        m = [residue_constants.restypes_with_x_and_gap.index(i) for i in mask]
        m = F.one_hot(torch.as_tensor(m),
                num_classes=len(residue_constants.restypes_with_x_and_gap))
        # Shape (k, c)
        m = ~(torch.sum(m, dim=0) > 0)
        protein['sequence_profile_mask'] = m
        # Shape (c)
        p = p * rearrange(m, 'c -> () () c')
    # Shape (b, l, c)
    if density:
        p = p / (torch.sum(p, dim=-1, keepdim=True) + epsilon)
    protein['sequence_profile'] = p
    return protein

@take1st
def make_bert_mask(protein,
                   fraction=0.12,
                   prepend_bos=True,
                   append_eos=True,
                   is_training=True):
    masked_shape = protein['seq'].shape
    mask = protein['mask']
    if prepend_bos:
        masked_shape = (*masked_shape[:-1], masked_shape[-1] + 1)
        mask = torch.cat((torch.ones((*masked_shape[:-1], 1), device=mask.device), mask), dim=-1)
    if append_eos:
        masked_shape = (*masked_shape[:-1], masked_shape[-1] + 1)
        mask = torch.cat((mask, torch.ones((*masked_shape[:-1], 1), device=mask.device)), dim=-1)
    masked_position = torch.rand(masked_shape) < fraction
    protein['bert_mask'] = masked_position * mask
    return protein

def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    """Create pseudo beta features."""

    is_gly = torch.eq(aatype, residue_constants.restype_order['G'])
    ca_idx = residue_constants.atom_order['CA']
    cb_idx = residue_constants.atom_order['CB']
    pseudo_beta = torch.where(
        torch.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :])

    if all_atom_masks is not None:
        pseudo_beta_mask = torch.where(
            is_gly, all_atom_masks[..., ca_idx], all_atom_masks[..., cb_idx])
        pseudo_beta_mask = pseudo_beta_mask.float()
        return pseudo_beta, pseudo_beta_mask

    return pseudo_beta

@take1st
def make_pseudo_beta(protein, prefix='', is_training=True):
    if prefix + 'seq' in protein and prefix + 'coord' in protein and prefix + 'coord_mask' in protein:
        protein[prefix + 'pseudo_beta'], protein[prefix + 'pseudo_beta_mask'] = (
                pseudo_beta_fn(protein[prefix + 'seq'], protein[prefix + 'coord'], protein[prefix + 'coord_mask']))
    return protein

@take1st
def make_backbone_affine(protein, is_training=True):
    assert (not is_training) or ('coord' in protein and 'coord_mask' in protein)
    if is_training or ('coord' in protein and 'coord_mask' in protein):
        assert protein['coord'].shape[-2] >= 3
        protein['backbone_affine'] = rigids_from_3x3(protein['coord'][...,:3,:])
        protein['backbone_affine_mask'] = torch.any(protein['coord_mask'][...,:3] != 0, dim=-1)
    return protein

@take1st
def make_random_seed_to_crop(protein, is_training=True):
    return protein

@take1st
def make_esm_embedd(protein, model, repr_layer, max_seq_len=None, device=None, field='embedds', is_training=True):
    esm_extractor = ESMEmbeddingExtractor.get(*model, device=device)
    data_in = list(zip(protein['pid'], map(lambda x: x[:max_seq_len], protein['str_seq'])))
    data_out = esm_extractor.extract(data_in, repr_layer=repr_layer, device=device)

    if len(data_out.shape) == 3:
        data_out = rearrange(data_out, 'b l c -> b () l c')
    assert len(data_out.shape) == 4
    protein[field] = data_out

    return protein

@take1st
def make_to_device(protein, fields, device, is_training=True):
    if isfunction(device):
        device = device()
    for k in fields:
        if k in protein:
            protein[k] = protein[k].to(device)
    return protein

@take1st
def make_selection(protein, fields, is_training=True):
    return {k: protein[k] for k in fields}

class FeatureBuilder:
    def __init__(self, config, is_training=True):
        self.config = config
        self.training = is_training

    def build(self, protein):
        for fn, kwargs in default(self.config, []):
            f = _feats_fn[fn](is_training=self.training, **kwargs)
            protein = f(protein)
        return protein

    def __call__(self, protein):
        return self.build(protein)
