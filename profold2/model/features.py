from functools import wraps

import torch
from einops import rearrange

from profold2.common import residue_constants
from profold2.utils import default,exists

_feats_fn = {}

def take1(fn):
    """Supply all arguments but the first."""

    @wraps(fn)
    def fc(*args, **kwargs):
        return lambda x: fn(x, *args, **kwargs)

    global _feats_fn
    _feats_fn[fn.__name__] = fc

    return fc

@take1
def make_seq_mask(protein):
    return protein

@take1
def make_msa_mask(protein):
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

@take1
def make_pseudo_beta(protein, prefix=''):
    protein[prefix + 'pseudo_beta'], protein[prefix + 'pseudo_beta_mask'] = (
            pseudo_beta_fn(protein[prefix + 'seq'], protein[prefix + 'coord'], protein[prefix + 'coord_mask']))
    return protein

@take1
def make_random_seed_to_crop(protein):
    return protein

@take1
def make_esm_embedd(protein, esm_extractor, repr_layer, device=None):
    data = list(zip(protein['pid'], protein['str_seq']))
    protein['emb_seq'] = rearrange(
            esm_extractor.extract(data, repr_layer=repr_layer, device=device),
            'b l c -> b () l c')
    return protein

@take1
def make_to_device(protein, fields, device):
    for k in fields:
        protein[k] = protein[k].to(device)
    return protein

@take1
def make_selection(protein, fields):
    return {k: protein[k] for k in fields}

class FeatureBuilder:
    def __init__(self, config):
        self.map_fn = [_feats_fn[k](**v) for k, v in config.items()] if config else []

    def build(self, protein):
        for f in self.map_fn:
            protein = f(protein)
        return protein
