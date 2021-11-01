import functools
import numpy as np
import torch
from einops import rearrange

import sidechainnet
from sidechainnet.utils.sequence import VOCAB
from sidechainnet.structure.build_info import NUM_COORDS_PER_RES,BB_BUILD_INFO,SC_BUILD_INFO

from profold2.common import residue_constants
from profold2.data import esm
from profold2.model.features import FeatureBuilder
from profold2.utils import *

def _make_cloud_mask(aa):
    """ relevent points will be 1. paddings will be 0. """
    mask = np.zeros(NUM_COORDS_PER_RES)
    # early stop if padding token
    if aa == "_":
        return mask
    # get num of atoms in aa
    n_atoms = 4+len(SC_BUILD_INFO[VOCAB.int2chars(VOCAB[aa])]["atom-names"])
    mask[:n_atoms] = 1
    return mask

CUSTOM_INFO = {aa: {"cloud_mask": _make_cloud_mask(aa)
                  } for aa in VOCAB.stdaas}


def cloud_mask(scn_seq, boolean=True, coords=None):
    """ Gets the boolean mask atom positions (not all aas have same atoms). 
        Inputs: 
        * scn_seq: (batch, length) sequence as provided by Sidechainnet package
        * boolean: whether to return as array of idxs or boolean values
        * coords: optional .(batch, lc, 3). sidechainnet coords.
                  returns the true mask (solves potential atoms that might not be provided)
        Outputs: (batch, length, NUM_COORDS_PER_RES) boolean mask 
    """
    scn_seq = expand_dims_to(scn_seq, 2 - len(scn_seq.shape))
    # early check for coords mask
    if exists(coords): 
        batch_mask = (coords == 0).sum(dim=-1) < coords.shape[-1]
        if boolean:
            return batch_mask.bool()
        return batch_mask.nonzero()

    # do loop in cpu
    device = scn_seq.device
    batch_mask = []
    scn_seq = scn_seq.cpu().tolist()
    for i, seq in enumerate(scn_seq):
        # get masks for each protein (points for each aa)
        batch_mask.append(torch.tensor([CUSTOM_INFO[VOCAB.int2char(aa)]['cloud_mask'] \
                                         for aa in seq]).bool().to(device))
    # concat in last dim
    batch_mask = torch.stack(batch_mask, dim=0)
    # return mask (boolean or indexes)
    if boolean:
        return batch_mask.bool()
    return batch_mask.nonzero()

def _collate_fn(insts, max_seq_len=None, aggregate_input=True, seqs_as_onehot=None, feat_builder=None):
    scn_collate_fn = sidechainnet.dataloaders.collate.get_collate_fn(aggregate_input, seqs_as_onehot)
    batch = scn_collate_fn(insts)

    # preprocess
    str_seqs = list(batch.str_seqs)
    int_seqs = batch.int_seqs
    mask = batch.msks
    coords = rearrange(batch.crds, 'b (l c) d -> b l c d', c=NUM_COORDS_PER_RES)

    clips = {}
    b, n = int_seqs.shape
    assert b == len(str_seqs)
    if exists(max_seq_len) and n > max_seq_len:
        assert max_seq_len > 0
        for k in range(b):
            if len(str_seqs[k]) <= max_seq_len:
                continue
            i, j = 0, max_seq_len
            if torch.sum(coords[k,...] != 0) > 0:
                while True:
                    i = np.random.randint(n - max_seq_len)
                    j = i + max_seq_len
                    if torch.sum(coords[k,i:j,...] != 0) > 0:
                        break
            clips[k] = dict(i=i, j=j, l=len(str_seqs[k]))
            str_seqs[k] = str_seqs[k][i:j]
            int_seqs[k,:j-i] = torch.clone(int_seqs[k,i:j])
            mask[k,:j-i] = torch.clone(mask[k,i:j])
            coords[k,:j-i,...] = torch.clone(coords[k,i:j,...])

        int_seqs = int_seqs[:,:max_seq_len]
        mask = mask[:,:max_seq_len]
        coords = coords[:,:max_seq_len,...]

    # postprocess
    int_seqs = int_seqs.apply_(
            lambda x: residue_constants.restype_order_with_x.get(VOCAB.int2char(x), residue_constants.unk_restype_index))

    batch = dict(pid=batch.pids, 
                seq=int_seqs,
                mask=mask,
                str_seq=str_seqs,
                coord=coords,
                coord_mask=cloud_mask(int_seqs, coords=coords),
                clips=clips)

    # build new features
    if feat_builder:
        batch = feat_builder.build(batch)

    return batch

def load(max_seq_len=None, aggregate_model_input=True, seq_as_onehot=None, collate_fn=None, feats=None, **kwargs):
    if collate_fn is None:
        collate_fn = functools.partial(_collate_fn, max_seq_len=max_seq_len,
                aggregate_input=aggregate_model_input,
                seqs_as_onehot=seq_as_onehot, feat_builder=FeatureBuilder(feats))
    if 'with_pytorch' not in kwargs:
        kwargs['with_pytorch'] = 'dataloaders'
    return sidechainnet.load(
            aggregate_model_input=aggregate_model_input, 
            collate_fn = collate_fn, 
            seq_as_onehot = seq_as_onehot, **kwargs)
