import torch
from einops import rearrange
import mp_nerf

from profold2.constants import *

def fold(seqs, backbones, atom_mask, cloud_mask=None, padding_tok=20,num_coords_per_res=NUM_COORDS_PER_RES):
    """ Gets a backbone of the protein, returns the whole coordinates
        with sidechains (same format as sidechainnet). Keeps differentiability.
        Inputs: 
        * seqs: (batch, L) either tensor or list
        * backbones: (batch, L*n_aa, 3): assume batch=1 (could be extended (?not tested)).
                     Coords for (N-term, C-alpha, C-term, (c_beta)) of every aa.
        * atom_mask: (num_coords_per_res,). int or bool tensor specifying which atoms are passed.
        * cloud_mask: (batch, l, c). optional. cloud mask from scn_cloud_mask`.
                      sets point outside of mask to 0. if passed, else c_alpha
        * padding: int. padding token. same as in sidechainnet: 20
        Outputs: whole coordinates of shape (batch, L, num_coords_per_res, 3)
    """
    atom_mask = atom_mask.bool().cpu().detach()
    cum_atom_mask = atom_mask.cumsum(dim=-1).tolist()

    device = backbones.device
    batch, length = backbones.shape[0], backbones.shape[1] // cum_atom_mask[-1]
    predicted  = rearrange(backbones, 'b (l back) d -> b l back d', l=length)

    # early check if whole chain is already pred
    if cum_atom_mask[-1] == num_coords_per_res:
        return predicted

    # build scaffold from (N, CA, C, CB) - do in cpu
    new_coords = torch.zeros(batch, length, NUM_COORDS_PER_RES, 3)
    predicted  = predicted.cpu() if predicted.is_cuda else predicted

    # fill atoms if they have been passed
    for i,atom in enumerate(atom_mask.tolist()):
        if atom:
            new_coords[:, :, i] = predicted[:, :, cum_atom_mask[i]-1]

    # generate sidechain if not passed
    for s,seq in enumerate(seqs): 
        # format seq accordingly
        if isinstance(seq, torch.Tensor):
            padding = (seq == padding_tok).sum().item()
            seq_str = ''.join([VOCAB._int2char[aa] for aa in seq.cpu().numpy()[:-padding or None]])
        elif isinstance(seq, str):
            padding = 0
            seq_str = seq
        # get scaffolds - will overwrite oxygen since its position is fully determined by N-C-CA
        scaffolds = mp_nerf.proteins.build_scaffolds_from_scn_angles(seq_str, angles=None, device="cpu")
        coords, _ = mp_nerf.proteins.sidechain_fold(wrapper = new_coords[s, :-padding or None].detach(),
                                                    **scaffolds, c_beta = cum_atom_mask[4]==5)
        # add detached scn
        for i,atom in enumerate(atom_mask.tolist()):
            if not atom:
                new_coords[:, :-padding or None, i] = coords[:, i]

    new_coords = new_coords.to(device)
    if cloud_mask is not None:
        new_coords[torch.logical_not(cloud_mask)] = 0.

    # replace any nan-s with previous point location (or N if pos is 13th of AA)
    nan_mask = list(torch.nonzero(new_coords!=new_coords, as_tuple=True))
    new_coords[nan_mask[0], nan_mask[1], nan_mask[2]] = new_coords[nan_mask[0], 
                                                                   nan_mask[1],
                                                                   (nan_mask[-2]+1) % new_coords.shape[-1]] 
    return new_coords.to(device)


