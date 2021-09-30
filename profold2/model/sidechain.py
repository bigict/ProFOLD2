import torch
from torch import nn
from einops import rearrange
import mp_nerf

from profold2.constants import *

def l2_normalize(x, dim=-1, epsilon=1e-12):
  return x / torch.sqrt(
      torch.max(torch.sum(x**2, dim=dim, keepdims=True), epsilon))

class InputProjection(nn.Module):
    def __init__(self, dim, channel):
        super().__init__()
        self.net = nn.Sequential(nn.ReLU(), nn.Linear(dim, channel))

    def forward(self, representations_list):
        act = [self.net(x) for x in representations_list]
        # Sum the activation list (equivalent to concat then Linear).
        return sum(act)

class ResidueBlock(nn.Module):
    def __init__(self, dim, channel):
        super().__init__()
        self.net = nn.Sequential(nn.ReLU(), nn.Linear(dim, channel), nn.ReLU(), nn.Linear(channel, channel))

    def forward(self, act, recycles=1):
        for _ in range(recycles):
            act += self.net(act)
        return act

class TorisonAngles(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.ReLU(), nn.Linear(dim, 14))

    def forward(self, act):
        angles = rearrange(self.net(act), 'b (l w) -> b l w', w=2)
        return l2_normalize(angles, dim=-1)

class MultiRigidSidechain(nn.Module):
    """Class to make side chain atoms."""
    def __init__(self, dim, channel, residual_recycles=1):
        super().__init__()

        self.input_projection = InputProjection(dim, channel)

        self.residual_block = ResidueBlock(dim, channel)
        self.residual_recycles = residual_recycles

        self.torison_angles = TorisonAngles(dim)

    def forward(self, rigids, representations_list, aatype):
        """Predict side chains using multi-rigid representations.

        Args:
          affine: The affines for each residue (translations in angstroms).
          representations_list: A list of activations to predict side chains from.
          aatype: Amino acid types.

        Returns:
          Dict containing atom positions and frames (in angstroms).
        """
        act = self.input_projection(representations_list)

        # Mapping with some residual blocks.
        act = self.residual_block(act, self.residual_recycles)

        # Map activations to torsion angles. Shape: (N, 7, 2).
        angles = self.torison_angles(act)

        # Map torsion angles to frames.
        all_frames_to_global = all_atom.torsion_angles_to_frames(
            aatype, rigids, angles)

        # Use frames and literature positions to create the final atom coordinates.
        # r3.Vecs with shape (N, 14).
        pred_positions = all_atom.frames_and_literature_positions_to_atom14_pos(
            aatype, all_frames_to_global)

        outputs.update({
            'angles_sin_cos': angles,  # (N, 7, 2)
            'atom_pos': pred_positions,  # r3.Vecs (N, 14, 3)
            'frames': all_frames_to_global,  # r3.Rigids (N, 8)
        })
        return outputs

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
    #batch, length = backbones.shape[0], backbones.shape[1] // cum_atom_mask[-1]
    batch, length = backbones.shape[0], backbones.shape[1]
    #predicted  = rearrange(backbones, 'b (l back) d -> b l back d', l=length)
    predicted = backbones

    # early check if whole chain is already pred
    if cum_atom_mask[-1] == num_coords_per_res:
        return predicted

    # build scaffold from (N, CA, C, CB) - do in cpu
    new_coords = torch.zeros(batch, length, num_coords_per_res, 3)
    predicted  = predicted.cpu() if predicted.is_cuda else predicted

    # fill atoms if they have been passed
    for i,atom in enumerate(atom_mask.tolist()):
        if atom:
            new_coords[:, :, i] = predicted[:, :, cum_atom_mask[i]-1]

    # generate sidechain if not passed
    for s, seq in enumerate(seqs): 
        # format seq accordingly
        if isinstance(seq, torch.Tensor):
            padding = (seq == padding_tok).sum().item()
            seq_str = ''.join([VOCAB._int2char[aa] for aa in seq.cpu().numpy()[:-padding or None]])
        elif isinstance(seq, str):
            padding = length - len(seq)
            seq_str = seq
        # get scaffolds - will overwrite oxygen since its position is fully determined by N-C-CA
        scaffolds = mp_nerf.proteins.build_scaffolds_from_scn_angles(seq_str, angles=None, device="cpu")
        coords, _ = mp_nerf.proteins.sidechain_fold(wrapper = new_coords[s, :-padding or None].detach(),
                                                    **scaffolds, c_beta = True)
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


