import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from profold2.model import functional, folding, sidechain
from profold2.utils import *

def softmax_cross_entropy(logits, labels):
  """Computes softmax cross entropy given logits and one-hot class labels."""
  loss = -torch.sum(labels * F.log_softmax(logits, dim=-1), dim=-1)
  return loss

class DistogramHead(nn.Module):
    """Head to predict a distogram.
    """
    def __init__(self, dim,
            buckets_first_break, buckets_last_break, buckets_num):
        super().__init__()

        self.num_buckets = buckets_num
        self.buckets = torch.linspace(buckets_first_break, buckets_last_break, steps=buckets_num-1)
        self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, buckets_num))

    def forward(self, headers, representations, batch):
        """Builds DistogramHead module.

       Arguments:
         representations: Dictionary of representations, must contain:
           * 'pair': pair representation, shape [N_res, N_res, c_z].
         batch: Batch, unused.

       Returns:
         Dictionary containing:
           * logits: logits for distogram, shape [N_res, N_res, N_bins].
        """
        x = representations['pair']
        trunk_embeds = (x + rearrange(x, 'b i j d -> b j i d')) * 0.5  # symmetrize
        return dict(logits=self.net(trunk_embeds))

    def loss(self, value, batch):
        """Log loss of a distogram."""
        logits = value['logits']
        assert len(logits.shape) == 4
        positions = batch['pseudo_beta']
        mask = batch['pseudo_beta_mask']

        assert positions.shape[-1] == 3

        sq_breaks = torch.square(self.buckets).to(positions.device)

        dist2 = torch.sum(
            torch.square(
                rearrange(positions, 'b l c -> b l () c') -
                rearrange(positions, 'b l c -> b () l c')),
            dim=-1,
            keepdims=True)

        true_bins = torch.sum(dist2 > sq_breaks, axis=-1)

        errors = softmax_cross_entropy(
            labels=F.one_hot(true_bins, self.num_buckets), logits=logits)


        square_mask = rearrange(mask, 'b l -> b () l') * rearrange(mask, 'b l -> b l ()')

        avg_error = (
            torch.sum(errors * square_mask) /
            (1e-6 + torch.sum(square_mask)))
        return dict(loss=avg_error, true_dist=torch.sqrt(dist2+1e-6))

class FoldingHead(nn.Module):
    """Head to predict 3d struct.
    """
    def __init__(self, dim, structure_module_depth, structure_module_heads, num_atoms=3, fape_min=1e-6, fape_max=15, fape_z=15, fape_weight=1.):
        super().__init__()
        self.struct_module = folding.StructureModule(dim, structure_module_depth, structure_module_heads)
        self.num_atoms = num_atoms
        assert self.num_atoms in [3, 14]

        self.fape_min = fape_min
        self.fape_max = fape_max
        self.fape_z = fape_z
        self.fape_weight = fape_weight
        self.criterion = nn.MSELoss()

    def forward(self, headers, representations, batch):
        frames, backbones, act = self.struct_module(representations, batch)

        if self.num_atoms > 3:
            atom_mask = torch.zeros(self.num_atoms).to(batch['seq'].device)
            atom_mask[..., 0] = 1
            atom_mask[..., 1] = 1
            atom_mask[..., 2] = 1

            # build SC container. set SC points to CA and optionally place carbonyl O
            coords = sidechain.fold(batch['str_seq'], backbones=backbones, atom_mask=atom_mask,
                                                  cloud_mask=batch.get('coord_mask'), num_coords_per_res=self.num_atoms)
        else:
            coords = backbones

        return dict(frames=frames, backbones=backbones, coords=coords, representations=dict(single=act))

    def loss(self, value, batch):
        coords, labels = value['coords'][...,:self.num_atoms,:], batch['coord'][...,:self.num_atoms,:]
        coord_mask = batch['coord_mask'][...,:self.num_atoms]
        flat_cloud_mask = rearrange(coord_mask, 'b l c -> b (l c)')

        # rotate / align
        coords_aligned, labels_aligned = Kabsch(
                rearrange(rearrange(coords, 'b l c d -> b (l c) d')[flat_cloud_mask], 'c d -> d c'),
                rearrange(rearrange(labels, 'b l c d -> b (l c) d')[flat_cloud_mask], 'c d -> d c'))

        loss = (1.0 - self.fape_weight) * torch.clamp(
                torch.sqrt(self.criterion(
                    rearrange(coords_aligned, 'd l -> l d'),
                    rearrange(labels_aligned, 'd l -> l d'))),
                self.fape_min, self.fape_max) / self.fape_z

        if self.num_atoms >= 3 and 'backbone_affine' in batch and 'backbone_affine_mask' in batch and self.fape_weight > 0:
            pred_frames, true_frames = value['frames'], batch['backbone_affine']
            frames_mask = batch['backbone_affine_mask']
            pred_points, true_points = coords, labels
            points_mask = coord_mask
            loss += self.fape_weight * functional.fape(
                    pred_frames, true_frames, frames_mask, pred_points, true_points, points_mask, self.fape_max)/self.fape_z

        return dict(loss=loss, coords_aligned=coords_aligned, labels_aligned=labels_aligned)

class LDDTHead(nn.Module):
    """Head to predict the pLDDT to be used as a per-residue configence score.
    """
    def __init__(self, dim, buckets_num=50):
        super().__init__()

        self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, buckets_num),
                nn.ReLU())
        self.buckets_num = buckets_num

    def forward(self, headers, representations, batch):
        if 'folding' in headers and 'act' in headers['folding']:
            x = headers['folding']['act']
        else:
            x = representations['single']
        return dict(logits=self.net(x))

    def loss(self, headers, value, batch):
        assert 'folding' in headers and 'coords' in headers['folding']

        ca_idx = residue_constants.atom_order['CA']

        # Shape (b, l, d)
        pred_points = rearrange(headers['folding']['coords'][...,ca_idx,:], 'b l c d -> b (l c) d')
        # Shape (b, l, d)
        true_points = rearrange(batch['coord'][...,ca_idx,:], 'b l c d -> b (l c) d')
        # Shape (b, l)
        points_mask = rearrange(batch['coord_mask'][...,ca_idx], 'b l c -> b (l c)')
        with torch.no_grade():
            # Shape (b, l)
            lddt_ca = functional.lddt(pred_points, true_points, points_mask)
            # protect against out of range for lddt_ca == 1
            bin_index = torch.min(
                    torch.floor(lddt_ca * self.buckets_num).int(),
                    self.buckets_num - 1)
            labels = F.one_hot(bin_index, self.buckets_num)
        errors = softmax_cross_entropy(labels=labels, logits=value['logits'])

        loss = (
            torch.sum(errors * true_points) /
            (1e-6 + torch.sum(true_points)))
        return dict(loss=avg_error)

class TMscoreHead(nn.Module):
    """Head to predict TM-score.
    """
    def __init__(self, dim, num_atoms=3):
        super().__init__()

        self.num_atoms = num_atoms
        assert self.num_atoms in [3, 14]

    def forward(self, headers, representations, batch):
        assert 'folding' in headers and 'coords' in headers['folding']

        if 'coords_aligned' in headers['folding'] and 'labels_aligned' in headers['folding']:
            coords_aligned, labels_aligned = headers['folding']['coords_aligned'], headers['folding']['labels_aligned']
            return dict(loss=TMscore(
                    rearrange(coords_aligned, 'd l -> () d l'), rearrange(labels_aligned, 'd l -> () d l'),
                    L=torch.sum(batch['mask'], dim=-1).item()))
        elif 'coord' in batch and 'coord_mask' in batch:
            pred, labels = headers['folding']['coords'][...,:self.num_atoms,:], batch['coord'][...,:self.num_atoms,:]
            coord_mask = batch['coord_mask'][...,:self.num_atoms]
            flat_cloud_mask = rearrange(coord_mask, 'b l c -> b (l c)')

            # rotate / align
            coords_aligned, labels_aligned = Kabsch(
                rearrange(rearrange(pred, 'b l c d -> b (l c) d')[flat_cloud_mask], 'c d -> d c'),
                rearrange(rearrange(labels, 'b l c d -> b (l c) d')[flat_cloud_mask], 'c d -> d c'))

            return dict(loss=TMscore(
                    rearrange(coords_aligned, 'd l -> () d l'), rearrange(labels_aligned, 'd l -> () d l'),
                    L=torch.sum(batch['mask'], dim=-1).item()))
        return None

class HeaderBuilder:
    _headers = dict(
            distogram = DistogramHead, 
            folding = FoldingHead,
            lddt = LDDTHead,
            tmscore = TMscoreHead)
    @staticmethod
    def build(dim, config, parent=None):
        def gen():
            for name, args, options in config:
                h = HeaderBuilder._headers[name](dim=dim, **args)
                if exists(parent) and isinstance(parent, nn.Module):
                    parent.add_module(f'head_{name}', h)
                yield name, h, options
        if exists(config):
            return list(gen())
        return []
