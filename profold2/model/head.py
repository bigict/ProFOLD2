import sys
import functools
import logging

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from profold2.common import residue_constants
from profold2.model import functional, folding, sidechain
from profold2.utils import *

logger = logging.getLogger(__name__)

def softmax_cross_entropy(logits, labels, mask=None):
    """Computes softmax cross entropy given logits and one-hot class labels."""
    if not exists(mask):
        mask = 1.0
    loss = -torch.sum(labels * F.log_softmax(logits, dim=-1) * mask, dim=-1)
    return loss

class ConfidenceHead(nn.Module):
    """Head to predict confidence.
    """
    def __init__(self, dim):
        super().__init__()

    def forward(self, headers, representations, batch):
        metrics = {}
        if 'lddt' in headers and 'logits' in headers['lddt']:
            metrics['plddt'] = functional.plddt(headers['lddt']['logits'])
        if 'plddt' in metrics:
            metrics['loss'] = torch.mean(metrics['plddt'], dim=-1)
        return metrics

class ContactHead(nn.Module):
    """Head to predict a contact.
    """
    def __init__(self, dim, diagonal=1, cutoff=8.):
        super().__init__()
 
        self.diagonal = diagonal
        self.cutoff = cutoff

    def forward(self, headers, representations, batch):
        assert 'mlm' in representations and 'contacts' in representations['mlm']
        return dict(logits=representations['mlm']['contacts'])

    def loss(self, value, batch):
        assert 'logits' in value
        logits = value['logits']
        assert len(logits.shape) == 3
        positions = batch['pseudo_beta']
        mask = batch['pseudo_beta_mask']

        assert positions.shape[-1] == 3

        dist2 = torch.cdist(positions, positions, p=2)

        targets = (dist2 <= self.cutoff).float()
        errors = F.binary_cross_entropy(logits, targets,
                reduction='none')

        square_mask = rearrange(mask, 'b l -> b () l') * rearrange(mask, 'b l -> b l ()')
        square_mask = torch.triu(square_mask, diagonal=self.diagonal) + torch.tril(square_mask, diagonal=-self.diagonal)

        avg_error = (
            torch.sum(errors * square_mask) /
            (1e-6 + torch.sum(square_mask)))
        logger.debug('ContactHead.loss: %s', avg_error.item())
        return dict(loss=avg_error)

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
        breaks = self.buckets.to(trunk_embeds.device)
        return dict(logits=self.net(trunk_embeds), breaks=breaks)

    def loss(self, value, batch):
        """Log loss of a distogram."""
        logits, breaks = value['logits'], value['breaks']
        assert len(logits.shape) == 4
        positions = batch['pseudo_beta']
        mask = batch['pseudo_beta_mask']

        assert positions.shape[-1] == 3

        sq_breaks = torch.square(breaks)

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
        logger.debug('DistogramHead.loss: %s', avg_error.item())
        return dict(loss=avg_error, true_dist=torch.sqrt(dist2+1e-6))

class FoldingHead(nn.Module):
    """Head to predict 3d struct.
    """
    def __init__(self, dim, structure_module_depth, structure_module_heads, num_atoms=3, fape_min=1e-6, fape_max=15, fape_z=15, fape_weight=1., fape_reduction=None):
        super().__init__()
        self.struct_module = folding.StructureModule(dim, structure_module_depth, structure_module_heads)
        self.num_atoms = num_atoms
        assert self.num_atoms in [3, 14]

        self.fape_min = fape_min
        self.fape_max = fape_max
        self.fape_z = fape_z
        self.fape_weight = fape_weight

        self.fape_reduction = fape_reduction

    def forward(self, headers, representations, batch):
        #(rotations, translations), act = self.struct_module(representations, batch)
        outputs = self.struct_module(representations, batch)
        (rotations, translations), act, backbones = map(lambda key: outputs[-1][key], ('frames', 'act', 'backbones'))

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

        return dict(frames=(rotations, translations), backbones=backbones, coords=coords, representations=dict(single=act), traj=outputs)

    def loss(self, value, batch):
        coords, labels = value['coords'][...,:self.num_atoms,:], batch['coord'][...,:self.num_atoms,:]
        coord_mask = batch['coord_mask'][...,:self.num_atoms]
        flat_cloud_mask = rearrange(coord_mask, 'b l c -> b (l c)')

        # rotate / align
        coords_aligned, labels_aligned = Kabsch(
                rearrange(rearrange(coords, 'b l c d -> b (l c) d')[flat_cloud_mask], 'c d -> d c'),
                rearrange(rearrange(labels, 'b l c d -> b (l c) d')[flat_cloud_mask], 'c d -> d c'))

        # loss
        loss = .0
        if 1.0 - self.fape_weight > 0:
            loss += (1.0 - self.fape_weight) * torch.clamp(
                    torch.sqrt(F.mse_loss(
                        rearrange(coords_aligned, 'd l -> l d'),
                        rearrange(labels_aligned, 'd l -> l d'))),
                    self.fape_min, self.fape_max) / self.fape_z
        if self.fape_weight > 0:
            assert self.num_atoms >= 3 and 'backbone_affine' in batch and 'backbone_affine_mask' in batch

            true_frames, true_points = batch['backbone_affine'], labels
            frames_mask, points_mask = batch['backbone_affine_mask'], coord_mask

            def yield_fape_loss(outputs):
                for i, traj in enumerate(outputs):
                    pred_frames, pred_points = map(lambda key: traj[key], ('frames', 'backbones'))
                    r = functional.fape(
                            pred_frames, true_frames, frames_mask, pred_points, true_points, points_mask, self.fape_max)/self.fape_z
                    logger.debug('FoldingHead.loss(%d): %s', i, r.item())
                    yield r

            loss += self.fape_weight*sum(yield_fape_loss(value['traj'][self.fape_reduction:])) / len(value['traj'][self.fape_reduction:])

        return dict(loss=loss, coords_aligned=coords_aligned, labels_aligned=labels_aligned)

class LDDTHead(nn.Module):
    """Head to predict the pLDDT to be used as a per-residue configence score.
    """
    def __init__(self, dim, buckets_num=50, min_resolution=.0, max_resolution=sys.float_info.max):
        super().__init__()

        self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, buckets_num))
        self.buckets_num = buckets_num

        self.min_resolution = min_resolution
        self.max_resolution = max_resolution

    def forward(self, headers, representations, batch):
        if 'folding' in headers and 'act' in headers['folding']:
            x = headers['folding']['act']
        else:
            x = representations['single']
        assert 'folding' in headers and 'coords' in headers['folding']
        return dict(logits=self.net(x), coords=headers['folding']['coords'])

    def loss(self, value, batch):
        assert 'coords' in value and 'logits' in value

        ca_idx = residue_constants.atom_order['CA']

        # Shape (b, l, d)
        pred_points = value['coords'][...,ca_idx,:]
        # Shape (b, l, d)
        true_points = batch['coord'][...,ca_idx,:]
        # Shape (b, l)
        points_mask = batch['coord_mask'][...,ca_idx]
        with torch.no_grad():
            # Shape (b, l)
            lddt_ca = functional.lddt(pred_points, true_points, points_mask)
            # protect against out of range for lddt_ca == 1
            bin_index = torch.clamp(
                    torch.floor(lddt_ca * self.buckets_num).long(),
                    max=self.buckets_num - 1)
            labels = F.one_hot(bin_index, self.buckets_num)

        errors = softmax_cross_entropy(labels=labels, logits=value['logits'])

        # Filter by resolution
        b = points_mask.shape[0]
        mask = torch.zeros(b, device=points_mask.device)
        if 'resolution' in batch and exists(batch['resolution']):
            assert len(batch['resolution']) == b
            for i in range(b):
                if exists(batch['resolution'][i]) and (self.min_resolution <= batch['resolution'][i] and batch['resolution'][i] <= self.max_resolution):
                    mask[i] = 1
        points_mask = torch.einsum('b,b ... -> b ...', mask, points_mask)
        loss = torch.sum(errors * points_mask) / (1e-6 + torch.sum(points_mask))
        logger.debug('LDDTHead.loss: %s', loss.item())
        return dict(loss=loss)

class MaskedLMHead(nn.Module):
    """Head to predict Masked Language Model
    """
    def __init__(self, dim):
        super().__init__()

    def forward(self, headers, representations, batch):
        assert 'mlm' in representations
        assert 'logits' in representations['mlm'] and 'labels' in representations['mlm']
        return dict(logits=representations['mlm']['logits'],
                labels=representations['mlm']['labels'])

    def loss(self, value, batch):
        assert 'bert_mask' in batch
        assert 'logits' in value and 'labels' in value

        logits, labels = value['logits'], value['labels']
        #mask = rearrange(batch['bert_mask'], 'b l -> b l ()')
        mask = batch['bert_mask']

        errors = softmax_cross_entropy(
                labels=F.one_hot(labels, logits.shape[-1]), logits=logits)

        avg_error = (
            torch.sum(errors * mask) /
            (1e-6 + torch.sum(mask)))

        logger.debug('MaskedLMHead.loss: %s', avg_error.item())
        return dict(loss=avg_error)

class MetricDict(dict):
    def __add__(self, o):
        n = MetricDict(**self)
        for k in o:
            if k in n:
                n[k] = n[k] + o[k]
            else:
                n[k] = o[k]
        return n

    def __mul__(self, o):
        n = MetricDict(**self)
        for k in n:
            n[k] = n[k] * o
        return n

    def __truediv__(self, o):
        n = MetricDict(**self)
        for k in n:
            n[k] = n[k] / o
        return n

class MetricDictHead(nn.Module):
    """Head to calculate metrics
    """
    def __init__(self, dim, **kwargs):
        super().__init__()

        self.params = kwargs

    def forward(self, headers, representations, batch):
        metrics = MetricDict()
        if 'distogram' in headers and 'pseudo_beta' in batch:
            assert 'logits' in headers['distogram'] and 'breaks' in headers['distogram']
            logits, breaks = headers['distogram']['logits'], headers['distogram']['breaks']
            positions = batch['pseudo_beta']
            mask = batch['pseudo_beta_mask']

            cutoff = self.params.get('contact_cutoff', 8.0)
            t =  torch.sum(breaks <= cutoff)
            pred = F.softmax(logits, dim=-1)
            pred = torch.sum(pred[...,:t+1], dim=-1)
            truth = torch.cdist(positions, positions, p=2)
            precision_list = contact_precision(
                    pred, truth, mask=mask,
                    ratios=self.params.get('contact_ratios'),
                    ranges=self.params.get('contact_ranges'),
                    cutoff=cutoff)
            metrics['contact'] = MetricDict()
            for (i, j), ratio, precision in precision_list:
                i, j = default(i, 0), default(j, 'inf')
                metrics['contact'][f'[{i},{j})_{ratio}'] = precision

        return dict(loss=metrics) if metrics else None

class SequenceProfileHead(nn.Module):
    """Head to predict sequence profile.
    """
    def __init__(self, dim, input_dim=None, single_repr=None):
        super().__init__()

        if not exists(input_dim):
            input_dim = dim
        if not exists(single_repr):
            single_repr = 'struct_module'
        assert single_repr in ('struct_module', 'mlm')
        self.single_repr = single_repr
        
        self.project = nn.Sequential(
                nn.Linear(input_dim, dim),
                nn.GELU(),
                nn.LayerNorm(dim),
                nn.Linear(dim, len(residue_constants.restypes_with_x_and_gap)))

    def forward(self, headers, representations, batch):
        assert 'sequence_profile' in batch

        if self.single_repr == 'mlm':
            assert 'mlm' in representations and 'representations' in representations['mlm']
            x = representations['mlm']['representations']
        else:
            x = representations['single']

        logits = self.project(x)
        return dict(logits=logits)

    def loss(self, value, batch):
        assert 'mask' in batch
        assert 'sequence_profile' in batch
        assert 'logits' in value

        logits, labels = value['logits'], batch['sequence_profile']
        mask = batch['mask']
        label_mask = None
        if 'sequence_profile_mask' in batch:
            label_mask = rearrange(batch['sequence_profile_mask'], 'c -> () () c')

        errors = softmax_cross_entropy(
                labels=labels, logits=logits, mask=label_mask)

        avg_error = (
            torch.sum(errors * mask) /
            (1e-6 + torch.sum(mask)))

        logger.debug('SequenceProfileHead.loss: %s', avg_error.item())
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
            confidence = ConfidenceHead,
            contact = ContactHead,
            distogram = DistogramHead, 
            folding = FoldingHead,
            lddt = LDDTHead,
            metric = MetricDictHead,
            mlm = MaskedLMHead,
            profile = SequenceProfileHead,
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
