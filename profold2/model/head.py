import sys
import logging

import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F
from einops import rearrange, repeat

from profold2.common import residue_constants
from profold2.model import functional, folding
from profold2.model.commons import embedd_dim_get
from profold2.utils import *

logger = logging.getLogger(__name__)


def clipped_sigmoid_cross_entropy(logits,
                                  labels,
                                  clip_negative_at_logit,
                                  clip_positive_at_logit,
                                  epsilon=1e-7):
  loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
  return loss

def softmax_cross_entropy(logits, labels, mask=None, gammar=0):
  """Computes softmax cross entropy given logits and one-hot class labels."""
  if not exists(mask):
    mask = 1.0
  if gammar > 0:  # focal loss enabled.
    prob = F.softmax(logits, dim=-1)
    labels = labels.float() * ((1 - prob)**gammar)
  loss = -torch.sum(labels * F.log_softmax(logits, dim=-1) * mask, dim=-1)
  return loss


def softmax_kl_diversity(logits, labels, mask=None):
  if not exists(mask):
    mask = 1.0
  loss = torch.sum(
      F.kl_div(F.log_softmax(logits, dim=-1), labels, reduction='none') * mask,
      dim=-1)
  return loss


def softmax_cosine_similarity(logits, labels, mask=None):
  if not exists(mask):
    mask = 1.0
  pred = F.softmax(logits, dim=-1)
  return F.cosine_similarity(pred * mask, labels * mask, dim=-1)


def _make_dynamic_errors(errors, batch, num_pivot, alpha=0.3):
  if exists(num_pivot) and num_pivot > 0 and 'num_msa' in batch:
    w = torch.clamp(batch['num_msa'], max=num_pivot) / num_pivot
    w = torch.pow(w, alpha)
    errors = torch.einsum('b ...,b -> b ...', errors, w)
    return errors, w
  return errors, [1.0] * errors.shape[0]

def batched_tmscore(pred_points, true_points, coord_mask, mask):
  assert len(pred_points.shape) == 4  # b i c d

  def _yield_tmscore(b):
    flat_cloud_mask = rearrange(coord_mask[b], 'i c -> (i c)')

    # rotate / align
    coords_aligned, labels_aligned = Kabsch(
        rearrange(
            rearrange(pred_points[b], 'i c d -> (i c) d')[flat_cloud_mask],
            'c d -> d c'),
        rearrange(
            rearrange(true_points[b], 'i c d -> (i c) d')[flat_cloud_mask],
            'c d -> d c'))

    return TMscore(rearrange(coords_aligned, 'd l -> () d l'),
                   rearrange(labels_aligned, 'd l -> () d l'),
                   L=torch.sum(mask[b], dim=-1))

  # tms = sum(map(_yield_tmscore, range(pred_points.shape[0])))
  tms = torch.cat(list(map(_yield_tmscore, range(pred_points.shape[0]))))
  return tms


class ConfidenceHead(nn.Module):
  """Head to predict confidence.
    """

  def __init__(self, dim):
    super().__init__()

  def forward(self, headers, representations, batch):
    metrics = {}
    with torch.no_grad():
      if 'lddt' in headers and 'logits' in headers['lddt']:
        metrics['plddt'] = functional.plddt(headers['lddt']['logits']) * 100
      if 'plddt' in metrics:
        mask = batch['mask']
        # metrics['loss'] = functional.masked_mean(value=metrics['plddt'], mask=mask)
        metrics['loss'] = functional.masked_mean(value=metrics['plddt'],
                                                 mask=mask,
                                                 dim=-1)
        logger.debug('ConfidenceHead.loss: %s', metrics['loss'])
      if 'pae' in headers and 'logits' in headers['pae']:
        pae, mae = functional.pae(headers['pae']['logits'],
                                  headers['pae']['breaks'],
                                  return_mae=True)
        metrics['pae'], metrics['mae'] = pae, mae
        ptm = functional.ptm(headers['pae']['logits'],
                             headers['pae']['breaks'],
                             mask=batch.get('mask'))
        metrics['ptm'] = ptm
    return metrics


class ContactHead(nn.Module):
  """Head to predict a contact.
    """

  def __init__(self, dim, diagonal=1, cutoff=8., loss_min=None, loss_max=None):
    super().__init__()

    self.diagonal = diagonal
    self.cutoff = cutoff
    self.loss_min = loss_min
    self.loss_max = loss_max

  def forward(self, headers, representations, batch):
    assert not self.training or ('mlm' in representations and
                                 'contacts' in representations['mlm'])
    if 'mlm' in representations and 'contacts' in representations[
        'mlm'] and exists(representations['mlm']['contacts']):
      return dict(logits=representations['mlm']['contacts'])
    return None

  def loss(self, value, batch):
    assert 'logits' in value
    logits = value['logits']
    assert len(logits.shape) == 3
    if 'pseudo_beta' in batch:
      positions = batch['pseudo_beta']
      mask = batch['pseudo_beta_mask']

      assert positions.shape[-1] == 3

      dist2 = torch.cdist(positions, positions, p=2)

      targets = (dist2 <= self.cutoff).float()

      with autocast(enabled=False):
        errors = F.binary_cross_entropy(logits, targets, reduction='none')

      square_mask, square_weight = (rearrange(mask, '... i -> ... i ()') *
                                    rearrange(mask, '... j -> ... () j'), 1.0)
      square_mask = torch.triu(square_mask,
                               diagonal=self.diagonal) + torch.tril(
                                   square_mask, diagonal=-self.diagonal)
      if 'coord_plddt' in batch:
        ca_idx = residue_constants.atom_order['CA']
        plddt_weight = torch.minimum(
            rearrange(batch['coord_plddt'][..., ca_idx], '... i->... i ()'),
            rearrange(batch['coord_plddt'][..., ca_idx], '... j->... () j'))
        if batch.get('coord_plddt_use_weighted_mask', True):
          square_mask *= plddt_weight
        else:
          square_weight = plddt_weight

      avg_error = functional.masked_mean(value=errors * square_weight,
                                         mask=square_mask,
                                         epsilon=1e-6)
      logger.debug('ContactHead.loss: %s', avg_error)
      if exists(self.loss_min) or exists(self.loss_max):
        avg_error = torch.clamp(avg_error, min=self.loss_min, max=self.loss_max)
      return dict(loss=avg_error)
    return None


class CoevolutionHead(nn.Module):
  """Head to predict Co-evolution.
    """

  def __init__(self,
               dim,
               mask='-',
               alpha=0.0,
               beta=0.0,
               gammar=0.0,
               num_pivot=1024,
               focal_loss=0,
               loss_min=None,
               loss_max=None):
    super().__init__()
    dim_single, dim_pairwise = embedd_dim_get(dim)

    num_class = len(residue_constants.restypes_with_x_and_gap)
    self.single = nn.Sequential(nn.Linear(dim_single, dim_single),
                                nn.GELU(),
                                nn.LayerNorm(dim_single),
                                nn.Linear(dim_single, num_class))
    self.pairwize = nn.Sequential(nn.LayerNorm(dim_pairwise),
                                  nn.Linear(dim_pairwise, num_class**2),
                                  nn.ReLU())
    self.mask = mask

    self.alpha = alpha
    self.beta = beta
    self.gammar = gammar
    self.num_pivot = num_pivot
    self.focal_loss = focal_loss
    self.loss_min = loss_min
    self.loss_max = loss_max

  def forward(self, headers, representations, batch):
    """Builds CoevolutionHead module.

        Arguments:
         representations: Dictionary of representations, must contain:
           * 'pair': pair representation, shape [N_res, N_res, c_z].
         batch: Batch, unused.

        Returns:
         Dictionary containing:
           * logits: logits for co-evolution, shape [N_res, N_res, N_bins^2].
        """
    num_class = len(residue_constants.restypes_with_x_and_gap)

    si, zij = representations['single'], representations['pair']
    m = functional.make_mask(residue_constants.restypes_with_x_and_gap,
                             self.mask,
                             device=si.device)

    ei = self.single(si)
    eij = self.pairwize(
        (zij + rearrange(zij, 'b i j d -> b j i d')) * 0.5)  # symmetrize

    eij = eij * rearrange(1 - torch.eye(zij.shape[-2], device=zij.device),
                          'i j -> i j ()')  # eii = 0

    ret = dict(wij=eij, bi=ei)
    if self.training or 'msa' in batch:
      assert 'msa' in batch
      hi = torch.einsum(
          'b m j d,b i j c d,d -> b m i c',
          F.one_hot(batch['msa'].long(), num_class).float(),
          rearrange(eij, 'b i j (c d) -> b i j c d', c=num_class, d=num_class),
          m)
      logits = rearrange(ei, 'b i c -> b () i c') + hi
      ret.update(logits=logits)
    return ret

  def loss(self, value, batch):
    """Log loss of a msa rebuilding."""
    num_class = len(residue_constants.restypes_with_x_and_gap)

    logits = value['logits']
    labels = F.one_hot(batch['msa'].long(), num_class)
    logger.debug('CoevolutionHead.loss.logits: %s, %s', logits.shape,
                 self.focal_loss)

    assert len(logits.shape) == 4
    assert 'msa' in batch
    label_mask = functional.make_mask(residue_constants.restypes_with_x_and_gap,
                                      self.mask,
                                      device=logits.device)

    errors = softmax_cross_entropy(labels=labels,
                                   logits=logits,
                                   mask=label_mask,
                                   gammar=self.focal_loss)
    mask = torch.einsum('b i,b m i c,c -> b m i', batch['mask'].float(),
                        labels.float(), label_mask)
    if 'msa_row_mask' in batch:
      mask = torch.einsum('b m i,b m -> b m i', mask, batch['msa_row_mask'])

    errors, w = _make_dynamic_errors(errors, batch, self.num_pivot)
    avg_error = functional.masked_mean(value=errors, mask=mask, epsilon=1e-6)
    logger.debug('CoevolutionHead.loss(%s): %s', w, avg_error)

    def _make_dynamic_regularization(w):
      if isinstance(w, float) and w > 0:
        b, device = value['wij'].shape[0], value['wij'].device
        return torch.full((b,), w, device=device)
      elif isinstance(w, list):
        assert 'num_msa' in batch
        min_w, max_w = w
        num_msa = torch.clamp(batch['num_msa'], max=self.num_pivot)
        return max_w + (min_w - max_w) * num_msa / self.num_pivot
      return None

    square_mask = (rearrange(batch['mask'], '... i -> ... i ()') *
                   rearrange(batch['mask'], '... j -> ... () j'))
    # L1 regularization
    alpha = _make_dynamic_regularization(self.alpha)
    if exists(alpha):
      r1 = torch.sum(torch.abs(value['wij']), dim=-1) * square_mask
      logger.debug('CoevolutionHead.loss.L1(%s): %s', alpha, torch.mean(r1))
      avg_error += torch.mean(alpha[..., None, None] * r1)

    # L2 regularization
    beta = _make_dynamic_regularization(self.beta)
    if exists(beta):
      r2 = torch.sum(torch.square(value['wij']), dim=-1) * square_mask
      logger.debug('CoevolutionHead.loss.L2(%s): %s', beta, torch.mean(r2))
      avg_error += torch.mean(beta[..., None, None] * r2)

    # LH regularization
    if self.gammar > 0:
      epsilon = 1e-10
      M = torch.sqrt(torch.sum(value['wij']**2, dim=-1) + epsilon)
      p = torch.sum(M, dim=-1)
      rlh = torch.sum(
          torch.square(
              torch.einsum('... i, ... i j, ... j', p, M, p) /
              (torch.einsum('... i,... i', p, p) + epsilon)))
      logger.debug('CoevolutionHead.loss.LH: %s', rlh)
      avg_error += 0.5 * self.gammar * rlh

    if exists(self.loss_min) or exists(self.loss_max):
      avg_error = torch.clamp(avg_error, min=self.loss_min, max=self.loss_max)
    return dict(loss=avg_error)


class DistogramHead(nn.Module):
  """Head to predict a distogram.
    """

  def __init__(self,
               dim,
               buckets_first_break,
               buckets_last_break,
               buckets_num,
               focal_loss=0):
    super().__init__()
    _, dim = embedd_dim_get(dim)

    self.num_buckets = buckets_num
    self.buckets = torch.linspace(buckets_first_break,
                                  buckets_last_break,
                                  steps=buckets_num - 1)
    self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, buckets_num))
    self.focal_loss = focal_loss

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
    if 'pseudo_beta' in batch:
      positions = batch['pseudo_beta']
      mask = batch['pseudo_beta_mask']

      assert positions.shape[-1] == 3

      sq_breaks = torch.square(breaks)

      dist2 = torch.sum(torch.square(
          rearrange(positions, 'b i c -> b i () c') -
          rearrange(positions, 'b j c -> b () j c')),
                        dim=-1,
                        keepdims=True)

      true_bins = torch.sum(dist2 > sq_breaks, axis=-1)
      errors = softmax_cross_entropy(labels=F.one_hot(true_bins,
                                                      self.num_buckets),
                                     logits=logits,
                                     gammar=self.focal_loss)
    else:
      b, l, device = *batch['seq'].shape, batch['seq'].device
      mask = torch.ones((b, l), device=device, dtype=torch.bool)

      with torch.no_grad():
        labels = torch.zeros(logits.shape,
                             device=logits.device)  #F.softmax(logits, dim=-1)

      errors = softmax_kl_diversity(labels=labels, logits=logits)

    square_mask, square_weight = (rearrange(mask, '... i -> ... i ()') *
                                  rearrange(mask, '... j -> ... () j'), 1.0)
    if 'coord_plddt' in batch:
      ca_idx = residue_constants.atom_order['CA']
      plddt_weight = torch.minimum(
          rearrange(batch['coord_plddt'][..., ca_idx], '... i->... i ()'),
          rearrange(batch['coord_plddt'][..., ca_idx], '... j->... () j'))
      if batch.get('coord_plddt_use_weighted_mask', True):
        square_mask *= plddt_weight
      else:
        square_weight = plddt_weight

    if 'loss.distogram.w' in batch:
      logger.debug('DistogramHead.loss.w: %s', batch['loss.distogram.w'])
      errors = torch.einsum('b ...,b -> b ...', errors,
                            batch['loss.distogram.w'])
    avg_error = functional.masked_mean(value=errors * square_weight,
                                       mask=square_mask,
                                       epsilon=1e-6)
    logger.debug('DistogramHead.loss: %s', avg_error)
    return dict(loss=avg_error)


class FoldingHead(nn.Module):
  """Head to predict 3d struct.
    """

  def __init__(self,
               dim,
               structure_module_depth,
               structure_module_heads,
               point_value_dim=4,
               qkv_use_bias=False,
               fape_min=1e-6,
               fape_max=15,
               fape_z=15,
               fape_backbone_clamp_ratio=1.0,
               fape_sidechain_clamp_ratio=1.0,
               dropout=.0,
               position_scale=1.0,
               **params):
    super().__init__()
    self.struct_module = folding.StructureModule(
        dim,
        structure_module_depth,
        structure_module_heads,
        point_value_dim=point_value_dim,
        qkv_use_bias=qkv_use_bias,
        dropout=dropout,
        position_scale=position_scale)

    self.fape_min = fape_min
    self.fape_max = fape_max
    self.fape_z = fape_z
    self.fape_backbone_clamp_ratio = fape_backbone_clamp_ratio
    self.fape_sidechain_clamp_ratio = fape_sidechain_clamp_ratio

    self.params = params

  def forward(self, headers, representations, batch):
    #(rotations, translations), act = self.struct_module(representations, batch)
    outputs = self.struct_module(representations, batch)
    (rotations, translations), act, atoms = map(lambda key: outputs[-1][key],
                                                ('frames', 'act', 'atoms'))

    return dict(frames=(rotations, translations),
                coords=atoms['coords'],
                representations=dict(single=act),
                traj=outputs)

  def loss(self, value, batch):

    def backbone_fape_loss(pred_frames_list, gt_frames, frames_mask):
      assert pred_frames_list and exists(frames_mask)

      dij_weight = None
      if 'coord_plddt' in batch:
        ca_idx = residue_constants.atom_order['CA']
        dij_weight = torch.minimum(
            rearrange(batch['coord_plddt'][..., ca_idx], '... i -> ... i ()'),
            rearrange(batch['coord_plddt'][..., ca_idx], '... j -> ... () j'))

      if 'loss.folding.w' in batch:
        logger.debug('FoldingHead.loss.w: %s', batch['loss.folding.w'])
        wij = rearrange(batch['loss.folding.w'], '... -> ... () ()')
        dij_weight = dij_weight * wij if exists(dij_weight) else wij

      for i, pred_frames in enumerate(pred_frames_list):
        with torch.no_grad():
          true_frames = default(gt_frames, pred_frames)

        _, pred_points = pred_frames
        _, true_points = true_frames
        r = functional.fape(pred_frames,
                            true_frames,
                            frames_mask,
                            pred_points,
                            true_points,
                            frames_mask,
                            self.fape_max,
                            clamp_ratio=self.fape_backbone_clamp_ratio,
                            dij_weight=dij_weight,
                            use_weighted_mask=batch.get(
                                'coord_plddt_use_weighted_mask',
                                False)) / self.fape_z
        logger.debug('FoldingHead.loss(backbone_fape_loss: %d): %s', i, r)
        yield r

    def sidechain_fape_loss(pred_frames, true_frames, frames_mask, pred_points,
                            true_points, point_mask):

      def frames_to_fape_shape(frames):
        rotations, translations = frames
        return (rearrange(rotations, '... i c h w -> ... (i c) h w'),
                rearrange(translations, '... i c h -> ... (i c) h'))

      def points_to_fape_shape(points):
        return rearrange(points, '... i c d -> ... (i c) d')

      def mask_to_fape_shape(masks):
        return rearrange(masks, '... i c -> ... (i c)')

      dij_weight = None
      if 'coord_plddt' in batch:
        # (N, 8, 3, 14)
        group_atom14_idx = F.one_hot(
            functional.batched_gather(
                residue_constants.restype_rigid_group_atom14_idx,
                batch['seq']).long(), residue_constants.atom14_type_num)
        # (N, 8)
        group_atom14_plddt = torch.amin(torch.einsum(
            '... g m n, ... n -> ... g m', group_atom14_idx.float(),
            batch['coord_plddt']),
                                        dim=-1)
        dij_weight = torch.minimum(
            rearrange(mask_to_fape_shape(group_atom14_plddt),
                      '... i -> ... i ()'),
            rearrange(mask_to_fape_shape(batch['coord_plddt']),
                      '... j -> ... () j'))
      if 'loss.folding.w' in batch:
        wij = rearrange(batch['loss.folding.w'], '... -> ... () ()')
        dij_weight = dij_weight * wij if exists(dij_weight) else wij

      with torch.no_grad():
        true_frames = default(true_frames, pred_frames)
        true_points = default(true_points, pred_points)

      r = functional.fape(frames_to_fape_shape(pred_frames),
                          frames_to_fape_shape(true_frames),
                          mask_to_fape_shape(frames_mask),
                          points_to_fape_shape(pred_points),
                          points_to_fape_shape(true_points),
                          mask_to_fape_shape(point_mask),
                          self.fape_max,
                          clamp_ratio=self.fape_sidechain_clamp_ratio,
                          dij_weight=dij_weight,
                          use_weighted_mask=batch.get(
                              'coord_plddt_use_weighted_mask',
                              False)) / self.fape_z
      logger.debug('FoldingHead.loss(sidechain_fape_loss): %s', r)
      return r

    def fape_loss(traj, gt):
      coords = gt.get('coord')
      if 'coord_mask' in gt:
        coord_mask = gt['coord_mask']
      else:
        assert 'coord_exists' in gt
        coord_mask = gt['coord_exists']
      backbone_affine, backbone_affine_mask = (gt.get('backbone_affine'),
                                               gt.get('backbone_affine_mask'))
      if not exists(backbone_affine_mask):
        n_idx = residue_constants.atom_order['N']
        ca_idx = residue_constants.atom_order['CA']
        c_idx = residue_constants.atom_order['C']
        backbone_affine_mask = torch.all(torch.stack(
            (coord_mask[..., c_idx], coord_mask[..., ca_idx],
             coord_mask[..., n_idx]),
            dim=-1) != 0,
                                         dim=-1)
      # backbone loss
      backbone_loss = sum(
          backbone_fape_loss([x['frames'] for x in traj], backbone_affine,
                             backbone_affine_mask)) / len(traj)
      # sidechine loss
      atoms = traj[-1]['atoms']
      atom_affine, atom_affine_mask = (gt.get('atom_affine'),
                                       gt.get('atom_affine_mask'))
      if not exists(atom_affine_mask):
        assert 'atom_affine_exists' in gt, gt.keys()
        atom_affine_mask = gt['atom_affine_exists']

      if exists(atom_affine):
        assert exists(coords), gt.keys()
        # Renamed frames
        alt_is_better = functional.symmetric_ground_truth_find_optimal(
            atoms['coords'], gt['coord_exists'], coords, coord_mask,
            gt['coord_alt'], gt['coord_alt_mask'], gt['coord_is_symmetric'])

        def to_renamed(x, x_alt):
          if isinstance(x, tuple):
            assert isinstance(x_alt, tuple) and len(x) == 2
            return to_renamed(x[0], x_alt[0]), to_renamed(x[1], x_alt[1])

          shape = len(x.shape) - len(alt_is_better.shape)
          assert shape >= 0
          if shape > 0:
            ext = ' '.join(['()'] * shape)
            m = rearrange(alt_is_better, f'... i -> ... i {ext}')
          else:
            m = alt_is_better
          return (~m) * x + m * x_alt

        atom_affine = to_renamed(atom_affine, gt['atom_affine_alt'])
        coords = to_renamed(coords, gt['coord_alt'])
        coord_mask = to_renamed(coord_mask, gt['coord_alt_mask'])

      sidechain_loss = sidechain_fape_loss(atoms['frames'], atom_affine,
                                           atom_affine_mask, atoms['coords'],
                                           coords, coord_mask)

      sidechain_w = self.params.get('sidechain_w', 0.5)
      assert 0 <= sidechain_w <= 1
      return (1. - sidechain_w) * backbone_loss + sidechain_w * sidechain_loss

    def torsion_angle_loss(traj, gt, epsilon=1e-6):

      def yield_norm_angle_loss(pred_angles_list):
        for i, pred_angles in enumerate(pred_angles_list):
          angle_norm = torch.sqrt(epsilon + torch.sum(pred_angles**2, dim=-1))
          errors = torch.abs(angle_norm - 1.)
          avg_error = torch.mean(errors)
          logger.debug('FoldingHead.loss(norm_angle_loss: %d): %s', i,
                       avg_error)
          yield avg_error

      def yield_pred_angle_loss(pred_angles_list, true_angles):
        for i, pred_angles in enumerate(pred_angles_list):
          pred_angles = functional.l2_norm(pred_angles)
          errors = torch.sum((pred_angles - true_angles)**2, dim=-1)
          yield errors

      pred_angles_list = [x['atoms']['angles'] for x in traj]
      # angle norm loss
      norm_loss = sum(yield_norm_angle_loss(pred_angles_list)) / len(traj)

      # angle pred loss
      pred_loss = []
      if 'torsion_angles_mask' in gt:
        if 'torsion_angles' in gt:
          pred_loss.append(
              sum(
                  yield_pred_angle_loss(pred_angles_list,
                                        gt.get('torsion_angles'))) / len(traj))
        if 'torsion_angles_alt' in gt:
          pred_loss.append(
              sum(
                  yield_pred_angle_loss(pred_angles_list,
                                        gt.get('torsion_angles_alt'))) /
              len(traj))
      angles_mask = gt.get('torsion_angles_mask')
      if exists(angles_mask):
        angles_mask[..., :3] = 0  # chi angle only
      if pred_loss:
        if len(pred_loss) == 2:
          pred_loss = functional.masked_mean(value=torch.minimum(*pred_loss),
                                             mask=angles_mask)
        elif len(pred_loss) == 1:
          pred_loss = functional.masked_mean(value=pred_loss[0],
                                             mask=angles_mask)
        logger.debug('FoldingHead.loss(pred_angle_loss): %s', pred_loss)
      else:
        pred_loss = .0
        logger.debug('FoldingHead.loss(pred_angle_loss): %s', pred_loss)

      return (self.params.get('angle_norm_w', 0.01) * norm_loss +
              self.params.get('angle_pred_w', 0.5) * pred_loss)

    assert 'traj' in value
    # loss
    loss = (fape_loss(value['traj'], batch) +
            torsion_angle_loss(value['traj'], batch))

    return dict(loss=loss)


class LDDTHead(nn.Module):
  """Head to predict the pLDDT to be used as a per-residue configence score.
    """

  def __init__(self,
               dim,
               num_channels=None,
               buckets_num=50,
               min_resolution=.0,
               max_resolution=sys.float_info.max):
    super().__init__()
    dim, _ = embedd_dim_get(dim)
    num_channels = default(num_channels, dim)

    self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_channels),
                             nn.ReLU(), nn.Linear(num_channels, num_channels),
                             nn.ReLU(), nn.Linear(num_channels, buckets_num))
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
    pred_points = value['coords'][..., ca_idx, :]

    if 'coord' in batch:
      # Shape (b, l, d)
      true_points = batch['coord'][..., ca_idx, :]
      # Shape (b, l)
      points_mask = batch['coord_mask'][..., ca_idx]
    else:
      with torch.no_grad():
        true_points = pred_points
      points_mask = torch.ones(pred_points.shape[:-1],
                               device=pred_points.device,
                               dtype=torch.bool)

    with torch.no_grad():
      # Shape (b, l)
      lddt_ca = functional.lddt(pred_points, true_points, points_mask)
      # protect against out of range for lddt_ca == 1
      bin_index = torch.clamp(torch.floor(lddt_ca * self.buckets_num).long(),
                              max=self.buckets_num - 1)
      labels = F.one_hot(bin_index, self.buckets_num)

    errors = softmax_cross_entropy(labels=labels, logits=value['logits'])

    # Filter by resolution
    mask = torch.logical_and(self.min_resolution <= batch['resolution'],
                             batch['resolution'] < self.max_resolution)
    points_mask = torch.einsum('b,b ... -> b ...', mask, points_mask)
    loss = torch.sum(errors * points_mask) / (1e-6 + torch.sum(points_mask))
    logger.debug('LDDTHead.loss: %s', loss)
    return dict(loss=loss)

class PAEHead(nn.Module):
  """Head for protein-protein interaction
    """

  def __init__(self,
               dim,
               buckets_num=64,
               buckets_first_break=0.,
               buckets_last_break=31.,
               min_resolution=.0,
               max_resolution=sys.float_info.max):
    super().__init__()
    _, dim = embedd_dim_get(dim)

    self.buckets = torch.linspace(buckets_first_break,
                                  buckets_last_break,
                                  steps=buckets_num - 1)
    self.net = nn.Sequential(nn.Linear(dim, buckets_num))

    self.num_buckets = buckets_num
    self.min_resolution = min_resolution
    self.max_resolution = max_resolution

  def forward(self, headers, representations, batch):
    assert 'folding' in headers and 'frames' in headers['folding']

    x = representations['pair']
    breaks = self.buckets.to(x.device)
    return dict(logits=self.net(x),
                breaks=breaks,
                frames=headers['folding']['frames'])

  def loss(self, value, batch):
    assert 'frames' in value and 'logits' in value

    # (1, num_bins - 1)
    breaks = value['breaks']
    # (1, num_bins)
    logits = value['logits']

    backbone_affine, backbone_affine_mask = (batch.get('backbone_affine'),
                                             batch.get('backbone_affine_mask'))
    if not exists(backbone_affine_mask):
      if 'coord_mask' in batch:
        coord_mask = batch['coord_mask']
      else:
        assert 'coord_exists' in gt
        coord_mask = batch['coord_exists']
      n_idx = residue_constants.atom_order['N']
      ca_idx = residue_constants.atom_order['CA']
      c_idx = residue_constants.atom_order['C']
      backbone_affine_mask = torch.all(torch.stack(
          (coord_mask[..., c_idx], coord_mask[..., ca_idx],
           coord_mask[..., n_idx]),
          dim=-1) != 0,
                                       dim=-1)

    pred_frames = value['frames']
    with torch.no_grad():
      true_frames = default(backbone_affine, pred_frames)

    # Compute the squared error for each alignment.
    def to_local(affine):
      rotations, translations = affine
      points = translations

      # inverse frames
      rotations = rearrange(rotations, '... h w -> ... w h')
      translations = -torch.einsum('... w,... h w -> ... h', translations,
                                   rotations)
      return torch.einsum(
          '... j w,... i h w -> ... i j h', points, rotations) + rearrange(
              translations, '... i h -> ... i () h')

    # Shape (num_res, num_res)
    # First num_res are alignment frames, second num_res are the residues.
    with torch.no_grad():
      ae_ca = torch.sum(
          torch.square(to_local(pred_frames) - to_local(true_frames)), dim=-1)
    sq_breaks = torch.square(breaks)
    true_bins = torch.sum(ae_ca[..., None] > sq_breaks, dim=-1)

    errors = softmax_cross_entropy(labels=F.one_hot(true_bins,
                                                    self.num_buckets),
                                   logits=logits)

    # Filter by resolution
    mask = torch.logical_and(self.min_resolution <= batch['resolution'],
                             batch['resolution'] < self.max_resolution)
    sq_mask = backbone_affine_mask[..., None] * backbone_affine_mask[...,
                                                                     None, :]
    sq_mask = torch.einsum('b,b ... -> b ...', mask, sq_mask)
    loss = torch.sum(errors * sq_mask) / (1e-8 + torch.sum(sq_mask))
    logger.debug('PAEHead.loss: %s', loss)
    return dict(loss=loss)

class PPIHead(nn.Module):
  """Head for protein-protein interaction
    """

  def __init__(self, dim, contact_cutoff=8., min_prob=0.15):
    super().__init__()

    del dim
    self.contact_cutoff = contact_cutoff
    self.min_prob = min_prob

  def forward(self, headers, representations, batch):
    assert 'distogram' in headers
    assert 'seq_color' in batch

    logits = headers['distogram']['logits']
    breaks = headers['distogram']['breaks']
    seq_mask = batch['mask']
    seq_color = batch['seq_color']

    # Probability that distance between i and j less than or equal
    # contact_cutoff
    probs = F.softmax(logits, dim=-1)
    t = torch.sum(breaks <= self.contact_cutoff)
    probs = torch.sum(probs[..., :t + 1], dim=-1)

    # Mask out intra-contact and padding
    crd_mask = (seq_mask[..., None] * seq_mask[..., None, :])
    clr_mask = (seq_color[..., None] != seq_color[..., None, :])
    probs = probs * clr_mask * crd_mask

    # Denoise
    probs = F.threshold(probs, self.min_prob, 0.0)

    # Prob. that amini acid i has contact with the others
    probs = 1.0 - torch.exp(torch.sum(torch.log(1.0 - probs), dim=-1))

    # Denoise
    probs = F.threshold(probs, self.min_prob, 0.0)

    # Prob. that chain i has contact with the other chains
    b, n = seq_mask.shape[0], torch.amax(seq_color)
    probs = 1.0 - torch.exp(
        torch.scatter_add(torch.zeros(b, n, device=probs.device), -1,
                          seq_color.long() - 1, torch.log(1.0 - probs)))

    logger.debug('PPIHead.probs: %s', probs)
    return dict(probs=probs)

  def loss(self, value, batch):
    if 'ppi_label' in batch:
      assert 'ppi_label' in batch and 'ppi_mask' in batch

      probs = value['probs']
      targets, mask = batch['ppi_label'], batch['ppi_mask']
      logger.debug('PPIHead.targets: %s', targets)
      with autocast(enabled=False):
        errors = F.binary_cross_entropy(probs.float(),
                                        targets.float(),
                                        reduction='none')
      avg_error = functional.masked_mean(value=errors, mask=mask, epsilon=1e-6)
      logger.debug('PPIHead.loss: %s', avg_error)
      return dict(loss=avg_error)
    return None


class RobertaLMHead(nn.Module):
  """Head for Masked Language Modeling
    """

  def __init__(self, dim, loss_min=None, loss_max=None):
    super().__init__()

    dim, _ = embedd_dim_get(dim)
    self.project = nn.Sequential(
        nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim),
        nn.Linear(dim, len(residue_constants.restypes_with_x)))
    self.loss_min = loss_min
    self.loss_max = loss_max

  def forward(self, headers, representations, batch):
    assert 'single' in representations
    return dict(logits=self.project(representations['single']))

  def loss(self, value, batch):
    assert 'bert_mask' in batch
    assert 'logits' in value and 'true_seq' in batch

    logits, labels = value['logits'], batch['true_seq']
    #mask = rearrange(batch['bert_mask'], 'b l -> b l ()')
    mask = batch['bert_mask']

    errors = softmax_cross_entropy(labels=F.one_hot(labels, logits.shape[-1]),
                                   logits=logits)

    avg_error = functional.masked_mean(value=errors, mask=mask, epsilon=1e-6)
    logger.debug('RobertaLMHead.loss: %s', avg_error)
    if exists(self.loss_min) or exists(self.loss_max):
      avg_error = torch.clamp(avg_error, min=self.loss_min, max=self.loss_max)
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
    del dim

    self.params = kwargs

  def forward(self, headers, representations, batch):
    metrics = MetricDict()
    with torch.no_grad():
      if 'distogram' in headers and 'pseudo_beta' in batch:
        assert 'logits' in headers['distogram'] and 'breaks' in headers[
            'distogram']
        logits, breaks = headers['distogram']['logits'], headers['distogram'][
            'breaks']
        positions = batch['pseudo_beta']
        mask = batch['pseudo_beta_mask']

        cutoff = self.params.get('contact_cutoff', 8.0)
        t = torch.sum(breaks <= cutoff)
        pred = F.softmax(logits, dim=-1)
        pred = torch.sum(pred[..., :t + 1], dim=-1)
        truth = torch.cdist(positions, positions, p=2)
        metrics['contact'] = MetricDict()
        for b in range(truth.shape[0]):
          precision_list = contact_precision(
              pred[b],
              truth[b],
              mask=mask[b],
              ratios=self.params.get('contact_ratios'),
              ranges=self.params.get('contact_ranges'),
              cutoff=cutoff)
          for (i, j), ratio, precision in precision_list:
            i, j = default(i, 0), default(j, 'inf')
            k = f'[{i},{j})_{ratio}'
            if k in metrics['contact']:
              metrics['contact'][k] = torch.cat(
                  (metrics['contact'][k], precision), dim=-1)
            else:
              metrics['contact'][k] = precision
        if '[24,inf)_1' in metrics['contact']:
          logger.debug('MetricDictHead.contact.[24,inf]@L: %s',
                       metrics['contact']['[24,inf)_1'])
      if ('profile' in headers or
          'coevolution' in headers) and 'sequence_profile' in batch:
        labels, label_mask = batch['sequence_profile'], 1.0
        if 'sequence_profile_mask' in batch:
          label_mask = rearrange(batch['sequence_profile_mask'], 'c -> () () c')

        mask = batch['mask']
        if 'profile' in headers:
          assert 'logits' in headers['profile']
          logits = headers['profile']['logits']
          sim = softmax_cosine_similarity(logits=logits,
                                          labels=labels,
                                          mask=label_mask)
          avg_sim = functional.masked_mean(value=sim, mask=mask)
          logger.debug('MetricDictHead.profile.cosine: %s', avg_sim)
          metrics['profile'] = MetricDict()
          metrics['profile']['cosine'] = avg_sim
        if 'coevolution' in headers:
          metrics['coevolution'] = MetricDict()

          assert 'logits' in headers['coevolution']
          prob = F.softmax(headers['coevolution']['logits'], dim=-1)

          pred = torch.sum(prob, dim=-3)
          pred = pred / (torch.sum(pred, dim=-1, keepdims=True) + 1e-6)
          sim = F.cosine_similarity(pred * label_mask,
                                    labels * label_mask,
                                    dim=-1)
          avg_sim = functional.masked_mean(value=sim, mask=mask)
          logger.debug('MetricDictHead.coevolution.cosine: %s', avg_sim)
          metrics['coevolution']['cosine'] = avg_sim

          if 'msa' in batch:
            num_class = prob.shape[-1]

            pred = torch.argmax(prob, dim=-1)
            mask = F.one_hot(batch['msa'].long(),
                             num_classes=num_class) * label_mask
            if 'msa_row_mask' in batch:
              mask = torch.einsum('b m i c, b m -> b m i c', mask,
                                  batch['msa_row_mask'])
            avg_sim = functional.masked_mean(value=F.one_hot(
                pred, num_classes=num_class),
                                             mask=mask)
            logger.debug('MetricDictHead.coevolution.identity: %s', avg_sim)
            metrics['coevolution']['identity'] = avg_sim

            errors = -torch.sum(mask * torch.log(prob + 10e-8), dim=-1)
            avg_error = torch.exp(
                functional.masked_mean(value=errors,
                                       mask=torch.sum(mask, dim=-1)))
            logger.debug('MetricDictHead.coevolution.perplexity: %s', avg_error)
            metrics['coevolution']['perplexity'] = avg_error

      if 'folding' in headers and 'coords' in headers['folding'] and (
          'coord_mask' in batch or 'coord_exists' in batch):
        points, point_mask = headers['folding']['coords'], batch.get(
            'coord_exists', batch.get('coord_mask'))
        assert exists(point_mask)

        seq_index = batch.get('seq_index')
        ca_ca_distance_error = functional.between_ca_ca_distance_loss(
            points, point_mask, seq_index)
        metrics['violation'] = MetricDict()
        metrics['violation']['ca_ca_distance'] = ca_ca_distance_error
        logger.debug('MetricDictHead.violation.ca_ca_distance: %s',
                     ca_ca_distance_error)

        if 'coord' in batch:
          ca_idx = residue_constants.atom_order['CA']
          # Shape (b, l, d)
          pred_points = points[..., ca_idx, :]
          true_points = batch['coord'][..., ca_idx, :]
          points_mask = point_mask[..., ca_idx]

          # Shape (b, l)
          lddt_ca = functional.lddt(pred_points, true_points, points_mask)
          avg_lddt_ca = functional.masked_mean(value=lddt_ca, mask=points_mask)
          metrics['lddt'] = avg_lddt_ca
          logger.debug('MetricDictHead.lddt: %s', avg_lddt_ca)

    return dict(loss=metrics) if metrics else None

class FitnessHead(nn.Module):
  """Head to predict fitness.
    """

  def __init__(self, dim, num_var_as_ref=0):
    super().__init__()
    del dim

    self.sigma = nn.Linear(1, 1)

    assert num_var_as_ref >= 0
    self.num_var_as_ref = num_var_as_ref

  def forward(self, headers, representations, batch):
    assert 'coevolution' in headers
    eij, ei = headers['coevolution']['wij'], headers['coevolution']['bi']

    num_class = len(residue_constants.restypes_with_x_and_gap)
    if 'variant' in batch:
      variant = batch['variant']
      variant_mask = batch['variant_mask']
    else:
      assert 'seq' in batch
      variant = rearrange(batch['seq'], 'b i -> b () i')
      variant_mask = rearrange(batch['mask'], 'b i -> b () i')
    variant = F.one_hot(variant.long(), num_class)
    hi = torch.einsum(
        'b m j d,b i j c d -> b m i c',
        variant.float(),
        rearrange(eij, 'b i j (c d) -> b i j c d', c=num_class, d=num_class))
    logits = rearrange(ei, 'b i c -> b () i c') + hi
    logits = torch.einsum('b m i c, b m i c -> b m i', logits, variant.float())
    logits = torch.sum(self.sigma(logits[..., None]), dim=-1)
    variant_logit = torch.sum(
        (logits - logits[:, :1, ...]) * variant_mask, dim=-1)
    # if self.training:
    #   variant_diff = torch.clamp(variant_diff, max=0.0)
    # variant_logit = torch.sum(self.sigma(variant_diff), dim=-1)
    return dict(logits=logits, variant_logit=variant_logit)

  def loss(self, value, batch):
    logits, logits_ref = value['logits'], None
    if 'variant' in batch:
      variant_mask = batch['variant_mask']
      variant_label = batch['variant_label']

      if self.num_var_as_ref > 0:
        # minimum of variants in batch
        label_mask = torch.sum(variant_mask, dim=-1) > 0
        label_num = torch.min(torch.sum(label_mask, dim=-1)).item()

        num_var_as_ref = min(self.num_var_as_ref, label_num)
        if num_var_as_ref > 0:
          variant_label = batch['variant_label'][:, :label_num] * label_mask

          b, _, n = logits.shape
          # sample reference based on variant_label
          ref_idx = torch.multinomial(variant_label + 1e-3,
                                      num_var_as_ref,
                                      replacement=False)
          ref_idx = torch.cat((torch.zeros(
              (b, 1), dtype=ref_idx.dtype, device=logits.device), ref_idx),
                              dim=-1)
          logger.debug("FitnessHead.ref_idx: %s", ref_idx)

          logits_ref = torch.gather(
              logits, 1, repeat(ref_idx, 'b m -> b m i', i=logits.shape[-1]))
          mask_ref = torch.gather(
              variant_mask, 1,
              repeat(ref_idx, 'b m -> b m i', i=logits.shape[-1]))
          label_ref = torch.gather(variant_label, 1, ref_idx)
    else:
      variant_mask = rearrange(torch.zeros_like(batch['mask']), 'b i -> b () i')
      b = variant_mask.shape[0]
      variant_label = torch.ones((b, 1), device=variant_mask.device)

    if not exists(logits_ref):
      logits_ref = logits[:, :1, ...]
      mask_ref = variant_mask[:, :1, ...]
      label_ref = variant_label[:, :1, ...]

    variant_logit = rearrange(logits, 'b m i -> b m () i') - rearrange(
        logits_ref, 'b n i -> b () n i')
    variant_mask = rearrange(variant_mask, 'b m i -> b m () i') * rearrange(
        mask_ref, 'b n i -> b () n i')
    variant_label = rearrange(variant_label, 'b m -> b m ()') - rearrange(
        label_ref, 'b n -> b () n')
    variant_label = torch.clamp((1. + variant_label) / 2, min=0, max=1)
    variant_logit = torch.sum(variant_logit * variant_mask, dim=-1)
    errors = F.binary_cross_entropy_with_logits(variant_logit,
                                                variant_label,
                                                reduction='none')
    variant_mask = torch.sum(variant_mask, dim=-1) > 0
    avg_error = functional.masked_mean(value=errors, mask=variant_mask)
    logger.debug('FitnessHead.loss: %s', avg_error)
    return dict(loss=avg_error)


class SequenceProfileHead(nn.Module):
  """Head to predict sequence profile.
    """

  def __init__(self, dim, input_dim=None, single_repr=None, num_pivot=None):
    super().__init__()
    dim, _ = embedd_dim_get(dim)

    if not exists(input_dim):
      input_dim = dim
    if not exists(single_repr):
      single_repr = 'struct_module'
    assert single_repr in ('struct_module', 'mlm')
    self.single_repr = single_repr
    self.num_pivot = num_pivot

    self.project = nn.Sequential(
        nn.Linear(input_dim, dim), nn.GELU(), nn.LayerNorm(dim),
        nn.Linear(dim, len(residue_constants.restypes_with_x_and_gap)))

  def forward(self, headers, representations, batch):
    assert 'sequence_profile' in batch

    if self.single_repr == 'mlm':
      assert 'mlm' in representations and 'representations' in representations[
          'mlm']
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

    errors = softmax_kl_diversity(labels=labels, logits=logits, mask=label_mask)

    errors, w = _make_dynamic_errors(errors, batch, self.num_pivot)
    avg_error = functional.masked_mean(value=errors, mask=mask, epsilon=1e-6)
    logger.debug('SequenceProfileHead.loss(%s): %s', w, avg_error)
    return dict(loss=avg_error)


class TMscoreHead(nn.Module):
  """Head to predict TM-score.
    """

  def __init__(self, dim, num_atoms=3):
    super().__init__()
    del dim

    self.num_atoms = num_atoms
    assert self.num_atoms in [3, 14]

  def forward(self, headers, representations, batch):
    assert 'folding' in headers and 'coords' in headers['folding']

    if 'coord' in batch and 'coord_mask' in batch:
      pred_points = headers['folding']['coords'][..., :self.num_atoms, :]
      true_points = batch['coord'][..., :self.num_atoms, :]
      coord_mask = batch['coord_mask'][..., :self.num_atoms]
      with torch.no_grad():
        tms = batched_tmscore(pred_points, true_points, coord_mask,
                              batch['mask'])
      return dict(loss=tms)
    return None


class ViolationHead(nn.Module):
  """Head to structure violations.
    """

  def __init__(self, dim):
    super().__init__()
    del dim

  def forward(self, headers, representations, batch):
    assert 'folding' in headers and 'coords' in headers['folding']
    return dict(coords=headers['folding']['coords'])

  def loss(self, value, batch):
    assert 'coords' in value
    seq, mask = batch['seq'], batch['mask']
    seq_index = batch.get('seq_index')
    if not exists(seq_index):
      b, n = seq.shape[:2]
      seq_index = repeat(torch.arange(n, device=seq.device), 'i -> b i', b=b)
    if 'coord_mask' in batch or 'coord_exists' in batch:
      points, point_mask = value['coords'], batch.get('coord_exists',
                                                      batch.get('coord_mask'))
      assert exists(point_mask)

      # loss_dict.update(ca_ca_distance_loss = functional.between_ca_ca_distance_loss(
      #         points, point_mask, seq_index))
      loss_dict = {}

      loss_dict.update(
          functional.between_residue_bond_loss(points,
                                               point_mask,
                                               seq_index,
                                               seq,
                                               loss_only=True))
      loss_dict.update(
          functional.between_residue_clash_loss(points,
                                                point_mask,
                                                seq_index,
                                                seq,
                                                loss_only=True))
      loss_dict.update(
          functional.within_residue_clash_loss(points,
                                               point_mask,
                                               seq_index,
                                               seq,
                                               loss_only=True))

      for k, v in loss_dict.items():
        logger.debug('ViolationHead.%s: %s', k, v)
      return dict(loss=sum(loss_dict.values()))
    return None


class HeaderBuilder:
  _headers = dict(coevolution=CoevolutionHead,
                  confidence=ConfidenceHead,
                  contact=ContactHead,
                  distogram=DistogramHead,
                  fitness=FitnessHead,
                  folding=FoldingHead,
                  lddt=LDDTHead,
                  metric=MetricDictHead,
                  pae=PAEHead,
                  ppi=PPIHead,
                  profile=SequenceProfileHead,
                  roberta=RobertaLMHead,
                  tmscore=TMscoreHead,
                  violation=ViolationHead)

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
