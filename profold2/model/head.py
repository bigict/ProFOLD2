import sys
import logging

import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F
from einops import rearrange, repeat

from profold2.common import residue_constants
from profold2.model import commons, functional, folding
from profold2.utils import default, env, exists

logger = logging.getLogger(__name__)


def binary_focal_loss_weight(probs, labels, gammar):
  assert gammar > 0
  p = probs * labels + (1. - probs) * (1. - labels)
  return (1. - p) ** gammar


def sigmoid_cross_entropy(probs, labels, gammar=0):
  errors = F.binary_cross_entropy(probs, labels, reduction='none')
  if gammar > 0:  # focal loss enabled.
    errors = errors * binary_focal_loss_weight(probs, labels, gammar)
  return errors


def sigmoid_cross_entropy_with_logits(
    logits, labels, alpha=None, gammar=0, epsilon=1e-7
):
  errors = F.binary_cross_entropy_with_logits(
      logits, labels, reduction='none', pos_weight=alpha
  )
  if gammar > 0:  # focal loss enabled.
    probs = torch.clamp(torch.sigmoid(logits), min=epsilon, max=1. - epsilon)
    errors = errors * binary_focal_loss_weight(probs, labels, gammar)
  return errors


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
      F.kl_div(F.log_softmax(logits, dim=-1), labels, reduction='none') * mask, dim=-1
  )
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

    if torch.any(flat_cloud_mask):
      # rotate / align
      coords_aligned, labels_aligned = functional.kabsch_align(
          rearrange(pred_points[b], 'i c d -> (i c) d')[flat_cloud_mask],
          rearrange(true_points[b], 'i c d -> (i c) d')[flat_cloud_mask]
      )

      return functional.tmscore(
          coords_aligned, labels_aligned, n=torch.sum(mask[b], dim=-1)
      )
    return torch.as_tensor(0., device=flat_cloud_mask.device)

  # tms = sum(map(_yield_tmscore, range(pred_points.shape[0])))
  tms = torch.stack(list(map(_yield_tmscore, range(pred_points.shape[0]))))
  return tms


class ConfidenceHead(nn.Module):
  """Head to predict confidence.
    """
  def __init__(self, dim):
    super().__init__()

  def forward(self, headers, representations, batch):
    metrics = {}
    with torch.no_grad():
      mask = batch['mask']
      if 'lddt' in headers and 'logits' in headers['lddt']:
        metrics['plddt'] = functional.plddt(headers['lddt']['logits']) * 100
      if 'plddt' in metrics:
        # metrics['loss'] = functional.masked_mean(value=metrics['plddt'], mask=mask)
        metrics['loss'] = functional.masked_mean(
            value=metrics['plddt'], mask=mask, dim=-1
        )
        logger.debug('ConfidenceHead.loss: %s', metrics['loss'])
      if 'pae' in headers and 'logits' in headers['pae']:
        logits, breaks = headers['pae']['logits'], headers['pae']['breaks']
        metrics['pae'], metrics['mae'] = functional.pae(
            logits, breaks, mask=mask, return_mae=True
        )
        metrics['ptm'] = functional.ptm(logits, breaks, mask=mask)
        metrics['iptm'] = functional.ptm(
           logits, breaks, mask=mask, seq_color=batch.get('seq_color')
        )
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
    assert not self.training or (
        'mlm' in representations and 'contacts' in representations['mlm']
    )
    if 'mlm' in representations and 'contacts' in representations['mlm'] and exists(
        representations['mlm']['contacts']
    ):
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

      square_mask, square_weight = (
          rearrange(mask, '... i -> ... i ()') * rearrange(mask, '... j -> ... () j'),
          1.0
      )
      square_mask = torch.triu(square_mask, diagonal=self.diagonal
                              ) + torch.tril(square_mask, diagonal=-self.diagonal)
      if 'coord_plddt' in batch:
        ca_idx = residue_constants.atom_order['CA']
        plddt_weight = torch.minimum(
            rearrange(batch['coord_plddt'][..., ca_idx], '... i->... i ()'),
            rearrange(batch['coord_plddt'][..., ca_idx], '... j->... () j')
        )
        if batch.get('coord_plddt_use_weighted_mask', True):
          square_mask *= plddt_weight
        else:
          square_weight = plddt_weight

      avg_error = functional.masked_mean(
          value=errors * square_weight, mask=square_mask, epsilon=1e-6
      )
      logger.debug('ContactHead.loss: %s', avg_error)
      if exists(self.loss_min) or exists(self.loss_max):
        avg_error = torch.clamp(avg_error, min=self.loss_min, max=self.loss_max)
      return dict(loss=avg_error)
    return None


class CoevolutionHead(nn.Module):
  """Head to predict Co-evolution.
    """
  def __init__(
      self,
      dim,
      gating=True,
      mask='-',
      alpha=0.0,
      beta=0.0,
      gammar=0.0,
      num_states=None,
      apc=False,
      num_pivot=1024,
      focal_loss=0,
      loss_min=None,
      loss_max=None
  ):
    super().__init__()
    del alpha, beta, gammar  # Deprecated
    dim_single, dim_pairwise = commons.embedd_dim_get(dim)

    num_class = len(residue_constants.restypes_with_x_and_gap)
    self.single = nn.Sequential(
        nn.Linear(dim_single, dim_single), nn.GELU(), nn.LayerNorm(dim_single),
        nn.Linear(dim_single, num_class)
    )
    if gating:
      if exists(num_states):
        assert num_states >= 1
        self.states = nn.Parameter(torch.randn(num_states, num_class, num_class))
      else:
        num_states = 1
        self.states = nn.Parameter(torch.randn(num_class, num_class))
      self.pairwise = nn.Sequential(
          nn.LayerNorm(dim_pairwise), nn.Linear(dim_pairwise, num_states)
      )

      commons.init_linear_(self.pairwise[1], b=1.)
    else:
      self.states = None
      self.pairwise = nn.Sequential(
          nn.LayerNorm(dim_pairwise), nn.Linear(dim_pairwise, num_class**2), nn.ReLU()
      )
    m = functional.make_mask(residue_constants.restypes_with_x_and_gap, mask)
    self.register_buffer('mask', m, persistent=False)

    self.apc = apc
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
    num_class = self.mask.shape[0]

    si, zij = representations['single'], representations['pair']
    zij = (zij + rearrange(zij, 'b i j d -> b j i d')) * 0.5  # symmetrize

    ei, eij = self.single(si), self.pairwise(zij)

    # FIX: batch_size > 1
    mij = (
        rearrange(batch['mask'], 'b i -> b i () ()') *
        rearrange(batch['mask'], 'b j -> b () j ()')
    )
    if exists(self.states):  # Average Product Correction
      if self.apc:
        eij = functional.apc(eij * mij, dim=(-3, -2))
      eij = F.sigmoid(eij)
    eij = eij * mij * rearrange(
        1 - torch.eye(zij.shape[-2], device=zij.device), 'i j -> i j ()'
    )  # eii = 0

    # pseudo msa if not exists
    if 'msa' in batch:
      msa = F.one_hot(batch['msa'].long(), num_class).float()
    else:
      assert 'seq' in batch
      msa = rearrange(
          F.one_hot(batch['seq'].long(), num_class).float(), 'b i d -> b () i d'
      )
    ret = dict(bi=ei)
    if exists(self.states):  # multi-state potts model
      states = self.states
      if len(states.shape) == 2:  # back compatible
        states = rearrange(states, 'c d -> () c d')
      states = (states + rearrange(states, 'q c d -> q d c')) * 0.5
      hi = torch.einsum(
          'b m j d,b i j q,q c d,d -> b m i c', msa, eij, states, self.mask
      )
      ret.update(wab=states)
      # eij = torch.einsum('b i j q,q c d -> b i j c d', eij, states)
    else:  # native potts model
      eij = rearrange(eij, 'b i j (c d) -> b i j c d', c=num_class, d=num_class)
      hi = torch.einsum('b m j d,b i j c d,d -> b m i c', msa, eij, self.mask)
    ret.update(wij=eij)

    logits = rearrange(ei, 'b i c -> b () i c') + hi
    ret.update(logits=logits, mask=self.mask)
    return ret

  def loss(self, value, batch):
    """Log loss of a msa rebuilding."""
    num_class = self.mask.shape[0]

    logits = value['logits']
    labels = F.one_hot(batch['msa'].long(), num_class)
    logger.debug('CoevolutionHead.loss.logits: %s, %s', logits.shape, self.focal_loss)

    assert len(logits.shape) == 4
    assert 'msa' in batch
    errors = softmax_cross_entropy(
        labels=labels, logits=logits, mask=self.mask, gammar=self.focal_loss
    )
    mask = torch.einsum(
        'b i,b m i c,c -> b m i', batch['mask'].float(), labels.float(), self.mask
    )
    if 'msa_mask' in batch:
      mask = mask * batch['msa_mask']  # (b m i)

    errors, w = _make_dynamic_errors(errors, batch, self.num_pivot)
    avg_error = functional.masked_mean(value=errors, mask=mask, epsilon=1e-6)
    logger.debug('CoevolutionHead.loss(%s): %s', w, avg_error)

    if exists(self.loss_min) or exists(self.loss_max):
      avg_error = torch.clamp(avg_error, min=self.loss_min, max=self.loss_max)
    return dict(loss=avg_error)


class DistogramHead(nn.Module):
  """Head to predict a distogram.
    """
  def __init__(
      self, dim, buckets_first_break, buckets_last_break, buckets_num, focal_loss=0
  ):
    super().__init__()
    _, dim = commons.embedd_dim_get(dim)

    self.num_buckets = buckets_num
    buckets = torch.linspace(
        buckets_first_break, buckets_last_break, steps=buckets_num - 1
    )
    self.register_buffer('buckets', buckets, persistent=False)
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
    return dict(logits=self.net(trunk_embeds), breaks=self.buckets)

  def loss(self, value, batch):
    """Log loss of a distogram."""
    logits, breaks = value['logits'], value['breaks']
    assert len(logits.shape) == 4
    if 'pseudo_beta' in batch:
      positions = batch['pseudo_beta']
      mask = batch['pseudo_beta_mask']

      assert positions.shape[-1] == 3

      sq_breaks = torch.square(breaks)

      dist2 = torch.sum(
          torch.square(
              rearrange(positions, 'b i c -> b i () c') -
              rearrange(positions, 'b j c -> b () j c')
          ),
          dim=-1,
          keepdim=True
      )

      true_bins = torch.sum(dist2 > sq_breaks, axis=-1)
      errors = softmax_cross_entropy(
          labels=F.one_hot(true_bins, self.num_buckets),
          logits=logits,
          gammar=self.focal_loss
      )
    else:
      mask = torch.zeros_like(batch['seq'], dtype=torch.bool)
      with torch.no_grad():
        labels = F.softmax(logits, dim=-1)
      errors = softmax_kl_diversity(labels=labels, logits=logits)

    square_mask, square_weight = (
        rearrange(mask, '... i -> ... i ()') * rearrange(mask, '... j -> ... () j'), 1.0
    )
    if 'coord_plddt' in batch:
      ca_idx = residue_constants.atom_order['CA']
      plddt_weight = torch.minimum(
          rearrange(batch['coord_plddt'][..., ca_idx], '... i->... i ()'),
          rearrange(batch['coord_plddt'][..., ca_idx], '... j->... () j')
      )
      if batch.get('coord_plddt_use_weighted_mask', True):
        square_mask *= plddt_weight
      else:
        square_weight = plddt_weight

    if 'loss.distogram.w' in batch:
      logger.debug('DistogramHead.loss.w: %s', batch['loss.distogram.w'])
      errors = torch.einsum('b ...,b -> b ...', errors, batch['loss.distogram.w'])
    avg_error = functional.masked_mean(
        value=errors * square_weight, mask=square_mask, epsilon=1e-6
    )
    logger.debug('DistogramHead.loss: %s', avg_error)
    return dict(loss=avg_error)


class FoldingHead(nn.Module):
  """Head to predict 3d struct.
    """
  def __init__(
      self,
      dim,
      structure_module_depth,
      structure_module_heads,
      point_value_dim=8,
      qkv_use_bias=False,
      fape_min=1e-6,
      fape_max=15,
      fape_z=15,
      fape_backbone_clamp_ratio=1.0,
      fape_sidechain_clamp_ratio=1.0,
      dropout=.0,
      position_scale=1.0,
      **params
  ):
    super().__init__()
    self.struct_module = folding.StructureModule(
        dim,
        structure_module_depth,
        structure_module_heads,
        point_value_dim=point_value_dim,
        qkv_use_bias=qkv_use_bias,
        dropout=dropout,
        position_scale=position_scale
    )

    self.fape_min = fape_min
    self.fape_max = fape_max
    self.fape_z = fape_z
    self.fape_backbone_clamp_ratio = fape_backbone_clamp_ratio
    self.fape_sidechain_clamp_ratio = fape_sidechain_clamp_ratio

    self.params = params

  def forward(self, headers, representations, batch):
    #(rotations, translations), act = self.struct_module(representations, batch)
    outputs = self.struct_module(representations, batch)
    (rotations, translations), act, atoms = map(
        lambda key: outputs[-1][key], ('frames', 'act', 'atoms')
    )

    return dict(
        frames=(rotations, translations),
        coords=atoms['coords'],
        representations=dict(single=act),
        traj=outputs
    )

  def loss(self, value, batch):
    def backbone_fape_loss(pred_frames_list, gt_frames, frames_mask):
      assert pred_frames_list and exists(frames_mask)

      dij_weight = None
      if 'coord_plddt' in batch:
        ca_idx = residue_constants.atom_order['CA']
        dij_weight = torch.minimum(
            rearrange(batch['coord_plddt'][..., ca_idx], '... i -> ... i ()'),
            rearrange(batch['coord_plddt'][..., ca_idx], '... j -> ... () j')
        )

      if 'loss.folding.w' in batch:
        logger.debug('FoldingHead.loss.w: %s', batch['loss.folding.w'])
        wij = rearrange(batch['loss.folding.w'], '... -> ... () ()')
        dij_weight = dij_weight * wij if exists(dij_weight) else wij

      for i, pred_frames in enumerate(pred_frames_list):
        with torch.no_grad():
          true_frames = default(gt_frames, pred_frames)

        clamp_ratio = self.fape_backbone_clamp_ratio
        if 0 < clamp_ratio <= 1:
          # clamp pairs between protein monomer only.
          clamp_ratio = torch.where(
              torch.logical_and(
                  batch['seq'] >= residue_constants.prot_from_idx,
                  batch['seq'] <= residue_constants.prot_to_idx
              ),
              clamp_ratio,
              0.
          )
          if 'seq_color' in batch:
            clamp_ratio = torch.where(
                batch['seq_color'][..., :, None] == batch['seq_color'][..., None, :],
                clamp_ratio,
                0.
            )

        _, pred_points = pred_frames
        _, true_points = true_frames
        r = functional.fape(
            pred_frames,
            true_frames,
            frames_mask,
            pred_points,
            true_points,
            frames_mask,
            self.fape_max,
            clamp_ratio=clamp_ratio,
            dij_weight=dij_weight,
            use_weighted_mask=batch.get('coord_plddt_use_weighted_mask', False)
        ) / self.fape_z
        logger.debug('FoldingHead.loss(backbone_fape_loss: %d): %s', i, r)
        yield r

    def sidechain_fape_loss(
        pred_frames, true_frames, frames_mask, pred_points, true_points, point_mask
    ):
      def frames_to_fape_shape(frames):
        rotations, translations = frames
        return (
            rearrange(rotations, '... i c h w -> ... (i c) h w'),
            rearrange(translations, '... i c h -> ... (i c) h')
        )

      def points_to_fape_shape(points):
        return rearrange(points, '... i c d -> ... (i c) d')

      def mask_to_fape_shape(masks):
        return rearrange(masks, '... i c -> ... (i c)')

      dij_weight = None
      if 'coord_plddt' in batch:
        # (N, 8, 3, 14)
        group_atom14_idx = F.one_hot(
            functional.batched_gather(
                residue_constants.restype_rigid_group_atom14_idx, batch['seq']
            ).long(), residue_constants.atom14_type_num
        )
        # (N, 8)
        group_atom14_plddt = torch.amin(
            torch.einsum(
                '... g m n, ... n -> ... g m', group_atom14_idx.float(),
                batch['coord_plddt']
            ),
            dim=-1
        )
        dij_weight = torch.minimum(
            rearrange(mask_to_fape_shape(group_atom14_plddt), '... i -> ... i ()'),
            rearrange(mask_to_fape_shape(batch['coord_plddt']), '... j -> ... () j')
        )
      if 'loss.folding.w' in batch:
        wij = rearrange(batch['loss.folding.w'], '... -> ... () ()')
        dij_weight = dij_weight * wij if exists(dij_weight) else wij

      with torch.no_grad():
        true_frames = default(true_frames, pred_frames)
        true_points = default(true_points, pred_points)

      r = functional.fape(
          frames_to_fape_shape(pred_frames),
          frames_to_fape_shape(true_frames),
          mask_to_fape_shape(frames_mask),
          points_to_fape_shape(pred_points),
          points_to_fape_shape(true_points),
          mask_to_fape_shape(point_mask),
          self.fape_max,
          clamp_ratio=self.fape_sidechain_clamp_ratio,
          dij_weight=dij_weight,
          use_weighted_mask=batch.get('coord_plddt_use_weighted_mask', False)
      ) / self.fape_z
      logger.debug('FoldingHead.loss(sidechain_fape_loss): %s', r)
      return r

    def fape_loss(traj, gt):
      coords = gt.get('coord')
      if 'coord_mask' in gt:
        coord_mask = gt['coord_mask']
      else:
        assert 'coord_exists' in gt
        coord_mask = gt['coord_exists']
      backbone_affine, backbone_affine_mask = (
          gt.get('backbone_affine'), gt.get('backbone_affine_mask')
      )
      if not exists(backbone_affine_mask):
        n_idx = residue_constants.atom_order['N']
        ca_idx = residue_constants.atom_order['CA']
        c_idx = residue_constants.atom_order['C']
        backbone_affine_mask = torch.all(
            torch.stack(
                (
                    coord_mask[..., c_idx], coord_mask[..., ca_idx], coord_mask[...,
                                                                                n_idx]
                ),
                dim=-1
            ) != 0,
            dim=-1
        )
      # backbone loss
      backbone_loss = sum(
          backbone_fape_loss(
              [x['frames'] for x in traj], backbone_affine, backbone_affine_mask
          )
      ) / len(traj)
      # sidechine loss
      atoms = traj[-1]['atoms']
      atom_affine, atom_affine_mask = (
          gt.get('atom_affine'), gt.get('atom_affine_mask')
      )
      if not exists(atom_affine_mask):
        assert 'atom_affine_exists' in gt, gt.keys()
        atom_affine_mask = gt['atom_affine_exists']

      if exists(atom_affine):
        assert exists(coords), gt.keys()
        # Renamed frames
        alt_is_better = functional.symmetric_ground_truth_find_optimal(
            atoms['coords'], gt['coord_exists'], coords, coord_mask, gt['coord_alt'],
            gt['coord_alt_mask'], gt['coord_is_symmetric']
        )

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

      sidechain_loss = sidechain_fape_loss(
          atoms['frames'], atom_affine, atom_affine_mask, atoms['coords'], coords,
          coord_mask
      )

      sidechain_w = self.params.get('sidechain_w', 0.5)
      assert 0 <= sidechain_w <= 1
      return (1. - sidechain_w) * backbone_loss + sidechain_w * sidechain_loss

    def torsion_angle_loss(traj, gt, epsilon=1e-6):
      def yield_norm_angle_loss(pred_angles_list):
        for i, pred_angles in enumerate(pred_angles_list):
          angle_norm = torch.sqrt(epsilon + torch.sum(pred_angles**2, dim=-1))
          errors = torch.abs(angle_norm - 1.)
          avg_error = torch.mean(errors)
          logger.debug('FoldingHead.loss(norm_angle_loss: %d): %s', i, avg_error)
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
              sum(yield_pred_angle_loss(pred_angles_list, gt.get('torsion_angles'))) /
              len(traj)
          )
        if 'torsion_angles_alt' in gt:
          pred_loss.append(
              sum(
                  yield_pred_angle_loss(pred_angles_list, gt.get('torsion_angles_alt'))
              ) / len(traj)
          )
      angles_mask = gt.get('torsion_angles_mask')
      if exists(angles_mask):
        angles_mask[..., :3] = 0  # chi angle only
      if pred_loss:
        if len(pred_loss) == 2:
          pred_loss = functional.masked_mean(
              value=torch.minimum(*pred_loss), mask=angles_mask
          )
        elif len(pred_loss) == 1:
          pred_loss = functional.masked_mean(value=pred_loss[0], mask=angles_mask)
        logger.debug('FoldingHead.loss(pred_angle_loss): %s', pred_loss)
      else:
        pred_loss = .0
        logger.debug('FoldingHead.loss(pred_angle_loss): %s', pred_loss)

      return (
          self.params.get('angle_norm_w', 0.01) * norm_loss +
          self.params.get('angle_pred_w', 0.5) * pred_loss
      )

    assert 'traj' in value
    # loss
    loss = (fape_loss(value['traj'], batch) + torsion_angle_loss(value['traj'], batch))

    return dict(loss=loss)


class LDDTHead(nn.Module):
  """Head to predict the pLDDT to be used as a per-residue configence score.
    """
  def __init__(
      self,
      dim,
      num_channels=None,
      buckets_num=50,
      min_resolution=.0,
      max_resolution=sys.float_info.max
  ):
    super().__init__()
    dim, _ = commons.embedd_dim_get(dim)
    num_channels = default(num_channels, dim)

    self.net = nn.Sequential(
        nn.LayerNorm(dim), nn.Linear(dim, num_channels), nn.ReLU(),
        nn.Linear(num_channels, num_channels), nn.ReLU(),
        nn.Linear(num_channels, buckets_num)
    )
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
      points_mask = torch.ones(
          pred_points.shape[:-1], device=pred_points.device, dtype=torch.bool
      )

    with torch.no_grad():
      # Shape (b, l)
      lddt_ca = functional.lddt(pred_points, true_points, points_mask)
      # protect against out of range for lddt_ca == 1
      bin_index = torch.clamp(
          torch.floor(lddt_ca * self.buckets_num).long(), max=self.buckets_num - 1
      )
      labels = F.one_hot(bin_index, self.buckets_num)

    errors = softmax_cross_entropy(labels=labels, logits=value['logits'])

    # Filter by resolution
    mask = torch.logical_and(
        self.min_resolution <= batch['resolution'],
        batch['resolution'] < self.max_resolution
    )
    points_mask = torch.einsum('b,b ... -> b ...', mask, points_mask)
    loss = torch.sum(errors * points_mask) / (1e-6 + torch.sum(points_mask))
    logger.debug('LDDTHead.loss: %s', loss)
    return dict(loss=loss)


class PAEHead(nn.Module):
  """Head for protein-protein interaction
    """
  def __init__(
      self,
      dim,
      buckets_num=64,
      buckets_first_break=0.,
      buckets_last_break=31.,
      min_resolution=.0,
      max_resolution=sys.float_info.max
  ):
    super().__init__()
    _, dim = commons.embedd_dim_get(dim)

    buckets = torch.linspace(
        buckets_first_break, buckets_last_break, steps=buckets_num - 1
    )
    self.register_buffer('buckets', buckets, persistent=False)
    self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, buckets_num, bias=False))

    self.num_buckets = buckets_num
    self.min_resolution = min_resolution
    self.max_resolution = max_resolution

  def forward(self, headers, representations, batch):
    assert 'folding' in headers and 'frames' in headers['folding']

    x = representations['pair']
    return dict(
        logits=self.net(x), breaks=self.buckets, frames=headers['folding']['frames']
    )

  def loss(self, value, batch):
    assert 'frames' in value and 'logits' in value

    # (1, num_bins - 1)
    breaks = value['breaks']
    # (1, num_bins)
    logits = value['logits']

    backbone_affine, backbone_affine_mask = (
        batch.get('backbone_affine'), batch.get('backbone_affine_mask')
    )
    if not exists(backbone_affine_mask):
      if 'coord_mask' in batch:
        coord_mask = batch['coord_mask']
      else:
        assert 'coord_exists' in gt
        coord_mask = batch['coord_exists']
      n_idx = residue_constants.atom_order['N']
      ca_idx = residue_constants.atom_order['CA']
      c_idx = residue_constants.atom_order['C']
      backbone_affine_mask = torch.all(
          torch.stack(
              (coord_mask[..., c_idx], coord_mask[..., ca_idx], coord_mask[..., n_idx]),
              dim=-1
          ) != 0,
          dim=-1
      )

    pred_frames = value['frames']
    with torch.no_grad():
      true_frames = default(backbone_affine, pred_frames)

    # Compute the squared error for each alignment.
    def to_local(affine):
      rotations, translations = affine
      points = translations

      # inverse frames
      rotations = rearrange(rotations, '... h w -> ... w h')
      translations = -torch.einsum('... w,... h w -> ... h', translations, rotations)
      return torch.einsum('... j w,... i h w -> ... i j h', points,
                          rotations) + rearrange(translations, '... i h -> ... i () h')

    # Shape (num_res, num_res)
    # First num_res are alignment frames, second num_res are the residues.
    with torch.no_grad():
      ae_ca = torch.sum(
          torch.square(to_local(pred_frames) - to_local(true_frames)), dim=-1
      )
    sq_breaks = torch.square(breaks)
    true_bins = torch.sum(ae_ca[..., None] > sq_breaks, dim=-1)

    errors = softmax_cross_entropy(
        labels=F.one_hot(true_bins, self.num_buckets), logits=logits
    )

    # Filter by resolution
    mask = torch.logical_and(
        self.min_resolution <= batch['resolution'],
        batch['resolution'] < self.max_resolution
    )
    sq_mask = backbone_affine_mask[..., None] * backbone_affine_mask[..., None, :]
    sq_mask = torch.einsum('b,b ... -> b ...', mask, sq_mask)
    loss = torch.sum(errors * sq_mask) / (1e-8 + torch.sum(sq_mask))
    logger.debug('PAEHead.loss: %s', loss)
    return dict(loss=loss)


class PairingHead(nn.Module):
  """Head to predict a nucleic acid base pairing.
    """
  def __init__(self, dim, alpha=0.5, focal_loss=0):
    super().__init__()
    _, dim = commons.embedd_dim_get(dim)

    # NOTE: 0 for unpaired
    self.net = nn.Sequential(
        nn.LayerNorm(dim), nn.Linear(dim, len(residue_constants.bptypes))
    )
    # NOTE: https://nakb.org/ndbmodule/bp-catalog/
    # pairing = torch.zeros(
    #     len(residue_constants.restypes_with_x), 3, dtype=torch.float32
    # )
    # a, c, g, u = [
    #     residue_constants.restype_order_with_x[(b, residue_constants.RNA)]
    #     for b in 'acgu'
    # ]
    # for i, (b1, b2) in enumerate([(a, u), (c, g), (g, u)]):
    #   pairing[b1, i] = 1.0
    #   pairing[b2, i] = 1.0
    # self.register_buffer('pairing', pairing, persistent=False)

    self.alpha = alpha
    self.focal_loss = focal_loss

  def forward(self, headers, representations, batch):
    """Builds ParingHead module.

        Arguments:
         representations: Dictionary of representations, must contain:
           * 'pair': pair representation, shape [N_res, N_res, c_z].
         batch: Batch, unused.

        Returns:
         Dictionary containing:
           * logits: logits for distogram, shape [N_res, N_res, N_bins].
        """
    x = self.net(representations['pair'])
    x = (x + rearrange(x, 'b i j d -> b j i d')) * 0.5  # symmetrize

    # build constrains
    # 1. Only three types of nucleotide combinations can form base pairs
    # 2. No sharp loop within three bases. For any adjacent bases within a
    #    distance of three nucleotides, they cannot form pairs with each
    #    other. for all |i - j| < 4, M_ij = 0
    # m = F.one_hot(batch['seq'].long(), len(residue_constants.restypes_with_x)).float()
    # m = rearrange(m, '... i d -> ... i () d') + rearrange(m, '... j d -> ... () j d')
    # m = torch.any(
    #     torch.einsum('... i j d,d c -> ... i j c', m, self.pairing) >= 2, dim=-1
    # )
    m = torch.abs(
        rearrange(batch['seq_index'], '... i -> ... i ()') -
        rearrange(batch['seq_index'], '... j -> ... () j')
    ) > 3
    # HACK: experience value
    mask_value = 1e4 if x.dtype == torch.float16 else 1e6  # max_neg_value(q)
    x = x.masked_fill(~m[..., None], -mask_value)

    l, d = x.shape[-2:]
    u = torch.zeros(x.shape[:-3] + (l, 1, d), dtype=x.dtype, device=x.device)
    v = torch.zeros(x.shape[:-3] + (1, l, d), dtype=x.dtype, device=x.device)
    w = torch.zeros(x.shape[:-3] + (l, l, 1), dtype=x.dtype, device=x.device)

    x = F.log_softmax(torch.cat((u, x), dim=-2), dim=-2)[..., 1:, :] + F.log_softmax(
        torch.cat((v, x), dim=-3), dim=-3
    )[..., 1:, :, :] + 2 * F.log_softmax(torch.cat((w, x), dim=-1), dim=-1)[..., 1:]

    return dict(logits=x)

  def loss(self, value, batch):
    """Log loss of a base pairing."""
    x = value['logits']
    assert len(x.shape) == 4
    l, eps = x.shape[-2], 1e-6

    # x = rearrange(x, '... i j d -> ... i (j d)')
    if 'sta' in batch:
      y = rearrange(
          F.one_hot(batch['sta'].long(), l + 1)[..., 1:], '... i d j -> ... i j d'
      )
      with autocast(enabled=False):
        probs = torch.clamp(
            1. - torch.exp(
                torch.sum(
                    torch.log(
                        torch.clamp(1. - torch.exp(x.float()), min=eps, max=1.)
                    ),
                    dim=-1
                )
            ),
            min=eps,
            max=1.
        )
        targets = torch.any(y, dim=-1).float()
        errors = sigmoid_cross_entropy(probs, targets, gammar=self.focal_loss)

        assert 'sta_type_mask' in batch
        if torch.any(batch['sta_type_mask']):
          assert y.shape[-1] == x.shape[-1]
          errors = (1. - self.alpha) * errors + self.alpha * torch.sum(
              y.float() * x.float(), dim=-1
          ) * batch['sta_type_mask']
    else:
      with torch.no_grad():
        y = torch.exp(x)
      errors = torch.sum(F.kl_div(x, y, reduction='none'), dim=-1)

    errors = torch.sum(errors, dim=-1)

    avg_error = functional.masked_mean(value=errors, mask=batch['mask'], epsilon=1e-6)
    logger.debug('PairingHead.loss: %s', avg_error)
    return dict(loss=avg_error)


class RobertaLMHead(nn.Module):
  """Head for Masked Language Modeling
    """
  def __init__(self, dim, loss_min=None, loss_max=None):
    super().__init__()

    dim, _ = commons.embedd_dim_get(dim)
    self.project = nn.Sequential(
        nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim),
        nn.Linear(dim, len(residue_constants.restypes_with_x))
    )
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

    errors = softmax_cross_entropy(
        labels=F.one_hot(labels, logits.shape[-1]), logits=logits
    )

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
        assert 'logits' in headers['distogram'] and 'breaks' in headers['distogram']
        logits, breaks = headers['distogram']['logits'], headers['distogram']['breaks']
        positions = batch['pseudo_beta']
        mask = batch['pseudo_beta_mask']

        cutoff = self.params.get('contact_cutoff', 8.0)
        t = torch.sum(breaks <= cutoff)
        pred = F.softmax(logits, dim=-1)
        pred = torch.sum(pred[..., :t + 1], dim=-1)
        truth = torch.cdist(positions, positions, p=2)
        metrics['contact'] = MetricDict()
        for b in range(truth.shape[0]):
          precision_list = functional.contact_precision(
              pred[b],
              truth[b],
              mask=mask[b],
              ratios=self.params.get('contact_ratios'),
              ranges=self.params.get('contact_ranges'),
              cutoff=cutoff
          )
          for (i, j), ratio, precision in precision_list:
            i, j = default(i, 0), default(j, 'inf')
            k = f'[{i},{j})_{ratio}'
            if k in metrics['contact']:
              metrics['contact'][k] = torch.cat(
                  (metrics['contact'][k], precision), dim=-1
              )
            else:
              metrics['contact'][k] = precision
        if '[24,inf)_1' in metrics['contact']:
          logger.debug(
              'MetricDictHead.contact.[24,inf]@L: %s', metrics['contact']['[24,inf)_1']
          )
      if (
          'profile' in headers or 'coevolution' in headers
      ) and 'sequence_profile' in batch:
        labels, label_mask = batch['sequence_profile'], 1.0
        if 'sequence_profile_mask' in batch:
          label_mask = rearrange(batch['sequence_profile_mask'], 'c -> () () c')

        mask = batch['mask']
        if 'profile' in headers:
          assert 'logits' in headers['profile']
          logits = headers['profile']['logits']
          sim = softmax_cosine_similarity(logits=logits, labels=labels, mask=label_mask)
          avg_sim = functional.masked_mean(value=sim, mask=mask)
          logger.debug('MetricDictHead.profile.cosine: %s', avg_sim)
          metrics['profile'] = MetricDict()
          metrics['profile']['cosine'] = avg_sim
        if 'coevolution' in headers:
          metrics['coevolution'] = MetricDict()

          assert 'logits' in headers['coevolution']
          if 'mask' in headers['coevolution']:
            label_mask = headers['coevolution']['mask']
          prob = F.softmax(headers['coevolution']['logits'], dim=-1)

          pred = torch.sum(prob, dim=-3)
          pred = pred / (torch.sum(pred, dim=-1, keepdim=True) + 1e-6)
          sim = F.cosine_similarity(pred * label_mask, labels * label_mask, dim=-1)
          avg_sim = functional.masked_mean(value=sim, mask=mask)
          logger.debug('MetricDictHead.coevolution.cosine: %s', avg_sim)
          metrics['coevolution']['cosine'] = avg_sim

          if 'msa' in batch:
            num_class = prob.shape[-1]

            pred = torch.argmax(prob, dim=-1)
            mask = F.one_hot(batch['msa'].long(), num_classes=num_class) * label_mask
            if 'msa_mask' in batch:
              mask = torch.einsum('b m i c,b m i -> b m i c', mask, batch['msa_mask'])
            avg_sim = functional.masked_mean(
                value=F.one_hot(pred, num_classes=num_class), mask=mask
            )
            logger.debug('MetricDictHead.coevolution.identity: %s', avg_sim)
            metrics['coevolution']['identity'] = avg_sim

            errors = -torch.sum(mask * torch.log(prob + 10e-8), dim=-1)
            avg_error = torch.exp(
                functional.masked_mean(value=errors, mask=torch.sum(mask, dim=-1))
            )
            logger.debug('MetricDictHead.coevolution.perplexity: %s', avg_error)
            metrics['coevolution']['perplexity'] = avg_error

      if 'folding' in headers and 'coords' in headers['folding'] and (
          'coord_mask' in batch or 'coord_exists' in batch
      ):
        points, point_mask = headers['folding']['coords'], batch.get(
            'coord_exists', batch.get('coord_mask')
        )
        assert exists(point_mask)

        seq_index = batch.get('seq_index')
        point_mask = point_mask * torch.logical_and(  # protein only
            batch['seq'] >= residue_constants.prot_from_idx,
            batch['seq'] <= residue_constants.prot_to_idx
        )[..., None]
        ca_ca_distance_error = functional.between_ca_ca_distance_loss(
            points, point_mask, seq_index
        )
        metrics['violation'] = MetricDict()
        metrics['violation']['ca_ca_distance'] = ca_ca_distance_error
        logger.debug(
            'MetricDictHead.violation.ca_ca_distance: %s', ca_ca_distance_error
        )

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
  def __init__(
      self,
      dim,
      mask='-',
      log_softmax=False,
      softplus=False,
      prior_w=0.,
      prior_b=None,
      pooling='sum',
      alpha=None,
      task_model=None,
      task_num=1,
      task_weight=None,
      task_dim_hidden=None,
      num_var_as_ref=0,
      label_threshold=0.,
      label_epsilon=0.,
      pos_weight=None,
      focal_loss=0.,
      shard_size=2048
  ):
    super().__init__()
    dim_single, _ = commons.embedd_dim_get(dim)

    self.task_num = task_num
    if not exists(task_weight):
      task_weight = [1.] * task_num
    assert len(task_weight) == task_num
    task_weight = torch.as_tensor(task_weight)
    self.register_buffer('task_weight', task_weight, persistent=False)
    if exists(task_model):
      assert task_model in ('MMoE', )
      dim_hidden = default(task_dim_hidden, dim_single)
      self.task_gating = nn.Sequential(
          nn.Linear(dim_single, dim_hidden), nn.GELU(), nn.LayerNorm(dim_hidden),
          nn.Linear(dim_hidden, task_num, bias=False)
      )
    else:
      self.task_gating = None

    # log(\pi_{\theta}(s_w|s)/\pi_{\theat}(s_l|s)) strictly.
    self.log_softmax = log_softmax
    if softplus:  # ensure that \sigma >= 0
      assert prior_w >= 0
      self.sigma = nn.Parameter(
          torch.log(torch.exp(torch.full((task_num, ), max(prior_w, 1e-4))) - 1.)
      )
    else:
      self.sigma = nn.Parameter(torch.full((task_num, ), prior_w))
    self.softplus = softplus
    self.prior_b = prior_b

    m = functional.make_mask(residue_constants.restypes_with_x_and_gap, mask)
    self.register_buffer('mask', m, persistent=False)

    assert pooling in ('sum', 'mean')
    self.pooling = pooling

    assert num_var_as_ref >= 0
    self.num_var_as_ref = num_var_as_ref
    self.label_threshold = label_threshold
    self.label_epsilon = label_epsilon
    if exists(pos_weight):
      self.register_buffer(
          'pos_weight', torch.full((task_num,), pos_weight), persistent=False
      )
    else:
      self.pos_weight = None
    self.alpha = alpha
    self.focal_loss = focal_loss
    self.shard_size = env('profold2_fitness_shard_size', defval=shard_size, dtype=int)
    self.return_motifs = env('profold2_fitness_return_motifs', defval=True, dtype=bool)

  def predict(self, variant_logit, variant_mask, gating=None):
    variant_logit = variant_logit[..., None]
    if exists(gating):
      variant_logit = variant_logit * gating
    if self.pooling == 'sum':
      variant_logit = torch.sum(variant_logit * variant_mask, dim=-2)
    else:
      assert self.pooling == 'mean'
      variant_logit = functional.masked_mean(
          value=variant_logit, mask=variant_mask, dim=-2
      )
    w = self.sigma
    if self.softplus:
      w = F.softplus(w)
      if exists(self.prior_b):
        w = w + self.prior_b
    elif exists(self.prior_b):
      w = torch.clamp(w, min=self.prior_b)
    return w * variant_logit

  def forward(self, headers, representations, batch):
    assert 'coevolution' in headers
    num_class = self.mask.shape[0]
    eij, ei = headers['coevolution']['wij'], headers['coevolution']['bi']
    if 'wab' in headers['coevolution']:
      wab = headers['coevolution']['wab']
    else:
      wab = None

    def _hamiton_run(variant):
      variant = F.one_hot(variant.long(), num_class)
      if exists(wab):
        hi = torch.einsum(
            'b m j d,b i j q,q c d,d -> b m i c', variant.float(), eij, wab, self.mask
        )
      else:
        hi = torch.einsum(
            'b m j d,b i j c d,d -> b m i c', variant.float(), eij, self.mask
        )
      motifs = rearrange(ei, 'b i c -> b () i c') + hi
      if self.log_softmax:
        logits = torch.einsum(
            'b m i c,b m i c -> b m i', F.log_softmax(motifs, dim=-1), variant.float()
        )
      else:
        logits = torch.einsum('b m i c,b m i c -> b m i', motifs, variant.float())
      if self.return_motifs:
        return motifs, logits
      return None, logits

    def _hamiton_cat(logits):
      motifs, logits = zip(*logits)
      motifs = torch.cat(motifs, dim=1) if self.return_motifs else None
      logits = torch.cat(logits, dim=1)
      return motifs, logits

    if 'variant' in batch:
      variant = batch['variant']
      variant_mask = batch['variant_mask']
    else:
      assert 'seq' in batch
      variant = rearrange(batch['seq'], 'b i -> b () i')
      variant_mask = rearrange(batch['mask'], 'b i -> b () i')

    motifs, logits = functional.sharded_apply(
        _hamiton_run, [variant],
        shard_size=None if self.training else self.shard_size,
        shard_dim=1,
        cat_dim=_hamiton_cat
    )

    r = dict(logits=logits)
    if exists(self.task_gating):
      gating = F.sigmoid(self.task_gating(representations['single']))
    else:
      gating = None
    r.update(gating=gating)
    if not self.training:
      if 'variant_task_mask' in batch:
        variant_mask = batch['variant_task_mask']
      else:
        variant_mask = variant_mask[..., None]
      variant_mask = variant_mask * variant_mask[:, :1, ...]
      variant_logit = logits - logits[:, :1, ...]
      variant_logit = self.predict(variant_logit, variant_mask, gating=gating)
      r.update(variant_logit=variant_logit)

    if exists(motifs):
      r.update(motifs=motifs)
    return r

  def loss(self, value, batch):
    logits, logits_ref = value['logits'], None
    gating = value.get('gating')
    num_class = self.mask.shape[0]
    avg_error_motif = 0

    if 'variant' in batch:
      variant_mask = batch['variant_mask']
      variant_label = batch['variant_label']
      variant_label_mask = batch['variant_label_mask']

      if exists(self.alpha) and self.alpha > 0:
        # predict motifs
        motifs = value['motifs']
        labels = F.one_hot(batch['variant'].long(), num_class)

        with autocast(enabled=False):
          errors = softmax_cross_entropy(
              labels=labels, logits=motifs, mask=self.mask, gammar=self.focal_loss
          )
        motif_mask = ((variant_label > self.label_threshold) &
                      variant_label_mask) | (~variant_label_mask)
        motif_mask = torch.all(motif_mask, dim=-1, keepdim=True) * variant_mask
        avg_error_motif = functional.masked_mean(value=errors, mask=motif_mask)
        logger.info('FitnessHead.motifs.loss: %s', avg_error_motif)
        avg_error_motif = self.alpha * avg_error_motif

      if self.num_var_as_ref > 0:
        # minimum of variants in batch
        label_mask = torch.sum(variant_mask, dim=-1) > 0
        label_num = torch.amin(torch.sum(label_mask, dim=-1))

        num_var_as_ref = min(self.num_var_as_ref, label_num)
        if num_var_as_ref > 0:
          variant_weight = (
              torch.any(
                  (variant_label > self.label_threshold) * variant_label_mask, dim=-1
              )
          ) * label_mask

      if 'variant_task_mask' in batch:
        variant_mask = batch['variant_task_mask']
      else:
        variant_mask = variant_mask[..., None]

      if self.num_var_as_ref > 0:
        if num_var_as_ref > 0:
          b, _, n = logits.shape
          # sample reference based on variant_label
          ref_idx = torch.multinomial(
              variant_weight + 1e-3, num_var_as_ref, replacement=False
          )
          ref_idx = torch.cat(
              (torch.zeros((b, 1), dtype=ref_idx.dtype, device=logits.device), ref_idx),
              dim=-1
          )
          logger.debug('FitnessHead.ref_idx: %s', ref_idx)

          logits_ref = torch.gather(
              logits, 1, repeat(ref_idx, 'b m -> b m i', i=logits.shape[-1])
          )
          mask_ref = torch.gather(
              variant_mask, 1,
              repeat(
                  ref_idx,
                  'b m -> b m i t',
                  i=variant_mask.shape[-2],
                  t=variant_mask.shape[-1]
              )
          )
          label_ref = torch.gather(
              variant_label, 1, repeat(ref_idx, 'b m -> b m t', t=self.task_num)
          )
          label_mask_ref = torch.gather(
              variant_label_mask, 1, repeat(ref_idx, 'b m -> b m t', t=self.task_num)
          )
    else:
      variant_mask = rearrange(torch.zeros_like(batch['mask']), 'b i -> b () i ()')
      b = variant_mask.shape[0]
      variant_label = torch.ones((b, 1, self.task_num), device=variant_mask.device)
      variant_label_mask = torch.zeros(b, 1, self.task_num, device=variant_mask.device)

    if not exists(logits_ref):
      logits_ref = logits[:, :1, ...]
      mask_ref = variant_mask[:, :1, ...]
      label_ref = variant_label[:, :1, ...]
      label_mask_ref = variant_label_mask[:, :1, ...]

    # pairwise logistic loss
    variant_logit = rearrange(logits, 'b m i -> b m () i'
                             ) - rearrange(logits_ref, 'b n i -> b () n i')
    variant_mask = rearrange(variant_mask, 'b m i t -> b m () i t'
                            ) * rearrange(mask_ref, 'b n i t -> b () n i t')
    variant_label = rearrange(variant_label, 'b m t -> b m () t'
                             ) - rearrange(label_ref, 'b n t -> b () n t')
    variant_label = torch.sign(variant_label
                              ) * (torch.abs(variant_label) > self.label_epsilon)
    variant_label = torch.clamp((1. + variant_label) / 2, min=0, max=1)
    variant_label_mask = rearrange(variant_label_mask, 'b m t -> b m () t'
                                  ) * rearrange(label_mask_ref, 'b n t -> b () n t')
    # variant_logit = torch.sum(variant_logit * variant_mask, dim=-1)
    variant_logit = self.predict(variant_logit, variant_mask, gating=gating)
    logger.debug('FitnessHead.logit: %s', str(variant_logit))
    logger.debug('FitnessHead.label: %s', str(variant_label))
    with autocast(enabled=False):
      errors = sigmoid_cross_entropy_with_logits(
          variant_logit.float(),
          variant_label.float(),
          alpha=self.pos_weight,
          gammar=self.focal_loss
      )
    avg_error_fitness = functional.masked_mean(
        value=errors, mask=variant_label_mask * self.task_weight
    )
    logger.debug('FitnessHead.loss: %s', avg_error_fitness)
    return dict(loss=avg_error_motif + avg_error_fitness)


class SequenceProfileHead(nn.Module):
  """Head to predict sequence profile.
    """
  def __init__(self, dim, input_dim=None, single_repr=None, num_pivot=None):
    super().__init__()
    dim, _ = commons.embedd_dim_get(dim)

    if not exists(input_dim):
      input_dim = dim
    if not exists(single_repr):
      single_repr = 'struct_module'
    assert single_repr in ('struct_module', 'mlm')
    self.single_repr = single_repr
    self.num_pivot = num_pivot

    self.project = nn.Sequential(
        nn.Linear(input_dim, dim), nn.GELU(), nn.LayerNorm(dim),
        nn.Linear(dim, len(residue_constants.restypes_with_x_and_gap))
    )

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
        tms = batched_tmscore(pred_points, true_points, coord_mask, batch['mask'])
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
    del mask
    seq_index = batch.get('seq_index')
    if not exists(seq_index):
      b, n = seq.shape[:2]
      seq_index = repeat(torch.arange(n, device=seq.device), 'i -> b i', b=b)
    if 'coord_mask' in batch or 'coord_exists' in batch:
      points, point_mask = value['coords'], batch.get(
          'coord_exists', batch.get('coord_mask')
      )
      assert exists(point_mask)
      point_mask = point_mask * torch.logical_and(  # protein only
          batch['seq'] >= residue_constants.prot_from_idx,
          batch['seq'] <= residue_constants.prot_to_idx
      )[..., None]

      # loss_dict.update(ca_ca_distance_loss = functional.between_ca_ca_distance_loss(
      #         points, point_mask, seq_index))
      loss_dict = {}

      loss_dict.update(
          functional.between_residue_bond_loss(
              points, point_mask, seq_index, seq, loss_only=True
          )
      )
      loss_dict.update(
          functional.between_residue_clash_loss(
              points, point_mask, seq_index, seq, loss_only=True
          )
      )
      loss_dict.update(
          functional.within_residue_clash_loss(
              points, point_mask, seq_index, seq, loss_only=True
          )
      )

      for k, v in loss_dict.items():
        logger.debug('ViolationHead.%s: %s', k, v)
      return dict(loss=sum(loss_dict.values()))
    return None


class HeaderBuilder:
  _headers = dict(
      coevolution=CoevolutionHead,
      confidence=ConfidenceHead,
      contact=ContactHead,
      distogram=DistogramHead,
      fitness=FitnessHead,
      folding=FoldingHead,
      lddt=LDDTHead,
      metric=MetricDictHead,
      pae=PAEHead,
      pairing=PairingHead,
      profile=SequenceProfileHead,
      roberta=RobertaLMHead,
      tmscore=TMscoreHead,
      violation=ViolationHead
  )

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
