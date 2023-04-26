import sys
import functools
import logging
import math

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat

from profold2.common import residue_constants
from profold2.model import functional, folding
from profold2.model.commons import embedd_dim_get
from profold2.utils import *

logger = logging.getLogger(__name__)


def softmax_cross_entropy(logits, labels, mask=None):
  """Computes softmax cross entropy given logits and one-hot class labels."""
  if not exists(mask):
    mask = 1.0
  loss = -torch.sum(labels * F.log_softmax(logits, dim=-1) * mask, dim=-1)
  return loss


def softmax_cross_entropy_with_probability(probs, labels, mask=None):
  """Computes softmax cross entropy given logits and one-hot class labels."""
  if not exists(mask):
    mask = 1.0
  loss = -torch.sum(labels * torch.log(probs) * mask, dim=-1)
  return loss


def softmax_cross_entropy_with_focal(logits, labels, gammar=0, mask=None):
  prob = F.softmax(logits, dim=-1)
  if gammar > 0:
    labels = labels.float() * ((1 - prob)**gammar)
  return softmax_cross_entropy_with_probability(prob, labels, mask)


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


def make_mask(restypes, mask, device=None):
  num_class = len(restypes)
  if exists(mask) and mask:
    m = [restypes.index(i) for i in mask]
    # Shape (k, c)
    m = F.one_hot(torch.as_tensor(m, device=device), num_class)
    # Shape (c)
    m = ~(torch.sum(m, dim=0) > 0)
    return m.float()
  return torch.as_tensor([1.0] * num_class, device=device)


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
      logger.debug('ConfidenceHead.loss: %s', metrics['loss'].item())
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
      logger.debug('ContactHead.loss: %s', avg_error.item())
      if exists(self.loss_min) or exists(self.loss_max):
        avg_error = torch.clamp(avg_error, min=self.loss_min, max=self.loss_max)
      return dict(loss=avg_error)
    return None


class CoevolutionDeletionHead(nn.Module):
  """Head to predict Co-evolution (Deletion).
    """

  def __init__(self,
               dim,
               mask='-',
               alpha=0.0,
               beta=0.0,
               gammar=0.0,
               loss_min=None,
               loss_max=None):
    super().__init__()
    dim_single, dim_pairwise = embedd_dim_get(dim)

    num_class = len(residue_constants.restypes_with_x_and_gap)
    self.single = nn.Sequential(nn.Linear(dim_single, dim_single),
                                nn.GELU(),
                                nn.LayerNorm(dim_single),
                                nn.Linear(dim_single, 1))
    self.pairwize = nn.Sequential(nn.LayerNorm(dim_pairwise),
                                  nn.Linear(dim_pairwise, num_class * 1))
    self.mask = mask

    self.alpha = alpha
    self.beta = beta
    self.gammar = gammar
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
    if self.training or 'msa' in batch:
      assert 'msa' in batch
      num_class = len(residue_constants.restypes_with_x_and_gap)

      si, zij = representations['single'], representations['pair']
      m = make_mask(residue_constants.restypes_with_x_and_gap,
                    self.mask,
                    device=si.device)

      ei = self.single(si)
      eij = self.pairwize(
          (zij + rearrange(zij, 'b i j d -> b j i d')) * 0.5)  # symmetrize

      eij = eij * rearrange(1 - torch.eye(zij.shape[-2], device=zij.device),
                            'i j -> i j ()')  # eii = 0
      hi = torch.einsum(
          'b m j d,b i j c d,d -> b m i c',
          F.one_hot(batch['msa'], num_class).float(),
          rearrange(eij, 'b i j (c d) -> b i j c d', c=1, d=num_class), m)
      logits = rearrange(ei, 'b i c -> b () i c') + hi
      return dict(logits=logits, wij=eij, bi=ei)
    return None

  def loss(self, value, batch):
    """Log loss of a msa rebuilding."""
    assert 'msa' in batch and 'del_msa' in batch

    logits = value['logits']
    labels = rearrange(
        torch.atan(batch['del_msa'] / 3.0) * 2.0 / math.pi, 'b m i -> b m i ()')
    assert len(logits.shape) == 4

    num_class = len(residue_constants.restypes_with_x_and_gap)
    label_mask = make_mask(residue_constants.restypes_with_x_and_gap,
                           self.mask,
                           device=logits.device)
    msa = F.one_hot(batch['msa'], num_class)

    errors = torch.sum(F.binary_cross_entropy_with_logits(logits,
                                                          labels,
                                                          reduction='none'),
                       dim=-1)
    mask = torch.einsum('b i,b m i c,c -> b m i', batch['mask'].float(),
                        msa.float(), label_mask)

    avg_error = functional.masked_mean(value=errors, mask=mask, epsilon=1e-6)
    logger.debug('CoevolutionHead(Deletion).loss: %s', avg_error.item())
    if self.alpha > 0 and 'wij' in value:
      r1 = torch.mean(torch.sum(torch.abs(value['wij']), dim=-1))
      logger.debug('CoevolutionHead(Deletion).loss.L1: %s', r1.item())
      avg_error += self.alpha * r1

    if self.beta > 0 and 'wij' in value:
      r2 = torch.mean(torch.sum(torch.square(value['wij']), dim=-1))
      logger.debug('CoevolutionHead(Deletion).loss.L2: %s', r2.item())
      avg_error += self.beta * r2

    if self.gammar > 0 and 'wij' in value:
      epsilon = 1e-10
      M = torch.sqrt(torch.sum(value['wij']**2, dim=-1) + epsilon)
      p = torch.sum(M, dim=-1)
      rlh = torch.sum(
          torch.square(
              torch.einsum('... i, ... i j, ... j', p, M, p) /
              (torch.einsum('... i,... i', p, p) + epsilon)))
      logger.debug('CoevolutionHead(Deletion).loss.LH: %s', rlh.item())
      avg_error += 0.5 * self.gammar * rlh

    if exists(self.loss_min) or exists(self.loss_max):
      avg_error = torch.clamp(avg_error, min=self.loss_min, max=self.loss_max)
    return dict(loss=avg_error)


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
                                  nn.Linear(dim_pairwise, num_class**2))
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
    if self.training or 'msa' in batch:
      assert 'msa' in batch
      num_class = len(residue_constants.restypes_with_x_and_gap)

      si, zij = representations['single'], representations['pair']
      m = make_mask(residue_constants.restypes_with_x_and_gap,
                    self.mask,
                    device=si.device)

      ei = self.single(si)
      eij = self.pairwize(
          (zij + rearrange(zij, 'b i j d -> b j i d')) * 0.5)  # symmetrize

      eij = eij * rearrange(1 - torch.eye(zij.shape[-2], device=zij.device),
                            'i j -> i j ()')  # eii = 0
      hi = torch.einsum(
          'b m j d,b i j c d,d -> b m i c',
          F.one_hot(batch['msa'], num_class).float(),
          rearrange(eij, 'b i j (c d) -> b i j c d', c=num_class, d=num_class),
          m)
      logits = rearrange(ei, 'b i c -> b () i c') + hi
      return dict(logits=logits, wij=eij, bi=ei)
    return None

  def loss(self, value, batch):
    """Log loss of a msa rebuilding."""
    num_class = len(residue_constants.restypes_with_x_and_gap)

    logits = value['logits']
    labels = F.one_hot(batch['msa'], num_class)
    logger.debug('CoevolutionHead.loss.logits: %s, %s', logits.shape,
                 self.focal_loss)

    assert len(logits.shape) == 4
    assert 'msa' in batch
    label_mask = make_mask(residue_constants.restypes_with_x_and_gap,
                           self.mask,
                           device=logits.device)

    if self.focal_loss > 0:
      errors = softmax_cross_entropy_with_focal(labels=labels,
                                                logits=logits,
                                                mask=label_mask,
                                                gammar=self.focal_loss)
    else:
      errors = softmax_cross_entropy(labels=labels,
                                     logits=logits,
                                     mask=label_mask)
    mask = torch.einsum('b i,b m i c,c -> b m i', batch['mask'].float(),
                        labels.float(), label_mask)

    avg_error = functional.masked_mean(value=errors, mask=mask, epsilon=1e-6)
    logger.debug('CoevolutionHead.loss: %s', avg_error.item())

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

    # L1 regularization
    alpha = _make_dynamic_regularization(self.alpha)
    if exists(alpha):
       r1 = torch.sum(torch.abs(value['wij']), dim=-1)
       logger.debug('CoevolutionHead.loss.L1(%s): %s',
                    alpha.tolist(), torch.mean(r1).item())
       avg_error += torch.mean(alpha[...,None,None] * r1)

    # L2 regularization
    beta = _make_dynamic_regularization(self.beta)
    if exists(beta):
       r2 = torch.sum(torch.square(value['wij']), dim=-1)
       logger.debug('CoevolutionHead.loss.L2(%s): %s',
                    beta.tolist(), torch.mean(r2).item())
       avg_error += torch.mean(beta[...,None,None] * r2)

    # LH regularization
    if self.gammar > 0:
      epsilon = 1e-10
      M = torch.sqrt(torch.sum(value['wij']**2, dim=-1) + epsilon)
      p = torch.sum(M, dim=-1)
      rlh = torch.sum(
          torch.square(
              torch.einsum('... i, ... i j, ... j', p, M, p) /
              (torch.einsum('... i,... i', p, p) + epsilon)))
      logger.debug('CoevolutionHead.loss.LH: %s', rlh.item())
      avg_error += 0.5 * self.gammar * rlh

    if exists(self.loss_min) or exists(self.loss_max):
      avg_error = torch.clamp(avg_error, min=self.loss_min, max=self.loss_max)
    return dict(loss=avg_error)


class DistillationHead(nn.Module):
  """Head to predict Co-evolution.
    """

  def __init__(self,
               dim,
               pij_model_dir,
               loss_fn=None,
               loss_min=None,
               loss_max=None):
    super().__init__()

    pij_model = np.load(pij_model_dir)
    self.pij = torch.from_numpy(pij_model['model'])

    if not exists(loss_fn):
      loss_fn = 'evoformer_module'
    assert loss_fn in ('evoformer_module', 'struct_module')
    self.loss_fn = loss_fn
    self.loss_min = loss_min
    self.loss_max = loss_max

  def forward(self, headers, representations, batch):
    assert 'distogram' in headers and 'breaks' in headers['distogram']
    if not 'coord' in batch and not 'coord_mask' in batch:
      if self.loss_fn == 'evoformer_module':
        assert 'logits' in headers['distogram']
        return dict(logits=headers['distogram']['logits'],
                    breaks=headers['distogram']['breaks'])
      else:
        assert 'folding' in headers and 'coords' in headers['folding']
        return dict(coords=headers['folding']['coords'],
                    breaks=headers['distogram']['breaks'])
    return None

  def loss(self, value, batch):
    sq_breaks = (value['breaks']**2)
    b, seq_len, device = *batch['seq'].shape, batch['seq'].device

    pij, pij_mask = self.pij_from_ref(seq_len, device=device)
    assert sq_breaks.shape[-1] + 1 == pij.shape[-1]
    if self.loss_fn == 'evoformer_module':
      logits = value['logits']
      errors = softmax_kl_diversity(labels=pij, logits=logits)
    else:
      ca_idx = residue_constants.atom_order['CA']
      # is_gly = torch.eq(batch['seq'], residue_constants.restype_order['G'])
      # cb_idx = residue_constants.atom_order['CB']
      # cb_coord = torch.where(
      #     repeat(is_gly, '... i -> ... i d', d=3),
      #     coords[..., ca_idx, :],
      #     coords[..., cb_idx, :])
      # cb_dist2 = torch.sum((rearrange(cb_coord, '... i d -> ... i () d') -
      #     rearrange(cb_coord, '... j d -> ... () j d'))**2, dim=-1, keepdims=True)
      # cb_bins = torch.sum(ca_dist2 > sq_breaks, dim=-1)
      coords = value['coords']
      ca_coord = coords[..., ca_idx, :]
      ca_dist2 = torch.sum((rearrange(ca_coord, '... i d -> ... i () d') -
                            rearrange(ca_coord, '... j d -> ... () j d'))**2,
                           dim=-1,
                           keepdims=True)
      ca_bins = torch.sum(ca_dist2 > sq_breaks, dim=-1)
      errors = softmax_cross_entropy_with_probability(
          labels=F.one_hot(ca_bins, num_classes=pij.shape[-1]),
          probs=repeat(pij, '... -> b ...', b=b))

    avg_error = functional.masked_mean(value=errors,
                                       mask=pij_mask,
                                       epsilon=1e-6)
    logger.debug('DistillationHead.loss(%s): %s', self.loss_fn,
                 avg_error.item())
    if exists(self.loss_min) or exists(self.loss_max):
      avg_error = torch.clamp(avg_error, min=self.loss_min, max=self.loss_max)
    return dict(loss=avg_error)

  def pij_from_ref(self, seq_len, device=None, epsilon=1e-10):
    max_sep_len, num_buckets = self.pij.shape
    value = torch.zeros(
        (seq_len, seq_len, num_buckets), device=device) + epsilon
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    for i in range(seq_len):
      for j in range(i + 1, min(seq_len, i + max_sep_len)):
        value[i, j] = self.pij[j - i].to(device=device)
        value[j, i] = value[i, j]
        mask[i, j] = True
        mask[j, i] = mask[i, j]
    return value, mask


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
      if self.focal_loss > 0:
        errors = softmax_cross_entropy_with_focal(labels=F.one_hot(
            true_bins, self.num_buckets),
                                                  logits=logits,
                                                  gammar=self.focal_loss)
      else:
        errors = softmax_cross_entropy(labels=F.one_hot(true_bins,
                                                        self.num_buckets),
                                       logits=logits)
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

    avg_error = functional.masked_mean(value=errors * square_weight,
                                       mask=square_mask,
                                       epsilon=1e-6)
    logger.debug('DistogramHead.loss: %s', avg_error.item())
    return dict(loss=avg_error)


class FoldingHead(nn.Module):
  """Head to predict 3d struct.
    """

  def __init__(self,
               dim,
               structure_module_depth,
               structure_module_heads,
               fape_min=1e-6,
               fape_max=15,
               fape_z=15,
               dropout=.0,
               **params):
    super().__init__()
    self.struct_module = folding.StructureModule(dim,
                                                 structure_module_depth,
                                                 structure_module_heads,
                                                 dropout=dropout)

    self.fape_min = fape_min
    self.fape_max = fape_max
    self.fape_z = fape_z

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
                            dij_weight=dij_weight,
                            use_weighted_mask=batch.get(
                                'coord_plddt_use_weighted_mask',
                                False)) / self.fape_z
        logger.debug('FoldingHead.loss(backbone_fape_loss: %d): %s', i,
                     r.item())
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
                residue_constants.restype_rigid_group_atom14_idx, batch['seq']),
            residue_constants.atom14_type_num)
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
                          dij_weight=dij_weight,
                          use_weighted_mask=batch.get(
                              'coord_plddt_use_weighted_mask',
                              False)) / self.fape_z
      logger.debug('FoldingHead.loss(sidechain_fape_loss): %s', r.item())
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
      return ((1. - sidechain_w) * backbone_loss + sidechain_w * sidechain_loss)

    def torsion_angle_loss(traj, gt, epsilon=1e-6):

      def yield_norm_angle_loss(pred_angles_list):
        for i, pred_angles in enumerate(pred_angles_list):
          angle_norm = torch.sqrt(epsilon + torch.sum(pred_angles**2, dim=-1))
          errors = torch.abs(angle_norm - 1.)
          avg_error = torch.mean(errors)
          logger.debug('FoldingHead.loss(norm_angle_loss: %d): %s', i,
                       avg_error.item())
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
        logger.debug('FoldingHead.loss(pred_angle_loss): %s', pred_loss.item())
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
               buckets_num=50,
               min_resolution=.0,
               max_resolution=sys.float_info.max):
    super().__init__()
    dim, _ = embedd_dim_get(dim)

    self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU(),
                             nn.Linear(dim, dim), nn.ReLU(),
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
    b = points_mask.shape[0]
    mask = torch.zeros(b, device=points_mask.device)
    if 'resolution' in batch and exists(batch['resolution']):
      assert len(batch['resolution']) == b
      for i in range(b):
        if exists(batch['resolution'][i]) and (
            self.min_resolution <= batch['resolution'][i] and
            batch['resolution'][i] <= self.max_resolution):
          mask[i] = 1
    points_mask = torch.einsum('b,b ... -> b ...', mask, points_mask)
    loss = torch.sum(errors * points_mask) / (1e-6 + torch.sum(points_mask))
    logger.debug('LDDTHead.loss: %s', loss.item())
    return dict(loss=loss)


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
    logger.debug('RobertaLMHead.loss: %s', avg_error.item())
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
      precision_list = contact_precision(
          pred,
          truth,
          mask=mask,
          ratios=self.params.get('contact_ratios'),
          ranges=self.params.get('contact_ranges'),
          cutoff=cutoff)
      metrics['contact'] = MetricDict()
      for (i, j), ratio, precision in precision_list:
        i, j = default(i, 0), default(j, 'inf')
        metrics['contact'][f'[{i},{j})_{ratio}'] = precision
      if '[24,inf)_1' in metrics['contact']:
        logger.debug('MetricDictHead.contact.[24,inf]@L: %s',
                     metrics['contact']['[24,inf)_1'].item())
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
        logger.debug('MetricDictHead.profile.cosine: %s', avg_sim.item())
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
        logger.debug('MetricDictHead.coevolution.cosine: %s', avg_sim.item())
        metrics['coevolution']['cosine'] = avg_sim

        if 'msa' in batch:
          num_class = prob.shape[-1]

          pred = torch.argmax(prob, dim=-1)
          mask = F.one_hot(batch['msa'], num_classes=num_class) * label_mask
          avg_sim = functional.masked_mean(value=F.one_hot(
              pred, num_classes=num_class),
                                           mask=mask)
          logger.debug('MetricDictHead.coevolution.identity: %s',
                       avg_sim.item())
          metrics['coevolution']['identity'] = avg_sim

          errors = -torch.sum(mask * torch.log(prob + 10e-8), dim=-1)
          avg_error = torch.exp(
              functional.masked_mean(value=errors, mask=torch.sum(mask,
                                                                  dim=-1)))
          logger.debug('MetricDictHead.coevolution.perplexity: %s',
                       avg_error.item())
          metrics['coevolution']['perplexity'] = avg_error

    return dict(loss=metrics) if metrics else None


class SequenceProfileGapHead(nn.Module):
  """Head to predict sequence profile (Gap).
    """

  def __init__(self, dim, input_dim=None, single_repr=None):
    super().__init__()
    dim, _ = embedd_dim_get(dim)

    if not exists(input_dim):
      input_dim = dim
    if not exists(single_repr):
      single_repr = 'struct_module'
    assert single_repr in ('struct_module', 'mlm')
    self.single_repr = single_repr

    self.project = nn.Sequential(nn.Linear(input_dim, dim), nn.GELU(),
                                 nn.LayerNorm(dim), nn.Linear(dim, 1))

  def forward(self, headers, representations, batch):
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
    assert 'sequence_profile_gap_value' in batch
    assert 'logits' in value

    logits = value['logits']
    labels = rearrange(batch['sequence_profile_gap_value'], 'b l -> b l ()')
    assert logits.shape == labels.shape
    mask = batch['mask']

    errors = torch.sum(F.binary_cross_entropy_with_logits(logits,
                                                          labels,
                                                          reduction='none'),
                       dim=-1)
    avg_error = functional.masked_mean(value=errors, mask=mask, epsilon=1e-6)
    logger.debug('SequenceProfileHead(Gap).loss: %s', avg_error.item())
    return dict(loss=avg_error)


class SequenceProfileHead(nn.Module):
  """Head to predict sequence profile.
    """

  def __init__(self,
               dim,
               input_dim=None,
               single_repr=None,
               loss_func='CrossEntropy'):
    super().__init__()
    dim, _ = embedd_dim_get(dim)

    if not exists(input_dim):
      input_dim = dim
    if not exists(single_repr):
      single_repr = 'struct_module'
    assert single_repr in ('struct_module', 'mlm')
    self.single_repr = single_repr

    self.project = nn.Sequential(
        nn.Linear(input_dim, dim), nn.GELU(), nn.LayerNorm(dim),
        nn.Linear(dim, len(residue_constants.restypes_with_x_and_gap)))

    assert loss_func in ('CrossEntropy', 'KLDiv')
    self.loss_func = loss_func

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

    if self.loss_func == 'CrossEntropy':
      errors = softmax_cross_entropy(labels=labels,
                                     logits=logits,
                                     mask=label_mask)
    else:
      errors = softmax_kl_diversity(labels=labels,
                                    logits=logits,
                                    mask=label_mask)

    avg_error = functional.masked_mean(value=errors, mask=mask, epsilon=1e-6)
    logger.debug('SequenceProfileHead.loss(%s): %s', self.loss_func,
                 avg_error.item())
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

    if 'coords_aligned' in headers['folding'] and 'labels_aligned' in headers[
        'folding']:
      coords_aligned, labels_aligned = headers['folding'][
          'coords_aligned'], headers['folding']['labels_aligned']
      return dict(loss=TMscore(rearrange(coords_aligned, 'd l -> () d l'),
                               rearrange(labels_aligned, 'd l -> () d l'),
                               L=torch.sum(batch['mask'], dim=-1).item()))
    elif 'coord' in batch and 'coord_mask' in batch:
      pred, labels = headers['folding']['coords'][
          ..., :self.num_atoms, :], batch['coord'][..., :self.num_atoms, :]
      coord_mask = batch['coord_mask'][..., :self.num_atoms]
      flat_cloud_mask = rearrange(coord_mask, 'b l c -> b (l c)')

      # rotate / align
      coords_aligned, labels_aligned = Kabsch(
          rearrange(
              rearrange(pred, 'b l c d -> b (l c) d')[flat_cloud_mask],
              'c d -> d c'),
          rearrange(
              rearrange(labels, 'b l c d -> b (l c) d')[flat_cloud_mask],
              'c d -> d c'))

      return dict(loss=TMscore(rearrange(coords_aligned, 'd l -> () d l'),
                               rearrange(labels_aligned, 'd l -> () d l'),
                               L=torch.sum(batch['mask'], dim=-1).item()))
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
      points, point_mask = value['coords'], batch.get('coord_mask',
                                                      batch.get('coord_exists'))
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
        logger.debug('ViolationHead.%s: %s', k, v.item())
      return dict(loss=sum(loss_dict.values()))
    return None


class HeaderBuilder:
  _headers = dict(coevolution=CoevolutionHead,
                  coevolutiond=CoevolutionDeletionHead,
                  confidence=ConfidenceHead,
                  contact=ContactHead,
                  distillation=DistillationHead,
                  distogram=DistogramHead,
                  folding=FoldingHead,
                  lddt=LDDTHead,
                  metric=MetricDictHead,
                  roberta=RobertaLMHead,
                  profile=SequenceProfileHead,
                  profileg=SequenceProfileGapHead,
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
