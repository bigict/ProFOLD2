"""An implementation of AlphaFold2 model
 """
from dataclasses import dataclass
import functools
import logging
import random

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from profold2.common import residue_constants
from profold2.model import commons, functional
from profold2.model.evoformer import Evoformer, TemplateEmbedding
from profold2.model.head import HeaderBuilder
from profold2.utils import exists

logger = logging.getLogger(__name__)


@dataclass
class Recyclables:
  msa_first_row_repr: torch.Tensor
  pairwise_repr: torch.Tensor
  coords: torch.Tensor
  frames: tuple = None

  def asdict(self):
    return dict(
        msa_first_row_repr=self.msa_first_row_repr,
        pairwise_repr=self.pairwise_repr,
        coords=self.coords,
        frames=self.frames
    )


@dataclass
class _ReturnValues:
  recyclables: Recyclables = None
  headers: dict = None
  loss: torch.Tensor = None


class ReturnValues(_ReturnValues):
  def __init__(self, **kwargs):
    if 'recyclables' in kwargs and exists(kwargs['recyclables']):
      kwargs['recyclables'] = Recyclables(**kwargs['recyclables'])
    super().__init__(**kwargs)

  def asdict(self):
    return dict(
        recyclables=self.recyclables.asdict()
        if exists(self.recyclables) else self.recyclables,
        headers=self.headers,
        loss=self.loss
    )


class InputEmbeddings(nn.Module):
  """InputEmbedder
   """
  def __init__(
      self,
      dim,
      num_seq_tokens=len(residue_constants.restypes_with_x),
      num_msa_tokens=0,
      max_rel_dist=0
  ):
    super().__init__()

    dim_msa, dim_pairwise = dim

    self.to_single_emb = nn.Linear(num_seq_tokens, dim_msa)
    self.to_msa_emb = nn.Linear(num_msa_tokens, dim_msa) if num_msa_tokens > 0 else None

    if num_msa_tokens > 0:
      self.to_pairwise_emb = commons.PairwiseEmbedding(
          num_seq_tokens, dim_pairwise, max_rel_dist=max_rel_dist
      )
    else:
      self.to_pairwise_emb = commons.PairwiseEmbedding(
          dim_msa, dim_pairwise, max_rel_dist=max_rel_dist
      )

  def forward(self, target_feat, target_mask, seq_index, msa_feat=None, msa_mask=None):
    s = self.to_single_emb(target_feat)
    if exists(self.to_msa_emb) and exists(msa_feat):
      m = self.to_msa_emb(msa_feat)
      m_mask = msa_mask
      assert exists(msa_mask)
    else:
      # NOTE: `target_feat` is a tensor of residue_id.
      m = self.to_single_emb(rearrange(target_feat, '... i -> ... () i'))
      m_mask = rearrange(target_mask, '... i -> ... () i')

    # add single representation to msa representation
    m = commons.tensor_add(m, rearrange(s, '... i d -> ... () i d'))

    if not exists(msa_feat):
      target_feat = s
    x, x_mask = self.to_pairwise_emb(target_feat, target_mask, seq_index=seq_index)
    return x, m, x_mask, m_mask

def _create_extra_msa_feature(batch):
  """Expand extra_msa into 1hot and concat with other extra msa features.

  We do this as late as possible as the one_hot extra msa can be very large.

  Arguments:
    batch: a dictionary with the following keys:
     * 'extra_msa': [N_extra_seq, N_res] MSA that wasn't selected as a cluster
       centre. Note, that this is not one-hot encoded.
     * 'extra_has_deletion': [N_extra_seq, N_res] Whether there is a deletion to
       the left of each position in the extra MSA.
     * 'extra_deletion_value': [N_extra_seq, N_res] The number of deletions to
       the left of each position in the extra MSA.

  Returns:
    Concatenated tensor of extra MSA features.
  """
  # 23 = 20 amino acids + 'X' for unknown + gap + bert mask
  msa_1hot = F.one_hot(batch['extra_msa'].long(), num_classes=23)
  msa_feat = [msa_1hot,
              batch['extra_has_deletion'][...,None],
              batch['extra_deletion_value'][...,None]]
  return torch.cat(msa_feat, axis=-1), batch['extra_msa_mask']

class AlphaFold2(nn.Module):
  """An implementation of the AlphaFold2 model
   """
  def __init__(
      self,
      *,
      dim,
      evoformer_depth=48,
      evoformer_msa_dim=(256, 32),
      evoformer_head_num=(8, 4),
      evoformer_head_dim=(32, 32),
      evoformer_extra_depth=4,
      evoformer_extra_msa_dim=(64, 32),
      evoformer_extra_heads=(8, 4),
      evoformer_extra_head_dim=(8, 32),
      template_depth=2,
      max_rel_dist=32,
      num_tokens=len(residue_constants.restypes_with_x),
      num_msa_tokens=0,
      attn_dropout=0.,
      ff_dropout=0.,
      accept_msa_attn=True,
      accept_frame_attn=False,
      accept_frame_update=False,
      recycling_single_repr=True,
      recycling_frames=False,
      recycling_pos=False,
      recycling_pos_min_bin=3.25,
      recycling_pos_max_bin=20.75,
      recycling_pos_num_bin=15,
      headers=None
  ):
    super().__init__()

    dim_single, dim_pairwise = dim

    dim_msa, dim_outer = evoformer_msa_dim
    self.dim = dim_single, dim_msa, dim_pairwise

    # input embedder
    self.embedder = InputEmbeddings(
        dim=(dim_msa, dim_pairwise),
        num_seq_tokens=num_tokens,
        num_msa_tokens=num_msa_tokens,
        max_rel_dist=max_rel_dist
    )
    self.template_emb = TemplateEmbedding(
        dim=dim_pairwise,
        depth=template_depth,
        dim_templ=64,
        heads=4,
        dim_head=16,
        dim_msa=dim_msa,
        attn_dropout=0.25,
        ff_dropout=0
    ) if template_depth > 0 else None

    # extra msa stack
    dim_msa_extra, dim_outer_extra = evoformer_extra_msa_dim
    if evoformer_extra_depth > 0:
      self.msa_activations_extra = nn.Linear(25, dim_msa_extra)
      self.evoformer_extra = Evoformer(
              depth=evoformer_extra_depth,
              dim_msa=dim_msa_extra,
              dim_pairwise=dim_pairwise,
              heads=evoformer_extra_heads,
              dim_head=evoformer_extra_head_dim,
              attn_dropout=attn_dropout,
              ff_dropout=ff_dropout,
              global_column_attn=True)
    # main trunk modules
    self.evoformer = Evoformer(
        depth=evoformer_depth,
        dim_msa=dim_msa,
        dim_pairwise=dim_pairwise,
        heads=evoformer_head_num,
        dim_head=evoformer_head_dim,
        accept_msa_attn=accept_msa_attn,
        accept_frame_attn=accept_frame_attn,
        accept_frame_update=accept_frame_update,
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout
    )

    # msa to single activations
    self.to_single_repr = nn.Linear(dim_msa, dim_single)

    # recycling params
    self.recycling_to_msa_repr = nn.Linear(
        dim_single, dim_msa
    ) if recycling_single_repr else None
    self.recycling_pos_linear = nn.Linear(
        recycling_pos_num_bin, dim_pairwise
    ) if recycling_pos else None
    self.recycling_frames = recycling_frames
    if recycling_pos:
      recycling_pos_breaks = torch.linspace(
          recycling_pos_min_bin, recycling_pos_max_bin, steps=recycling_pos_num_bin
      )
      self.register_buffer('recycling_pos_breaks', recycling_pos_breaks, persistent=False)
    self.recycling_msa_norm = nn.LayerNorm(dim_msa)
    self.recycling_pairwise_norm = nn.LayerNorm(dim_pairwise)

    self.headers = HeaderBuilder.build((dim_single, dim_pairwise), headers, parent=self)

  def embeddings(self):
    return dict(
        token=self.embedder.to_single_emb.weight,
        pairwise=self.embedder.to_pairwise_emb.embeddings()
    )

  def forward(
      self, batch, *, return_recyclables=False, compute_loss=True, shard_size=None
  ):
    seq, mask, seq_embed, seq_index = map(
        batch.get, ('seq', 'mask', 'emb_seq', 'seq_index')
    )
    target_feat, = map(batch.get, ('target_feat',))
    if not exists(target_feat):
      target_feat = seq
    msa_enabled = batch.get('msa_enabled', True)
    if msa_enabled:
      msa, msa_mask, msa_embed = map(batch.get, ('msa_feat', 'msa_mask', 'emb_msa'))
    else:
      msa, msa_mask, msa_embed = None, None, None  # msa as features disabled
    del seq_embed, msa_embed
    recyclables, = map(batch.get, ('recyclables', ))

    # variables
    # b, n, device = *seq.shape[:2], seq.device

    representations = {}

    # input embeddings
    x, m, x_mask, m_mask = self.embedder(
        target_feat, mask, seq_index, msa_feat=msa, msa_mask=msa_mask
    )

    # add recyclables, if present
    if exists(recyclables):
      if exists(recyclables.coords) and exists(self.recycling_pos_linear):
        pseudo_beta = functional.pseudo_beta_fn(seq, recyclables.coords)
        dgram = functional.distogram_from_positions(
            pseudo_beta, self.recycling_pos_breaks
        )
        x = commons.tensor_add(x, self.recycling_pos_linear(dgram))  # pylint: disable=not-callable
      m[..., 0, :, :] = commons.tensor_add(
          m[..., 0, :, :], self.recycling_msa_norm(recyclables.msa_first_row_repr)
      )
      x = commons.tensor_add(x, self.recycling_pairwise_norm(recyclables.pairwise_repr))

    # template
    if exists(self.template_emb):
      template_batch = {k: batch[k] for k in batch if k.startswith('template_')}
      x, m, _, m_mask = self.template_emb(x, x_mask, m, m_mask, template_batch,
                                          shard_size=shard_size)

    # extra
    if exists(self.evoformer_extra):
      # Embed extra MSA features.
      extra_msa_feat, extra_msa_mask = _create_extra_msa_feature(batch)
      extra_msa_activations = self.msa_activations_extra(extra_msa_feat)
      x, *_ = self.evoformer_extra(x,
                                  extra_msa_activations,
                                  None,
                                  mask=x_mask,
                                  msa_mask=extra_msa_mask,
                                  shard_size=shard_size)

    # add recyclables, if present
    if exists(recyclables) and exists(recyclables.frames):
      quaternions, translations = recyclables.frames
    else:
      # black hole frames
      b, n, device = m.shape[:-3], m.shape[-2], m.device
      quaternions = torch.tensor([1., 0., 0., 0.], device=device)
      quaternions = torch.tile(quaternions, b + (n, 1))
      translations = torch.zeros(b + (n, 3), device=device)

    # trunk
    x, m, t = self.evoformer(
        x, m, (quaternions, translations),
        mask=x_mask,
        msa_mask=m_mask,
        shard_size=shard_size
    )

    s = self.to_single_repr(m[..., 0, :, :])

    # ready output container
    ret = ReturnValues()

    representations.update(msa=m, pair=x, single=s, frames=t, single_init=s)

    ret.headers = {}
    for name, module, options in self.headers:
      value = module(ret.headers, representations, batch)
      if not exists(value):
        continue
      ret.headers[name] = value
      if 'representations' in value:
        representations.update(value['representations'])
      if self.training and compute_loss and hasattr(module, 'loss'):
        loss = module.loss(ret.headers[name], batch)
        if exists(loss):
          ret.headers[name].update(loss)
          lossw = loss['loss'] * options.get('weight', 1.0)
          if exists(ret.loss):
            ret.loss = commons.tensor_add(ret.loss, lossw)
          else:
            ret.loss = lossw

    if return_recyclables:
      msa_first_row_repr, pairwise_repr = m[..., 0, :, :], representations['pair']
      if exists(self.recycling_to_msa_repr):
        msa_first_row_repr = self.recycling_to_msa_repr(representations['single'])
      msa_first_row_repr, pairwise_repr = map(
          torch.detach, (msa_first_row_repr, pairwise_repr)
      )
      coords = None
      if 'folding' in ret.headers and 'coords' in ret.headers['folding']:
        coords = ret.headers['folding']['coords'].detach()
      frames = representations['frames'] if self.recycling_frames else None
      ret.recyclables = Recyclables(msa_first_row_repr, pairwise_repr, coords, frames)

    return ret.asdict()


class AlphaFold2WithRecycling(nn.Module):
  """Wrap the AlphaFold2 with recycling
   """
  def __init__(self, **kwargs):
    super().__init__()

    self.impl = AlphaFold2(**kwargs)
    logger.debug(self)

  def embeddings(self):
    return self.impl.embeddings()

  def forward(self, batch, *, num_recycle=0, **kwargs):
    assert num_recycle >= 0

    # variables
    seq = batch['seq']
    b, n, device = seq.shape[:-1], seq.shape[-1], seq.device
    # FIXME: fake recyclables
    if 'recyclables' not in batch:
      _, dim_msa, dim_pairwise = self.impl.dim  # embedd_dim_get(self.impl.dim)
      batch['recyclables'] = Recyclables(
          msa_first_row_repr=torch.zeros(b + (n, dim_msa), device=device),
          pairwise_repr=torch.zeros(b + (n, n, dim_pairwise), device=device),
          coords=torch.zeros(b + (n, residue_constants.atom_type_num, 3), device=device)
      )

    if self.training:
      num_recycle = random.randint(0, num_recycle)
    cycling_function = functools.partial(
        self.impl, return_recyclables=True, compute_loss=False, **kwargs
    )

    with torch.no_grad():
      for i in range(num_recycle):
        ret = ReturnValues(**cycling_function(batch))
        if 'tmscore' in ret.headers:
          logger.debug(
              '%s/%s pid: %s, tmscore: %s', i, num_recycle, ','.join(batch['pid']),
              ret.headers['tmscore']['loss']
          )
        batch['recyclables'] = ret.recyclables

    ret = ReturnValues(
        **self.impl(batch, return_recyclables=False, compute_loss=True, **kwargs)
    )
    metrics = {}
    if 'plddt_mean' in batch:
      metrics['plddt_mean'] = batch['plddt_mean']
    if 'confidence' in ret.headers:
      metrics['confidence'] = ret.headers['confidence']['loss']
    if 'metric' in ret.headers and 'contact' in ret.headers['metric']['loss']:
      contacts = ret.headers['metric']['loss']['contact']
      if '[24,inf)_1' in contacts:
        metrics['P@L'] = contacts['[24,inf)_1']
    if 'tmscore' in ret.headers:
      metrics['tmscore'] = ret.headers['tmscore']['loss']
    if metrics:
      msg = ', '.join(['%s: %s'] * len(metrics))
      logger.debug(
          f'%s/%s pid: %s, {msg}', num_recycle, num_recycle, ','.join(batch['pid']),
          *functools.reduce(lambda x, y: x + y, metrics.items())
      )

    return ret.asdict()
