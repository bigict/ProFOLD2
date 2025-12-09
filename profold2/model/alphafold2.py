"""An implementation of AlphaFold2-style model
 """
from dataclasses import dataclass
import functools
import logging
import random
from typing import Optional

from tqdm.auto import tqdm

import torch
from torch import nn
from einops import rearrange

from profold2.common import residue_constants
from profold2.model import commons, folding, functional
from profold2.model.evoformer import Evoformer
from profold2.model.head import HeaderBuilder
from profold2.utils import env, exists, status

logger = logging.getLogger(__name__)


@dataclass
class Recyclables:
  msa_first_row_repr: torch.Tensor
  pairwise_repr: torch.Tensor
  coords: torch.Tensor
  frames: Optional[tuple[torch.Tensor, torch.Tensor]] = None

  def asdict(self) -> dict:
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

  def asdict(self) -> dict:
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
      r_max=0,
      s_max=0
  ):
    super().__init__()

    dim_msa, dim_pairwise = dim

    self.to_single_emb = nn.Embedding(num_seq_tokens + 1, dim_msa)
    self.to_msa_emb = nn.Linear(num_msa_tokens, dim_msa) if num_msa_tokens > 0 else None

    if num_msa_tokens > 0:
      self.to_pairwise_emb = commons.PairwiseEmbedding(
          num_seq_tokens, dim_pairwise, r_max=r_max, s_max=s_max
      )
    else:
      self.to_pairwise_emb = commons.PairwiseEmbedding(
          dim_msa, dim_pairwise, r_max=r_max, s_max=s_max
      )

  def forward(
      self,
      target_feat,
      target_mask,
      seq_index,
      seq_color=None,
      seq_sym=None,
      seq_entity=None,
      token_index=None,
      msa_feat=None,
      msa_mask=None
  ):
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
    x, x_mask = self.to_pairwise_emb(
        target_feat,
        target_mask,
        seq_index=seq_index,
        seq_color=seq_color,
        seq_sym=seq_sym,
        seq_entity=seq_entity,
        token_index=token_index
    )
    return x, m, x_mask, m_mask


class AlphaFold2(nn.Module):
  """An implementation of the AlphaFold2-style model
   """
  def __init__(
      self,
      *,
      dim,
      evoformer_depth=48,
      evoformer_head_num=8,
      evoformer_head_dim=32,
      relative_pos_r_max=32,
      relative_pos_s_max=env('profold2_relative_pos_s_max', defval=0, dtype=int),
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

    self.dim = dim
    dim_single, dim_msa, dim_pairwise = dim

    # input embedder
    self.embedder = InputEmbeddings(
        dim=(dim_msa, dim_pairwise),
        num_seq_tokens=num_tokens,
        num_msa_tokens=num_msa_tokens,
        r_max=relative_pos_r_max,
        s_max=relative_pos_s_max
    )

    # main trunk modules
    self.evoformer = Evoformer(
        depth=evoformer_depth,
        checkpoint_segment_size=env(
            'profold2_evoformer_checkpoint_segment_size', defval=1, dtype=int
        ),
        dim_msa=dim_msa,
        dim_pairwise=dim_pairwise,
        heads=evoformer_head_num,
        dim_head=evoformer_head_dim,
        accept_msa_attn=accept_msa_attn,
        accept_frame_attn=accept_frame_attn,
        accept_frame_update=accept_frame_update,
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout
    ) if evoformer_depth > 0 else None

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
      self.register_buffer('recycling_pos_breaks', recycling_pos_breaks)
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
    seq, mask, seq_embed, seq_index, seq_color, seq_sym, seq_entity = map(
        batch.get,
        ('seq', 'mask', 'emb_seq', 'seq_index', 'seq_color', 'seq_sym', 'seq_entity')
    )
    token_index = batch.get('token_index', batch['seq_index'])
    target_feat, = map(batch.get, ('target_feat', ))
    if not exists(target_feat):
      target_feat = seq
    msa_enabled = batch.get('msa_enabled', False)
    if msa_enabled:
      msa, msa_mask, msa_embed = map(batch.get, ('msa_feat', 'msa_mask', 'emb_msa'))
    else:
      msa, msa_mask, msa_embed = None, None, None  # msa as features disabled
    del seq_embed, msa_embed
    recyclables, = map(batch.get, ('recyclables', ))

    representations = {'recycling': return_recyclables}

    # input embeddings
    x, m, x_mask, msa_mask = self.embedder(
        target_feat,
        mask,
        seq_index,
        seq_color=seq_color,
        seq_sym=seq_sym,
        seq_entity=seq_entity,
        token_index=token_index,
        msa_feat=msa,
        msa_mask=msa_mask
    )

    # add recyclables, if present
    if exists(recyclables):
      if exists(recyclables.coords) and exists(self.recycling_pos_linear):
        pseudo_beta = functional.pseudo_beta_fn(seq, recyclables.coords)
        dgram = functional.distogram_from_positions(
            self.recycling_pos_breaks, pseudo_beta
        )
        x = commons.tensor_add(x, self.recycling_pos_linear(dgram))  # pylint: disable=not-callable
      m[..., 0, :, :] = commons.tensor_add(
          m[..., 0, :, :], self.recycling_msa_norm(recyclables.msa_first_row_repr)
      )
      x = commons.tensor_add(x, self.recycling_pairwise_norm(recyclables.pairwise_repr))

    # add recyclables, if present
    if exists(recyclables) and exists(recyclables.frames):
      quaternions, translations = recyclables.frames
    else:
      # black hole frames
      b, n, device = m.shape[:-3], m.shape[-2], m.device
      quaternions = torch.tensor([1., 0., 0., 0.], device=device)
      quaternions = torch.tile(quaternions, b + (n, 1))
      translations = torch.zeros(b + (n, 3), device=device)
    t = (quaternions, translations)

    # trunk
    if exists(self.evoformer):
      x, m, t = self.evoformer(
          x, m, t, mask=x_mask, msa_mask=msa_mask, shard_size=shard_size
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
    if 'folding' in ret.headers:
      batch = folding.multi_chain_permutation_alignment(ret.headers['folding'], batch)
    if self.training and compute_loss:
      for name, module, options in self.headers:
        if not hasattr(module, 'loss') or name not in ret.headers:
          continue
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

    self.config = kwargs
    self.impl = AlphaFold2(**kwargs)
    logger.debug(self)

  @staticmethod
  def from_config(config):
    kwargs = dict(
        dim=config['dim'],
        evoformer_depth=config['evoformer_depth'],
        evoformer_head_num=config['evoformer_head_num'],
        evoformer_head_dim=config['evoformer_head_dim'],
        accept_msa_attn=config.get('evoformer_accept_msa_attn', True),
        accept_frame_attn=config.get('evoformer_accept_frame_attn', False),
        accept_frame_update=config.get('evoformer_accept_frame_update', False),
        headers=config['headers']
    )

    # optional args.
    for key in (
        'template_depth',
        'num_tokens',
        'num_msa_tokens',
        'recycling_single_repr',
        'recycling_pos',
    ):
      if key in config:
        kwargs[key] = config[key]

    return AlphaFold2WithRecycling(**kwargs)

  def to_config(self):
    return self.config

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

    pid = ','.join(batch['pid'])
    with torch.no_grad():
      with status(batch, recycling=True):
        for i in tqdm(
            range(num_recycle), disable=self.training, desc=f'Trunk Recycling [{pid}]'
        ):
          ret = ReturnValues(**cycling_function(batch))
          if 'tmscore' in ret.headers:
            logger.debug(
                '%s/%s pid: %s, tmscore: %s', i, num_recycle, pid,
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
          f'%s/%s pid: %s, {msg}', num_recycle, num_recycle, pid,
          *functools.reduce(lambda x, y: x + y, metrics.items())
      )

    return ret.asdict()
