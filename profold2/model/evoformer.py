"""main evoformer class
 """
import torch
from torch import nn
from torch.cuda.amp import autocast
from einops import rearrange

from profold2.model.commons import (
    Attention, checkpoint_sequential_nargs, FeedForward, FrameAttentionBlock,
    FrameUpdater, MsaAttentionBlock, PairwiseAttentionBlock, tensor_add
)
from profold2.model import profiler
from profold2.utils import exists


class TemplateEmbedding(nn.Module):
  """The Template representation in AlphaFold2
   """
  def __init__(
      self,
      dim,
      heads=8,
      dim_head=64,
      attn_dropout=0.,
      templates_dim=32,
      templates_embed_layers=4,
      templates_angles_feats_dim=55
  ):
    super().__init__()

    self.to_template_embed = nn.Linear(templates_dim, dim)
    self.templates_embed_layers = templates_embed_layers
    self.template_pairwise_embedder = PairwiseAttentionBlock(
        dim=dim, dim_head=dim_head, heads=heads
    )

    self.template_pointwise_attn = Attention(
        dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout
    )

    self.template_angle_mlp = nn.Sequential(
        nn.Linear(templates_angles_feats_dim, dim), nn.GELU(), nn.Linear(dim, dim)
    )

  def forward(
      self,
      x,
      x_mask,
      m,
      msa_mask,
      templates_feats=None,
      templates_angles=None,
      templates_mask=None
  ):
    # embed templates, if present
    if exists(templates_feats):
      assert exists(templates_mask)
      _, num_templates, n, *_ = templates_feats.shape

      # embed template

      t = self.to_template_embed(templates_feats)
      t_mask_crossed = rearrange(templates_mask, 'b t i -> b t i ()'
                                ) * rearrange(templates_mask, 'b t j -> b t () j')

      t = rearrange(t, 'b t ... -> (b t) ...')
      t_mask_crossed = rearrange(t_mask_crossed, 'b t ... -> (b t) ...')

      for _ in range(self.templates_embed_layers):
        t = self.template_pairwise_embedder(t, mask=t_mask_crossed)

      t = rearrange(t, '(b t) ... -> b t ...', t=num_templates)
      t_mask_crossed = rearrange(
          t_mask_crossed, '(b t) ... -> b t ...', t=num_templates
      )

      # template pos emb

      x_point = rearrange(x, 'b i j d -> (b i j) () d')
      t_point = rearrange(t, 'b t i j d -> (b i j) t d')
      x_mask_point = rearrange(x_mask, 'b i j -> (b i j) ()')
      t_mask_point = rearrange(t_mask_crossed, 'b t i j -> (b i j) t')

      template_pooled = self.template_pointwise_attn(
          x_point, context=t_point, mask=x_mask_point, context_mask=t_mask_point
      )

      template_pooled_mask = rearrange(t_mask_point.sum(dim=-1) > 0, 'b -> b () ()')
      template_pooled = template_pooled * template_pooled_mask

      template_pooled = rearrange(template_pooled, '(b i j) () d -> b i j d', i=n, j=n)
      x = tensor_add(x, template_pooled)

    # add template angle features to MSAs by passing through MLP and then concat
    if exists(templates_angles):
      assert exists(templates_mask)
      t_angle_feats = self.template_angle_mlp(templates_angles)
      m = torch.cat((m, t_angle_feats), dim=1)
      msa_mask = torch.cat((msa_mask, templates_mask), dim=1)
    return x, x_mask, m, msa_mask


class EvoformerBlock(nn.Module):
  """One Evoformer Layer
   """
  def __init__(
      self,
      *,
      dim_msa,
      dim_pairwise,
      heads,
      dim_head,
      attn_dropout,
      ff_dropout,
      global_column_attn=False,
      accept_msa_attn=True,
      accept_frame_attn=False,
      accept_frame_update=False,
      **kwargs
  ):
    super().__init__()

    self.pair_attn = PairwiseAttentionBlock(
        dim_msa=dim_msa,
        dim_pairwise=dim_pairwise,
        heads=heads,
        dim_head=dim_head,
        dropout=attn_dropout
    )
    self.pair_ff = FeedForward(dim=dim_pairwise, dropout=ff_dropout)
    if accept_msa_attn:
      self.msa_attn = MsaAttentionBlock(
          dim_msa=dim_msa,
          dim_pairwise=dim_pairwise,
          heads=heads,
          dim_head=dim_head,
          dropout=attn_dropout,
          global_column_attn=global_column_attn,
          **kwargs
      )
      self.msa_ff = FeedForward(dim=dim_msa, dropout=ff_dropout)
    if accept_frame_attn:
      self.frame_attn = FrameAttentionBlock(
          dim_msa=dim_msa,
          dim_pairwise=dim_pairwise,
          heads=heads,
          scalar_key_dim=dim_head,
          scalar_value_dim=dim_head,
          dropout=attn_dropout,
          gating=True,
          point_weight_init=5e-3,
          require_pairwise_repr=False
      )
      self.frame_ff = FeedForward(dim=dim_msa, dropout=ff_dropout)
    if accept_frame_update:
      self.frame_update = FrameUpdater(dim=dim_msa, dropout=attn_dropout)

  def forward(self, inputs, shard_size=None):
    x, m, t, mask, msa_mask = inputs

    assert hasattr(self, 'msa_attn') or m.shape[1] == 1

    # msa attention and transition
    if hasattr(self, 'msa_attn'):
      with profiler.record_function('msa_attn'):
        m = self.msa_attn(
            m,
            mask=msa_mask,
            pairwise_repr=x,
            pairwise_mask=mask,
            shard_size=shard_size
        )
        m = tensor_add(m, self.msa_ff(m, shard_size=shard_size))

    # frame attention and transition
    if hasattr(self, 'frame_attn'):
      with profiler.record_function('frame_attn'):
        with autocast(enabled=False):
          # to default float
          m, x, t = m.float(), x.float(), tuple(map(lambda x: x.float(), t))
          s = self.frame_attn(m[:, 0], mask=msa_mask[:, 0], pairwise_repr=x, frames=t)
          s = tensor_add(s, self.frame_ff(s, shard_size=shard_size))
        m = torch.cat((s[:, None, ...], m[:, 1:, ...]), dim=1)

    # frame update
    if hasattr(self, 'frame_update'):
      with profiler.record_function('frame_update'):
        with autocast(enabled=False):
          # to default float
          m, t = m.float(), tuple(map(lambda x: x.float(), t))
          t = self.frame_update(m[:, 0], frames=t)

    # pairwise attention and transition
    with profiler.record_function('pair_attn'):
      # with autocast(enabled=False):
      x = self.pair_attn(
          x, mask=mask, msa_repr=m, msa_mask=msa_mask, shard_size=shard_size
      )
      x = tensor_add(x, self.pair_ff(x, shard_size=shard_size))

    return x, m, t, mask, msa_mask


class Evoformer(nn.Module):
  """The Evoformer in AlphaFold2
   """
  def __init__(self, *, depth, **kwargs):
    super().__init__()
    self.layers = nn.ModuleList([EvoformerBlock(**kwargs) for _ in range(depth)])  # pylint: disable=missing-kwoa

  def forward(self, x, m, frames=None, mask=None, msa_mask=None, shard_size=None):
    with profiler.record_function('evoformer'):
      inp = (x, m, frames, mask, msa_mask)
      x, m, t, *_ = checkpoint_sequential_nargs(
          self.layers, len(self.layers), inp, shard_size=shard_size
      )
      return x, m, t
