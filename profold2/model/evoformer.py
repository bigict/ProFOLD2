"""main evoformer class
 """
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
from einops import rearrange, repeat

from profold2.common import residue_constants
from profold2.model import commons, functional, profiler
from profold2.utils import exists


class TemplatePairBlock(nn.Module):
  """The pair stack block for templates in AlphaFold2
   """
  def __init__(self, *, dim, heads, dim_head, attn_dropout, ff_dropout):
    super().__init__()

    self.attn = commons.PairwiseAttentionBlock(
        dim_msa=dim,
        dim_pairwise=dim,
        heads=heads,
        dim_head=dim_head,
        dropout=attn_dropout,
        disabled_outer_mean=True,
        multiplication_first=False
    )
    self.ff = commons.FeedForward(dim=dim, mult=2, dropout=ff_dropout)

  def forward(self, x, mask, shard_size=None):
    x = self.attn(x, mask=mask, shard_size=shard_size)
    x = commons.tensor_add(x, self.ff(x, shard_size=shard_size))
    return x, mask


class SingleTemplateEmbedding(nn.Module):
  """The pair stack for templates in AlphaFold2
   """
  def __init__(
      self,
      *,
      depth,
      dim,
      dim_templ_feat=88,
      templ_dgram_breaks_min=3.25,
      templ_dgram_breaks_max=50.57,
      templ_dgram_breaks_num=39,
      use_template_unit_vector=False,
      **kwargs
  ):
    super().__init__()

    dgram_breaks = torch.linspace(
        templ_dgram_breaks_min, templ_dgram_breaks_max, steps=templ_dgram_breaks_num
    )
    self.register_buffer('dgram_breaks', dgram_breaks, persistent=False)
    self.use_template_unit_vector = use_template_unit_vector

    self.to_pair = nn.Linear(dim_templ_feat, dim)
    self.pair_stack = commons.layer_stack(TemplatePairBlock, depth, dim=dim, **kwargs)
    self.to_out_norm = nn.LayerNorm(dim)

  def forward(self, batch, mask_2d, shard_size=None):
    _, m, n = batch['template_seq'].shape[:3]
    template_mask = batch['template_pseudo_beta_mask']
    template_mask_2d = rearrange(template_mask, '... i -> ... i ()'
                                ) * rearrange(template_mask, '... j -> ... () j')
    template_dgram = functional.distogram_from_positions(
        batch['template_pseudo_beta'], self.dgram_breaks
    )
    to_concat = [template_dgram, template_mask_2d[..., None]]

    aatype = F.one_hot(batch['template_seq'].long(), num_classes=22)
    to_concat += [
        repeat(aatype, '... j d -> ... i j d', i=n),
        repeat(aatype, '... i d -> ... i j d', j=n)
    ]

    n_idx = residue_constants.atom_order['N']
    ca_idx = residue_constants.atom_order['CA']
    c_idx = residue_constants.atom_order['C']
    R, t = functional.rigids_from_3x3(  # pylint: disable=invalid-name
        batch['template_coord'], indices=(c_idx, ca_idx, n_idx)
    )

    local_points = torch.einsum(
        '... i w h,... i j w -> ... i j h', R,
        rearrange(t, '... j d -> ... () j d') - rearrange(t, '... i d ->... i () d')
    )
    inv_distance_scalar = torch.rsqrt(
        1e-6 + torch.sum(torch.square(local_points), dim=-1, keepdims=True)
    )

    # Backbone affine mask: whether the residue has C, CA, N
    # (the template mask defined above only considers pseudo CB).
    template_mask = (
        batch['template_coord_mask'][..., n_idx] *
        batch['template_coord_mask'][..., ca_idx] *
        batch['template_coord_mask'][..., c_idx]
    )
    template_mask_2d = template_mask[..., :, None] * template_mask[..., None, :]
    inv_distance_scalar *= template_mask_2d[..., None]
    unit_vector = local_points * inv_distance_scalar
    if not self.use_template_unit_vector:
      unit_vector = torch.zeros_like(unit_vector)
    to_concat += [unit_vector, template_mask_2d[..., None]]

    x = torch.cat(to_concat, dim=-1)
    # Mask out non-template regions so we don't get arbitrary values in the
    # distogram for these regions.
    x *= template_mask_2d[..., None]

    x = self.to_pair(x)

    x = rearrange(x, 'b m ... -> (b m) ...')
    x, *_ = self.pair_stack(
        x, repeat(mask_2d, 'b ... -> (b m) ...', m=m), shard_size=shard_size
    )
    x = rearrange(x, '(b m) ... -> b m ...', m=m)

    x = self.to_out_norm(x)
    return x


class TemplateEmbedding(nn.Module):
  """The Template representation in AlphaFold2
   """
  def __init__(
      self,
      dim,
      depth,
      dim_templ,
      heads=4,
      dim_head=16,
      attn_dropout=0.,
      dim_msa=None,
      **kwargs
  ):
    super().__init__()

    self.template_pairwise_embedder = SingleTemplateEmbedding(
        depth=depth,
        dim=dim_templ,
        dim_head=dim_head,
        heads=heads,
        attn_dropout=attn_dropout,
        **kwargs
    )

    self.template_pointwise_attn = commons.Attention(
        dim_q=dim,
        dim_kv=dim_templ,
        heads=heads,
        dim_head=dim_head,
        dropout=attn_dropout,
        gating=False
    )

    self.template_single_embedder = nn.Sequential(
        nn.Linear(22 + 14 * 2 + 7, dim_msa), nn.ReLU(), nn.Linear(dim_msa, dim_msa)
    ) if dim_msa else None

  def forward(self, x, x_mask, m, m_mask, batch, shard_size=None):

    n = batch['template_seq'].shape[2]
    template_mask = batch['template_mask']

    # Make sure the weights are shared across templates by constructing the
    # embedder here.
    template_pair_representation = self.template_pairwise_embedder(
        batch, x_mask, shard_size=shard_size
    )
    t = rearrange(x, '... i j d -> ... (i j) () d')
    context = rearrange(template_pair_representation, '... m i j d -> ... (i j) m d')
    t = self.template_pointwise_attn(t, context=context, context_mask=template_mask)
    t = rearrange(t, '... (i j) m d -> ... i j (m d)', i=n)
    x += t

    if exists(self.template_single_embedder):
      aatype = F.one_hot(batch['template_seq'].long(), num_classes=22)
      to_concat = [
          aatype.float(),
          rearrange(batch['template_torsion_angles'], '... r d -> ... (r d)'),
          rearrange(batch['template_torsion_angles_alt'], '... r d -> ... (r d)'),
          batch['template_torsion_angles_mask'].float(),
      ]
      s = torch.cat(to_concat, dim=-1)
      s = self.template_single_embedder(s.float())  # pylint: disable=not-callable
      # Concatenate the templates to the msa.
      m = torch.cat((m, s), dim=1) if exists(m) else s
      # Concatenate templates masks to the msa masks.
      # Use mask from the psi angle, as it only depends on the backbone atoms
      # from a single residue.
      s_mask = batch['template_torsion_angles_mask'][..., 2]
      m_mask = torch.cat((m_mask, s_mask), dim=1) if exists(m_mask) else s_mask
    return x, m, x_mask, m_mask


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

    heads_msa, heads_pair = commons.default_list_get(heads, 2)
    dim_head_msa, dim_head_pair = commons.default_list_get(dim_head, 2)

    self.pair_attn = commons.PairwiseAttentionBlock(
        dim_msa=dim_msa,
        dim_pairwise=dim_pairwise,
        heads=heads_pair,
        dim_head=dim_head_pair,
        dropout=attn_dropout
    )
    self.pair_ff = commons.FeedForward(dim=dim_pairwise, dropout=ff_dropout)
    if accept_msa_attn:
      self.msa_attn = commons.MsaAttentionBlock(
          dim_msa=dim_msa,
          dim_pairwise=dim_pairwise,
          heads=heads_msa,
          dim_head=dim_head_msa,
          dropout=attn_dropout,
          global_column_attn=global_column_attn,
          **kwargs
      )
      self.msa_ff = commons.FeedForward(dim=dim_msa, dropout=ff_dropout)
    if accept_frame_attn:
      self.frame_attn = commons.FrameAttentionBlock(
          dim_msa=dim_msa,
          dim_pairwise=dim_pairwise,
          heads=heads_msa,
          scalar_key_dim=dim_head_msa,
          scalar_value_dim=dim_head_msa,
          dropout=attn_dropout,
          gating=True,
          point_weight_init=5e-3,
          require_pairwise_repr=False
      )
      self.frame_ff = commons.FeedForward(dim=dim_msa, dropout=ff_dropout)
    if accept_frame_update:
      self.frame_update = commons.FrameUpdater(dim=dim_msa, dropout=attn_dropout)

  def forward(self, x, m, t, mask=None, msa_mask=None, shard_size=None):
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
        m = commons.tensor_add(m, self.msa_ff(m, shard_size=shard_size))

    # frame attention and transition
    if hasattr(self, 'frame_attn'):
      with profiler.record_function('frame_attn'):
        with autocast(enabled=False):
          # to default float
          m, x, t = m.float(), x.float(), tuple(map(lambda x: x.float(), t))
          s = self.frame_attn(
              m[..., 0, :, :], mask=msa_mask[..., 0, :, :], pairwise_repr=x, frames=t
          )
          s = commons.tensor_add(s, self.frame_ff(s, shard_size=shard_size))

        m = torch.cat((s[..., None, :, :], m[..., 1:, :, :]), dim=-3)

    # frame update
    if hasattr(self, 'frame_update'):
      with profiler.record_function('frame_update'):
        with autocast(enabled=False):
          # to default float
          m, t = m.float(), tuple(map(lambda x: x.float(), t))
          t = self.frame_update(m[..., 0, :, :], frames=t)

    # pairwise attention and transition
    with profiler.record_function('pair_attn'):
      # with autocast(enabled=False):
      x = self.pair_attn(
          x, mask=mask, msa_repr=m, msa_mask=msa_mask, shard_size=shard_size
      )
      x = commons.tensor_add(x, self.pair_ff(x, shard_size=shard_size))

    return x, m, t


def Evoformer(depth, *args, **kwargs):
  """The Evoformer in AlphaFold2
   """
  return commons.layer_stack(EvoformerBlock, depth, *args, **kwargs)


class PairformerBlock(nn.Module):
  """One Evoformer Layer
   """
  def __init__(
      self,
      *,
      dim_single,
      dim_pairwise,
      heads,
      dim_head,
      attn_dropout,
      ff_dropout,
      **kwargs
  ):
    super().__init__()

    self.pair_attn = commons.PairwiseAttentionBlock(
        dim_msa=dim_single,
        dim_pairwise=dim_pairwise,
        heads=heads,
        dim_head=dim_head,
        disabled_outer_mean=True,
        dropout=attn_dropout
    )
    self.pair_ff = commons.FeedForward(
        dim=dim_pairwise, dropout=ff_dropout, activation='SwiGLU'
    )
    self.seq_attn = commons.AttentionPairBias(
        dim_node=dim_single,
        dim_edge=dim_pairwise,
        heads=heads,
        dim_head=dim_head,
        dropout=attn_dropout,
        **kwargs
    )
    self.seq_ff = commons.FeedForward(
        dim=dim_single, dropout=ff_dropout, activation='SwiGLU'
    )

  def forward(self, s, x, cond=None, mask=None, seq_mask=None, shard_size=None):
    # pairwise attention and transition
    with profiler.record_function('pair_attn'):
      x = self.pair_attn(x, mask=mask, shard_size=shard_size)
      x = commons.tensor_add(x, self.pair_ff(x, shard_size=shard_size))

    # seq attention and transition
    with profiler.record_function('seq_attn'):
      s = self.seq_attn(s, x, cond=cond, mask=seq_mask, edge_mask=mask)
      s = commons.tensor_add(s, self.seq_ff(s, shard_size=shard_size))

    return s, x


def Pairformer(depth, *args, **kwargs):
  """The Pairformer in AlphaFold3
   """
  return commons.layer_stack(PairformerBlock, depth, *args, **kwargs)
