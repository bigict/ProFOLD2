"""main evoformer class
 """
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat

from profold2.common import residue_constants
from profold2.model.commons import (Attention,
                                    checkpoint_sequential_nargs,
                                    embedd_dim_get,
                                    heads_get,
                                    FeedForward,
                                    MsaAttentionBlock,
                                    PairwiseAttentionBlock)

from profold2.model.functional import (distogram_from_positions,
                                       rigids_from_3x3)
from profold2.utils import exists

class TemplatePairBlock(nn.Module):
  def __init__(self,
               *,
               dim,
               heads,
               dim_head,
               attn_dropout,
               ff_dropout):
    super().__init__()

    self.layer = nn.ModuleList([
        PairwiseAttentionBlock(dim=dim,
                               heads=heads,
                               dim_head=dim_head,
                               dropout=attn_dropout,
                               disabled_outer_mean=True,
                               multiplication_first=False),
        FeedForward(dim=dim, mult=2, dropout=ff_dropout),
    ])

  def forward(self, inputs, shard_size=None):
    x, mask = inputs
    attn, ff = self.layer
    x = attn(x, mask=mask, shard_size=shard_size)
    x = x + ff(x)
    return x, mask

class TemplatePairStack(nn.Module):
  """The pair stack for templates in Alphafold2
   """
  def __init__(self, *, depth, **kwargs):
    super().__init__()

    self.layers = nn.ModuleList(
        [TemplatePairBlock(**kwargs) for _ in range(depth)])  # pylint: disable=missing-kwoa

  def forward(self, x, mask=None, shard_size=None):
    inp = (x, mask)
    x, *_ = checkpoint_sequential_nargs(self.layers,
                                        len(self.layers),
                                        inp,
                                        shard_size=shard_size)
    return x

class SingleTemplateEmbedding(nn.Module):
  """The pair stack for templates in Alphafold2
   """
  def __init__(self, *, depth, dim, dim_templ_feat=88, templ_dgram_breaks_min=3.25, templ_dgram_breaks_max=50.57, templ_dgram_breaks_num=39, use_template_unit_vector=False, **kwargs):
    super().__init__()

    self.dgram_breaks = torch.linspace(templ_dgram_breaks_min,
                                       templ_dgram_breaks_max,
                                       steps=templ_dgram_breaks_num)
    self.use_template_unit_vector = use_template_unit_vector

    self.to_pair = nn.Linear(dim_templ_feat, dim)
    self.pair_stack = TemplatePairStack(depth=depth, dim=dim, **kwargs)
    self.to_out_norm = nn.LayerNorm(dim)

  def forward(self, batch, mask_2d, shard_size=None):
    _, m, n = batch['template_seq'].shape[:3]
    template_mask = batch['template_pseudo_beta_mask']
    template_mask_2d = rearrange(template_mask, '... i -> ... i ()') * rearrange(template_mask, '... j -> ... () j')
    template_dgram = distogram_from_positions(batch['template_pseudo_beta'],
                                              self.dgram_breaks.to(device=batch['template_pseudo_beta'].device))
    to_concat = [template_dgram, template_mask_2d[...,None]]

    aatype = F.one_hot(batch['template_seq'].long(), num_classes=22)
    to_concat += [repeat(aatype, '... j d -> ... i j d', i=n), repeat(aatype, '... i d -> ... i j d', j=n)]

    n_idx = residue_constants.atom_order['N']
    ca_idx = residue_constants.atom_order['CA']
    c_idx = residue_constants.atom_order['C']
    R, t = rigids_from_3x3(batch['template_coord'], indices=(c_idx, ca_idx, n_idx))

    local_points = torch.einsum('... i w h,... i j w -> ... i j h', R, rearrange(t, '... j d -> ... () j d') - rearrange(t, '... i d ->... i () d'))
    inv_distance_scalar = torch.rsqrt(1e-6 + torch.sum(
        torch.square(local_points), dim=-1, keepdims=True))

    # Backbone affine mask: whether the residue has C, CA, N
    # (the template mask defined above only considers pseudo CB).
    template_mask = (
        batch['template_coord_mask'][..., n_idx] *
        batch['template_coord_mask'][..., ca_idx] *
        batch['template_coord_mask'][..., c_idx])
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
    x = self.pair_stack(x, mask=repeat(mask_2d, 'b ... -> (b m) ...', m=m), shard_size=shard_size)
    x = rearrange(x, '(b m) ... -> b m ...', m=m)

    x = self.to_out_norm(x)
    return x

class TemplateEmbedding(nn.Module):
  """The Template representation in Alphafold2
   """
  def __init__(self,
               dim,
               depth,
               dim_templ,
               heads=4,
               dim_head=16,
               attn_dropout=0.,
               dim_msa=None,
               **kwargs):
    super().__init__()

    self.template_pairwise_embedder = SingleTemplateEmbedding(depth=depth,
                                                              dim=dim_templ,
                                                              dim_head=dim_head,
                                                              heads=heads,
                                                              attn_dropout=attn_dropout,
                                                              **kwargs)

    self.template_pointwise_attn = Attention(dim=(dim, dim_templ),
                                             heads=heads,
                                             dim_head=dim_head,
                                             dropout=attn_dropout,
                                             gating=False)

    self.template_single_embedder = nn.Sequential(
        nn.Linear(22+14*2+7, dim_msa),
        nn.ReLU(),
        nn.Linear(dim_msa, dim_msa)) if dim_msa else None

  def forward(self,
              x, x_mask,
              m, m_mask,
              batch,
              shard_size=None):

    n = batch['template_seq'].shape[2]
    template_mask = batch['template_mask']

    # Make sure the weights are shared across templates by constructing the
    # embedder here.
    template_pair_representation = self.template_pairwise_embedder(
        batch, x_mask, shard_size=shard_size)
    t = rearrange(x, '... i j d -> ... (i j) () d')
    context = rearrange(template_pair_representation, '... m i j d -> ... (i j) m d')
    t = self.template_pointwise_attn(t,
                                     context=context,
                                     context_mask=template_mask)
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
      s = self.template_single_embedder(s.float())
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
  def __init__(self,
               *,
               dim,
               heads,
               dim_head,
               attn_dropout,
               ff_dropout,
               global_column_attn=False):
    super().__init__()

    dim_msa, dim_pairwise = embedd_dim_get(dim, 'msa', 'pairwise')
    heads_msa, heads_pairwsie = heads_get(heads, 'msa', 'pairwise')
    heads_dim_msa, heads_dim_pairwsie = heads_get(dim_head, 'msa', 'pairwise')
    self.layer = nn.ModuleList([
        PairwiseAttentionBlock(dim=dim,
                               heads=heads_pairwsie,
                               dim_head=heads_dim_pairwsie,
                               dropout=attn_dropout),
        FeedForward(dim=dim_pairwise, dropout=ff_dropout),
        MsaAttentionBlock(dim=dim,
                          heads=heads_msa,
                          dim_head=heads_dim_msa,
                          dropout=attn_dropout,
                          global_column_attn=global_column_attn),
        FeedForward(dim=dim_msa, dropout=ff_dropout),
    ])

  def forward(self, inputs, shard_size=None):
    x, m, mask, msa_mask = inputs
    attn, ff, msa_attn, msa_ff = self.layer

    # msa attention and transition
    m = msa_attn(m, mask=msa_mask, pairwise_repr=x, shard_size=shard_size)
    m = msa_ff(m) + m

    # pairwise attention and transition
    x = attn(x, mask=mask, msa_repr=m, msa_mask=msa_mask, shard_size=shard_size)
    x = ff(x) + x

    return x, m, mask, msa_mask


class Evoformer(nn.Module):
  """The Evoformer in Alphafold2
   """
  def __init__(self, *, depth, **kwargs):
    super().__init__()
    self.layers = nn.ModuleList(
        [EvoformerBlock(**kwargs) for _ in range(depth)])  # pylint: disable=missing-kwoa

  def forward(self, x, m, mask=None, msa_mask=None, shard_size=None):
    inp = (x, m, mask, msa_mask)
    x, m, *_ = checkpoint_sequential_nargs(self.layers,
                                           len(self.layers),
                                           inp,
                                           shard_size=shard_size)
    return x, m
