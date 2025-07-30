"""Diffusion model for generating 3D-structure"""
import functools
import logging
import math
from typing import Any, Optional, Union

from tqdm.auto import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from profold2.common import chemical_components, residue_constants
from profold2.model import accelerator, atom_layout, commons, functional
from profold2.utils import compose, default, env, exists

logger = logging.getLogger(__name__)


class InputFeatureEmbedder(nn.Module):
  """InputFeatureEmbedder
    """
  def __init__(
      self,
      dim=(128, 16),
      dim_token=384,
      num_tokens=len(residue_constants.restypes_with_x),
      depth=3,
      heads=4,
      atom_query_window_size=32,
      atom_key_window_size=128,
      atom_feats=None
  ):
    super().__init__()

    self.atom_encoder = AtomAttentionEncoder(
        dim=dim,
        dim_token=dim_token,
        depth=depth,
        heads=heads,
        atom_query_window_size=atom_query_window_size,
        atom_key_window_size=atom_key_window_size,
        atom_feats=atom_feats
    )
    self.num_tokens = num_tokens

  def forward(self, batch, shard_size=None):
    single_repr, *_ = self.atom_encoder(batch, shard_size=shard_size)
    return torch.cat(
        (F.one_hot(batch['seq'].long(), self.num_tokens + 1), single_repr), dim=-1  # pylint: disable=not-callable
    )


class RelativePositionEncoding(nn.Module):
  """RelativePositionEncoding
    """
  def __init__(self, dim, r_max=32, s_max=2):
    super().__init__()

    _, dim_pair = commons.embedd_dim_get(dim)
    self.r_max = r_max
    self.s_max = s_max

    # (d_{ij}^{seq_index}, d_{ij}^{token_index}, b_{ij}^{seq_entity}, d{ij}^{seq_sym}
    # 2*(r_{max} + 1) + 2*(r_{max} + 1) + 1 + 2*(s_{max} + 1)
    self.proj = nn.Linear(2 * 2 * self.r_max + 2 * self.s_max + 7, dim_pair, bias=False)

  def forward(
      self, seq_index, seq_color, seq_sym, seq_entity, token_index, shard_size=None
  ):
    def run_proj(seq_index_j, seq_color_j, seq_sym_j, seq_entity_j, token_index_j):
      bij_seq_index = (seq_index[..., :, None] == seq_index_j[..., None, :])
      bij_seq_color = (seq_color[..., :, None] == seq_color_j[..., None, :])
      bij_seq_entity = (seq_entity[..., :, None] == seq_entity_j[..., None, :])

      dij_seq_index = F.one_hot(  # pylint: disable=not-callable
          torch.where(
              bij_seq_color,
              torch.clamp(
                  seq_index[..., :, None] - seq_index_j[..., None, :],
                  min=-self.r_max, max=self.r_max
              ) + self.r_max,
              2 * self.r_max + 1
          ).long(),
          2 * (self.r_max + 1)
      )
      dij_token_index = F.one_hot(  # pylint: disable=not-callable
          torch.where(
              bij_seq_color * bij_seq_index,
              torch.clamp(
                  token_index[..., :, None] - token_index_j[..., None, :],
                  min=-self.r_max, max=self.r_max
              ) + self.r_max,
              2 * self.r_max + 1
          ).long(),
          2 * (self.r_max + 1)
      )
      dij_seq_sym = F.one_hot(  # pylint: disable=not-callable
          torch.where(
              seq_entity[..., :, None] == seq_entity_j[..., None, :],
              torch.clamp(
                  seq_sym[..., :, None] - seq_sym_j[..., None, :],
                  min=-self.s_max, max=self.s_max
              ) + self.s_max,
              2 * self.s_max + 1
          ).long(),
          2 * (self.s_max + 1)
      )
      with accelerator.autocast(enabled=False):
        feats = torch.cat(
            (dij_seq_index, dij_token_index, bij_seq_entity[..., None], dij_seq_sym),
            dim=-1
        ).float()
        return self.proj(feats)

    return functional.sharded_apply(
        run_proj, [seq_index, seq_color, seq_sym, seq_entity, token_index],
        shard_size=None if self.training else shard_size,
        shard_dim=-1,
        cat_dim=-2
    )


class AtomPairwiseEmbedding(nn.Module):
  """AtomPairwiseEmbedding
    """
  def __init__(self, dim):
    super().__init__()

    dim_single, dim_pair = commons.embedd_dim_get(dim)  # dim_atom_{single, pair}
    self.to_pairwise_repr = nn.Linear(dim_single, dim_pair * 2, bias=False)

  def forward(self, x, query_window_size=32, key_window_size=128):
    x_i, x_j = torch.chunk(self.to_pairwise_repr(x), 2, dim=-1)
    x_i, x_j, *_ = atom_layout.unfold(
        query_window_size, key_window_size, dim=-2, q=x_i, k=x_j
    )
    return x_i[..., None, :] + x_j[..., None, :, :]


class DiffusionTransformerBlock(nn.Module):
  """DiffusionTransformerBlock"""
  def __init__(self, *, dim, dim_cond, heads, group_size=1, dropout=0., **kwargs):
    super().__init__()

    dim_single, dim_pair = commons.embedd_dim_get(dim)

    class _Block(nn.Module):
      """Wrap AttentionWithBias and ConditionedFeedForward to Block"""
      def __init__(self):
        super().__init__()

        self.attn = commons.AttentionWithBias(
            dim=dim_single,
            heads=heads,
            dim_cond=dim_cond,
            q_use_bias=True,
            o_use_bias=False,
            g_use_bias=False,
            **kwargs
        )
        self.ff = commons.ConditionedFeedForward(dim_single, dim_cond)
        self.dropout_fn = functools.partial(commons.shaped_dropout, p=dropout)

      def forward(
          self,
          x,
          *,
          cond,
          mask,
          context,
          context_cond,
          context_mask,
          pair_bias,
          pair_mask,
          shard_size=None
      ):
        def dropout_wrap(f, *args, **kwargs):
          shape = x.shape[:-2] + (1, 1)
          return self.dropout_fn(
              f(*args, **kwargs), shape=shape, training=self.training
          )

        # run attn and ff parallel: x += attn(x) + ff(x)
        x = commons.tensor_add(
            x,
            dropout_wrap(
                self.attn,
                x,
                cond=cond,
                mask=mask,
                context=context,
                context_cond=context_cond,
                context_mask=context_mask,
                pair_bias=pair_bias,
                pair_mask=pair_mask
            ) + dropout_wrap(self.ff, x, cond, shard_size=shard_size)
        )

        return x

    self.net = commons.layer_stack(
        _Block, group_size, checkpoint_segment_size=group_size
    )
    self.edges_to_attn_bias = nn.Sequential(
        nn.LayerNorm(dim_pair, bias=False),
        nn.Linear(dim_pair, heads, bias=False),
        Rearrange('... i j h -> ... h i j')
    )

  def forward(
      self,
      x,
      single_cond,
      pair_cond,
      *,
      mask=None,
      pair_bias=None,
      pair_mask=None,
      context_creator=None,
      shard_size=None
  ):
    cond, context, context_cond, context_mask, padding = (
        single_cond, None, None, None, 0
    )
    if exists(context_creator):
      x, cond, mask, context, context_cond, context_mask, padding = context_creator(
          x, single_cond, mask
      )

    pair_bias = commons.tensor_add(
        self.edges_to_attn_bias(pair_cond), default(pair_bias, 0)
    )

    x = self.net(
        x,
        cond=cond,
        mask=mask,
        context=context,
        context_cond=context_cond,
        context_mask=context_mask,
        pair_bias=pair_bias,
        pair_mask=pair_mask,
        shard_size=shard_size
    )

    if exists(context_creator):
      x = rearrange(x, '... c i d -> ... (c i) d')
      if padding != 0:
        x = x[..., :-padding, :]

    return x, single_cond, pair_cond


class AtomTransformer(nn.Module):
  """AtomTransformer"""
  def __init__(
      self,
      dim,
      depth=3,
      heads=4,
      dim_head=32,
      query_window_size=32,
      key_window_size=128
  ):
    super().__init__()

    dim_single, _ = commons.embedd_dim_get(dim)  # dim_atom_{single, pair}

    self.query_window_size = query_window_size
    self.key_window_size = key_window_size

    self.difformer = commons.layer_stack(
        DiffusionTransformerBlock,
        depth,
        checkpoint_segment_size=env(
            'profold2_atomtransformer_checkpoint_segment_size', defval=1, dtype=int
        ),
        dim=dim,
        dim_cond=dim_single,
        has_context=True,
        heads=heads,
        dim_head=dim_head
    )

  def query_context_create(self, query, cond, mask=None):
    # HACK: split query to (query, context)
    query, context, padding = atom_layout.unfold(
        self.query_window_size, self.key_window_size, dim=-2, q=query, k=query
    )
    query_cond, context_cond, *_ = atom_layout.unfold(
        self.query_window_size, self.key_window_size, dim=-2, q=cond, k=cond
    )
    if exists(mask):
      query_mask, context_mask, *_ = atom_layout.unfold(
          self.query_window_size, self.key_window_size, dim=-1, q=mask, k=mask
      )
    else:
      query_mask, context_mask = None, None
    return query, query_cond, query_mask, context, context_cond, context_mask, padding

  def forward(
      self,
      single_repr,
      single_cond,
      pair_cond,
      mask=None,
      pair_mask=None,
      shard_size=None
  ):
    query, *_ = self.difformer(
        single_repr,
        single_cond,
        pair_cond,
        mask=mask,
        pair_mask=pair_mask,
        context_creator=self.query_context_create,
        shard_size=shard_size
    )
    if exists(mask):
      query = query * mask[..., None]
    return query


class AtomAttentionEncoder(nn.Module):
  """AtomAttentionEncoder"""
  def __init__(
      self,
      dim,
      dim_cond=(384, 128),
      dim_token=768,
      depth=3,
      heads=4,
      atom_query_window_size=32,
      atom_key_window_size=128,
      atom_feats=None,
      has_coords=False
  ):
    super().__init__()

    dim_single, dim_pair = commons.embedd_dim_get(dim)  # dim_atom_{single, pair}
    dim_cond_single, dim_cond_pair = commons.embedd_dim_get(dim_cond)

    self.atom_feats = default(
        atom_feats,
        (
            ('ref_pos', 3, nn.Identity()),
            ('ref_charge', 1, compose(torch.arcsinh, Rearrange('... -> ... ()'))),
            ('ref_mask', 1, Rearrange('... -> ... ()')),
            (
                'ref_element',
                chemical_components.elem_type_num,
                functools.partial(
                    F.one_hot, num_classes=chemical_components.elem_type_num
                )
            ),
            (
                'ref_atom_name_chars',
                chemical_components.name_char_channel * chemical_components.name_char_num,  # pylint: disable=line-too-long
                compose(
                    functools.partial(
                        F.one_hot, num_classes=chemical_components.name_char_num
                    ), Rearrange('... c d -> ... (c d)')
                )
            )
        )
    )
    dim_atom_feats = sum(d for _, d, _ in self.atom_feats)
    self.to_single_cond = nn.Linear(dim_atom_feats, dim_single, bias=False)
    self.to_pair_cond = nn.Linear(3 + 1 + 1, dim_pair, bias=False)

    if has_coords:
      self.from_trunk_single_cond = nn.Sequential(
          nn.LayerNorm(dim_cond_single, bias=False),
          nn.Linear(dim_cond_single, dim_single, bias=False)
      )
      self.from_trunk_pair_cond = nn.Sequential(
          nn.LayerNorm(dim_cond_pair, bias=False),
          nn.Linear(dim_cond_pair, dim_pair, bias=False)
      )

    self.outer_add = AtomPairwiseEmbedding(dim)
    # self.outer_ff = commons.FeedForward(dim_pair)
    self.outer_ff = commons.layer_stack(
        nn.Sequential, 3, nn.ReLU(), nn.Linear(dim_pair, dim_pair, bias=False)
    )

    self.transformer = AtomTransformer(
        dim,
        depth=depth,
        heads=heads,
        query_window_size=atom_query_window_size,
        key_window_size=atom_key_window_size
    )

    self.to_out = nn.Linear(dim_single, dim_token, bias=False)

  def forward(
      self,
      batch,
      r_noisy=None,
      trunk_single_cond=None,
      trunk_pair_cond=None,
      mask=None,
      batch_size=None,
      shard_size=None
  ):
    atom_to_token_idx = batch['atom_to_token_idx']

    # create the atom single conditioning: Embed per-atom meta data
    atom_single_cond = self.to_single_cond(
        torch.cat([f(batch[k]) for k, _, f in self.atom_feats], dim=-1)
    ) * batch['ref_mask'][..., None]

    # TODO: add ref_mask
    # embed offsets between atom reference position, pairwise inverse squared
    # distances, and the valid mask.
    ref_pos_i, ref_pos_j, *_ = atom_layout.unfold(
        self.transformer.query_window_size,
        self.transformer.key_window_size,
        dim=-2,
        q=batch['ref_pos'],
        k=batch['ref_pos']
    )
    ref_space_uid_i, ref_space_uid_j, *_ = atom_layout.unfold(
        self.transformer.query_window_size,
        self.transformer.key_window_size,
        dim=-1,
        q=batch['ref_space_uid'],
        k=batch['ref_space_uid']
    )
    ref_mask_i, ref_mask_j, *_ = atom_layout.unfold(
        self.transformer.query_window_size,
        self.transformer.key_window_size,
        dim=-1,
        q=batch['ref_mask'],
        k=batch['ref_mask']
    )
    dij_ref = ref_pos_i[..., :, None, :] - ref_pos_j[..., None, :, :]
    vij_ref = (ref_space_uid_i[..., :, None] == ref_space_uid_j[..., None, :])
    bij_ref = (ref_mask_i[..., :, None] * ref_mask_j[..., None, :])
    pair_cond = self.to_pair_cond(
        torch.cat(
            (
                dij_ref * vij_ref[..., None] * bij_ref[..., None],
                1 / (1 + torch.sum(dij_ref**2, dim=-1, keepdim=True)) * vij_ref[..., None],  # pylint: disable=line-too-long
                vij_ref[..., None]
            ),
            dim=-1
        )
    )

    # initialise the atom single representation as the single conditioning.
    query, query_cond = atom_single_cond, atom_single_cond

    # if provided, add trunk embeddings and noisy positions.
    assert not hasattr(self, 'from_trunk_single_cond') ^ exists(trunk_single_cond)
    if exists(trunk_single_cond):
      # broadcast the single embedding from the trunk
      query_cond = commons.tensor_add(
          atom_layout.gather(
              self.from_trunk_single_cond(trunk_single_cond), atom_to_token_idx
          ), query_cond
      )
    assert not hasattr(self, 'from_trunk_pair_cond') ^ exists(trunk_pair_cond)
    if exists(trunk_pair_cond):
      # broadcast the pair embedding from the trunk
      pair_cond = commons.tensor_add(
          pair_cond,
          atom_layout.gather(
              self.from_trunk_pair_cond(trunk_pair_cond),
              atom_to_token_idx,
              self.transformer.query_window_size,
              self.transformer.key_window_size
          )
      )
    if exists(batch_size):
      query_cond = rearrange(query_cond, '... i d -> ... () i d')
    if exists(r_noisy):
      # add the noisy positions.
      query = commons.tensor_add(r_noisy, query_cond)
    if exists(batch_size):
      pair_cond = rearrange(pair_cond, '... c i j d -> ... () c i j d')

    # add the combined single conditioning to the pair representation.
    pair_cond = commons.tensor_add(
        pair_cond,
        self.outer_add(
            F.relu(query_cond),
            self.transformer.query_window_size,
            self.transformer.key_window_size
        )
    )
    # run a small MLP on the pair activations.
    pair_cond = commons.tensor_add(pair_cond, self.outer_ff(pair_cond))
    # cross attention transformer.
    query = self.transformer(
        query,
        query_cond,
        pair_cond,
        mask=mask[..., None, :] if exists(batch_size) else mask,
        shard_size=shard_size
    )
    # aggregate per-atom representation to per-token representation.
    token_single_cond = F.relu(self.to_out(query))
    atom_to_token_idx = repeat(
        atom_to_token_idx, '... i -> ... i d', d=token_single_cond.shape[-1]
    )
    if exists(r_noisy) and exists(batch_size):
      atom_to_token_idx = repeat(
          atom_to_token_idx, '... i d -> ... m i d', m=token_single_cond.shape[-3]
      )
    with accelerator.autocast(enabled=False):
      token_single_cond = functional.scatter_mean(
          atom_to_token_idx, token_single_cond.float(), dim=-2
      )

    query_skip, query_cond_skip, pair_skip = query, query_cond, pair_cond
    return token_single_cond, query_skip, query_cond_skip, pair_skip


class AtomAttentionDecoder(nn.Module):
  """AtomAttentionDecoder"""
  def __init__(
      self,
      dim,
      dim_token=768,
      depth=3,
      heads=4,
      atom_query_window_size=32,
      atom_key_window_size=128,
  ):
    super().__init__()

    dim_single, _ = commons.embedd_dim_get(dim)  # dim_atom_{single, pair}

    self.from_token = nn.Linear(dim_token, dim_single, bias=False)
    self.transformer = AtomTransformer(
        dim,
        depth=depth,
        heads=heads,
        query_window_size=atom_query_window_size,
        key_window_size=atom_key_window_size
    )

  def forward(
      self,
      batch,
      token_single_cond,
      query_skip,
      context_skip,
      pair_skip,
      mask=None,
      batch_size=None,
      shard_size=None
  ):
    atom_to_token_idx = batch['atom_to_token_idx']
    if exists(batch_size):
      atom_to_token_idx = repeat(
          atom_to_token_idx, '... i -> ... m i', m=token_single_cond.shape[-3]
      )
    # broadcast per-token activations to per-atom activations and add the skip
    # connection
    atom_single_cond = commons.tensor_add(
        atom_layout.gather(self.from_token(token_single_cond), atom_to_token_idx),
        query_skip
    )
    # cross attention transformer
    atom_single_cond = self.transformer(
        atom_single_cond,
        context_skip,
        pair_skip,
        mask=mask[..., None, :] if exists(batch_size) else mask,
        shard_size=shard_size
    )
    return atom_single_cond


class FourierEmbedding(nn.Module):
  """FourierEmbedding
    """
  def __init__(self, dim, seed=2147483647):
    super().__init__()

    generator = torch.Generator()
    generator.manual_seed(
        env('profold2_fourier_embedding_seed', defval=seed, dtype=int)
    )
    # randomly generate weight/bias once before training
    self.w = nn.Parameter(torch.randn(dim, generator=generator), requires_grad=False)
    self.b = nn.Parameter(torch.randn(dim, generator=generator), requires_grad=False)

  def forward(self, t):
    # compute embeddings. scale w by t
    v = t[..., None] * self.w + self.b
    return torch.cos(2 * torch.pi * v)


class DiffusionConditioning(nn.Module):
  """DiffusionConditioning"""
  def __init__(self, dim, dim_noise=256, dim_inputs=449, sigma_data=16.0):
    super().__init__()

    dim_single, dim_pair = commons.embedd_dim_get(dim)  # dim_cond_{single, pair}

    self.from_single = nn.Sequential(
        nn.LayerNorm(dim_single + dim_inputs, bias=False),
        nn.Linear(dim_single + dim_inputs, dim_single, bias=False)
    )
    self.from_pairwise = nn.Sequential(
        nn.LayerNorm(dim_pair * 2, bias=False),
        nn.Linear(dim_pair * 2, dim_pair, bias=False)
    )
    self.from_pos_emb = RelativePositionEncoding(dim=dim)
    self.from_noise = nn.Sequential(
        FourierEmbedding(dim_noise),
        nn.LayerNorm(dim_noise, bias=False),
        nn.Linear(dim_noise, dim_single, bias=False)
    )
    self.to_single = commons.residue_stack(
        commons.FeedForward, 2, dim_single, mult=2, activation='SwiGLU', use_bias=False
    )
    self.to_pairwise = commons.residue_stack(
        commons.FeedForward, 2, dim_pair, mult=2, activation='SwiGLU', use_bias=False
    )

    self.sigma_data = sigma_data

  def forward(
      self,
      batch,
      *,
      noise_level,
      inputs,
      trunk_single_cond,
      trunk_pair_cond,
      batch_size=None,
      shard_size=None
  ):
    # single conditioning
    s = self.from_single(torch.cat((trunk_single_cond, inputs), dim=-1))

    # HACK: fix Automatic Mixed Precision
    with accelerator.autocast(enabled=False):
      t = self.from_noise(
          (torch.log(noise_level.float()) - math.log(self.sigma_data)) / 4.
      )

    s = commons.tensor_add(
        rearrange(s, '... i d -> ... () i d') if exists(batch_size) else s,
        rearrange(t, '... d -> ... () d')
    )
    s = self.to_single(s, shard_size=shard_size)

    # pair conditioning
    x = self.from_pairwise(
        torch.cat(
            (
                trunk_pair_cond,
                self.from_pos_emb(
                    batch['seq_index'],
                    batch['seq_color'],
                    batch['seq_sym'],
                    batch['seq_entity'],
                    default(batch.get('token_index'), batch['seq_index']),
                    shard_size=shard_size
                )
            ),
            dim=-1
        )
    )
    x = self.to_pairwise(x, shard_size=shard_size)

    return s, x


class DiffusionModule(nn.Module):
  """DiffusionModule"""
  def __init__(
      self,
      dim,
      dim_atom=(128, 16),
      dim_token=768,
      dim_inputs=449,
      dim_noise=256,
      sigma_data=16.0,
      atom_encoder_depth=3,
      atom_encoder_head_num=4,
      transformer_depth=24,
      transformer_group_size=4,
      transformer_head_num=16,
      transformer_dim_head=48,
      atom_decoder_depth=3,
      atom_decoder_head_num=4,
      atom_query_window_size=32,
      atom_key_window_size=128,
  ):
    super().__init__()

    dim_single, dim_pair = commons.embedd_dim_get(dim)  # dim_cond_{single, pair}
    dim_atom_single, *_ = commons.embedd_dim_get(dim_atom)

    self.conditioning = DiffusionConditioning(
        dim, dim_inputs=dim_inputs, dim_noise=dim_noise, sigma_data=sigma_data
    )

    self.from_coord = nn.Linear(3, dim_atom_single, bias=False)
    self.atom_encoder = AtomAttentionEncoder(
        dim=dim_atom,
        dim_cond=dim,
        dim_token=dim_token,
        depth=atom_encoder_depth,
        heads=atom_encoder_head_num,
        atom_query_window_size=atom_query_window_size,
        atom_key_window_size=atom_key_window_size,
        has_coords=True
    )
    self.transformer_in = nn.Sequential(
        nn.LayerNorm(dim_single, bias=False),
        nn.Linear(dim_single, dim_token, bias=False)
    )
    assert transformer_depth % transformer_group_size == 0
    self.transformer = commons.layer_stack(
        DiffusionTransformerBlock,
        depth=transformer_depth // transformer_group_size,
        checkpoint_segment_size=env(
            'profold2_diffuser_checkpoint_segment_size', defval=1, dtype=int
        ),
        dim=(dim_token, dim_pair),
        dim_cond=dim_single,
        group_size=transformer_group_size,
        heads=transformer_head_num,
        dim_head=transformer_dim_head
    )
    self.transformer_out = nn.LayerNorm(dim_token, bias=False)
    self.atom_decoder = AtomAttentionDecoder(
        dim=dim_atom,
        dim_token=dim_token,
        depth=atom_decoder_depth,
        heads=atom_decoder_head_num,
        atom_query_window_size=atom_query_window_size,
        atom_key_window_size=atom_key_window_size
    )
    self.to_coord = nn.Sequential(
        nn.LayerNorm(dim_atom_single, bias=False),
        nn.Linear(dim_atom_single, 3, bias=False)
    )

  @property
  def sigma_data(self):
    return self.conditioning.sigma_data

  def forward(
      self,
      batch,
      x_noisy,
      *,
      x_mask,
      noise_level,
      inputs,
      trunk_single_cond,
      trunk_pair_cond,
      use_conditioning=True,
      batch_size=None,
      shard_size=None
  ):
    # mask conditioning features if use_conditioning is False
    inputs = inputs * use_conditioning
    trunk_single_cond = trunk_single_cond * use_conditioning
    trunk_pair_cond = trunk_pair_cond * use_conditioning

    # conditioning
    single_cond, pair_cond = self.conditioning(
        batch,
        noise_level=noise_level,
        inputs=inputs,
        trunk_single_cond=trunk_single_cond,
        trunk_pair_cond=trunk_pair_cond,
        batch_size=batch_size,
        shard_size=shard_size
    )

    ##################################################
    # EDM: r_noisy = c_in * x_noisy
    #      where c_in = 1 / sqrt(sigma_data^2 + sigma^2)
    ##################################################

    # scale positions to dimensionless
    with accelerator.autocast(enabled=False):
      r_noisy = x_noisy.float() / torch.sqrt(
          self.conditioning.sigma_data**2 + noise_level.float()**2
      )[..., None, None]
      r_noisy = self.from_coord(r_noisy)

    ##################################################
    # EDM: r_update = F_theta(r_noisy, c_noise(sigma))
    ##################################################

    # sequence-local Atom Attention and aggregation to coasrse-grained tokens
    token_cond, query_skip, context_skip, pair_skip = self.atom_encoder(
        batch,
        r_noisy,
        trunk_single_cond=trunk_single_cond,
        trunk_pair_cond=pair_cond,
        mask=x_mask,
        batch_size=batch_size,
        shard_size=shard_size
    )

    # full self-attention on token level
    token_cond = commons.tensor_add(token_cond, self.transformer_in(single_cond))
    token_cond, *_ = self.transformer(
        token_cond,
        single_cond,
        pair_cond[..., None, :, :, :] if exists(batch_size) else pair_cond,
        mask=batch['mask'][..., None, :] if exists(batch_size) else batch['mask']
    )
    token_cond = self.transformer_out(token_cond)

    # broadcast token activations to atoms and run Sequence-local Atom Attention
    r_update = self.atom_decoder(
        batch,
        token_cond,
        query_skip,
        context_skip,
        pair_skip,
        mask=x_mask,
        batch_size=batch_size,
        shard_size=shard_size
    )

    ##################################################
    # EDM: D = c_skip * x_noisy + c_out * r_update
    #      c_skip = sigma_data^2 / (sigma_data^2 + sigma^2)
    #      c_out = (sigma_data * sigma) / sqrt(sigma_data^2 + sigma^2)
    #      s_ratio = 1 + (sigma / sigma_data)^2
    #      c_skip = 1 / s_ratio
    #      c_out = sigma / sqrt(s_ratio)
    ##################################################

    # rescale updates to positions and combine with input positions
    with accelerator.autocast(enabled=False):
      r_update = self.to_coord(r_update.float())
      noise_level = noise_level[..., None, None].float()
      s_ratio = 1 + (noise_level / self.conditioning.sigma_data)**2
      x_denoised = (
          x_noisy.float() / s_ratio + r_update * noise_level / torch.sqrt(s_ratio)
      )

    return x_denoised

  def loss_scale(self, noise_level):
    ##################################################
    # EDM: L = \lambda(sigma) || x_pred - x_true ||
    #    where \lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
    ##################################################
    return (noise_level**2 + self.sigma_data**2) / (noise_level * self.sigma_data)**2


class DiffusionSampler(nn.Module):
  """DiffusionSampler"""
  def __init__(
      self,
      dim,
      dim_atom=(128, 16),
      dim_token=768,
      dim_inputs=449,
      dim_noise=256,
      sigma_data=16.0,
      sigma_data_ca_w=0.0,
      sigma_mean=-1.2,
      sigma_std=1.5,
      sigma_min=4e-4,
      sigma_max=160,
      rho=7.,
      trans_scale_factor=1.0,
      diffuser_atom_encoder_depth=3,
      diffuser_atom_encoder_head_num=4,
      diffuser_transformer_depth=24,
      diffuser_transformer_group_size=4,
      diffuser_transformer_head_num=16,
      diffuser_atom_decoder_depth=3,
      diffuser_atom_decoder_head_num=4,
      diffuser_atom_query_window_size=32,
      diffuser_atom_key_window_size=128,
  ):
    super().__init__()

    self.diffuser = DiffusionModule(
        dim,
        dim_atom=dim_atom,
        dim_token=dim_token,
        dim_inputs=dim_inputs,
        dim_noise=dim_noise,
        sigma_data=sigma_data,
        atom_encoder_depth=diffuser_atom_encoder_depth,
        atom_encoder_head_num=diffuser_atom_encoder_head_num,
        transformer_depth=diffuser_transformer_depth,
        transformer_group_size=diffuser_transformer_group_size,
        transformer_head_num=diffuser_transformer_head_num,
        atom_decoder_depth=diffuser_atom_decoder_depth,
        atom_decoder_head_num=diffuser_atom_decoder_head_num,
        atom_query_window_size=diffuser_atom_query_window_size,
        atom_key_window_size=diffuser_atom_key_window_size
    )

    assert 0 <= sigma_data_ca_w <= 1
    self.sigma_data_ca_w = sigma_data_ca_w
    # noise sampler - train
    self.sigma_mean = sigma_mean
    self.sigma_std = sigma_std
    # noise scheduler - inference
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.rho = rho
    # centre random augmenter
    self.trans_scale_factor = trans_scale_factor

  def forward(
      self,
      batch: dict[str, Any],
      inputs: Optional[torch.Tensor] = None,
      trunk_single_cond: Optional[torch.Tensor] = None,
      trunk_pair_cond: Optional[torch.Tensor] = None,
      use_conditioning: bool = True,
      diffuser_batch_size: Optional[int] = None,
      shard_size: Optional[int] = None
  ):
    x_true, x_mask = atom_layout.flatten(
        batch['atom_to_token_idx'],
        batch['atom_within_token_idx'],
        batch['coord'],
        batch['coord_mask']
    )
    assert x_true.shape == batch['ref_pos'].shape
    assert x_mask.shape == batch['ref_mask'].shape
    # apply random rotation and translation
    x_noisy = self.centre_random_augmenter(
        x_true, mask=x_mask, batch_size=diffuser_batch_size
    )
    # sigma: independent noise-level [..., N_sample]
    noise_level = self.noise_sampler(
        *x_true.shape[:-2], batch_size=diffuser_batch_size, device=x_true.device
    )
    # noise
    noise = self.noise_apply(
        batch,
        torch.randn_like(x_noisy),
        noise_level[..., None, None],
        batch_size=diffuser_batch_size
    )
    # denoised_x
    x_denoised = self.diffuser(
        batch,
        x_noisy=x_noisy + noise,
        x_mask=x_mask,
        noise_level=noise_level,
        inputs=inputs,
        trunk_single_cond=trunk_single_cond,
        trunk_pair_cond=trunk_pair_cond,
        use_conditioning=use_conditioning,
        batch_size=diffuser_batch_size,
        shard_size=shard_size
    )
    return x_true, x_mask, x_noisy, x_denoised, noise_level

  def loss_scale(self, noise_level):
    return self.diffuser.loss_scale(noise_level)

  def noise_apply(self, batch, noise, noise_level, batch_size=None):
    w = self.sigma_data_ca_w
    with accelerator.autocast(enabled=False):
      if w > 0:
        atom_to_token_idx, atom_within_token_idx, atom_padding_token_idx = map(
            lambda key: batch[key],
            ('atom_to_token_idx', 'atom_within_token_idx', 'atom_padding_token_idx')
        )
        if exists(batch_size):
          atom_to_token_idx, atom_within_token_idx, atom_padding_token_idx = map(
              lambda t: repeat(t, '... i -> ... m i', m=batch_size),
              (atom_to_token_idx, atom_within_token_idx, atom_padding_token_idx)
          )
        ca_idx = residue_constants.atom_order['CA']
        noise = atom_layout.unflatten(
            atom_to_token_idx, atom_padding_token_idx, noise.float()
        )
        noise = repeat(
            noise[..., ca_idx, :], '... i d -> ... i c d', c=noise.shape[-2]
        )
        noise = atom_layout.flatten(atom_to_token_idx, atom_within_token_idx, noise)
        noise = (noise * w + torch.randn_like(noise) * (1 - w)) / math.sqrt(
            w**2 + (1 - w)**2
        )
      if isinstance(noise_level, torch.Tensor):
        return noise * noise_level.float()
      return noise * noise_level

  def noise_sampler(self, *size, batch_size=None, device=None):
    """DiffusionNoiseSampler: sample the noise-level."""
    if exists(batch_size):
      size = size + (batch_size, )
    with accelerator.autocast(enabled=False):
      x = torch.randn(*size, device=device)
      return torch.exp(self.sigma_mean + self.sigma_std * x) * self.diffuser.sigma_data

  def noise_scheduler(self, steps=200):
    """DiffusionNoiseScheduler: schedule the noise-level (time steps)."""
    # t = torch.linspace(0, 1, steps + 1, device=device, dtype=dtype)
    s_min, s_max = self.sigma_min**(1. / self.rho), self.sigma_max**(1. / self.rho)
    return [
        self.diffuser.sigma_data * (s_max + (s_min - s_max) * t / steps)**self.rho
        for t in range(steps + 1)
    ]
    # return self.sigma_data * (s_max + (s_min - s_max) * t)**self.rho

  def initial_sampler(self, *size, batch_size=None, device=None):
    if exists(batch_size):
      size = size[:-1] + (batch_size, size[-1])
    return torch.randn(*size, 3, device=device)

  def centre_random_augmenter(self, x, mask=None, batch_size=None):
    if exists(mask):
      c = functional.masked_mean(value=x, mask=mask[..., None], dim=-2, keepdim=True)
    else:
      c = torch.mean(x, dim=-2, keepdim=True)

    x = commons.tensor_sub(x, c)
    if exists(batch_size):
      x = repeat(x, '... i d -> ... n i d', n=batch_size)

    with accelerator.autocast(enabled=False):
      R, t = functional.rigids_from_randn(*x.shape[:-2], device=x.device)  # pylint: disable=invalid-name
      x = functional.rigids_apply(
          (R[..., None, :, :], t[..., None, :] * self.trans_scale_factor), x.float()
      )

    return x

  def sample(
      self,
      batch,
      inputs=None,
      trunk_single_cond=None,
      trunk_pair_cond=None,
      steps=200,
      gamma0: float = 0.8,
      gamma_min: float = 1.0,
      noise_scale_lambda: float = 1.003,
      step_scale_eta: float = 1.5,
      use_conditioning: bool = True,
      diffuser_batch_size: Optional[int] = None,
      shard_size: Optional[int] = None
  ):
    assert not self.training

    diffuser_shard_size = None
    if exists(diffuser_batch_size):
      diffuser_shard_size = min(
          env('profold2_diffuser_sampling_shard_size', defval=1, dtype=int),
          diffuser_batch_size
      )

    tau_list = self.noise_scheduler(steps=steps)

    x_mask = batch['ref_mask']
    x = self.noise_apply(
        batch,
        self.initial_sampler(
            *x_mask.shape, batch_size=diffuser_batch_size, device=x_mask.device
        ),
        tau_list[0],
        batch_size=diffuser_batch_size
    )
    for tau_idx in tqdm(range(1, len(tau_list)), desc='Diffusion Sampling'):
      x = self.centre_random_augmenter(
          x, mask=x_mask[..., None, :] if exists(diffuser_batch_size) else x_mask
      )
      gamma = gamma0 if tau_list[tau_idx] > gamma_min else 0
      t_hat = tau_list[tau_idx - 1] * (gamma + 1)
      delta = noise_scale_lambda * math.sqrt(t_hat**2 - tau_list[tau_idx - 1]**2)
      x_noisy = x + self.noise_apply(
          batch, torch.randn_like(x), delta, batch_size=diffuser_batch_size
      )

      noise_level = torch.full(x_mask.shape[:-1], t_hat, device=x_mask.device)
      if exists(diffuser_batch_size):
        noise_level = noise_level[..., None]

      # denoise
      x_denoised = functional.sharded_apply(
          functools.partial(
              self.diffuser,
              batch,
              noise_level=noise_level,
              x_mask=x_mask,
              inputs=inputs,
              trunk_single_cond=trunk_single_cond,
              trunk_pair_cond=trunk_pair_cond,
              use_conditioning=use_conditioning,
              batch_size=diffuser_shard_size,
              shard_size=shard_size
          ), [x_noisy],
          shard_size=diffuser_shard_size,
          shard_dim=-3,
          cat_dim=-3
      )

      x_delta = (x_noisy - x_denoised) * (tau_list[tau_idx] - t_hat) / t_hat
      x = x_noisy + step_scale_eta * x_delta

    atom_to_token_idx, atom_within_token_idx = map(
        lambda key: batch[key], ('atom_to_token_idx', 'atom_padding_token_idx')
    )
    if exists(diffuser_batch_size):
      atom_to_token_idx, atom_within_token_idx = map(
          lambda t: repeat(t, '... i -> ... m i', m=diffuser_batch_size),
          (atom_to_token_idx, atom_within_token_idx)
      )
    return atom_layout.unflatten(atom_to_token_idx, atom_within_token_idx, x)


def multi_chain_permutation_alignment(value, batch):
  return batch  # FIX: disable chain permutation.
