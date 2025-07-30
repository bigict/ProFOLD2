"""Diffusion model for generating 3d-structure"""

from torch import nn
from torch.nn import functional as F

from profold2.model import commons, functional
from profold2.utils import default, exists


class InputFeatureEmbedder(nn.Module):
  """InputFeatureEmbedder
    """
  def __init__(self, dim, dim_atom=(128, 16), dim_token=None):
    super().__init__()

    dim_single, _ = commons.embedd_dim_get(dim)
    dim_token = default(dim_token, dim_single)

    self.atom_encoder = AtomAttentionEncoder(dim_atom)


class RelativePositionEncoding(nn.Module):
  """RelativePositionEncoding
    """
  def __init__(self, dim, r_max=32, s_max=2):
    super().__init__()

    _, dim_pairwise = commons.embedd_dim_get(dim)
    self.r_max = r_max
    self.s_max = s_max

    # (d_{ij}^{seq_index}, d_{ij}^{token_index}, b_{ij}^{seq_entity}, d{ij}^{seq_sym}
    # 2*(r_{max} + 1) + 2*(r_{max} + 1) + 1 + 2*(s_{max} + 1)
    self.proj = nn.Linear(
        2 * 2 * self.r_max + 2 * self.s_max + 7, dim_pairwise, bias=False
    )

  def forward(
      self, seq_index, seq_color, seq_sym, seq_entity, token_index, shard_size=None
  ):
    return functional.sharded_apply(
        self._forward, [seq_index, seq_color, seq_sym, seq_entity, token_index],
        shard_size=None if self.training else shard_size,
        shard_dim=-1,
        cat_dim=-2
    )

  def _forward(
      self, seq_index, seq_color, seq_sym, seq_entity, token_index, shard_size=None
  ):
    bij_seq_index = (seq_index[..., :, None] == seq_index[..., None, :])
    bij_seq_color = (seq_color[..., :, None] == seq_color[..., None, :])
    bij_seq_entity = (seq_entity[..., :, None] == seq_entity[..., None, :])

    dij_seq_index = F.one_hot(
        torch.where(
            bij_seq_color,
            torch.clamp(
                seq_index[..., :, None] - seq_index[..., None, :],
                min=-self.r_max, max=self.r_max
            ) + self.r_max,
            2 * self.r_max + 1
        ),
        2 * self.r_max + 1
    )
    dij_token_index = F.one_hot(
        torch.where(
            bij_seq_color * bij_seq_index,
            torch.clamp(
                token_index[..., :, None] - token_index[..., None, :],
                min=-self.r_max, max=self.r_max
            ) + self.r_max,
            2 * self.r_max + 1
        ),
        2 * self.r_max + 1
    )
    dij_seq_sym = F.one_hot(
        torch.where(
            seq_entity[..., :, None] == seq_entity[..., None, :],
            torch.clamp(
                seq_sym[..., :, None] - seq_sym[..., None, :],
                min=-self.s_max, max=self.s_max
            ) + self.s_max,
            2 * self.s_max + 1
        ),
        2 * self.s_max + 1
    )
    return self.proj(
        torch.cat(
            (dij_seq_index, dij_token_index, bij_seq_entity, dij_seq_sym), dim=-1
        )
    )


class DiffusionConditioning(nn.Module):
  def __init__(self, dim, dim_noise=256, sigma_data=16.0):
    super().__init__()

    dim_single, dim_pairwise = commons.embedd_dim_get(dim)

    self.from_single = nn.Sequential(
        nn.LayerNorm(dim=dim_single), nn.Linear(dim_single, dim_single, bias=False)
    )
    self.from_pairwise = nn.Sequential(
        nn.LayerNorm(dim=dim_pairwise),
        nn.Linear(dim_pairwise, dim_pairwise, bias=False)
    )
    self.from_noise = nn.Sequential(
        commons.FourierEmbedding(dim_noise),
        nn.LayerNorm(dim=dim_noise),
        nn.Linear(dim_noise, dim_single, bias=False)
    )
    self.to_single = nn.ModuleList(
        commons.FeedForward(dim_single, mult=2) for _ in range(2)
    )
    self.to_pairwise = nn.ModuleList(
        commons.FeedForward(dim_pairwise, mult=2) for _ in range(2)
    )

    self.sigma_data = sigma_data

  def forward(self, single_repr, pairwise_repr, t_hat):
    # single conditioning
    s = self.from_single(single_repr)
    s = commons.tensor_add(s, self.from_noise(torch.log(t_hat / self.sigma_data) / 4.))
    for ff in self.to_single:
      s = commons.tensor_add(s, ff(s))

    # pair conditioning
    x = self.from_pairwise(pairwise_repr)
    for ff in self.to_pairwise:
      x = commons.tensor_add(x, ff(x))

    return s, x


class AtomTransformer(nn.Module):
  def __init__(self, dim, depth=3, heads=4, dim_atom=(128, 16)):
    super().__init__()

    self.difformer = commons.layer_stack(
        commons.DiffusionTransformerBlock, depth=depth, heads=heads
    )

  def forward(self, x):
    return self.difformer(x)


class AtomAttentionEncoder(nn.Module):
  def __init__(self, dim, dim_token=384, depth=3, heads=4, atom_feats=None):
    super().__init__()

    dim_single, dim_pairwise = commons.embedd_dim_get(dim)

    self.atom_feats = default(
        atom_feats, (
            ('ref_pos', 3),
            ('ref_charge', 1),
            ('ref_mask', 1),
            ('ref_element', 128),
            ('ref_atom_name_chars', 4 * 64)
        )
    )
    dim_atom_feats = sum(map(lambda kv: kv[1], self.atom_feats))
    self.to_single_repr = nn.Linear(dim_atom_feats, dim_single, bias=False)
    self.to_pairwise_repr = nn.Linear(3 + 1 + 1, dim_pairwise, bias=False)

    self.outer_add = commons.PairwiseEmbedding(dim)  # no position encoding
    self.outer_ff = commons.FeedForward(dim)

    self.transformer = AtomTransformer(dim, depth=depth, heads=heads)

  def forward(self, batch, single_repr=None, pairwise_repr=None):
    # create the atom single conditioning: Embed per-atom meta data
    atom_single_repr = self.to_single_repr(
        torch.cat([batch[k] for k in self.atom_feats], dim=-1)
    )

    # embed offsets between atom reference position, pairwise inverse squared
    # distances, and the valid mask.
    dij_ref = (batch['ref_pos'][..., :, None] - batch['ref_pos'][..., None, :])
    bij_ref = (
        batch['ref_space_uid'][..., :, None] == batch['ref_space_uid'][..., None, :]
    )
    atom_pairwise_repr = functional.sharded_apply(
        self.to_pairwise_repr,
        torch.cat(
            (
                dij_ref,
                1 / ( 1 + torch.sum(dij_ref**2, dim=-1, keepdim=True)),
                bij_ref[..., None]
            ),
            dim=-1
        )
        shard_size=None if self.training else shard_size,
        shard_dim=-1,
        cat_dim=-2
    ) * bij_ref[..., None]

    # initialise the atom single representation as the single conditioning.

    # add the combined single conditioning to the pair representation.
    atom_pairwise_repr = commons.tensor_add(
        atom_pairwise_repr, self.outer_add(F.relu(atom_single_repr))
    )
    # run a small MLP on the pair activations.
    atom_single_repr = commons.tensor_add(
        atom_pairwise_repr, self.outer_ff(atom_pairwise_repr)
    )
    # cross attention transformer.

    # broadcase per-token activations to per-atom activations and add the skip connection
    # cross attention transformer
    # map to positions update
    return self.transformer(x)


class AtomAttentionDecoder(nn.Module):
  def __init__(self, dim, depth, heads):
    super().__init__()

    self.transformer = AtomTransformer(dim, depth, heads)

  def forward(self, x):
    # broadcase per-token activations to per-atom activations and add the skip connection
    # cross attention transformer
    # map to positions update
    return self.transformer(x)


class DiffusionSchedule:
  pass


class DiffusionModule(nn.Module):
  def __init__(
      self,
      dim,
      dim_noise=256,
      sigma_data=16.0,
      difformer_depth=24,
      difformer_head_num=16
  ):
    super().__init__()

    dim_single, dim_pairwise = commons.embedd_dim_get(dim)

    self.conditioning = DiffusionSchedule(
        dim, dim_noise=dim_noise, sigma_data=sigma_data
    )

    self.atom_encoder = None
    self.difformer = commons.layer_stack(
        commons.DiffusionTransformerBlock,
        depth=difformer_depth,
        heads=difformer_head_num
    )
    self.atom_decoder = None
