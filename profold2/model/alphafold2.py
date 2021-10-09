from dataclasses import dataclass
import logging
import random

import torch
from torch import nn
from einops import rearrange,repeat

from profold2 import constants
from profold2.model.evoformer import *
from profold2.model.features import FeatureBuilder
from profold2.model.head import HeaderBuilder
from profold2.model.mlm import MLM
from profold2.utils import *

# constants

@dataclass
class Recyclables:
    single_msa_repr_row: torch.Tensor
    pairwise_repr: torch.Tensor

    def asdict(self):
        return dict(single_msa_repr_row=self.single_msa_repr_row, pairwise_repr=self.pairwise_repr)

@dataclass
class _ReturnValues:
    theta: torch.Tensor = None
    phi: torch.Tensor = None
    omega: torch.Tensor = None
    msa_mlm_loss: torch.Tensor = None
    recyclables: Recyclables = None
    headers: dict = None
    loss: torch.Tensor = None

class ReturnValues(_ReturnValues):
    def __init__(self, **kwargs):
        if 'recyclables' in kwargs and exists(kwargs['recyclables']):
            kwargs['recyclables'] = Recyclables(**kwargs['recyclables'])
        super().__init__(**kwargs)
    
    def asdict(self):
        return dict(theta=self.theta,
                phi=self.phi,
                omega=self.omega,
                msa_mlm_loss=self.msa_mlm_loss,
                recyclables=self.recyclables.asdict() if exists(self.recyclables) else self.recyclables,
                headers=self.headers,
                loss=self.loss)

class Alphafold2(nn.Module):
    def __init__(
        self,
        *,
        dim,
        max_seq_len = 2048,
        depth = 6,
        heads = 8,
        dim_head = 64,
        max_rel_dist = 32,
        num_tokens = constants.NUM_AMINO_ACIDS,
        num_embedds = constants.NUM_EMBEDDS_TR,
        max_num_msas = constants.MAX_NUM_MSA,
        max_num_templates = constants.MAX_NUM_TEMPLATES,
        extra_msa_evoformer_layers = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        templates_dim = 32,
        templates_embed_layers = 4,
        templates_angles_feats_dim = 55,
        predict_angles = False,
        symmetrize_omega = False,
        disable_token_embed = False,
        mlm_mask_prob = 0.15,
        mlm_random_replace_token_prob = 0.1,
        mlm_keep_token_same_prob = 0.1,
        mlm_exclude_token_ids = (0,),
        device = None,
        feats = None,
        headers = None
    ):
        super().__init__()
        self.dim = dim

        # token embedding
        self.token_emb = nn.Embedding(num_tokens + 1, dim) if not disable_token_embed else Always(0)
        self.disable_token_embed = disable_token_embed
        self.to_pairwise_repr = PairwiseEmbedding(dim, max_rel_dist)

        # extra msa embedding
        self.extra_msa_evoformer = Evoformer(
            dim = dim,
            depth = depth,
            seq_len = max_seq_len,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            global_column_attn = True
        )

        # template embedding
        self.template_embedding = TemplateEmbedding(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                max_seq_len = max_seq_len,
                attn_dropout = attn_dropout,
                templates_dim = templates_dim,
                templates_embed_layers = templates_embed_layers,
                templates_angles_feats_dim = templates_angles_feats_dim)

        # projection for angles, if needed
        self.predict_angles = predict_angles
        self.symmetrize_omega = symmetrize_omega

        if predict_angles:
            self.to_prob_theta = nn.Linear(dim, constants.THETA_BUCKETS)
            self.to_prob_phi   = nn.Linear(dim, constants.PHI_BUCKETS)
            self.to_prob_omega = nn.Linear(dim, constants.OMEGA_BUCKETS)

        # custom embedding projection
        self.embedd_project = nn.Linear(num_embedds, dim)

        # main trunk modules
        self.evoformer = Evoformer(
            dim = dim,
            depth = depth,
            seq_len = max_seq_len,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        # MSA SSL MLM

        self.mlm = MLM(
            dim = dim,
            num_tokens = num_tokens,
            mask_id = num_tokens, # last token of embedding is used for masking
            mask_prob = mlm_mask_prob,
            keep_token_same_prob = mlm_keep_token_same_prob,
            random_replace_token_prob = mlm_random_replace_token_prob,
            exclude_token_ids = mlm_exclude_token_ids
        )

        # recycling params
        self.recycling_msa_norm = nn.LayerNorm(dim)
        self.recycling_pairwise_norm = nn.LayerNorm(dim)

        self.feat_builder = FeatureBuilder(feats)

        self.headers = HeaderBuilder.build(dim, headers, device=device)
        for n, h, _ in self.headers:
          self.add_module(f'head_{n}', h)

        self.to(device=device)

    def forward(
        self,
        extra_msa = None,
        extra_msa_mask = None,
        templates_feats = None,
        templates_mask = None,
        templates_angles = None,
        recyclables = None,
        return_recyclables = False,
        batch = None,
        compute_loss = True
    ):
        batch = self.feat_builder(batch)

        seq, mask, seq_embed, seq_index = map(batch.get, ('seq', 'mask', 'emb_seq', 'seq_index'))
        msa, msa_mask, msa_embed = map(batch.get, ('msa', 'msa_mask', 'emb_msa'))
        embedds, = map(batch.get, ('embedds',))

        assert not (self.disable_token_embed and not exists(seq_embed)), 'sequence embedding must be supplied if one has disabled token embedding'
        assert not (self.disable_token_embed and not exists(msa_embed)), 'msa embedding must be supplied if one has disabled token embedding'

        # if MSA is not passed in, just use the sequence itself
        if not exists(embedds) and not exists(msa):
            msa = rearrange(seq, 'b n -> b () n')
            msa_mask = rearrange(mask, 'b n -> b () n')

        # assert on sequence length
        assert not exists(msa) or msa.shape[-1] == seq.shape[-1], 'sequence length of MSA and primary sequence must be the same'

        # variables
        b, n, device = *seq.shape[:2], seq.device

        # embed main sequence
        x = self.token_emb(seq)

        if exists(seq_embed):
            x += seq_embed

        # mlm for MSAs
        if self.training and exists(msa):
            original_msa = msa
            msa_mask = default(msa_mask, lambda: torch.ones_like(msa).bool())

            noised_msa, replaced_msa_mask = self.mlm.noise(msa, msa_mask)
            msa = noised_msa

        # embed multiple sequence alignment (msa)
        if exists(msa):
            m = self.token_emb(msa)

            if exists(msa_embed):
                m = m + msa_embed

            # add single representation to msa representation
            m = m + rearrange(x, 'b n d -> b () n d')

            # get msa_mask to all ones if none was passed
            msa_mask = default(msa_mask, lambda: torch.ones_like(msa).bool())

        elif exists(embedds):
            m = self.embedd_project(embedds)
            
            # get msa_mask to all ones if none was passed
            msa_mask = default(msa_mask, lambda: torch.ones_like(embedds[..., -1]).bool())
        else:
            raise Error('either MSA or embeds must be given')

        # derive pairwise representation
        x, x_mask = self.to_pairwise_repr(x, mask, seq_index)

        # add recyclables, if present
        if exists(recyclables):
            m[:, 0] = m[:, 0] + self.recycling_msa_norm(recyclables.single_msa_repr_row)
            x = x + self.recycling_pairwise_norm(recyclables.pairwise_repr)

        # embed templates, if present
        x, x_mask, m, msa_mask = self.template_embedding(
                x, x_mask, m, msa_mask,
                templates_feats=templates_feats,
                templates_angles=templates_angles,
                templates_mask=templates_mask)

        # embed extra msa, if present
        if exists(extra_msa):
            extra_m = self.token_emb(msa)
            extra_msa_mask = default(extra_msa_mask, torch.ones_like(extra_m).bool())

            x, extra_m = self.extra_msa_evoformer(
                x,
                extra_m,
                mask = x_mask,
                msa_mask = extra_msa_mask
            )

        # trunk
        x, m = self.evoformer(
            x,
            m,
            mask = x_mask,
            msa_mask = msa_mask)

        # ready output container
        ret = ReturnValues()

        # calculate theta and phi before symmetrization
        if self.predict_angles:
            ret.theta_logits = self.to_prob_theta(x)
            ret.phi_logits = self.to_prob_phi(x)

        representations = {'pair': x, 'single': m[:, 0]}
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
                ret.headers[name].update(loss)
                if exists(ret.loss):
                    ret.loss += loss['loss'] * options.get('weight', 1.0)
                else:
                    ret.loss = loss['loss'] * options.get('weight', 1.0)

        # calculate mlm loss, if training
        if self.training and exists(msa):
            num_msa = original_msa.shape[1]
            ret.msa_mlm_loss = self.mlm(m[:, :num_msa], original_msa, replaced_msa_mask)

        # determine angles, if specified
        if self.predict_angles:
            omega_input = trunk_embeds if self.symmetrize_omega else x
            ret.omega_logits = self.to_prob_omega(omega_input)

        if return_recyclables:
            single_msa_repr_row, pairwise_repr = map(torch.detach, (representations['single'], representations['pair']))
            ret.recyclables = Recyclables(single_msa_repr_row, pairwise_repr)

        return ret.asdict()

class Alphafold2WithRecycling(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.impl = Alphafold2(**kwargs)
        logging.debug('{}'.format(self.impl))

    def forward(self, num_recycle=0, **kwargs):
        assert num_recycle >= 0

        ret = ReturnValues()
        if self.training:
            num_recycle = random.randint(0, num_recycle)

        for i in range(num_recycle):
            ret = ReturnValues(**self.impl(recyclables=ret.recyclables, return_recyclables=True, compute_loss=False, **kwargs))
            if 'tmscore' in ret.headers:
                logging.debug('{}/{} tmscore: {}'.format(i, num_recycle, ret.headers['tmscore']['loss'].item()))

        ret = ReturnValues(**self.impl(recyclables=ret.recyclables, return_recyclables=False, compute_loss=True, **kwargs))
        if 'tmscore' in ret.headers:
            logging.debug('{}/{} tmscore: {}'.format(num_recycle, num_recycle, ret.headers['tmscore']['loss'].item()))

        return ret.asdict()
