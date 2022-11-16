from dataclasses import dataclass
import functools
import logging
import random

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat

from profold2.common import residue_constants
from profold2.model.commons import embedd_dim_get
from profold2.model.evoformer import *
from profold2.model.head import HeaderBuilder
from profold2.model.sequence import (
    ESMEmbedding,
    ESM_EMBED_DIM,
    ESM_EMBED_LAYER,
    ESM_MODEL_PATH)
from profold2.utils import *

logger = logging.getLogger(__name__)

@dataclass
class Recyclables:
    single_msa_repr_row: torch.Tensor
    pairwise_repr: torch.Tensor

    def asdict(self):
        return dict(single_msa_repr_row=self.single_msa_repr_row, pairwise_repr=self.pairwise_repr)

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
                recyclables=self.recyclables.asdict() if exists(self.recyclables) else self.recyclables,
                headers=self.headers,
                loss=self.loss)

class Alphafold2(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=6,
        heads=8,
        dim_head=64,
        max_rel_dist=32,
        num_tokens=len(residue_constants.restypes_with_x),
        embedd_dim=ESM_EMBED_DIM,
        attn_dropout=0.,
        ff_dropout=0.,
        disable_token_embed=False,
        headers=None
    ):
        super().__init__()

        self.dim = dim
        dim_single, dim_pairwise = embedd_dim_get(dim)

        # token embedding
        self.token_emb = nn.Embedding(num_tokens + 1, dim_single) if not disable_token_embed else Always(0)
        self.disable_token_embed = disable_token_embed
        self.to_pairwise_repr = PairwiseEmbedding(dim, max_rel_dist)

        # custom embedding projection
        if embedd_dim > 0:
            self.embedd_project = nn.Linear(embedd_dim, dim_single)
            self.sequence = ESMEmbedding(*ESM_MODEL_PATH)

        # main trunk modules
        self.evoformer = Evoformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        # recycling params
        self.recycling_msa_norm = nn.LayerNorm(dim_single)
        self.recycling_pairwise_norm = nn.LayerNorm(dim_pairwise)

        self.headers = HeaderBuilder.build(dim, headers, parent=self)

    def embeddings(self):
        return dict(token=self.token_emb.weight, pairwise=self.to_pairwise_repr.embeddings())

    def forward(
        self,
        batch,
        *,
        sequence_max_input_len=None,
        sequence_max_step_len=None,
        return_recyclables=False,
        compute_loss=True,
        shard_size=None
    ):
        seq, mask, seq_embed, seq_index = map(batch.get, ('seq', 'mask', 'emb_seq', 'seq_index'))
        msa, msa_mask, msa_embed = map(batch.get, ('msa', 'msa_mask', 'emb_msa'))
        msa, msa_mask, msa_embed = None, None, None
        embedds, = map(batch.get, ('embedds',))
        recyclables, = map(batch.get, ('recyclables',))

        # variables
        b, n, device = *seq.shape[:2], seq.device

        assert not (self.disable_token_embed and not exists(seq_embed)), 'sequence embedding must be supplied if one has disabled token embedding'
        assert not (self.disable_token_embed and not exists(msa_embed)), 'msa embedding must be supplied if one has disabled token embedding'

        representations = {}

        # embed multiple sequence alignment (msa)
        if hasattr(self, 'sequence'):
            embedds, contacts = self.sequence(
                    batch, sequence_max_input_len=self.sequence_max_input_len, sequence_max_step_len=self.sequence_max_step_len)
            representations['mlm'] = dict(representations=embedds,
                    contacts=contacts)

            embedds = rearrange(embedds, 'b l c -> b () l c')

        # if MSA is not passed in, just use the sequence itself
        if not exists(embedds) and not exists(msa):
            msa = rearrange(seq, 'b n -> b () n')
            msa_mask = rearrange(mask, 'b n -> b () n')

        # assert on sequence length
        assert not exists(msa) or msa.shape[-1] == seq.shape[-1], 'sequence length of MSA and primary sequence must be the same'

        # embed main sequence
        x = self.token_emb(seq)

        if exists(seq_embed):
            x += seq_embed

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
            m = rearrange(x, 'b n d -> b () n d')
            msa_mask = rearrange(mask, 'b n -> b () n')
            #raise Error('either MSA or embeds must be given')

        # derive pairwise representation
        x, x_mask = self.to_pairwise_repr(x, mask, seq_index)

        # add recyclables, if present
        if exists(recyclables):
            m[:, 0] = m[:, 0] + self.recycling_msa_norm(recyclables.single_msa_repr_row)
            x = x + self.recycling_pairwise_norm(recyclables.pairwise_repr)

        # trunk
        x, m = self.evoformer(
            x,
            m,
            mask = x_mask,
            msa_mask = msa_mask,
            shard_size = shard_size)

        # ready output container
        ret = ReturnValues()

        representations.update(pair=x, single=m[:, 0], single_init=m[:, 0])

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
                    if exists(ret.loss):
                        ret.loss += loss['loss'] * options.get('weight', 1.0)
                    else:
                        ret.loss = loss['loss'] * options.get('weight', 1.0)

        if return_recyclables:
            single_msa_repr_row, pairwise_repr = map(torch.detach, (representations['single'], representations['pair']))
            ret.recyclables = Recyclables(single_msa_repr_row, pairwise_repr)

        return ret.asdict()

class Alphafold2WithRecycling(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.impl = Alphafold2(**kwargs)
        logger.debug('{}'.format(self))

    def embeddings(self):
        return self.impl.embeddings()

    def forward(self, batch, *, num_recycle=0, **kwargs):
        assert num_recycle >= 0

        # variables
        seq = batch['seq']
        b, n, device = *seq.shape[:2], seq.device
        # FIXME: fake recyclables
        if 'recyclables' not in batch:
            dim_single, dim_pairwise = embedd_dim_get(self.impl.dim)
            batch['recyclables'] = Recyclables(single_msa_repr_row=torch.zeros(b, n, dim_single, device=device),
                        pairwise_repr=torch.zeros(b, n, n, dim_pairwise, device=device))

        ret = ReturnValues()
        if self.training:
            num_recycle = random.randint(0, num_recycle)
        cycling_function = functools.partial(self.impl, return_recyclables=True, compute_loss=False, **kwargs)

        with torch.no_grad():
            for i in range(num_recycle):
                ret = ReturnValues(**cycling_function(batch))
                if 'tmscore' in ret.headers:
                    logger.debug('{}/{} pid:{} tmscore: {}'.format(i, num_recycle, ','.join(batch['pid']), ret.headers['tmscore']['loss'].item()))
                batch['recyclables'] = ret.recyclables

        ret = ReturnValues(**self.impl(batch, return_recyclables=False, compute_loss=True, **kwargs))
        if 'tmscore' in ret.headers:
            logger.debug('{}/{} pid:{} tmscore: {}'.format(num_recycle, num_recycle, ','.join(batch['pid']), ret.headers['tmscore']['loss'].item()))

        return ret.asdict()
