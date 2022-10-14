from dataclasses import dataclass
import functools
import logging
import random

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat

from profold2.common import residue_constants
from profold2.data import esm
from profold2.model.commons import embedd_dim_get
from profold2.model.evoformer import *
from profold2.model.head import HeaderBuilder
from profold2.model.mlm import MLM
from profold2.model.sequence import ESMEmbedding
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
        return dict(
                msa_mlm_loss=self.msa_mlm_loss,
                recyclables=self.recyclables.asdict() if exists(self.recyclables) else self.recyclables,
                headers=self.headers,
                loss=self.loss)

class Alphafold2(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth = 6,
        heads = 8,
        dim_head = 64,
        max_rel_dist = 32,
        num_tokens = len(residue_constants.restypes_with_x),
        embedd_dim = esm.ESM_EMBED_DIM,
        extra_msa_evoformer_layers = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        templates_dim = 32,
        templates_embed_layers = 4,
        templates_angles_feats_dim = 55,
        disable_token_embed = False,
        mlm_mask_prob = 0.15,
        mlm_random_replace_token_prob = 0.1,
        mlm_keep_token_same_prob = 0.1,
        mlm_exclude_token_ids = (0,),
        headers = None
    ):
        super().__init__()

        self.dim = dim
        dim_single, dim_pairwise = embedd_dim_get(dim)

        # token embedding
        self.token_emb = nn.Embedding(num_tokens + 1, dim_single) if not disable_token_embed else Always(0)
        self.disable_token_embed = disable_token_embed
        self.to_pairwise_repr = PairwiseEmbedding(dim, max_rel_dist)

        # extra msa embedding
        self.extra_msa_evoformer = Evoformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            global_column_attn = True
        )

        # template embedding
        self.template_embedding = TemplateEmbedding(
                dim = dim_single,
                dim_head = dim_head,
                heads = heads,
                attn_dropout = attn_dropout,
                templates_dim = templates_dim,
                templates_embed_layers = templates_embed_layers,
                templates_angles_feats_dim = templates_angles_feats_dim)

        # custom embedding projection
        self.embedd_project = nn.Linear(embedd_dim, dim_single)

        self.sequence = ESMEmbedding(*esm.ESM_MODEL_PATH)

        # main trunk modules
        self.evoformer = Evoformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        # MSA SSL MLM

        self.mlm = MLM(
            dim = dim_single,
            num_tokens = num_tokens,
            mask_id = num_tokens, # last token of embedding is used for masking
            mask_prob = mlm_mask_prob,
            keep_token_same_prob = mlm_keep_token_same_prob,
            random_replace_token_prob = mlm_random_replace_token_prob,
            exclude_token_ids = mlm_exclude_token_ids
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
        extra_msa = None,
        extra_msa_mask = None,
        templates_feats = None,
        templates_mask = None,
        templates_angles = None,
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
        if not self.training:  # and n > self.sequence.max_input_len:
            embedds, contacts = [], None  # torch.zeros((b, n, n), device=device)
            max_input_len = sequence_max_input_len if exists(sequence_max_input_len) else self.sequence.max_input_len
            max_step_len = sequence_max_step_len if exists(sequence_max_step_len) else self.sequence.max_step_len
            for k in range(0, n, max_step_len):
                i, j = k, min(k + max_input_len, n)
                delta = 0 if i == 0 else max_input_len - max_step_len
                if i > 0 and j < i + max_input_len:
                    delta += i + max_input_len - n
                    i = n - max_input_len
                labels = self.sequence.batch_convert(
                        [s[i:j] for s in batch['str_seq']], device=device)
                clips = dict([(s, dict(i=i, j=j, l=n)) for s in range(len(batch['str_seq']))])
                # x, y = self.sequence(labels, repr_layer=esm.ESM_EMBED_LAYER, return_contacts=False)
                x = self.sequence(labels,
                        repr_layer=esm.ESM_EMBED_LAYER,
                        clips=clips,
                        return_contacts=False)
                p = delta // 2
                if embedds and p > 0:
                    l = embedds[-1].shape[-2]
                    assert l > p
                    embedds[-1] = embedds[-1][...,:l-p,:]
                embedds.append(x[...,delta-p:j-i,:])
                # contacts[...,i:j,i:j] = y
                if j < k + max_input_len:
                    break
            embedds = torch.cat(embedds, dim=-2)
        else:
            labels = self.sequence.batch_convert(batch['str_seq'], device=device)
            embedds, contacts = self.sequence(
                        labels,
                        repr_layer=esm.ESM_EMBED_LAYER,
                        clips = batch.get('clips'),
                        return_contacts=True)

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

        # calculate mlm loss, if training
        if self.training and exists(msa):
            num_msa = original_msa.shape[1]
            ret.msa_mlm_loss = self.mlm(m[:, :num_msa], original_msa, replaced_msa_mask)

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
