import functools
import logging

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat

from profold2.common import residue_constants
from profold2.utils import default, exists

# embedding related constants
MSA_EMBED_LAYER = 12
MSA_EMBED_DIM = 768
MSA_MODEL_PATH = ["facebookresearch/esm:main", "esm_msa1b_t12_100M_UR50S"]

ESM_EMBED_LAYER = 33
ESM_EMBED_DIM = 1280
ESM_MODEL_PATH = ["facebookresearch/esm:main", "esm1b_t33_650M_UR50S"]

logger = logging.getLogger(__name__)

def _esm_lookup(self, alphabet, tokens, clips=None):
    padding_mask = tokens.eq(self.padding_idx)  # B, T

    x = self.embed_scale * self.embed_tokens(tokens)
    
    if getattr(self.args, "token_dropout", False):
        x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
        # x: B x T x C
        mask_ratio_train = 0.15 * 0.8
        src_lengths = (~padding_mask).sum(-1)
        mask_ratio_observed = (tokens == self.mask_idx).sum(-1).float() / src_lengths
        x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]
    
    def pseudo_embed_positions(b):
        pos_idx = 0
        if b in clips:
            pos_idx = clips[b].get('i', 0)

        if pos_idx > 0:
            l = tokens.shape[1]
            logger.debug('_esm_lookup: batch=%d, pos_idx=%d, l=%d', b, pos_idx, l)
            pos_idx = min(pos_idx, self.embed_positions.max_positions - l)
            unks = torch.full((1, pos_idx) + tokens.shape[2:],
                    alphabet.unk_idx, dtype=tokens.dtype, device=tokens.device)
            pseudo_tokens = torch.cat((tokens[b:b+1,:1,...], unks, tokens[b:b+1,1:,...]), dim=1)
            pseudo_pos_embed = self.embed_positions(pseudo_tokens)
            return torch.cat((
                pseudo_pos_embed[:,:1,...],
                pseudo_pos_embed[:,pos_idx+1:,...]), dim=1)
        return self.embed_positions(tokens[b:b+1,...])

    if exists(clips) and len(clips) > 0:
        x = x + torch.cat(
                [pseudo_embed_positions(b) for b in range(tokens.shape[0])], dim=0)
    else:
        x = x + self.embed_positions(tokens)
    return x

def _esm_forward(self, x, tokens, repr_layers=[], need_head_weights=False, return_contacts=False):
    if return_contacts:
        need_head_weights = True

    assert tokens.ndim == 2
    padding_mask = tokens.eq(self.padding_idx)  # B, T

    #x = self.embed_scale * self.embed_tokens(tokens)
    #
    #if getattr(self.args, "token_dropout", False):
    #    x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
    #    # x: B x T x C
    #    mask_ratio_train = 0.15 * 0.8
    #    src_lengths = (~padding_mask).sum(-1)
    #    mask_ratio_observed = (tokens == self.mask_idx).sum(-1).float() / src_lengths
    #    x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]
    #
    #x = x + self.embed_positions(tokens)

    if self.model_version == "ESM-1b":
        if self.emb_layer_norm_before:
            x = self.emb_layer_norm_before(x)
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

    repr_layers = set(repr_layers)
    hidden_representations = {}
    if 0 in repr_layers:
        hidden_representations[0] = x

    if need_head_weights:
        attn_weights = []

    # (B, T, E) => (T, B, E)
    x = x.transpose(0, 1)

    if not padding_mask.any():
        padding_mask = None

    for layer_idx, layer in enumerate(self.layers):
        x, attn = layer(
            x, self_attn_padding_mask=padding_mask, need_head_weights=need_head_weights
        )
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x.transpose(0, 1)
        if need_head_weights:
            # (H, B, T, T) => (B, H, T, T)
            attn_weights.append(attn.transpose(1, 0))

    if self.model_version == "ESM-1b":
        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        x = self.lm_head(x)
    else:
        x = F.linear(x, self.embed_out, bias=self.embed_out_bias)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

    result = {"logits": x, "representations": hidden_representations}
    if need_head_weights:
        # attentions: B x L x H x T x T
        attentions = torch.stack(attn_weights, 1)
        if self.model_version == "ESM-1":
            # ESM-1 models have an additional null-token for attention, which we remove
            attentions = attentions[..., :-1]
        if padding_mask is not None:
            attention_mask = 1 - padding_mask.type_as(attentions)
            attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
            attentions = attentions * attention_mask[:, None, None, :, :]
        result["attentions"] = attentions
        if return_contacts:
            contacts = self.contact_head(tokens, attentions)
            result["contacts"] = contacts

    return result

class ESMEmbedding(nn.Module):
    def __init__(self, repo_or_dir, model):
        super().__init__()

        self.model, self.alphabet = torch.hub.load(repo_or_dir, model)

        if (hasattr(self.model, 'args') and
                hasattr(self.model.args, 'max_positions')):
            if exists(self.model.padding_idx):
                self.max_input_len = self.model.args.max_positions - self.model.padding_idx - 1 # HACK: :(
            else:
                self.max_input_len = self.model.args.max_positions
        else:
            self.max_input_len = 1022
        self.max_step_len = self.max_input_len // 2

    def batch_convert(self, seqs, device=None):
        batch_converter = self.alphabet.get_batch_converter()
        _, _, batch_tokens = batch_converter(
                list(zip([f'{i}' for i in range(len(seqs))], seqs)))
        if exists(device):
            batch_tokens = batch_tokens.to(device)
        return batch_tokens

    def forward(self, batch, clips=None, repr_layer=ESM_EMBED_LAYER, return_contacts=False):
        """ Returns the ESM embeddings for a protein.
            Inputs:
            * seq: ( (b,) L,) tensor of ints (in sidechainnet int-char convention)
            Outputs: tensor of (batch, n_seqs, L, embedd_dim)
                * n_seqs: number of sequences in the MSA. 1 for ESM-1b
                * embedd_dim: number of embedding dimensions. 1280 for ESM-1b
        """
        # use ESM transformer
        assert not return_contacts or exists(repr_layer)

        def run_function(repr_layer, return_contacts):
            def forward(x, batch_tokens):
                results = _esm_forward(self.model, x, batch_tokens, repr_layers=[repr_layer], return_contacts=return_contacts)
                # index 0 is for start token. so take from 1 one
                representations = results['representations'][repr_layer][...,1:-1,:]
                # logits = results['logits']
                if return_contacts:
                    return representations, results['contacts']
                return representations
            return forward

        # Extract per-residue representations
        x = _esm_lookup(self.model, self.alphabet, batch, clips=clips)
        if torch.is_grad_enabled():
            return checkpoint(run_function(repr_layer, return_contacts), x, batch)
        return run_function(repr_layer, return_contacts)(x, batch)
