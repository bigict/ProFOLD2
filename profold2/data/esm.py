import torch

from profold2.utils import default,exists

# embedding related constants

MSA_EMBED_LAYER = 12
MSA_EMBED_DIM = 768
MSA_MODEL_PATH = ["facebookresearch/esm:main", "esm_msa1b_t12_100M_UR50S"]

ESM_EMBED_LAYER = 33
ESM_EMBED_DIM = 1280
ESM_MODEL_PATH = ["facebookresearch/esm:main", "esm1b_t33_650M_UR50S"]

# adapted from https://github.com/facebookresearch/esm

_extractor_dict = {}

class ESMEmbeddingExtractor:
    def __init__(self, repo_or_dir, model):
        self.model, alphabet = torch.hub.load(repo_or_dir, model)
        self.batch_converter = alphabet.get_batch_converter()
        self.max_input_len = 1022
        self.max_step_len = 511

    def extract(self, seqs, repr_layer=None, return_contacts=False, device=None):
        """ Returns the ESM embeddings for a protein.
            Inputs:
            * seq: ( (b,) L,) tensor of ints (in sidechainnet int-char convention)
            Outputs: tensor of (batch, n_seqs, L, embedd_dim)
                * n_seqs: number of sequences in the MSA. 1 for ESM-1b
                * embedd_dim: number of embedding dimensions. 1280 for ESM-1b
        """
        # use ESM transformer
        device = default(device, getattr(seqs, 'device', None))
        max_seq_len = max(map(lambda x: len(x[1]), seqs))
        # repr_layer = default(repr_layer, ESM_EMBED_LAYER)

        assert not return_contacts or max_seq_len <= self.max_input_len
        assert not return_contacts or exists(repr_layer)

        representations, contacts = [], []
        # Extract per-residue representations
        with torch.no_grad():
            if exists(repr_layer):
                for i in range(0, max_seq_len, self.max_step_len):
                    j = min(i + self.max_input_len, max_seq_len)
                    delta = 0 if i == 0 else self.max_input_len - self.max_step_len
                    if i > 0 and j < i + self.max_input_len:
                        delta += i + self.max_input_len - max_seq_len
                        i = max_seq_len - self.max_input_len

                    batch_seqs = [(label, seq[i:j]) for label, seq in seqs]
                    batch_labels, batch_strs, batch_tokens = self.batch_converter(batch_seqs)

                    if exists(device):
                        batch_tokens = batch_tokens.to(device)

                    results = self.model(batch_tokens, repr_layers=[repr_layer], return_contacts=return_contacts)
                    # index 0 is for start token. so take from 1 one
                    representations.append(results['representations'][repr_layer][...,delta+1:j-i+1,:])
                    if return_contacts:
                        contacts.append(results['contacts'])
                    
                    if j >= max_seq_len:
                        break
            else:
                batch_tokens, batch_strs, batch_tokens = self.batch_converter(seqs)
                results = self.model.embed_tokens(batch_tokens)
                representations.append(results[...,1:max_seq_len+1,:])

        if return_contacts:
            return torch.cat(representations, dim=1), torch.cat(contacts, dim=1)
        return torch.cat(representations, dim=1)

    @staticmethod
    def get(repo_or_dir, model, device=None):
        global _extractor_dict

        if (repo_or_dir, model) not in _extractor_dict:
            obj = ESMEmbeddingExtractor(repo_or_dir, model)
            if exists(device):
                obj.model.to(device=device)
            _extractor_dict[(repo_or_dir, model)] = obj
        return _extractor_dict[(repo_or_dir, model)]
