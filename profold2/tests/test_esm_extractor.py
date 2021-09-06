import unittest

import torch

from profold2.data.esm import *

class TestUtils(unittest.TestCase):
    def test_esm_extractor(self):
        esm = ESMEmbeddingExtractor(*ESM_MODEL_PATH)
        # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
        data = [
            ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
            ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE")
        ]
        representations, contacts = esm.extract(data, repr_layer=ESM_EMBED_LAYER, return_contacts=True)
        # nodes
        # representations: (b l d)
        # contacts: (b l l)
        max_seq_len = max(map(lambda x: len(x[1]), data))
        self.assertEqual(representations.shape, (len(data), max_seq_len, ESM_EMBED_DIM))
        self.assertEqual(contacts.shape, (len(data), max_seq_len, max_seq_len))

    def test_msa_extractor(self):
        esm = ESMEmbeddingExtractor(*MSA_MODEL_PATH)
        # Make an "MSA" of size 3
        data = [
            ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
            ("protein2", "MHTVRQSRLKSIVRILEMSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
            ("protein3", "MHTVRQSRLKSIVRILEMSKEPVSGAQL---LSVSRQVIVQDIAYLRSLGYNIVAT----VLAGG"),
        ]
        representations, contacts = esm.extract(data, repr_layer=MSA_EMBED_LAYER, return_contacts=True)
        # nodes
        # representations: (b n l d)
        # contacts: (b l l)
        self.assertEqual(representations.shape, (1, 3, 65, MSA_EMBED_DIM))
        self.assertEqual(contacts.shape, (1, 65, 65))

if __name__ == '__main__':
    unittest.main()
