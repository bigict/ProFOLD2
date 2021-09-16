import unittest

import torch
from profold2.data import scn


class TestDataSet(unittest.TestCase):
    def test_scn_dataset(self):
        scn_seq = torch.tensor([[0, 1]])
        mask = scn.cloud_mask(scn_seq, boolean=True)
        result = torch.tensor(
            [[[True, True, True, True, True, False, False, False, False, False, False, False, False, False], \
              [True, True, True, True, True, True, False, False, False, False, False, False, False, False]]])
        self.assertTrue(torch.equal(mask, result))

    def test_scn_loader(self):
        print(scn.VOCAB.pad_id)
        self.assertTrue(True)
        

if __name__ == '__main__':
    unittest.main()
