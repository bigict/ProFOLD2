import unittest
import os
from datetime import datetime

from profold2.data.dataset import *

class TestDataSet(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'data/custom')

    # def test_custom_dataset(self):
    #     data = ProteinStructureDataset(self.data_dir)
    #     self.assertTrue(True)

    # def test_custom_dataset_msa(self):
    #     data = ProteinStructureDataset(self.data_dir, feat_flags=ProteinStructureDataset.FEAT_ALL)
    #     for item in data:
    #         print(item)
    #         self.assertTrue('seq' in item)
    #         self.assertTrue('msa' in item)
    #         self.assertTrue(item['seq'].shape == item['msa'].shape[1:])
    #     self.assertTrue(True)

    def test_custom_loader(self):
        feats = [("make_coord_mask", dict(includes=['N', 'CA', 'C', 'CB']))]
        data = load(self.data_dir, batch_size=1, max_crop_len=255, feats=feats)
        s = datetime.now()
        for i, batch in enumerate(iter(data)):
            if i >= 500:
                break
            self.assertTrue('seq' in batch)
            self.assertTrue(batch['seq'].shape[0], 2)
            self.assertTrue('coord' in batch)
            self.assertTrue('coord_mask' in batch)
            print('x', batch['coord_mask'])
            print('y', batch['coord_exists'])
        e = datetime.now()
        print(e - s)
        self.assertTrue(True)

    #def test_custom_loader_msa(self):
    #    feats = [("make_seq_profile_pairwise", dict(mask='-', density=True))]
    #    data = load(self.data_dir, batch_size=1, max_crop_len=255, feat_flags=ProteinStructureDataset.FEAT_ALL, feats=feats)
    #    s = datetime.now()
    #    for i, batch in enumerate(iter(data)):
    #        if i >= 1:
    #            break
    #        #print(batch)
    #        self.assertTrue('seq' in batch)
    #        self.assertTrue(batch['seq'].shape[0], 2)
    #        self.assertTrue('coord' in batch)
    #    e = datetime.now()
    #    print(e - s)
    #    self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()

