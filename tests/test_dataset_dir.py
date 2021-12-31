import unittest
import os
from datetime import datetime

from profold2.data.custom import *
#from profold2.data.custom_new import *

class TestDataSet(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'data/custom')

    def test_custom_dataset(self):
        data = ProteinStructureDataset(self.data_dir)
        self.assertTrue(True)

    def test_custom_loader(self):
        data = load(self.data_dir, batch_size=1, max_seq_len=255, feat_flags=ProteinStructureDataset.FEAT_ALL)
        s = datetime.now()
        for i, batch in enumerate(iter(data)):
            if i >= 500:
                break
            self.assertTrue('seq' in batch)
            self.assertTrue(batch['seq'].shape[0], 2)
            self.assertTrue('coord' in batch)
            print(batch)
        e = datetime.now()
        print(e - s)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()

