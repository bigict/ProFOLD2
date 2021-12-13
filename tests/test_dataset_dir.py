import unittest
from datetime import datetime

from profold2.data.custom import *
#from profold2.data.custom_new import *

class TestDataSet(unittest.TestCase):
    def test_custom_dataset(self):
        data = ProteinStructureDataset('casp14')
        print(data[0])
        self.assertTrue(True)

    def test_custom_loader(self):
        data = load('casp14', batch_size=1, max_seq_len=255, feat_flags=ProteinStructureDataset.FEAT_ALL)
        s = datetime.now()
        for i, batch in enumerate(iter(data)):
            if i >= 500:
                break
            self.assertTrue('seq' in batch)
            self.assertTrue(batch['seq'].shape[0], 2)
            self.assertTrue('coord' in batch)
        e = datetime.now()
        print(e - s)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()

