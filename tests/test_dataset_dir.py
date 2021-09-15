import unittest

from profold2.data.custom import *

class TestDataSet(unittest.TestCase):
    def test_custom_dataset(self):
        data = ProteinStructureDataset('casp14')
        #print(data[0])
        self.assertTrue(True)

    def test_custom_loader(self):
        data = load('casp14', batch_size=2)
        for batch in iter(data):
            self.assertTrue('seq' in batch)
            self.assertTrue(batch['seq'].shape[0], 2)
            self.assertTrue('coord' in batch)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()

