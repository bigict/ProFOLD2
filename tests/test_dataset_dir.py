import unittest

from profold2.data.custom import *

class TestDataSet(unittest.TestCase):
    def test_custom_dataset(self):
        data = ProteinStructureDataset('casp14')
        print(data[0])
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()

