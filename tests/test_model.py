import unittest

import numpy as np
import torch
from einops import repeat, rearrange

from profold2.model.commons import *
from profold2.model.folding import InvariantPointAttention, StructureModule
from profold2.model.functional import (
        quaternion_to_matrix,
        matrix_to_quaternion,
        quaternion_multiply)

class TestUtils(unittest.TestCase):
    def test_outer_product_mean(self):
        b, m, n = 1, 3, 125
        dim_single, dim_pairwise = 256, 128
        outer_mean = OuterMean((dim_single, dim_pairwise))
        outer_mean.eval()

        x, mask = torch.rand(b, m, n, dim_single), torch.ones(b, m, n)
        with torch.no_grad():
            x1 = outer_mean(x, mask=mask)
            x2 = outer_mean(x, mask=mask, shard_size=2)
        self.assertTrue(torch.allclose(x1, x2))

    def test_ipa(self):
        dim = 256
        b, n = 1, 10
        ipa = InvariantPointAttention(dim=dim)
        x = torch.rand(b, n, dim)
        z = torch.rand(b, n, n, dim)
        quaternions = torch.rand(b, n, 4)
        quaternions = quaternions / torch.linalg.norm(quaternions, dim=-1, keepdim=True)
        rotations = quaternion_to_matrix(quaternions)
        translates = torch.rand(b, n, 3)
        with torch.no_grad():
            r1 = ipa.forward(x, pairwise_repr=z, rotations=rotations, translations=translates)
            r2 = ipa.forward(x, pairwise_repr=z, rotations=rotations, translations=translates)
            self.assertTrue(torch.allclose(r1, r2))

    def test_structure_module(self):
        dim = 256
        b, n = 1, 10
        struct_module = StructureModule(dim, 4, 4)
        x = torch.rand(b, n, dim)
        z = torch.rand(b, n, n, dim)
        representations = dict(single=x, pair=z)
        batch = dict(seq=torch.randint(0, 20, size=(b, n)), mask=torch.ones(b, n))
        outputs = struct_module(representations, batch)
        pass

if __name__ == '__main__':
    unittest.main()
