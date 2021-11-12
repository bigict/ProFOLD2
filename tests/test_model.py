import unittest

import numpy as np
import torch
from pytorch3d.transforms import quaternion_apply, quaternion_to_matrix, matrix_to_quaternion, random_quaternions, quaternion_multiply
from einops import repeat, rearrange

from profold2.model.folding import InvariantPointAttention, StructureModule

class TestUtils(unittest.TestCase):
    def test_ipa(self):
        dim = 256
        b, n = 1, 10
        ipa = InvariantPointAttention(dim=dim)
        x = torch.rand(b, n, dim)
        z = torch.rand(b, n, n, dim)
        quaternions = rearrange(random_quaternions(b*n), '(b n) r -> b n r', b=b, n=n)
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
        (rotations, translates), backbones, act = struct_module(representations, batch)
        print(translates)
        print(backbones)
        print(backbones.shape)
        print(translates.shape)
        print(torch.einsum('b l n c,b l r c -> b l n r', backbones-repeat(translates, 'b l c -> b l n c', n=3), rotations.transpose(-1, -2)))
        pass

if __name__ == '__main__':
    unittest.main()
