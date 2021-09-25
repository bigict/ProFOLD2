import unittest

import torch
import numpy as np

from profold2.constants import *
from profold2.utils import *

def test_mat_to_masked():
    # nodes
    x = torch.ones(19, 3)
    x_mask = torch.randn(19) > -0.3
    # edges
    edges_mat = torch.randn(19, 19) < 1
    edges = torch.nonzero(edges_mat, as_tuple=False).t()

    # test normal edges / nodes
    cleaned = mat_input_to_masked(x, x_mask, edges=edges)
    cleaned_2 = mat_input_to_masked(x, x_mask, edges_mat=edges_mat)

    # test batch dimension
    x_ = torch.stack([x]*2, dim=0)
    x_mask_ = torch.stack([x_mask]*2, dim=0)
    edges_mat_ = torch.stack([edges_mat]*2, dim=0)

    cleaned_3 = mat_input_to_masked(x_, x_mask_, edges_mat=edges_mat_)
    assert True


def test_center_distogram_median():
    distogram = torch.randn(1, 128, 128, 37)
    distances, weights = center_distogram_torch(distogram, center = 'median')
    assert True

def test_masks():
    seqs = torch.randint(20, size=(2, 50))
    # cloud point mask - can't test bc it needs sidechainnet installed
    # cloud_masks = scn_cloud_mask(seqs, boolean=True)
    # atom masking
    N_mask, CA_mask, C_mask = scn_backbone_mask(seqs, boolean = True)
    assert True

def test_mds_and_mirrors():
    distogram = torch.randn(2, 32*3, 32*3, 37)

    distances, weights = center_distogram_torch(distogram)
    # set out some points (due to padding)
    paddings = [7,0]
    for i,pad in enumerate(paddings):
        if pad > 0:
            weights[i, -pad:, -pad:] = 0.

    # masks
    masker  = torch.arange(distogram.shape[1]) % 3
    N_mask  = (masker==0).bool()
    CA_mask = (masker==1).bool()
    coords_3d, _ = MDScaling(distances, 
        weights = weights,
        iters = 5, 
        fix_mirror = 2,
        N_mask = N_mask,
        CA_mask = CA_mask,
        C_mask = None
    )
    assert list(coords_3d.shape) == [2, 3, 32*3], 'coordinates must be of the right shape after MDS'

def test_sidechain_container():
    seqs = torch.tensor([[0]*137, [3]*137]).long()
    bb = torch.randn(2, 137*4, 3)
    atom_mask = torch.tensor( [1]*4 + [0]*(14-4) )
    proto_3d = sidechain_container(seqs, bb, atom_mask=atom_mask)
    assert list(proto_3d.shape) == [2, 137, 14, 3]


def test_distmat_loss():
    a = torch.randn(2, 137, 14, 3)
    b = torch.randn(2, 137, 14, 3)
    loss = distmat_loss_torch(a, b, p=2, q=2) # mse on distmat
    assert True

def test_lddt():
    a = torch.randn(2, 137, 14, 3)
    b = torch.randn(2, 137, 14, 3)
    cloud_mask = torch.ones(a.shape[:-1]).bool()
    lddt_result = lddt_ca_torch(a, b, cloud_mask)

    assert list(lddt_result.shape) == [2, 137]

def test_kabsch():
    a  = torch.randn(3, 8)
    b  = torch.randn(3, 8) 
    a_, b_ = Kabsch(a,b)
    assert a.shape == a_.shape

def test_tmscore():
    a = torch.randn(2, 3, 8)
    b = torch.randn(2, 3, 8)
    out = TMscore(a, b, L=8)
    assert True

def test_gdt():
    a = torch.randn(1, 3, 8)
    b = torch.randn(1, 3, 8)
    GDT(a, b, weights = 1)
    assert True

class TestUtils(unittest.TestCase):
    def test_dist_loss(self):
        X = torch.tensor([[1.0,2.0,3.0], [2.0,1.0,3.0]])
        Y = torch.tensor([[1.0,2.0,3.0], [1.0,2.0,3.0]])
        self.assertAlmostEqual(distmat_loss_torch(X, Y), 1.0, delta=1e-5)

    def test_gdt(self):
        a = np.array([[[1.0], [2.0], [3.0]]])
        b = np.array([[[1.0], [2.0], [3.0]]])
        r = GDT(a, b, weights = 1)
        self.assertAlmostEqual(r, 1.0, delta=1e-5)

        a = torch.from_numpy(a)
        b = torch.from_numpy(b)
        r = GDT(a, b, weights = 1)
        self.assertAlmostEqual(r, 1.0, delta=1e-5)

    def test_rmsd(self):
        a = np.array([[[1.0], [2.0], [3.0]]])
        b = np.array([[[1.0], [2.0], [3.0]]])
        r = RMSD(a, b)
        self.assertAlmostEqual(r, 0.0, delta=1e-5)

        a = torch.from_numpy(a)
        b = torch.from_numpy(b)
        r = RMSD(a, b)
        self.assertAlmostEqual(r, 0.0, delta=1e-5)

    def test_tmscore(self):
        a = torch.randn(2, 3, 8)
        b = torch.randn(2, 3, 8)
        out = TMscore(a, b, L=8)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
