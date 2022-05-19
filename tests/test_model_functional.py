import unittest
import os
import sys

import numpy as np
import torch
from einops import repeat, rearrange

from profold2.common import residue_constants, protein
from profold2.model import functional as F

class TestUtils(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'data/relax')

    def test_lddt(self):
        params = [
                ('same',
                    [[[0, 0, 0], [5, 0, 0], [10, 0, 0]]],
                    [[[0, 0, 0], [5, 0, 0], [10, 0, 0]]],
                    [[1, 1, 1]]),
                ('all_shifted',
                    [[[0, 0, 0], [5, 0, 0], [10, 0, 0]]],
                    [[[-1, 0, 0], [4, 0, 0], [9, 0, 0]]],
                    [[1, 1, 1]]),
                ('all_rotated',
                    [[[0, 0, 0], [5, 0, 0], [10, 0, 0]]],
                    [[[0, 0, 0], [0, 5, 0], [0, 10, 0]]],
                    [[1, 1, 1]]),
                ('half_a_dist',
                    [[[0, 0, 0], [5, 0, 0]]],
                    [[[0, 0, 0], [5.5-1e-5, 0, 0]]],
                    [[1, 1]]),
                ('one_a_dist',
                    [[[0, 0, 0], [5, 0, 0]]],
                    [[[0, 0, 0], [6-1e-5, 0, 0]]],
                    [[0.75, 0.75]]),
                ('two_a_dist',
                    [[[0, 0, 0], [5, 0, 0]]],
                    [[[0, 0, 0], [7-1e-5, 0, 0]]],
                    [[0.5, 0.5]]),
                ('four_a_dist',
                    [[[0, 0, 0], [5, 0, 0]]],
                    [[[0, 0, 0], [9-1e-5, 0, 0]]],
                    [[0.25, 0.25]]),
                ('five_a_dist',
                    [[[0, 0, 0], [16-1e-5, 0, 0]]],
                    [[[0, 0, 0], [11, 0, 0]]],
                    [[0, 0]]),
                ('no_pairs',
                    [[[0, 0, 0], [20, 0, 0]]],
                    [[[0, 0, 0], [25-1e-5, 0, 0]]],
                    [[1, 1]]),
            ]
        for name, pred_points, true_points, expect_lddt in params:
            pred_points, true_points, expect_lddt = map(
                    lambda x: torch.as_tensor(x, dtype=torch.float),
                    (pred_points, true_points, expect_lddt))
            b, l, _ = pred_points.shape
            points_mask = torch.ones(b, l)
            with self.subTest(name=name):
                result_lddt = F.lddt(pred_points, true_points, points_mask)
                self.assertTrue(torch.allclose(expect_lddt, result_lddt))
    
    def test_plddt(self):
        params = [
                ('one',
                    [[[0, 0, 0], [10, 0, 0], [0, 10, 0]]],
                    [[0.5, 1.0/6, 0.5]]),
            ]
        for name, logits, expect_plddt in params:
            logits, expect_plddt = map(
                    lambda x: torch.as_tensor(x, dtype=torch.float),
                    (logits, expect_plddt))
            with self.subTest(name=name):
                result_plddt = F.plddt(logits)
                self.assertTrue(torch.allclose(result_plddt, expect_plddt, atol=1e-4))

    def test_fape(self):
        def get_rotation_matrix(indexes):
            """(B, 3, 3) <- (B, 3)"""
            indexs = torch.cat([
                    torch.ones(indexes.size(0), 1).type_as(indexes),
                    indexes
                ], dim=-1)  # (B, 4) <- (B, 3)
            norms = indexs.norm(dim=-1, keepdim=True)  # (B, 1) <- (B, 4)
            indexs = indexs / norms # (B, 4) <<- (B,)
            
            a, b, c, d = indexs.T  # 4 * (B,) <- (B, 4)
            r_00 = a ** 2 + b ** 2 - c ** 2 - d ** 2
            r_01 = 2 * b * c - 2 * a * d
            r_02 = 2 * b * d + 2 * a * c
            r_10 = 2 * b * c + 2 * a * d
            r_11 = a ** 2 - b ** 2 + c ** 2 - d ** 2
            r_12 = 2 * c * d - 2 * a * b
            r_20 = 2 * b * d - 2 * a * c
            r_21 = 2 * c * d + 2 * a * b
            r_22 = a ** 2 - b ** 2 - c ** 2 + d ** 2
            result = torch.stack([
                torch.stack([r_00, r_01, r_02], dim=1),
                torch.stack([r_10, r_11, r_12], dim=1),
                torch.stack([r_20, r_21, r_22], dim=1)], dim=1
            )
            return result

        batch = 10
        length = 11
        points_per_frame = 3
        predicted_rotations = torch.eye(3).repeat(batch, length, 1, 1)  # (B, L, 3, 3)
        predicted_transitions = torch.randn(batch, length, 3)  # (B, L, 3)
        predicted_frame = predicted_rotations, predicted_transitions  # [Tuple]
        predicted_points = torch.randn(batch, length, points_per_frame, 3)
        mask = torch.ones_like(predicted_points[..., 0]).type(torch.bool)
        frames_mask = torch.ones(*mask.shape[:-1]).bool()
        true_frames = predicted_frame
        true_points = predicted_points
        loss = F.fape(predicted_frame, true_frames, frames_mask, predicted_points, true_points, mask, epsilon=0)
        self.assertEqual(loss, 0)

        true_transitions = predicted_transitions
        loss = []
        for i in range(5):
            true_transitions = true_transitions + 0.1 * torch.randn_like(true_transitions)
            true_frames = predicted_rotations, true_transitions
            loss.append(F.fape(predicted_frame, true_frames, frames_mask, predicted_points, true_points, mask, epsilon=0))
        for i in range(len(loss) - 1):
            self.assertTrue(loss[i] < loss[i + 1])

        true_points = predicted_points
        loss = []
        for i in range(5):
            true_points = true_points + 0.1 * torch.randn_like(true_points)
            loss.append(F.fape(predicted_frame, true_frames, frames_mask, predicted_points, true_points, mask, epsilon=0))
        for i in range(len(loss) - 1):
            self.assertTrue(loss[i] < loss[i + 1])

        random_rotation = get_rotation_matrix(0.05 * torch.ones(1, 3))[0]  # (3, 3)
        a_unit_vector = torch.tensor([[1, 0, 0]]).type(torch.float)
        cos = a_unit_vector @ random_rotation @ a_unit_vector.T
        #print(np.pi / torch.acos(cos))
        true_rotations = predicted_rotations
        loss = []
        for i in range(100):
            true_rotations = true_rotations @ random_rotation  # (B, L, 3, 3)  <<- (3, 3)
            true_frames = true_rotations, true_transitions
            loss.append(F.fape(predicted_frame, true_frames, frames_mask, predicted_points, true_points, mask, epsilon=0).item())
        #print(loss)
        loss_incress = torch.tensor(loss[1:]) > torch.tensor(loss[:-1])
        #print(loss_incress)
        # for i in range(len(loss) - 1):
        #     self.assertTrue(loss[i] < loss[i + 1])
        shift_positions = (loss_incress[1:] ^ loss_incress[:-1]).nonzero().flatten()
        #print((shift_positions[1:] - shift_positions[:-1]).float().mean())

    def test_rigids_from_3x3(self):
        points = torch.rand(1, 5, 3, 3)
        x_diff = rearrange(points, 'b l i d -> b l i () d') - rearrange(points, 'b l j d -> b l () j d')
        R, t = F.rigids_from_3x3(points)
        y = torch.einsum('b l n w, b l h w -> b l n h', points - repeat(t, 'b l w -> b l n w', n=3), rearrange(R, 'b l h w -> b l w h'))
        y_diff = rearrange(y, 'b l i d -> b l i () d') - rearrange(y, 'b l j d -> b l () j d')
        self.assertTrue(torch.allclose(torch.sum(x_diff*x_diff, dim=-1), torch.sum(y_diff*y_diff, dim=-1)))
        def to_local(rotations, transitions, points):
            rotations = rearrange(rotations, 'b l h w -> b l w h')
            transitions = -torch.einsum('b l w,b l h w -> b l h', transitions, rotations)
            _, l, n, _ = points.shape
            return torch.einsum('b j n w,b i h w -> b i j n h', points, rotations) + repeat(transitions, 'b i h -> b i j n h', j=l, n=n)
        xij = to_local(R, t, points)
        m_diff = rearrange(xij, 'b l i n w -> b l (i n) () w') - rearrange(xij, 'b l j n w -> b l () (j n) w')
        n_diff = rearrange(points, 'b i n d -> b (i n) () d')  - rearrange(points, 'b j n d -> b () (j n) d')
        #for i in range(5):
        #    print('m_diff', torch.sum(m_diff*m_diff, dim=-1)[:,i,...])
        #print('n_diff', torch.sum(n_diff*n_diff, dim=-1))
        ##print(torch.sum(xij*xij, -1))
        #print(torch.sum(m_diff*m_diff, dim=-1).shape)

    def torsion_random(self, b, n):
        alpha_f = torch.rand(b, n, 7, 2)
        alpha_f = alpha_f / torch.linalg.norm(alpha_f, dim=-1, keepdim=True)
        return alpha_f

    def rigids_random(self, b, n):
        quaternions = torch.rand(b, n, 4)
        quaternions = quaternions / torch.linalg.norm(quaternions, dim=-1, keepdim=True)
        rotations = F.quaternion_to_matrix(quaternions)
        translations = torch.rand(b, n, 3)
        return rotations, translations

    def rigids_to_af2(self, frames):
        assert 'AF2_HOME' in os.environ
        if not os.environ['AF2_HOME'] in sys.path:
            sys.path.append(os.environ['AF2_HOME'])
        from alphafold.model import r3

        rotations, trans = frames
        # 1. Select Rots "xx,xy,xz,yx,yy,yz,xz,yz,zz" from rotations_np
        def rotations_to_array(rotations, i, j):
            t = rearrange(rotations[...,i,j], 'b n ...->(b n) ...')
            return np.asarray(t)
        rots = [rotations_to_array(rotations, i//3, i%3) for i in range(3*3)]
        rots = r3.Rots(*rots)
        
        # 2. Select Trans "x,y,z" from rotations_np
        def trans_to_array(trans, i):
            t = rearrange(trans[...,i], 'b n ...->(b n) ...')
            return np.asarray(t)
        trans = [trans_to_array(trans, i) for i in range(3)]
        trans = r3.Vecs(*trans)

        return r3.Rigids(rots, trans)

    def rigids_to_pf2(self, frames):
        assert 'AF2_HOME' in os.environ
        if not os.environ['AF2_HOME'] in sys.path:
            sys.path.append(os.environ['AF2_HOME'])
        from alphafold.model import r3

        assert isinstance(frames, r3.Rigids)

        rots = [np.stack(frames.rot[i*3:(i+1)*3], axis=-1) for i in range(3)]
        rots = np.stack(rots, axis=-2)
        trans = np.stack(frames.trans[:], axis=-1)

        return torch.from_numpy(rots), torch.from_numpy(trans)

    def test_rigids_from_angles(self):
        b, n = 1, 10

        if 'AF2_HOME' in os.environ:
            if not os.environ['AF2_HOME'] in sys.path:
                sys.path.append(os.environ['AF2_HOME'])
            from alphafold.model import all_atom

            alpha_f = self.torsion_random(b, n)
            aatype = torch.randint(0, 20, (b, n))
            backb_frames = self.rigids_random(b, n)

            # Test for ProFOLD_Function: rigids_from_angles 
            rotations_pf2, translations_pf2 = F.rigids_from_angles(aatype, backb_frames, alpha_f)
            #print("rotations.shape:{} \nrotations.xx:{}".format(rotations.shape, rotations[0, :, :, 0, 0]))
            #print("translations.shape:{} \ntrans.x:{}".format(translations.shape, translations[0, :, :, 0]))
            #print(rotations.shape)

            #print('-'*70)

            # Test for AlphaFold_Function: all_atoms.rigids_from_torsion_angles
            print("Validate by AlphaFold2 ")
            # 1. Change data type for alphafold input 
            alpha_f_np = alpha_f.view(-1,7,2).numpy() 
            aatype_np = aatype.view(-1).numpy()
            rigids = self.rigids_to_af2(backb_frames)

            # 2. Output: some change for logs in alphafold.model.all_atoms in lines 522
            out = all_atom.torsion_angles_to_frames(aatype_np, rigids, alpha_f_np)

            rotations_af2, translations_af2 = self.rigids_to_pf2(out)
            rotations_af2 = rearrange(rotations_af2, '(b n) ...->b n ...', b=b)
            translations_af2 = rearrange(translations_af2, '(b n) ...->b n ...', b=b)

            self.assertTrue(torch.allclose(rotations_pf2, rotations_af2))
            self.assertTrue(torch.allclose(translations_pf2, translations_af2))

    def test_rigids_to_positions(self):
        b, n = 1, 10

        alpha_f = self.torsion_random(b, n)
        aatype = torch.randint(0, 20, (b, n))
        backb_frames = self.rigids_random(b, n)
        all_atom_frames_pf2 = F.rigids_from_angles(aatype, backb_frames, alpha_f)

        if 'AF2_HOME' in os.environ:
            if not os.environ['AF2_HOME'] in sys.path:
                sys.path.append(os.environ['AF2_HOME'])
            from alphafold.model import all_atom

            all_atom_frames_af2 = self.rigids_to_af2(all_atom_frames_pf2)

            aatype_np = aatype.view(-1).numpy()
            coords_af2 = all_atom.frames_and_literature_positions_to_atom14_pos(aatype_np, all_atom_frames_af2)
            coords_af2 = torch.from_numpy(np.stack(coords_af2[:], axis=-1))

            coords_pf2 = F.rigids_to_positions(all_atom_frames_pf2, aatype)
            self.assertTrue(torch.allclose(coords_af2, coords_pf2))

    def test_fape2(self):
        true_points = torch.rand(1, 5, 3, 3)
        true_frames = F.rigids_from_3x3(true_points)
        quaternions = torch.rand(1, 1, 4)
        quaternions = quaternions / torch.linalg.norm(quaternions, dim=-1, keepdim=True)
        transitions = torch.rand(1, 1, 3)
        rotations = F.quaternion_to_matrix(quaternions)
        pred_points = torch.einsum('b l n w, b l h w -> b l n h', true_points, repeat(rotations, 'b l h w-> b (l c) h w', c=5)) + repeat(transitions, 'b l h -> b (l c) n h', c=5, n=3)
        pred_frames = F.rigids_from_3x3(pred_points)
        frames_mask = torch.ones(1, 5)
        points_mask = torch.ones(1, 5, 3)
        loss = F.fape(true_frames, true_frames, frames_mask, true_points, true_points, points_mask, epsilon=0.)
        # print('loss: ', loss)
        self.assertAlmostEqual(loss, .0)

    def test_pytorch3d(self):
        quaternions = torch.rand(1, 1, 4)
        quaternions = quaternions / torch.linalg.norm(quaternions, dim=-1, keepdim=True)
        quaternion_update = torch.rand(1, 1, 4)
        quaternion_update = quaternion_update / torch.linalg.norm(quaternion_update, dim=-1, keepdim=True)
        #quaternion_update = rearrange(random_quaternions(1), 'l d -> () l d')
        rotations = F.quaternion_to_matrix(quaternions)
        rotation_update = F.quaternion_to_matrix(quaternion_update)
        quaternions = F.quaternion_multiply(quaternions, quaternion_update)
        rotations = torch.einsum('b l h w,b l w d -> b l h d', rotations, rotation_update)
        self.assertTrue(torch.allclose(rotations, F.quaternion_to_matrix(quaternions)))

    def test_batched_gather(self):
        with open(os.path.join(self.data_dir, 'T1024-D1.pdb')) as f:
            prot = protein.from_pdb_string(f.read())
        batch = dict(seq=torch.from_numpy(prot.aatype),
                coord=torch.tensor(prot.atom_positions, dtype=torch.float),
                coord_mask=torch.from_numpy(prot.atom_mask))
        for k, v in batch.items():
            batch[k] = rearrange(v, '... -> () ...')
        print('---------')
        print(F.rigids_from_positions(batch['seq'],
                batch['coord'], batch['coord_mask']))
        print('---------')

    def test_rigids_from_positions(self):
        with open(os.path.join(self.data_dir, 'T1024-D1.pdb')) as f:
            prot = protein.from_pdb_string(f.read())
    def test_between_ca_ca_distance_loss(self):
        with open(os.path.join(self.data_dir, 'T1024-D1.pdb')) as f:
            prot = protein.from_pdb_string(f.read())
        loss = F.between_ca_ca_distance_loss(
            torch.tensor(prot.atom_positions, dtype=torch.float),
            torch.tensor(prot.atom_mask, dtype=torch.bool),
            torch.tensor(prot.residue_index, dtype=torch.int))
        self.assertAlmostEqual(loss, .0)

    def test_between_residue_bond_loss(self):
        with open(os.path.join(self.data_dir, 'T1024-D1.pdb')) as f:
            prot = protein.from_pdb_string(f.read())
        loss_dict = F.between_residue_bond_loss(
            torch.tensor(prot.atom_positions,  dtype=torch.float),
            torch.tensor(prot.atom_mask, dtype=torch.bool),
            torch.tensor(prot.residue_index, dtype=torch.int),
            torch.tensor(prot.aatype, dtype=torch.int))
        for k, v in loss_dict.items():
            self.assertAlmostEqual(v, .0)
        #print('-------------')

    def test_between_residue_clash_loss(self):
        with open(os.path.join(self.data_dir, 'T1024-D1.pdb')) as f:
            prot = protein.from_pdb_string(f.read())
        batch = dict(seq=torch.from_numpy(prot.aatype),
                coord=torch.tensor(prot.atom_positions, dtype=torch.float),
                coord_mask=torch.from_numpy(prot.atom_mask),
                seq_index=torch.tensor(prot.residue_index, dtype=torch.int))
        for k, v in batch.items():
            batch[k] = rearrange(v, '... -> () ...')

        loss_dict = F.between_residue_clash_loss(
            batch['coord'],
            batch['coord_mask'],
            batch['seq_index'],
            batch['seq'])
        for k, v in loss_dict.items():
            print(k, v)
            self.assertAlmostEqual(v, .0)
        #print('-------------')

    def test_within_residue_clash_loss(self):
        with open(os.path.join(self.data_dir, 'T1024-D1.pdb')) as f:
            prot = protein.from_pdb_string(f.read())
        batch = dict(seq=torch.from_numpy(prot.aatype),
                coord=torch.tensor(prot.atom_positions, dtype=torch.float),
                coord_mask=torch.from_numpy(prot.atom_mask),
                seq_index=torch.tensor(prot.residue_index, dtype=torch.int))
        for k, v in batch.items():
            batch[k] = rearrange(v, '... -> () ...')

        loss_dict = F.within_residue_clash_loss(
            batch['coord'],
            batch['coord_mask'],
            batch['seq_index'],
            batch['seq'])
        for k, v in loss_dict.items():
            print(k, v)
            self.assertAlmostEqual(v, .0)
        #print('-------------')

    def test_symmetric_ground_truth_rename(self):
        with open(os.path.join(self.data_dir, 'T1024-D1.pdb')) as f:
            prot = protein.from_pdb_string(f.read())
        batch = dict(seq=torch.from_numpy(prot.aatype),
                coord=torch.tensor(prot.atom_positions, dtype=torch.float),
                coord_mask=torch.from_numpy(prot.atom_mask))
        for k, v in batch.items():
            batch[k] = rearrange(v, '... -> () ...')
        batch.update(F.symmetric_ground_truth_create_alt(batch['seq'],
                batch['coord'], batch['coord_mask']))
        #print(F.symmetric_ground_truth_rename(batch['coord'],
        #    batch['coord_exits'],
        #    batch['coord'], batch['coord_mask'],
        #    batch['coord_alt'], batch['coord_alt_mask'],
        #    batch['coord_is_symmetric']))

if __name__ == '__main__':
    unittest.main()
