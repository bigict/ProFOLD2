import unittest
import os
import sys

import numpy as np
import torch
from einops import repeat, rearrange

from profold2.common import residue_constants, protein
from profold2.model import functional as F
from profold2.model.features import make_backbone_affine, make_coord_mask

class TestFunctional(unittest.TestCase):
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
        predicted_points[:,:,0,:] = predicted_transitions
        predicted_points = rearrange(predicted_points, 'b l n d -> b (l n) d')
        mask = torch.ones_like(predicted_points[..., 0]).type(torch.bool)
        frames_mask = torch.ones(*predicted_transitions.shape[:-1]).bool()
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

        with open(os.path.join(self.data_dir, 'T1024-D1.pdb')) as f:
            prot = protein.from_pdb_string(f.read())
        if 'AF2_HOME' in os.environ:
            if not os.environ['AF2_HOME'] in sys.path:
                sys.path.append(os.environ['AF2_HOME'])
            from alphafold.model import all_atom, r3, quat_affine
            from alphafold.model.tf.data_transforms import make_atom14_masks
            af2_affine = quat_affine.make_transform_from_reference(
                    n_xyz=prot.atom_positions[...,0,:], 
                    ca_xyz=prot.atom_positions[...,1,:],
                    c_xyz=prot.atom_positions[...,2,:])

            pf2_batch = dict(seq=torch.from_numpy(prot.aatype),
                    coord=torch.tensor(prot.atom_positions, dtype=torch.float),
                    coord_mask=torch.from_numpy(prot.atom_mask))
            for k, v in pf2_batch.items():
                pf2_batch[k] = rearrange(v, '... -> () ...')
            #print('==============')
            #print(af2_affine[0])
            pf2_affine = F.rigids_from_3x3(pf2_batch['coord'][...,:3,:], indices=(2, 1, 0))
            #print(pf2_affine[0].numpy())

            self.assertTrue(np.allclose(af2_affine[0], pf2_affine[0].numpy(), atol=1e-7))
            self.assertTrue(np.allclose(af2_affine[1], pf2_affine[1].numpy(), atol=1e-7))

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
            #print('backb_frames', backb_frames[0])
            #print('rotations_pf2', rotations_pf2[0])

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
        pred_points = torch.einsum('b l n w, b l h w -> b l n h', true_points, rotations) + repeat(transitions, 'b l h -> b l n h', n=3)
        pred_frames = F.rigids_from_3x3(pred_points)
        frames_mask = torch.ones(1, 5)
        points_mask = torch.ones(1, 5, 3)

        true_points = true_points + 0.1 * torch.randn_like(true_points)

        def points_to_fape_shape(points):
            return rearrange(points, 'b l n d -> b (l n) d')
        true_points = points_to_fape_shape(true_points)
        pred_points = points_to_fape_shape(pred_points)
        points_mask = rearrange(points_mask, 'b l n -> b (l n)')

        epsilon = .0

        if 'AF2_HOME' in os.environ:
            if not os.environ['AF2_HOME'] in sys.path:
                sys.path.append(os.environ['AF2_HOME'])
            from alphafold.model import all_atom, r3
            from alphafold.model.tf.data_transforms import make_atom14_masks

            af2_true_frames = self.rigids_to_af2(true_frames)
            af2_pred_frames = self.rigids_to_af2(pred_frames)
            af2_frames_mask = frames_mask[0].numpy()
            af2_true_points = r3.vecs_from_tensor(true_points[0].numpy())
            af2_pred_points = r3.vecs_from_tensor(pred_points[0].numpy())
            af2_points_mask = points_mask[0].numpy()
            print('================')
            af2_loss = all_atom.frame_aligned_point_error(
                    af2_true_frames,
                    af2_pred_frames,
                    af2_frames_mask,
                    af2_true_points,
                    af2_pred_points,
                    af2_points_mask,
                    1.0,
                    epsilon=epsilon)

            pf2_loss = F.fape(
                    true_frames,
                    pred_frames,
                    frames_mask,
                    true_points,
                    pred_points,
                    points_mask,
                    epsilon=epsilon)
            print('xxx loss: ', af2_loss, pf2_loss)
            self.assertTrue(np.allclose(af2_loss, pf2_loss, atol=1e-6))

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
        self.assertTrue(torch.allclose(rotations, F.quaternion_to_matrix(quaternions), atol=1e-7))

    def test_batched_gather(self):
        with open(os.path.join(self.data_dir, 'T1024-D1.pdb')) as f:
            prot = protein.from_pdb_string(f.read())
        batch = dict(seq=torch.from_numpy(prot.aatype),
                coord=torch.tensor(prot.atom_positions, dtype=torch.float),
                coord_mask=torch.from_numpy(prot.atom_mask))
        for k, v in batch.items():
            batch[k] = rearrange(v, '... -> () ...')
        #print('---------')
        #print(F.rigids_from_positions(batch['seq'],
        #        batch['coord'], batch['coord_mask']))
        #print('---------')

        batch = make_backbone_affine()(batch)
        frames_from_points = F.rigids_from_positions(batch['seq'], batch['coord'], batch['coord_mask'])
        x = frames_from_points['atom_affine'][0] * rearrange(frames_from_points['atom_affine_mask'], '... -> ... () ()')
        angles = F.angles_from_positions(batch['seq'],
                batch['coord'], batch['coord_mask'])
        frames_from_angles = F.rigids_from_angles(batch['seq'], batch['backbone_affine'], angles['torsion_angles'])
        pf2_positions = F.rigids_to_positions(frames_from_angles, batch['seq'])
        #print('x', pf2_positions)
        if 'AF2_HOME' in os.environ:
            if not os.environ['AF2_HOME'] in sys.path:
                sys.path.append(os.environ['AF2_HOME'])
            from alphafold.model import all_atom, r3
            from alphafold.model.tf.data_transforms import make_atom14_masks
            af2_positions = all_atom.frames_and_literature_positions_to_atom14_pos(prot.aatype, self.rigids_to_af2(frames_from_angles))
            x = pf2_positions
            y = r3.vecs_to_tensor(af2_positions)
            z = batch['coord']
            #for i in range(x.shape[1]):
            #    print('-------------')
            #    print(i, x[0,i,:3,:].numpy())
            #    print(i, y[i,:3,:])
            #    print(i, z[0,i,:3,:].numpy())

        #print(positions.shape)
        #print(batch['coord_mask'].shape)
        #positions = positions * rearrange(batch['coord_mask'], '... -> ... ()')
        #print('x', batch['coord'])
        #print('y', positions)
        ##frames_from_points = F.rigids_from_positions(batch['seq'], batch['coord'], batch['coord_mask'])
        ##F.rigids_to_positions(

    def test_rigids_from_positions(self):
        with open(os.path.join(self.data_dir, 'T1024-D1.pdb')) as f:
            prot = protein.from_pdb_string(f.read())

        if 'AF2_HOME' in os.environ:
            if not os.environ['AF2_HOME'] in sys.path:
                sys.path.append(os.environ['AF2_HOME'])
            from alphafold.model import all_atom, r3
            from alphafold.model.tf.data_transforms import make_atom14_masks
            af2_batch = make_atom14_masks(dict(aatype=prot.aatype))
            for k, v in af2_batch.items():
                af2_batch[k] = np.array(v)
            print('================')
            atom37_positions = all_atom.atom14_to_atom37(prot.atom_positions, af2_batch)
            self.assertTrue(np.allclose(all_atom.atom37_to_atom14(atom37_positions, af2_batch), prot.atom_positions))
            atom37_masks = all_atom.atom14_to_atom37(prot.atom_mask, af2_batch)
            self.assertTrue(np.allclose(all_atom.atom37_to_atom14(atom37_masks, af2_batch), prot.atom_mask))
            af2_frames = all_atom.atom37_to_frames(prot.aatype, atom37_positions, atom37_masks)
            af2_angles = all_atom.atom37_to_torsion_angles(prot.aatype[None,...], atom37_positions[None,...], atom37_masks[None,...])
            #print(af2_frames)

            #print('---------')
            pf2_batch = dict(seq=torch.from_numpy(prot.aatype),
                    coord=torch.tensor(prot.atom_positions, dtype=torch.float),
                    coord_mask=torch.from_numpy(prot.atom_mask))
            for k, v in pf2_batch.items():
                pf2_batch[k] = rearrange(v, '... -> () ...')

            pf2_frames = F.rigids_from_positions(pf2_batch['seq'],
                    pf2_batch['coord'], pf2_batch['coord_mask'])
            def pf2_squeeze_affine(frames, backb_only=False):
                # Adapt backbone frame to old convention (mirror x-axis and z-axis). 
                rots = torch.eye(3, dtype=torch.float)
                if not backb_only:
                    rots = torch.tile(rots, (8, 1, 1))
                    rots[1:, 0, 0] = -1
                    rots[1:, 2, 2] = -1
                    frames = F.rigids_rotate(frames, rots)
                r, t = frames
                return rearrange(r, 'b i ... -> (b i) ...'), rearrange(t, 'b i ... -> (b i) ...')

            for k, v in pf2_frames.items():
                if k == 'atom_affine' or k == 'atom_affine_alt':
                    pf2_frames[k] = pf2_squeeze_affine(v)
                else:
                    pf2_frames[k] = rearrange(v, 'b i ... -> (b i) ...')
            pf2_batch = make_backbone_affine()(pf2_batch)
            assert 'backbone_affine' in pf2_batch
            # C, Ca, N
            pf2_backb_frames = F.rigids_from_3x3(pf2_batch['coord'], indices=(2, 1, 0))
            pf2_backb_frames = pf2_squeeze_affine(pf2_backb_frames, backb_only=True)

            #af2_group_atom37_idx = af2_frames['residx_rigidgroup_base_atom37_idx']
            #g, n, c = pf2_frames['group_atom14_idx'].shape[-3:]
            #pf2_group_atom14_idx = rearrange(pf2_frames['group_atom14_idx'],'... g n c -> ... c (g n)')
            #pf2_group_atom37_idx = all_atom.atom14_to_atom37(pf2_group_atom14_idx.numpy(), af2_batch)
            #pf2_group_atom37_idx = rearrange(pf2_group_atom37_idx, '... c (g n) -> ... g n c', g=g, n=n)
            #self.assertTrue(np.allclose(af2_group_atom37_idx, pf2_group_atom37_idx.argmax(axis=-1)))
            #af2_group_positions = af2_frames['base_atom_pos']
            #pf2_group_positions = pf2_frames['group_points']
            #self.assertTrue(np.allclose(af2_group_positions, pf2_group_positions.numpy()))
            #print('^^')


            af2_atom_affine = np.array(af2_frames['rigidgroups_gt_frames'])
            
            af2_atom_affine_exists = np.array(af2_frames['rigidgroups_group_exists'])
            af2_atom_affine_mask = np.array(af2_frames['rigidgroups_gt_exists'])
            af2_atom_affine_alt = np.array(af2_frames['rigidgroups_alt_gt_frames'])
            af2_atom_affine_is_ambiguous = np.array(af2_frames['rigidgroups_group_is_ambiguous'])

            pf2_atom_affine = torch.cat((
                    rearrange(pf2_frames['atom_affine'][0], '... h w -> ... (h w)'),
                    pf2_frames['atom_affine'][1]),
                    dim=-1)
            pf2_atom_affine_exists = pf2_frames['atom_affine_exists']
            pf2_atom_affine_mask = pf2_frames['atom_affine_mask']
            pf2_atom_affine_alt = torch.cat((
                    rearrange(pf2_frames['atom_affine_alt'][0], '... h w -> ... (h w)'),
                    pf2_frames['atom_affine_alt'][1]),
                    dim=-1)
            pf2_atom_affine_is_ambiguous = pf2_frames['atom_affine_is_ambiguous']
            backb_affine = torch.cat((
                    rearrange(pf2_backb_frames[0], '... h w -> ... (h w)'),
                    pf2_backb_frames[1]),
                    dim=-1)
            backb_affine_from_feat = pf2_squeeze_affine(pf2_batch['backbone_affine'], backb_only=True)
            backb_affine_from_feat = torch.cat((
                    rearrange(backb_affine_from_feat[0], '... h w -> ... (h w)'),
                    backb_affine_from_feat[1]),
                    dim=-1)
            #for i in range(af2_atom_affine.shape[0]):
            #    print(i, prot.aatype[i], af2_atom_affine[i])
            #    print(i, prot.aatype[i], pf2_atom_affine[i].numpy())
            #    print('d', prot.aatype[i], np.abs(af2_atom_affine[i] - pf2_atom_affine[i].numpy()))
            #    self.assertTrue(np.allclose(af2_atom_affine[i], pf2_atom_affine[i].numpy(), atol=1e-7))
            #print('-------')
            #print(af2_atom_affine)
            #print(pf2_atom_affine)
            #print(pf2_atom_affine[...,0,:])
            #print(pf2_backb_frames)

            self.assertTrue(np.allclose(af2_atom_affine,
                    pf2_atom_affine.numpy(), atol=1e-7))
            self.assertTrue(np.allclose(backb_affine.numpy(),
                    pf2_atom_affine[...,0,:].numpy(), atol=1e-7))
            self.assertTrue(np.allclose(backb_affine.numpy(),
                    backb_affine_from_feat.numpy(), atol=1e-7))
            self.assertTrue(np.allclose(af2_atom_affine_exists,
                    pf2_atom_affine_exists.numpy(), atol=1e-7))
            self.assertTrue(np.allclose(af2_atom_affine_mask,
                    pf2_atom_affine_mask.numpy(), atol=1e-7))
            self.assertTrue(np.allclose(af2_atom_affine_alt,
                    pf2_atom_affine_alt.numpy(), atol=1e-7))
            self.assertTrue(np.allclose(af2_atom_affine_is_ambiguous,
                    pf2_atom_affine_is_ambiguous.numpy(), atol=1e-7))

    def test_angles_from_positions(self):
        with open(os.path.join(self.data_dir, 'T1024-D1.pdb')) as f:
            prot = protein.from_pdb_string(f.read())

        if 'AF2_HOME' in os.environ:
            if not os.environ['AF2_HOME'] in sys.path:
                sys.path.append(os.environ['AF2_HOME'])
            from alphafold.model import all_atom, r3
            from alphafold.model.tf.data_transforms import make_atom14_masks
            af2_batch = make_atom14_masks(dict(aatype=prot.aatype))
            for k, v in af2_batch.items():
                af2_batch[k] = np.array(v)
            print('================')
            atom37_positions = all_atom.atom14_to_atom37(prot.atom_positions, af2_batch)
            self.assertTrue(np.allclose(all_atom.atom37_to_atom14(atom37_positions, af2_batch), prot.atom_positions))
            atom37_masks = all_atom.atom14_to_atom37(prot.atom_mask, af2_batch)
            self.assertTrue(np.allclose(all_atom.atom37_to_atom14(atom37_masks, af2_batch), prot.atom_mask))
            af2_angles = all_atom.atom37_to_torsion_angles(prot.aatype[None,...], atom37_positions[None,...], atom37_masks[None,...])
            #print(af2_torsion_angles)

            #print('---------')
            pf2_batch = dict(seq=torch.from_numpy(prot.aatype),
                    coord=torch.tensor(prot.atom_positions, dtype=torch.float),
                    coord_mask=torch.from_numpy(prot.atom_mask))
            for k, v in pf2_batch.items():
                pf2_batch[k] = rearrange(v, '... -> () ...')

            pf2_angles = F.angles_from_positions(pf2_batch['seq'],
                    pf2_batch['coord'], pf2_batch['coord_mask'])

            af2_torsion_angles = af2_angles['torsion_angles_sin_cos'] * af2_angles['torsion_angles_mask'][...,None]
            pf2_torsion_angles = pf2_angles['torsion_angles'] * pf2_angles['torsion_angles_mask'][...,None]
            self.assertTrue(np.allclose(np.array(af2_torsion_angles), pf2_torsion_angles.numpy(), atol=1e-4))

            af2_torsion_angles = af2_angles['torsion_angles_mask']
            pf2_torsion_angles = pf2_angles['torsion_angles_mask']
            self.assertTrue(np.allclose(np.array(af2_torsion_angles[0]), pf2_torsion_angles[0].numpy(), atol=1e-7))

            af2_torsion_angles = af2_angles['alt_torsion_angles_sin_cos'] * af2_angles['torsion_angles_mask'][...,None]
            pf2_torsion_angles = pf2_angles['torsion_angles_alt'] * pf2_angles['torsion_angles_mask'][...,None]
            self.assertTrue(np.allclose(np.array(af2_torsion_angles), pf2_torsion_angles.numpy(), atol=1e-4))

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

    def test_symmetric_ground_truth(self):
        with open(os.path.join(self.data_dir, 'T1024-D1.pdb')) as f:
            prot = protein.from_pdb_string(f.read())
        prot = protein.Protein(aatype=prot.aatype,
                residue_index=prot.residue_index,
                b_factors=prot.b_factors,
                atom_positions=np.random.randn(*prot.atom_positions.shape) * prot.atom_mask[...,None],
                atom_mask=prot.atom_mask)

        pf2_batch = dict(seq=torch.from_numpy(prot.aatype),
                coord=torch.tensor(prot.atom_positions, dtype=torch.float),
                coord_mask=torch.from_numpy(prot.atom_mask))
        for k, v in pf2_batch.items():
            pf2_batch[k] = rearrange(v, '... -> () ...')
        pf2_batch.update(F.symmetric_ground_truth_create_alt(pf2_batch['seq'],
                pf2_batch['coord'], pf2_batch['coord_mask']))

        coord_pred = torch.randn(*pf2_batch['coord'].shape)
        pf2_batch.update(F.symmetric_ground_truth_rename(coord_pred, *map(
                lambda key: pf2_batch[key],
                ('coord_exists', 'coord', 'coord_mask', 'coord_alt', 'coord_alt_mask', 'coord_is_symmetric'))))

        if 'AF2_HOME' in os.environ:
            if not os.environ['AF2_HOME'] in sys.path:
                sys.path.append(os.environ['AF2_HOME'])
            from alphafold.model import all_atom, r3, quat_affine
            from alphafold.model.tf.data_transforms import make_atom14_masks
            from alphafold.model.label_pipeline import make_atom14_data
            from alphafold.model.folding import compute_renamed_ground_truth
            af2_batch = make_atom14_masks(dict(aatype=prot.aatype))
            for k, v in af2_batch.items():
                af2_batch[k] = np.array(v)
            print('================')
            atom37_positions = all_atom.atom14_to_atom37(prot.atom_positions, af2_batch)
            atom37_mask = all_atom.atom14_to_atom37(prot.atom_mask, af2_batch)
            #coord_alt, coord_mask_alt = all_atom.get_alt_atom14(prot.aatype, r3.vecs_from_tensor(prot.atom_positions), prot.atom_mask)
            #af2_batch['atom14_alt_gt_positions'] = coord_alt
            #af2_batch['atom14_alt_gt_mask'] = coord_mask_alt
            af2_batch = make_atom14_data(dict(aatype_index=prot.aatype, all_atom_positions=atom37_positions, all_atom_mask=atom37_mask))
            af2_batch.update(compute_renamed_ground_truth(af2_batch, rearrange(coord_pred, 'b i ... -> (b i)...').numpy()))

            for k, v in af2_batch.items():
                af2_batch[k] = np.array(v)
            #print(af2_batch['pred_dist'].shape)
            #print(pf2_batch['pred_dist'].shape)
            ##self.assertTrue(np.allclose(af2_batch['pred_dist'], pf2_batch['pred_dist'][0].numpy()))
            #for i in range(pf2_batch['pred_dist'].shape[1]):
            #    print(i, af2_batch['pred_dist'][i], af2_batch['pred_xx'][i])
            #    print(i, pf2_batch['pred_dist'][0][i].numpy(), pf2_batch['pred_xx'][0][i])
            #    self.assertTrue(np.allclose(af2_batch['pred_dist'][i], pf2_batch['pred_dist'][0][i].numpy(), atol=1e-5))
            #for i in range(pf2_batch['pred_dist'].shape[1]):
            #    for j in range(pf2_batch['pred_dist'].shape[2]):
            #        print(i, j, af2_batch['pred_dist'][i][j])
            #        print(i, j, pf2_batch['pred_dist'][0][i][j].numpy())
            #        self.assertTrue(np.allclose(af2_batch['pred_dist'][i][j], pf2_batch['pred_dist'][0][i][j].numpy()))
            #for i in range(pf2_batch['coord_renamed'].shape[1]):
            #    #with self.subTest(name=f'coord_alt_{i}'):
            #    print(i, af2_batch['renamed_atom14_gt_positions'][i])
            #    print(i, pf2_batch['coord_renamed'][0][i].numpy())
            #    print(i, af2_batch['renamed_atom14_gt_positions'][i] - pf2_batch['coord_renamed'][0][i].numpy())
            #    self.assertTrue(np.allclose(af2_batch['renamed_atom14_gt_positions'][i], pf2_batch['coord_renamed'][0][i].numpy(), atol=1e-7))
            self.assertTrue(np.allclose(af2_batch['renamed_atom14_gt_positions'], pf2_batch['coord_renamed'][0].numpy(), atol=1e-7))
        #print(F.symmetric_ground_truth_rename(batch['coord'],
        #    batch['coord_exits'],
        #    batch['coord'], batch['coord_mask'],
        #    batch['coord_alt'], batch['coord_alt_mask'],
        #    batch['coord_is_symmetric']))

if __name__ == '__main__':
    unittest.main()
