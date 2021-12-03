import unittest

import numpy as np
import torch
from pytorch3d.transforms import quaternion_apply, quaternion_to_matrix, matrix_to_quaternion, random_quaternions, quaternion_multiply
from einops import repeat, rearrange

from profold2.model import functional as F

class TestUtils(unittest.TestCase):
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

    def test_rigids_from_angles(self):
        b, n = 1, 10
        aatypes = torch.randint(0, 20, size=(b, n))
        backb_points = torch.rand(b, n, 3, 3)
        backb_frames = F.rigids_from_3x3(backb_points)
        angles = torch.rand(b, n, 3, 2)
        angles = angles / torch.linalg.norm(angles, dim=-1, keepdim=True) + 1e-12
        x = F.rigids_from_angles(aatypes, backb_frames, angles)
        print('test_rigids_from_angles: ', x[0].shape, x[1].shape)

    def test_rigids_to_positions(self):
        b, n = 1, 10
        aatypes = torch.randint(0, 20, size=(b, n))
        backb_points = torch.rand(b, n, 14, 3, 3)
        backb_frames = F.rigids_from_3x3(backb_points)
        x = F.rigids_to_positions(backb_frames, aatypes)
        #print(x)

    def test_fape2(self):
        true_points = torch.rand(1, 5, 3, 3)
        true_frames = F.rigids_from_3x3(true_points)
        quaternions = rearrange(random_quaternions(1), 'l d -> () l d')
        transitions = torch.rand(1, 1, 3)
        rotations = quaternion_to_matrix(quaternions)
        pred_points = torch.einsum('b l n w, b l h w -> b l n h', true_points, repeat(rotations, 'b l h w-> b (l c) h w', c=5)) + repeat(transitions, 'b l h -> b (l c) n h', c=5, n=3)
        pred_frames = F.rigids_from_3x3(pred_points)
        frames_mask = torch.ones(1, 5)
        points_mask = torch.ones(1, 5, 3)
        loss = F.fape(true_frames, true_frames, frames_mask, true_points, true_points, points_mask, epsilon=0.)
        print('loss: ', loss)
        self.assertAlmostEqual(loss, .0)

    def test_pytorch3d(self):
        quaternions = rearrange(random_quaternions(1), 'l d -> () l d')
        quaternion_update = rearrange(random_quaternions(1), 'l d -> () l d')
        rotations = quaternion_to_matrix(quaternions)
        rotation_update = quaternion_to_matrix(quaternion_update)
        quaternions = quaternion_multiply(quaternions, quaternion_update)
        rotations = torch.einsum('b l h w,b l w d -> b l h d', rotations, rotation_update)
        self.assertTrue(torch.allclose(rotations, quaternion_to_matrix(quaternions)))

if __name__ == '__main__':
    unittest.main()
