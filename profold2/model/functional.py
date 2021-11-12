import torch
from einops import rearrange, repeat

def lddt(pred_points, true_points, points_mask, cutoff=15.):
    """Computes the lddt score for a batch of coordinates.
        https://academic.oup.com/bioinformatics/article/29/21/2722/195896
        Inputs: 
        * pred_coords: (b, l, d) array of predicted 3D points.
        * true_points: (b, l, d) array of true 3D points.
        * points_mask : (b, l) binary-valued array. 1 for points that exist in
            the true points
        * cutoff: maximum inclusion radius in reference struct.
        Outputs:
        * (b, l) lddt scores ranging between 0 and 1
    """
    assert len(pred_points.shape) == 3 and pred_points.shape[-1] == 3
    assert len(true_points.shape) == 3 and true_points.shape[-1] == 3

    eps = 1e-10

    # Compute true and predicted distance matrices. 
    pred_cdist = torch.cdist(pred_points, pred_points, p=2) + eps
    true_cdist = torch.cdist(true_points, true_points, p=2) + eps

    cdist_to_score = ((true_cdist < cutoff) *
            (rearrange(points_mask, 'b i -> b i ()')*rearrange(points_mask, 'b j -> b () j')) *
            (1.0 - torch.eye(true_cdist.shape[1])))  # Exclude self-interaction

    # Shift unscored distances to be far away
    dist_l1 = torch.abs(true_cdist - pred_cdist)

    # True lDDT uses a number of fixed bins.
    # We ignore the physical plausibility correction to lDDT, though.
    score = 0.25 * sum([dist_l1 < t for t in (0.5, 1.0, 2.0, 4.0)])

    # Normalize over the appropriate axes.
    norm = 1. / (eps + torch.sum(cdist_to_score, dim=-1))
    return norm * (eps + torch.sum(cdist_to_score * score, dim=-1))


def rigids_from_3x3(points, epsilon=1e-6):
    """Create rigids from 3 points.
    This creates a set of rigid transformations from 3 points by Gram Schmidt
    orthogonalization.
    """
    # Shape (b, l, 3, 3)
    assert points.shape[-2:] == (3, 3)
    v1 = points[...,2,:] - points[...,1,:]
    v2 = points[...,0,:] - points[...,1,:]

    e1 = v1 / torch.clamp(torch.linalg.norm(v1, dim=-1, keepdim=True), min=epsilon)
    c = torch.sum(e1 * v2, dim=-1, keepdim=True)
    u2 = v2 - e1*c
    e2 = u2 / torch.clamp(torch.linalg.norm(u2, dim=-1, keepdim=True), min=epsilon)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack((e1, e2, e3), dim=-1)
    t = points[...,1,:]

    return R, t

def fape(pred_frames, true_frames, frames_mask, pred_points, true_points, points_mask, clamp_distance=None, epsilon=1e-8):
    """ FAPE(Frame Aligined Point Error) - Measure point error under different alignments
    """
    # Shape (b, l, 3 3), (b, l, 3)
    pred_rotations, pred_trans = pred_frames
    assert pred_rotations.shape[-2:] == (3, 3) and pred_trans.shape[-1] == 3
    # Shape (b, l, 3 3), (b, l, 3)
    true_rotations, true_trans = true_frames
    assert true_rotations.shape[-2:] == (3, 3) and true_trans.shape[-1] == 3
    # Shape (b, l)
    assert frames_mask.shape == points_mask.shape[:-1]
    # Shape (b, l, n, 3)
    assert pred_points.shape[-1] == 3 and true_points.shape[-1] == 3
    # Shape (b, l, n)
    assert pred_points.shape[:3] == points_mask.shape

    def to_local(rotations, translations, points):
        rotations = rearrange(rotations, 'b l h w -> b l w h')
        translations = -torch.einsum('b l w,b l h w -> b l h', translations, rotations)
        _, l, n, _ = points.shape
        return torch.einsum('b j n w,b i h w -> b i j n h', points, rotations) + repeat(translations, 'b i h -> b i j n h', j=l, n=n)
    
    pred_xij = to_local(pred_rotations, pred_trans, pred_points)
    true_xij = to_local(true_rotations, true_trans, true_points)

    # Shape (b, l, l, n)
    dij = torch.sqrt(
            torch.sum((pred_xij - true_xij)**2, dim=-1) + epsilon)
    if clamp_distance:
        dij = torch.clip(dij, 0, clamp_distance)
    dij_mask = rearrange(frames_mask, 'b i -> b i () ()') * rearrange(points_mask, 'b j n -> b () j n')

    return torch.sum(dij * dij_mask) / (epsilon + torch.sum(dij_mask))
