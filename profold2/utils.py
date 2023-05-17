"""Utils for profold2
"""
import contextlib
from functools import wraps
from inspect import isfunction
import time
import uuid

import numpy as np
import torch
from torch.cuda.amp import autocast


# helpers
def exists(val):
  return val is not None


def default(val, d):
  if exists(val):
    return val
  return d() if isfunction(d) else d


def unique_id():
  """Generate a unique ID as specified in RFC 4122."""
  # See https://docs.python.org/3/library/uuid.html
  return str(uuid.uuid4())


@contextlib.contextmanager
def timing(msg, print_fn, prefix='', callback_fn=None):
  print_fn(f'{prefix}Started {msg}')
  tic = time.time()
  yield
  toc = time.time()
  if exists(callback_fn):
    callback_fn(tic, toc)
  print_fn(f'{prefix}Finished {msg} in {(toc-tic):>.3f} seconds')


# decorators


def set_backend_kwarg(fn):

  @wraps(fn)
  def inner(*args, backend='auto', **kwargs):
    if backend == 'auto':
      backend = 'torch' if isinstance(args[0], torch.Tensor) else 'numpy'
    kwargs.update(backend=backend)
    return fn(*args, **kwargs)

  return inner


def expand_dims_to(t, length=3):
  if length == 0:
    return t
  return t.reshape(*((1,) * length),
                   *t.shape)  # will work with both torch and numpy


def expand_arg_dims(dim_len=3):
  """ pack here for reuse.
        turns input into (B x D x N)
    """

  def outer(fn):

    @wraps(fn)
    def inner(x, y, **kwargs):
      assert len(x.shape) == len(y.shape), 'Shapes of A and B must match.'
      remaining_len = dim_len - len(x.shape)
      x = expand_dims_to(x, length=remaining_len)
      y = expand_dims_to(y, length=remaining_len)
      return fn(x, y, **kwargs)

    return inner

  return outer


def invoke_torch_or_numpy(torch_fn, numpy_fn):

  def outer(fn):

    @wraps(fn)
    def inner(*args, **kwargs):
      backend = kwargs.pop('backend')
      passed_args = fn(*args, **kwargs)
      passed_args = list(passed_args)
      if isinstance(passed_args[-1], dict):
        passed_kwargs = passed_args.pop()
      else:
        passed_kwargs = {}
      backend_fn = torch_fn if backend == 'torch' else numpy_fn
      return backend_fn(*passed_args, **passed_kwargs)

    return inner

  return outer


@contextlib.contextmanager
def torch_default_dtype(dtype):
  prev_dtype = torch.get_default_dtype()
  torch.set_default_dtype(dtype)
  yield
  torch.set_default_dtype(prev_dtype)


@contextlib.contextmanager
def torch_allow_tf32(allow=True):
  if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda.matmul,
                                                 'allow_tf32'):
    matmul_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow
  if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn,
                                                  'allow_tf32'):
    cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = allow
  yield
  if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn,
                                                  'allow_tf32'):
    torch.backends.cudnn.allow_tf32 = cudnn_allow_tf32
  if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda.matmul,
                                                 'allow_tf32'):
    torch.backends.cuda.matmul.allow_tf32 = matmul_allow_tf32


# pylint: disable=line-too-long
# distance utils (distogram to dist mat + masking)
# distance matrix to 3d coords: https://github.com/scikit-learn/scikit-learn/blob/42aff4e2e/sklearn/manifold/_mds.py#L279
# pylint: enable=line-too-long


def mds_torch(pre_dist_mat,
              weights=None,
              iters=10,
              tol=1e-5,
              eigen=False,
              verbose=2):
  """ Gets distance matrix. Outputs 3d. See below for wrapper.
        Assumes (for now) distogram is (N x N) and symmetric
        Outs:
        * best_3d_coords: (batch x 3 x N)
        * historic_stresses: (batch x steps)
    """
  device = pre_dist_mat.device
  # ensure batched MDS
  pre_dist_mat = expand_dims_to(pre_dist_mat,
                                length=(3 - len(pre_dist_mat.shape)))
  # start
  batch, n, _ = pre_dist_mat.shape
  diag_idxs = np.arange(n)
  his = [torch.tensor([np.inf] * batch, device=device)]

  # initialize by eigendecomposition:
  # https://www.lptmc.jussieu.fr/user/lesne/bioinformatics.pdf
  # follow :
  # https://www.biorxiv.org/content/10.1101/2020.11.27.401232v1.full.pdf
  d = pre_dist_mat**2
  m = 0.5 * (d[:, :1, :] + d[:, :, :1] - d)
  # do loop svd bc it's faster: (2-3x in CPU and 1-2x in GPU)
  # pylint: disable=line-too-long
  # https://discuss.pytorch.org/t/batched-svd-lowrank-being-much-slower-than-loop-implementation-both-cpu-and-gpu/119336
  # pylint: enable=line-too-long
  svds = [torch.svd_lowrank(mi) for mi in m]
  u = torch.stack([svd[0] for svd in svds], dim=0)
  s = torch.stack([svd[1] for svd in svds], dim=0)
  # v = torch.stack([svd[2] for svd in svds], dim=0)
  best_3d_coords = torch.bmm(u, torch.diag_embed(s).abs().sqrt())[..., :3]

  # only eigen - way faster but not weights
  if weights is None and eigen:
    return torch.transpose(best_3d_coords, -1,
                           -2), torch.zeros_like(torch.stack(his, dim=0))
  elif eigen:
    if verbose:
      print(
          'Can\'t use eigen flag if weights are active. Fallback to iterative')

  # continue the iterative way
  if weights is None:
    weights = torch.ones_like(pre_dist_mat)

  # iterative updates:
  for i in range(iters):
    # compute distance matrix of coords and stress
    best_3d_coords = best_3d_coords.contiguous()
    dist_mat = torch.cdist(best_3d_coords, best_3d_coords, p=2).clone()

    stress = (weights * (dist_mat - pre_dist_mat)**2).sum(dim=(-1, -2)) * 0.5
    # perturb - update X using the Guttman transform - sklearn-like
    dist_mat[dist_mat <= 0] += 1e-7
    ratio = weights * (pre_dist_mat / dist_mat)
    b = -ratio
    b[:, diag_idxs, diag_idxs] += ratio.sum(dim=-1)

    # update
    coords = (1. / n * torch.matmul(b, best_3d_coords))
    dis = torch.norm(coords, dim=(-1, -2))

    if verbose >= 2:
      print(f'it: {i}, stress {stress}')
    # update metrics if relative improvement above tolerance
    if (his[-1] - stress / dis).mean() <= tol:
      if verbose:
        print(f'breaking at iteration {i} with stress {stress / dis}')
      break

    best_3d_coords = coords
    his.append(stress / dis)

  return torch.transpose(best_3d_coords, -1, -2), torch.stack(his, dim=0)


def mds_numpy(pre_dist_mat,
              weights=None,
              iters=10,
              tol=1e-5,
              eigen=False,
              verbose=2):
  """ Gets distance matrix. Outputs 3d. See below for wrapper.
        Assumes (for now) distrogram is (N x N) and symmetric
        Out:
        * best_3d_coords: (3 x N)
        * historic_stress
    """
  del eigen
  if weights is None:
    weights = np.ones_like(pre_dist_mat)

  # ensure batched MDS
  pre_dist_mat = expand_dims_to(pre_dist_mat,
                                length=(3 - len(pre_dist_mat.shape)))
  # start
  batch, n, _ = pre_dist_mat.shape
  his = [np.inf]
  # init random coords
  best_stress = np.inf * np.ones(batch)
  best_3d_coords = 2 * np.random.rand(batch, 3, n) - 1
  # iterative updates:
  for i in range(iters):
    # compute distance matrix of coords and stress
    dist_mat = np.linalg.norm(best_3d_coords[:, :, :, None] -
                              best_3d_coords[:, :, None, :],
                              axis=-3)
    stress = ((weights * (dist_mat - pre_dist_mat))**2).sum(axis=(-1, -2)) * 0.5
    # perturb - update X using the Guttman transform - sklearn-like
    dist_mat[dist_mat == 0] = 1e-7
    ratio = weights * (pre_dist_mat / dist_mat)
    b = -ratio
    b[:, np.arange(n), np.arange(n)] += ratio.sum(axis=-1)
    # update - double transpose. TODO: consider fix
    coords = (1. / n * np.matmul(best_3d_coords, b))
    dis = np.linalg.norm(coords, axis=(-1, -2))
    if verbose >= 2:
      print(f'it: {i}, stress {stress}')
    # update metrics if relative improvement above tolerance
    if (best_stress - stress / dis).mean() <= tol:
      if verbose:
        print(f'breaking at iteration {i} with stress {stress / dis}')
      break

    best_3d_coords = coords
    best_stress = stress / dis
    his.append(best_stress)

  return best_3d_coords, np.array(his)


def get_dihedral_torch(c1, c2, c3, c4):
  """ Returns the dihedral angle in radians.
        Will use atan2 formula from:
        https://en.wikipedia.org/wiki/Dihedral_angle#In_polymer_physics
        Can't use torch.dot bc it does not broadcast
        Inputs:
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
    """
  u1 = c2 - c1
  u2 = c3 - c2
  u3 = c4 - c3

  return torch.atan2(
      ((torch.norm(u2, dim=-1, keepdim=True) * u1) *
       torch.cross(u2, u3, dim=-1)).sum(dim=-1),
      (torch.cross(u1, u2, dim=-1) * torch.cross(u2, u3, dim=-1)).sum(dim=-1))


def get_dihedral_numpy(c1, c2, c3, c4):
  """ Returns the dihedral angle in radians.
        Will use atan2 formula from:
        https://en.wikipedia.org/wiki/Dihedral_angle#In_polymer_physics
        Inputs:
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
    """
  u1 = c2 - c1
  u2 = c3 - c2
  u3 = c4 - c3

  return np.arctan2(
      ((np.linalg.norm(u2, axis=-1, keepdims=True) * u1) *
       np.cross(u2, u3, axis=-1)).sum(axis=-1),
      (np.cross(u1, u2, axis=-1) * np.cross(u2, u3, axis=-1)).sum(axis=-1))


def calc_phis_torch(pred_coords,
                    n_mask,
                    ca_mask,
                    c_mask=None,
                    prop=True,
                    verbose=0):
  """ Filters mirrors selecting the 1 with most N of negative phis.
        Used as part of the MDScaling wrapper if arg is passed. See below.
        Angle Phi between planes:
        (Cterm{-1}, N, Ca{0}) and (N{0}, Ca{+1}, Cterm{+1})
        Inputs:
        * pred_coords: (batch, 3, N) predicted coordinates
        * n_mask: (batch, N) boolean mask for N-term positions
        * ca_mask: (batch, N) boolean mask for C-alpha positions
        * c_mask: (batch, N) or None. boolean mask for C-alpha positions or
                    automatically calculate from n_mask and ca_mask if None.
        * prop: bool. whether to return as a proportion of negative phis.
        * verbose: bool. verbosity level
        Output: (batch, N) containing the phi angles or (batch,) containing
                the proportions.
        Note: use [0] since all prots in batch have same backbone
    """
  del verbose
  # detach gradients for angle calculation - mirror selection
  pred_coords_ = torch.transpose(pred_coords.detach(), -1, -2).cpu()
  # ensure dims
  n_mask = expand_dims_to(n_mask, 2 - len(n_mask.shape))
  ca_mask = expand_dims_to(ca_mask, 2 - len(ca_mask.shape))
  if c_mask is not None:
    c_mask = expand_dims_to(c_mask, 2 - len(c_mask.shape))
  else:
    c_mask = torch.logical_not(torch.logical_or(n_mask, ca_mask))
  # select points
  n_terms = pred_coords_[:, n_mask[0].squeeze()]
  c_alphas = pred_coords_[:, ca_mask[0].squeeze()]
  c_terms = pred_coords_[:, c_mask[0].squeeze()]
  # compute phis for every pritein in the batch
  phis = [
      get_dihedral_torch(c_terms[i, :-1], n_terms[i, 1:], c_alphas[i, 1:],
                         c_terms[i, 1:]) for i in range(pred_coords.shape[0])
  ]

  # return percentage of lower than 0
  if prop:
    return torch.stack([(x < 0).float().mean() for x in phis], dim=0)
  return phis


def calc_phis_numpy(pred_coords,
                    n_mask,
                    ca_mask,
                    c_mask=None,
                    prop=True,
                    verbose=0):
  """ Filters mirrors selecting the 1 with most N of negative phis.
        Used as part of the MDScaling wrapper if arg is passed. See below.
        Angle Phi between planes:
        (Cterm{-1}, N, Ca{0}) and (N{0}, Ca{+1}, Cterm{+1})
        Inputs:
        * pred_coords: (batch, 3, N) predicted coordinates
        * n_mask: (N, ) boolean mask for N-term positions
        * ca_mask: (N, ) boolean mask for C-alpha positions
        * c_mask: (N, ) or None. boolean mask for C-alpha positions or
                    automatically calculate from n_mask and ca_mask if None.
        * prop: bool. whether to return as a proportion of negative phis.
        * verbose: bool. verbosity level
        Output: (batch, N) containing the phi angles or (batch,) containing
                the proportions.
    """
  del verbose
  # detach gradients for angle calculation - mirror selection
  pred_coords_ = np.transpose(pred_coords, (0, 2, 1))
  n_terms = pred_coords_[:, n_mask.squeeze()]
  c_alphas = pred_coords_[:, ca_mask.squeeze()]
  # select c_term auto if not passed
  if c_mask is not None:
    c_terms = pred_coords_[:, c_mask]
  else:
    c_terms = pred_coords_[:, (np.ones_like(n_mask) - n_mask -
                               ca_mask).squeeze().astype(bool)]
  # compute phis for every pritein in the batch
  phis = [
      get_dihedral_numpy(c_terms[i, :-1], n_terms[i, 1:], c_alphas[i, 1:],
                         c_terms[i, 1:]) for i in range(pred_coords.shape[0])
  ]

  # return percentage of lower than 0
  if prop:
    return np.array([(x < 0).mean() for x in phis])
  return phis


# alignment by centering + rotation to compute optimal RMSD
# adapted from : https://github.com/charnley/rmsd/


def kabsch_torch(x, y, cpu=False):
  """ Kabsch alignment of x into y.
        Assumes x, y are both (Dims x N_points). See below for wrapper.
    """
  device = x.device
  with autocast(enabled=False):
    x, y = x.float(), y.float()
    # center x and y to the origin
    x_ = x - x.mean(dim=-1, keepdim=True)
    y_ = y - y.mean(dim=-1, keepdim=True)
    # calculate convariance matrix (for each prot in the batch)
    c = torch.matmul(x_, y_.t()).detach()
    if cpu:
      c = c.cpu()
    # Optimal rotation matrix via SVD
    if int(torch.__version__.split('.')[1]) < 8:
      # Warning! int torch 1.<8 : W must be transposed
      v, s, w = torch.svd(c)
      w = w.t()
    else:
      v, s, w = torch.linalg.svd(c)

    # determinant sign for direction correction
    d = (torch.det(v) * torch.det(w)) < 0.0
    if d:
      s[-1] = s[-1] * (-1)
      v[:, -1] = v[:, -1] * (-1)
    # Create Rotation matrix U
    u = torch.matmul(v, w).to(device)
    # calculate rotations
    x_ = torch.matmul(x_.t(), u).t()
  # return centered and aligned
  return x_, y_


def kabsch_numpy(x, y):
  """ Kabsch alignment of x into y.
        Assumes x,y are both (Dims x N_points). See below for wrapper.
    """
  # center x and y to the origin
  x_ = x - x.mean(axis=-1, keepdims=True)
  y_ = y - y.mean(axis=-1, keepdims=True)
  # calculate convariance matrix (for each prot in the batch)
  c = np.dot(x_, y_.transpose())
  # Optimal rotation matrix via SVD
  v, s, w = np.linalg.svd(c)
  # determinant sign for direction correction
  d = (np.linalg.det(v) * np.linalg.det(w)) < 0.0
  if d:
    s[-1] = s[-1] * (-1)
    v[:, -1] = v[:, -1] * (-1)
  # Create Rotation matrix U
  u = np.dot(v, w)
  # calculate rotations
  x_ = np.dot(x_.T, u).T
  # return centered and aligned
  return x_, y_


# metrics - more formulas here: http://predictioncenter.org/casp12/doc/help.html


def rmsd_torch(x, y):
  """ Assumes x,y are both (B x D x N). See below for wrapper. """
  return torch.sqrt(torch.mean((x - y)**2, axis=(-1, -2)))


def rmsd_numpy(x, y):
  """ Assumes x,y are both (B x D x N). See below for wrapper. """
  return np.sqrt(np.mean((x - y)**2, axis=(-1, -2)))


def gdt_torch(x, y, cutoffs, weights=None):
  """ Assumes x,y are both (B x D x N). see below for wrapper.
        * cutoffs is a list of `K` thresholds
        * weights is a list of `K` weights (1 x each threshold)
    """
  device = x.device
  if weights is None:
    weights = torch.ones(1, len(cutoffs))
  else:
    weights = torch.tensor([weights]).to(device)
  # set zeros and fill with values
  score = torch.zeros(x.shape[0], len(cutoffs), device=device)
  dist = ((x - y)**2).sum(dim=1).sqrt()
  # iterate over thresholds
  for i, cutoff in enumerate(cutoffs):
    score[:, i] = (dist <= cutoff).float().mean(dim=-1)
  # weighted mean
  return (score * weights).mean(-1)


def gdt_numpy(x, y, cutoffs, weights=None):
  """ Assumes x, y are both (B x D x N). see below for wrapper.
        * cutoffs is a list of `K` thresholds
        * weights is a list of `K` weights (1 x each threshold)
    """
  if weights is None:
    weights = np.ones((1, len(cutoffs)))
  else:
    weights = np.array([weights])
  # set zeros and fill with values
  score = np.zeros((x.shape[0], len(cutoffs)))
  dist = np.sqrt(((x - y)**2).sum(axis=1))
  # iterate over thresholds
  for i, cutoff in enumerate(cutoffs):
    score[:, i] = (dist <= cutoff).mean(axis=-1)
  # weighted mean
  return (score * weights).mean(-1)


def tmscore_torch(x, y, n):
  """ Assumes x,y are both (B x D x N). see below for wrapper. """
  n = max(21, n)
  d0 = 1.24 * (n - 15)**(1.0 / 3.0) - 1.8
  # get distance
  dist = ((x - y)**2).sum(dim=1).sqrt()
  # formula (see wrapper for source):
  return (1.0 / (1.0 + (dist / d0)**2)).mean(dim=-1)


def tmscore_numpy(x, y, n):
  """ Assumes x,y are both (B x D x N). see below for wrapper. """
  n = max(21, n)
  d0 = 1.24 * np.cbrt(n - 15) - 1.8
  # get distance
  dist = np.sqrt(((x - y)**2).sum(axis=1))
  # formula (see wrapper for source):
  return (1 / (1 + (dist / d0)**2)).mean(axis=-1)


def mdscaling_torch(pre_dist_mat,
                    weights=None,
                    iters=10,
                    tol=1e-5,
                    fix_mirror=True,
                    n_mask=None,
                    ca_mask=None,
                    c_mask=None,
                    eigen=False,
                    verbose=2):
  """ Handles the specifics of MDS for proteins (mirrors, ...) """
  # batched mds for full parallel
  preds, stresses = mds_torch(pre_dist_mat,
                              weights=weights,
                              iters=iters,
                              tol=tol,
                              eigen=eigen,
                              verbose=verbose)
  if not fix_mirror:
    return preds, stresses

  # no need to caculate multiple mirrors - just correct Z axis
  phi_ratios = calc_phis_torch(preds, n_mask, ca_mask, c_mask, prop=True)
  to_correct = torch.nonzero((phi_ratios < 0.5)).view(-1)
  # fix mirrors by (-1)*Z if more (+) than (-) phi angles
  preds[to_correct, -1] = (-1) * preds[to_correct, -1]
  if verbose == 2:
    print('Corrected mirror idxs:', to_correct)

  return preds, stresses


def mdscaling_numpy(pre_dist_mat,
                    weights=None,
                    iters=10,
                    tol=1e-5,
                    fix_mirror=True,
                    n_mask=None,
                    ca_mask=None,
                    c_mask=None,
                    verbose=2):
  """ Handles the specifics of MDS for proteins (mirrors, ...) """
  # batched mds for full parallel
  preds, stresses = mds_numpy(pre_dist_mat,
                              weights=weights,
                              iters=iters,
                              tol=tol,
                              verbose=verbose)
  if not fix_mirror:
    return preds, stresses

  # no need to caculate multiple mirrors - just correct Z axis
  phi_ratios = calc_phis_numpy(preds, n_mask, ca_mask, c_mask, prop=True)
  for i, _ in enumerate(preds):
    # fix mirrors by (-1)*Z if more (+) than (-) phi angles
    if phi_ratios < 0.5:
      preds[i, -1] = (-1) * preds[i, -1]
      if verbose == 2:
        print('Corrected mirror in struct no.', i)

  return preds, stresses


################
### WRAPPERS ###
################

# pylint: disable=invalid-name

@set_backend_kwarg
@invoke_torch_or_numpy(mdscaling_torch, mdscaling_numpy)
def MDScaling(pre_dist_mat, **kwargs):
  """ Gets distance matrix (-ces). Outputs 3d.
        Assumes (for now) distrogram is (N x N) and symmetric.
        For support of ditograms: see `center_distogram_torch()`
        Inputs:
        * pre_dist_mat: (1, N, N) distance matrix.
        * weights: optional. (N x N) pairwise relative weights .
        * iters: number of iterations to run the algorithm on
        * tol: relative tolerance at which to stop the algorithm if no better
               improvement is achieved
        * backend: one of ["numpy", "torch", "auto"] for backend choice
        * fix_mirror: int. number of iterations to run the 3d generation and
                      pick the best mirror (highest number of negative phis)
        * n_mask: indexing array/tensor for indices of backbone N.
                  Only used if fix_mirror > 0.
        * ca_mask: indexing array/tensor for indices of backbone C_alpha.
                   Only used if fix_mirror > 0.
        * verbose: whether to print logs
        Outputs:
        * best_3d_coords: (3 x N)
        * historic_stress: (timesteps, )
    """
  pre_dist_mat = expand_dims_to(pre_dist_mat, 3 - len(pre_dist_mat.shape))
  return pre_dist_mat, kwargs


@expand_arg_dims(dim_len=2)
@set_backend_kwarg
@invoke_torch_or_numpy(kabsch_torch, kabsch_numpy)
def Kabsch(A, B):
  """ Returns Kabsch-rotated matrices resulting
        from aligning A into B.
        Adapted from: https://github.com/charnley/rmsd/
        * Inputs:
            * A,B are (3 x N)
            * backend: one of ["numpy", "torch", "auto"] for backend choice
        * Outputs: tensor/array of shape (3 x N)
    """
  # run calcs - pick the 0th bc an additional dim was created
  return A, B


@expand_arg_dims()
@set_backend_kwarg
@invoke_torch_or_numpy(rmsd_torch, rmsd_numpy)
def RMSD(A, B):
  """ Returns RMSD score as defined here (lower is better):
        https://en.wikipedia.org/wiki/
        Root-mean-square_deviation_of_atomic_positions
        * Inputs:
            * A,B are (B x 3 x N) or (3 x N)
            * backend: one of ["numpy", "torch", "auto"] for backend choice
        * Outputs: tensor/array of size (B,)
    """
  return A, B


@expand_arg_dims()
@set_backend_kwarg
@invoke_torch_or_numpy(gdt_torch, gdt_numpy)
def GDT(A, B, *, mode='TS', cutoffs=None, weights=None):
  """ Returns GDT(Global Distance Test) score as defined
        here (highre is better):
        http://predictioncenter.org/casp12/doc/help.html
        Supports both TS and HA
        * Inputs:
            * A,B are (B x 3 x N) (np.array or torch.tensor)
            * cutoffs: defines thresholds for gdt
            * weights: list containing the weights
            * mode: one of ["numpy", "torch", "auto"] for backend
        * Outputs: tensor/array of size (B,)
    """
  # define cutoffs for each type of gdt and weights
  cutoffs = default(cutoffs,
                    [0.5, 1, 2, 4] if mode in ['HA', 'ha'] else [1, 2, 4, 8])
  # calculate GDT
  return A, B, cutoffs, {'weights': weights}


@expand_arg_dims()
@set_backend_kwarg
@invoke_torch_or_numpy(tmscore_torch, tmscore_numpy)
def TMscore(A, B, *, L):
  """ Returns TMscore as defined here (higher is better):
        >0.5 (likely) >0.6 (highly likely) same folding.
        = 0.2. https://en.wikipedia.org/wiki/Template_modeling_score
        Warning! It's not exactly the code in:
        https://zhanglab.ccmb.med.umich.edu/TM-score/TMscore.cpp
        but will suffice for now.
        Inputs:
            * A, B are (B x 3 x N) (np.array or torch.tensor)
            * mode: one of ["numpy", "torch", "auto"] for backend
        Outputs: tensor/array of size (B,)
    """
  return A, B, dict(n=L)

# pylint: enable=invalid-name

def contact_precision_torch(pred, truth, ratios, ranges, mask=None, cutoff=8):
  # (..., l, l)
  assert truth.shape[-1] == truth.shape[-2]
  assert pred.shape == truth.shape

  seq_len = truth.shape[-1]
  mask1s = torch.ones_like(truth, dtype=torch.int8)
  if exists(mask):
    mask1s = mask1s * (mask[..., :, None] * mask[..., None, :])
  mask_ranges = map(
      lambda r: torch.triu(mask1s, default(r[0], 0)) - torch.triu(
          mask1s, default(r[1], seq_len)), ranges)

  pred_truth = torch.stack((pred, truth), dim=-1)
  for (i, j), m in zip(ranges, mask_ranges):
    masked_pred_truth = pred_truth[m.bool()]
    sorter_idx = (-masked_pred_truth[:, 0]).argsort()
    sorted_pred_truth = masked_pred_truth[sorter_idx]

    num_corrects = ((0 < sorted_pred_truth[:,1]) &
                    (sorted_pred_truth[:,1] <= cutoff)).sum()
    for ratio in ratios:
      num_tops = max(1, min(num_corrects, int(seq_len * ratio)))
      assert 0 < num_tops <= seq_len
      top_labels = sorted_pred_truth[:num_tops, 1]
      pred_corrects = ((0 < top_labels) & (top_labels <= cutoff)).sum()
      yield (i, j), ratio, pred_corrects / float(num_tops)


def contact_precision_numpy(pred, truth, ratios, ranges, mask=None, cutoff=8):
  # (l, l)
  assert truth.shape[-1] == truth.shape[-2]
  assert pred.shape == truth.shape

  seq_len = truth.shape[-1]
  mask1s = np.ones_like(truth, dtype=np.int8)
  if exists(mask):
    mask1s = mask1s * (mask[...:, None] * mask[..., None, :])
  mask_ranges = map(
      lambda r: np.triu(mask1s, default(r[0], 0)) - np.triu(
          mask1s, default(r[1], seq_len)), ranges)

  pred_truth = np.stack((pred, truth), axis=-1)
  for (i, j), m in zip(ranges, mask_ranges):
    masked_pred_truth = pred_truth[m.nonzero()]
    sorter_idx = (-masked_pred_truth[:, 0]).argsort()
    sorted_pred_truth = masked_pred_truth[sorter_idx]

    num_corrects = ((0 < sorted_pred_truth[:,1]) &
                    (sorted_pred_truth[:,1] <= cutoff)).sum()
    for ratio in ratios:
      num_tops = max(1, min(num_corrects, int(seq_len * ratio)))
      assert 0 < num_tops <= seq_len
      top_labels = sorted_pred_truth[:num_tops, 1]
      pred_corrects = ((0 < top_labels) & (top_labels <= cutoff)).sum()
      yield (i, j), ratio, pred_corrects / float(num_tops)


@set_backend_kwarg
@invoke_torch_or_numpy(contact_precision_torch, contact_precision_numpy)
def contact_precision(pred, truth, ratios=None, ranges=None, **kwargs):
  if not exists(ratios):
    ratios = [1, .5, .2, .1]
  if not exists(ranges):
    ranges = [(6, 12), (12, 24), (24, None)]
  return pred, truth, ratios, ranges, kwargs
