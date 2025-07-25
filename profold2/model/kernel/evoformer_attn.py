# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import numpy as np

from profold2.model.kernel import builder

kernel_ = None


def _on_cuda(tensor):
  return hasattr(tensor, 'device') and tensor.device.type == 'cuda'


def attention_fwd(Q, K, V, bias1, bias2):
  assert Q.shape[-3] > 16, 'seq_len must be greater than 16'
  O = torch.empty_like(Q, dtype=Q.dtype)
  assert _on_cuda(Q), 'Q must be on cuda'
  assert _on_cuda(K), 'K must be on cuda'
  assert _on_cuda(V), 'V must be on cuda'
  assert _on_cuda(bias1), 'bias1 must be on cuda'
  assert _on_cuda(bias2), 'bias2 must be on cuda'
  global kernel_
  if kernel_ is None:
    kernel_ = builder.build(builder.ATTENTION_CORE_NAME)
  nheads = Q.shape[-2]
  nq = (Q.shape[-3] + 31) // 32 * 32
  nb = np.prod(Q.shape[:-3])
  lse = torch.empty((nb, nheads, nq), dtype=torch.float32, device=Q.device)
  kernel_.attention_fwd(Q, K, V, bias1, bias2, O, lse)
  return O, lse


def attention_bwd(dO, Q, K, V, O, lse, bias1, bias2, bias1_grad, bias2_grad):
  assert max(
      Q.shape[-1], V.shape[-1]
  ) <= 64, 'Hidden size is too large. Need to change kMax to a larger value'
  dQ = torch.empty_like(Q, dtype=Q.dtype)
  dK = torch.empty_like(K, dtype=K.dtype)
  dV = torch.empty_like(V, dtype=V.dtype)
  assert _on_cuda(dO), 'dO must be on cuda'
  assert _on_cuda(Q), 'Q must be on cuda'
  assert _on_cuda(K), 'K must be on cuda'
  assert _on_cuda(V), 'V must be on cuda'
  assert _on_cuda(O), 'O must be on cuda'
  global kernel_
  if kernel_ is None:
    kernel_ = builder.build(builder.ATTENTION_CORE_NAME)
  delta = torch.empty_like(lse)
  if bias1_grad:
    dB1 = torch.zeros_like(bias1, dtype=torch.float32)
  else:
    dB1 = torch.tensor([], dtype=torch.float32, device=bias1.device)
  if bias2_grad:
    dB2 = torch.zeros_like(bias2, dtype=torch.float32)
  else:
    dB2 = torch.tensor([], dtype=torch.float32, device=bias2.device)
  kernel_.attention_bwd(dO, Q, K, V, O, lse, delta, bias1, bias2, dQ, dK, dV, dB1, dB2)
  return dQ, dK, dV, dB1.to(dO.dtype), dB2.to(dO.dtype)


class EvoformerFusedAttention(torch.autograd.Function):
  @staticmethod
  def forward(ctx, q, k, v, bias1=None, bias2=None):
    """
        q, k, v: are in shape [*, L, H, D]
    """
    bias1_ = bias1.contiguous(
    ) if bias1 is not None else torch.tensor([], dtype=q.dtype, device=q.device)
    bias2_ = bias2.contiguous(
    ) if bias2 is not None else torch.tensor([], dtype=q.dtype, device=q.device)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    o, lse = attention_fwd(q, k, v, bias1_, bias2_)
    ctx.save_for_backward(q, k, v, o, lse, bias1_, bias2_)
    return o

  @staticmethod
  def backward(ctx, grad_output):
    q, k, v, o, lse, bias1, bias2 = ctx.saved_tensors
    is_b1_grad = bias1.numel() != 0 and ctx.needs_input_grad[3]
    is_b2_grad = bias2.numel() != 0 and ctx.needs_input_grad[4]
    dQ, dK, dV, dB1, dB2 = attention_bwd(
        grad_output, q, k, v, o, lse, bias1, bias2, is_b1_grad, is_b2_grad
    )
    if not is_b1_grad:
      dB1 = None
    if not is_b2_grad:
      dB2 = None
    return dQ, dK, dV, dB1, dB2


def evoformer_attn(Q, K, V, biases):
  assert len(biases) <= 2

  if (len(biases) == 0):
    biases.append(None)

  if (len(biases) == 1):
    biases.append(None)

  # bias_1_shape = lambda x: (x.shape[0], x.shape[1], 1, 1, x.shape[2])
  # bias_2_shape = lambda x: (x.shape[0], 1, x.shape[3], x.shape[2], x.shape[2])
  bias_1_shape = lambda x: (*x.shape[:-4], x.shape[-4], 1, 1, x.shape[-3])
  bias_2_shape = lambda x: (*x.shape[:-4], 1, x.shape[-2], x.shape[-3], x.shape[-3])

  if biases[0] is not None:
    assert biases[0].shape == bias_1_shape(Q), 'bias1 shape is incorrect'

  if biases[1] is not None:
    assert biases[1].shape == bias_2_shape(Q), 'bias2 shape is incorrect'

  return EvoformerFusedAttention.apply(Q, K, V, biases[0], biases[1])
