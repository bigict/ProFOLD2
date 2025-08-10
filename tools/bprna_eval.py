import os

import torch
from torch.nn import functional as F
from einops import rearrange

from profold2.data import dataset
from profold2.data.dataset import ProteinStructureDataset
from profold2.utils import exists, timing


def evaluate(x, y, threshold=0.5, eps=1e-11):
  pred = torch.clamp(
      1. - torch.exp(
          torch.sum(
              torch.log(torch.clamp(1. - torch.exp(x.float()), min=eps, max=1.)),
              dim=-1
          )
      ),
      min=eps,
      max=1.
  ) >= threshold
  truth = torch.any(y, dim=-1)

  pred_p = torch.sum(pred)
  true_p = torch.sum(truth)

  tp = torch.sum(pred * truth)
  fp = pred_p - tp
  fn = true_p - tp

  recall = (tp + eps) / (tp + fn + eps)
  precision = (tp + eps) / (tp + fp + eps)
  f1_score = (2 * tp + eps) / (2 * tp + fp + fn + eps)

  return precision, recall, f1_score


def main(args):
  data = ProteinStructureDataset(data_dir=args.data_dir, data_idx=args.data_idx)

  pid_to_idx = {}
  for idx, pid_list in enumerate(data.pids):
    for k, pid in enumerate(pid_list):
      pid_to_idx[pid] = (idx, k)

  for pth_file in args.pth_file:
    with timing(f"processing {pth_file}", print):
      r = torch.load(pth_file, map_location="cpu")
      x = torch.squeeze(r.headers["pairing"]["logits"], dim=0)

      pid = pth_file.split(os.sep)[args.pid_idx]
      if args.pid_idx == -1:
        pid, _ = os.path.splitext(pid)
      feat = data[pid_to_idx[pid]]

      l, eps = feat['seq'].shape[-1], 1e-6
      y = rearrange(
          F.one_hot(feat['sta'].long(), l + 1)[..., 1:], '... i d j -> ... i j d'
      )

      precision, recall, f1_score = evaluate(x, y, threshold=args.threshold)
      print(f"{pid}\t{x.shape}\t{y.shape}\t{precision}\t{recall}\t{f1_score}")


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument("pth_file", type=str, nargs="+", help="prediction file (.pth).")
  parser.add_argument(
      "--data_dir", type=str, default=None, help="train dataset dir."
  )
  parser.add_argument(
      "--data_idx", type=str, default=None, help="dataset idx."
  )
  parser.add_argument("--threshold", type=float, default=.5, help="threhold.")
  parser.add_argument("--pid_idx", type=int, default=-2, help="extract pid from path.")
  args = parser.parse_args()

  main(args)
