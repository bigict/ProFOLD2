"""for train, run
     ```bash
     $python trainer.py -h
     ```
     for further help.
"""
import os
import logging
from logging.handlers import QueueHandler, QueueListener
import resource

import torch
import torch.multiprocessing as mp
from torch import nn

from profold2.model import Alphafold2

class _WorkerLogFilter(logging.Filter):
  def __init__(self, rank=-1):
    super().__init__()
    self._rank = rank

  def filter(self, record):
    if self._rank != -1:
      record.msg = f'Rank {self._rank} | {record.msg}'
    return True

class _WorkerLogging(object):
  """Initialize distibuted logger
  """
  def __init__(self, work_fn, args):  # pylint: disable=redefined-outer-name
    # logging
    os.makedirs(args.prefix, exist_ok=True)

    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(
                args.prefix,
                f'{work_fn.__name__}.log'))]

    def handler_apply(h, f, *arg):
      f(*arg)
      return h
    level=logging.DEBUG if args.verbose else logging.INFO
    handlers = list(map(
        lambda x: handler_apply(x, x.setLevel, level),
        handlers))
    fmt = '%(asctime)-15s [%(levelname)s] (%(filename)s:%(lineno)d) %(message)s'
    handlers = list(map(lambda x: handler_apply(
        x, x.setFormatter, logging.Formatter(fmt)), handlers))

    logging.basicConfig(
        format=fmt,
        level=level,
        handlers=handlers)

    self.queue = mp.Queue(-1)
    self.listener = QueueListener(self.queue, *handlers,
        respect_handler_level=True)

  def start(self):
    self.listener.start()

  def stop(self):
    self.listener.stop()

class _WorkerFunction(object):
  """Wrap the distibuted function
  """
  def __init__(self, work_fn, log_queue=None):
    self.work_fn = work_fn
    self.log_queue = log_queue

  def __call__(self, rank, args):  # pylint: disable=redefined-outer-name
    #--------------
    # setup distributed env if needed
    #--------------

    # logging
    if self.log_queue:
      root = logging.getLogger()
      ctx_handler = QueueHandler(self.log_queue)
      if args.gpu_list:
        ctx_filter = _WorkerLogFilter(args.gpu_list[rank])
        ctx_handler.addFilter(ctx_filter)
      root.addHandler(ctx_handler)

      level=logging.DEBUG if args.verbose else logging.INFO
      root.setLevel(level)

    if args.gpu_list:
      logger = logging.getLogger(__name__)
      logger.info(
              'torch.distributed.init_process_group: rank=%d@%d, world_size=%d',
              rank, args.gpu_list[rank], len(args.gpu_list))
      torch.distributed.init_process_group(
              backend='nccl',
              init_method=f'file://{args.ipc_file}',
              rank=rank,
              world_size=len(args.gpu_list))
      torch.cuda.set_device(args.gpu_list[rank])

    self.work_fn(rank, args)

    #--------------
    # cleanup
    #--------------
    if args.gpu_list:
      torch.distributed.destroy_process_group()

class WorkerModel(object):
  """Wrap distibuted model
  """
  def __init__(self, rank, args):  # pylint: disable=redefined-outer-name
    self.rank = rank
    self.args = args

  def device(self):
    if self.args.gpu_list and self.rank < len(self.args.gpu_list):
      assert self.args.gpu_list[self.rank] < torch.cuda.device_count()
      #torch.cuda.set_device(self.args.gpu_list[self.rank])
      return self.args.gpu_list[self.rank]
    return torch.device('cpu')

  def wrap(self, **kwargs):
    if self.args.gpu_list and self.rank < len(self.args.gpu_list):
      logger = logging.getLogger(__name__)

      model = Alphafold2(**kwargs)
      device = self.device()
      model.to(device)

      logger.info('wrap model with nn.parallel.DistributedDataParallel class')
      model = nn.parallel.DistributedDataParallel(
          model,
          device_ids=[self.args.gpu_list[self.rank]],
          output_device=self.args.gpu_list[self.rank],
          find_unused_parameters=False)
      model._set_static_graph()  # pylint: disable=protected-access
    return model

  def load(self, f):
    device = self.device()
    checkpoint = torch.load(f, map_location=self.args.map_location)
    model = Alphafold2(dim=checkpoint['dim'],
                       depth=checkpoint['evoformer_depth'],
                       heads=checkpoint['evoformer_head_num'],
                       dim_head=checkpoint['evoformer_head_dim'],
                       embedd_dim=checkpoint['mlm_dim'],
                       headers=checkpoint['headers'])
    model.load_state_dict(checkpoint['model'])
    model = model.to(device=device)
    model.eval()

    return checkpoint['feats'], model

def main(args, fn):  # pylint: disable=redefined-outer-name
  mp.set_start_method('spawn', force=True)

  if hasattr(fn, 'preprocess'):
    fn.preprocess(args)

  work_log = _WorkerLogging(fn, args)
  work_log.start()

  logger = logging.getLogger(__name__)
  logger.info('-----------------')
  logger.info('Arguments: %s', args)
  logger.info('-----------------')

  work_fn = _WorkerFunction(fn, work_log.queue)
  mp.spawn(work_fn, args=(args,),
          nprocs=len(args.gpu_list) if args.gpu_list else 1,
          join=True)

  if hasattr(fn, 'postprocess'):
    fn.postprocess(args)

  logger.info('-----------------')
  logger.info('Resources(myself): %s',
      resource.getrusage(resource.RUSAGE_SELF))
  logger.info('Resources(children): %s',
      resource.getrusage(resource.RUSAGE_CHILDREN))
  logger.info('-----------------')

  work_log.stop()

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('-g', '--gpu_list', type=int, nargs='+',
      help='list of GPU IDs')
  parser.add_argument('--ipc_file', type=str, default='/tmp/profold2.dist',
      help='ipc file to initialize the process group')
  parser.add_argument('-o', '--prefix', type=str, default='.',
      help='prefix of out directory, default=\'.\'')
  parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
  args = parser.parse_args()

  main(args, print)
