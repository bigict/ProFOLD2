"""Tools for inference, run
        ```bash
        $python evaluator.py -h
        ```
        for further help.
"""
import os
from dataclasses import dataclass, make_dataclass
import logging
import pickle

import torch
from einops import rearrange

# models & data
from profold2.data import dataset
from profold2.data.utils import tensor_to_numpy
from profold2.model import (
    accelerator, functional, profiler, snapshot, FeatureBuilder, ReturnValues
)
from profold2.utils import exists, timing

from profold2.command import worker


@dataclass
class Args(worker.Args):
  pass


def run(rank, args):  # pylint: disable=redefined-outer-name
  wm = worker.WorkerModel(rank, args)
  feats, model = wm.load(args.model)
  features = FeatureBuilder(feats).to(wm.device())
  logging.info('feats: %s', feats)

  kwargs = {}
  if rank.is_available() and accelerator.world_size(args.nnodes) > 1:
    kwargs['num_replicas'] = accelerator.world_size(args.nnodes)
    kwargs['rank'] = rank.rank
  test_loader = dataset.load(
      data_dir=args.eval_data,
      data_idx=args.eval_idx,
      attr_idx=args.eval_attr,
      pseudo_linker_prob=args.pseudo_linker_prob,
      max_msa_depth=args.max_msa_size,
      max_var_depth=args.max_var_size,
      var_task_num=args.num_var_task,
      min_crop_len=args.min_crop_len,
      max_crop_len=args.max_crop_len,
      crop_algorithm=args.crop_algorithm,
      crop_probability=args.crop_probability,
      msa_as_seq_prob=args.msa_as_seq_prob,
      msa_as_seq_topn=args.msa_as_seq_topn,
      msa_as_seq_clustering=args.msa_as_seq_clustering,
      msa_as_seq_min_alr=args.msa_as_seq_min_alr,
      feat_flags=(~dataset.FEAT_PDB if args.eval_without_pdb else dataset.FEAT_ALL),
      batch_size=args.batch_size,
      num_workers=args.num_workers,
      **kwargs
  )

  def data_cond(batch):
    return (
        args.min_protein_len <= batch['seq'].shape[1] and
        batch['seq'].shape[1] < args.max_protein_len
    )

  def data_eval(idx, batch):
    fasta_name, fasta_len = ','.join(batch['pid']), batch['seq'].shape[1]
    with timing(
        f'Building features for model on {fasta_name} {fasta_len}', logging.debug
    ):
      batch = features(batch, is_training=False)

    # predict - out is (batch, L * 3, 3)
    with timing(f'Running model on {fasta_name} {fasta_len}', logging.debug):
      with torch.no_grad():
        with accelerator.amp(args.amp_enabled):  # Automatic Mixed Precision
          r = ReturnValues(
              **model(
                  batch=batch,  # pylint: disable=not-callable
                  num_recycle=args.model_recycles,
                  shard_size=args.model_shard_size
              )
          )

    metric_dict = {}
    if 'confidence' in r.headers:
      metric_dict['confidence'] = r.headers['confidence']['loss'].item()
      logging.debug(
          '%d pid: %s Confidence: %s', idx, fasta_name,
          r.headers['confidence']['loss'].item()
      )
    if 'fitness' in r.headers:
      fitness = torch.sigmoid(r.headers['fitness']['variant_logit'])
      logging.info(
          'no: %d pid: %s, fitness: pred=%s', idx, fasta_name, fitness.tolist()
      )
      dump_pkl = {'variant_pred': fitness}
      if 'motifs' in r.headers['fitness']:
        dump_pkl['motifs'] = tensor_to_numpy(r.headers['fitness']['motifs'])
        # logging.info('no: %d pid: %s, motifs: motif=%s', idx, fasta_name,
        #              r.headers['fitness']['motifs'].tolist())
      if 'gating' in r.headers['fitness'] and exists(r.headers['fitness']['gating']):
        dump_pkl['gating'] = tensor_to_numpy(r.headers['fitness']['gating'])
      for key in ('seq', 'mask', 'seq_color'):
        if key in batch:
          dump_pkl[key] = tensor_to_numpy(batch[key])
        # logging.info('no: %d pid: %s, fitness: color=%s', idx, fasta_name,
        #              batch['seq_color'].tolist())
      if 'variant_label' in batch:
        dump_pkl['label'] = tensor_to_numpy(batch['variant_label'])
        logging.info(
            'no: %d pid: %s, fitness: true=%s', idx, fasta_name,
            batch['variant_label'].tolist()
        )
      if 'variant_label_mask' in batch:
        dump_pkl['label_mask'] = tensor_to_numpy(batch['variant_label_mask'])
        logging.info(
            'no: %d pid: %s, fitness: mask=%s', idx, fasta_name,
            batch['variant_label_mask'].tolist()
        )
      if 'variant_pid' in batch:
        dump_pkl['pid'] = batch['variant_pid']
        logging.info(
            'no: %d pid: %s, fitness: desc=%s', idx, fasta_name, batch['variant_pid']
        )
      assert 'coevolution' in r.headers
      if 'wij' in r.headers['coevolution']:
        wij = tensor_to_numpy(r.headers['coevolution']['wij'])
        bi = tensor_to_numpy(r.headers['coevolution']['bi'])
        dump_pkl['coevolution'] = {'wij': wij, 'bi': bi}
        if 'wab' in r.headers['coevolution']:
          dump_pkl['coevolution']['wab'] = tensor_to_numpy(
              r.headers['coevolution']['wab']
          )

      with open(os.path.join(args.prefix, f'{fasta_name}_var.pkl'), 'wb') as f:
        pickle.dump(dump_pkl, f)
    if 'metric' in r.headers:
      metrics = r.headers['metric']['loss']
      if 'contact' in metrics:
        if '[24,inf)_1' in metrics['contact']:
          metric_dict['P@L'] = metrics['contact']['[24,inf)_1'].item()
      if 'coevolution' in metrics:
        if 'perplexity' in metrics['coevolution']:
          metric_dict['perplexity'] = metrics['coevolution']['perplexity']
    if 'folding' in r.headers:
      assert 'coords' in r.headers['folding']
      if 'coord' in batch:
        coords = r.headers['folding']['coords']  # (b l c d)
        _, _, num_atoms, _ = coords.shape

        labels = batch['coord'][..., :num_atoms, :]
        flat_cloud_mask = rearrange(
            batch['coord_mask'][..., :num_atoms], 'b l c -> b (l c)'
        )

        # rotate / align
        with accelerator.autocast(enabled=False):
          coords_aligned, labels_aligned = functional.kabsch_align(
              rearrange(coords.float(), 'b l c d -> b (l c) d')[flat_cloud_mask],
              rearrange(labels.float(), 'b l c d -> b (l c) d')[flat_cloud_mask]
          )

        tms = functional.tmscore(
            coords_aligned, labels_aligned, n=torch.sum(batch['mask'], dim=-1)
        )
        metric_dict['tmscore'] = tms.item()
        logging.debug('%d pid: %s TM-score: %f', idx, fasta_name, tms.item())

        # tmscore, n = tmscore + tms.item(), n + 1

      logging.info(
          'no: %d pid: %s, %s', idx, fasta_name,
          ', '.join(f'{k}: {v}' for k, v in metric_dict.items())
      )

      return tms.item()
    else:
      raise ValueError('folding are not implemented yet!')

  tmscore, n = 0, 0

  with profiler.profile(
      enabled=args.enable_profiler,
      record_shapes=True,
      profile_memory=True,
      with_stack=True
  ) as prof:
    with snapshot.memory_snapshot(
        enabled=args.enable_memory_snapshot, device=rank.device
    ):
      # eval loop
      for idx, batch in enumerate(filter(data_cond, iter(test_loader))):
        try:
          tmscore += data_eval(idx, batch)
          n += 1
        except RuntimeError as e:
          logging.error('%d %s', idx, str(e))

        if hasattr(prof, 'step'):
          prof.step()
  if hasattr(prof, 'key_averages'):
    logging.debug('%s', prof.key_averages().table(sort_by='cuda_time_total'))

  if n > 0:
    logging.info('%d TM-score: %f (average)', n, tmscore / n)


def add_arguments(parser):  # pylint: disable=redefined-outer-name
  parser.add_argument(
      '--map_location',
      type=str,
      default=None,
      help='remapped to an alternative set of devices.'
  )
  parser.add_argument(
      '--model', type=str, default='model.pth', help='model of profold2.'
  )

  parser.add_argument('--eval_data', type=str, default=None, help='eval dataset.')
  parser.add_argument('--eval_idx', type=str, default=None, help='eval dataset idx.')
  parser.add_argument(
      '--eval_attr', type=str, default=None, help='eval dataset attr idx.'
  )
  parser.add_argument(
      '--eval_without_pdb', action='store_true', help='DO NOT load pdb data.'
  )
  parser.add_argument(
      '--min_protein_len',
      type=int,
      default=0,
      help='filter out proteins whose length<LEN.'
  )
  parser.add_argument(
      '--max_protein_len',
      type=int,
      default=1024,
      help='filter out proteins whose length>LEN.'
  )
  parser.add_argument(
      '--max_msa_size', type=int, default=1024, help='filter out MSAs whose size>SIZE.'
  )
  parser.add_argument(
      '--max_var_size', type=int, default=None, help='filter out VARs whose size>SIZE.'
  )
  parser.add_argument(
      '--num_var_task', type=int, default=1, help='number of tasks in VARs.'
  )
  parser.add_argument(
      '--min_crop_len',
      type=int,
      default=None,
      help='filter out proteins whose length<LEN.'
  )
  parser.add_argument(
      '--max_crop_len',
      type=int,
      default=None,
      help='filter out proteins whose length>LEN.'
  )
  parser.add_argument(
      '--crop_algorithm',
      type=str,
      default='random',
      choices=['random', 'domain', 'knn'],
      help='type of crop algorithm.'
  )
  parser.add_argument(
      '--crop_probability',
      type=float,
      default=0.0,
      help='crop protein with probability CROP_PROBABILITY when it\'s '
      'length>MIN_CROP_LEN.'
  )
  parser.add_argument(
      '--pseudo_linker_prob',
      type=float,
      default=0.0,
      help='enable loading complex data.'
  )
  parser.add_argument(
      '--msa_as_seq_prob',
      type=float,
      default=0.0,
      help='take msa_{i} as sequence with probability DATA_MSA_AS_SEQ_PROB.'
  )
  parser.add_argument(
      '--msa_as_seq_topn',
      type=int,
      default=None,
      help='take msa_{i} as sequence belongs to DATA_MSA_AS_SEQ_TOPN.'
  )
  parser.add_argument(
      '--msa_as_seq_clustering',
      action='store_true',
      help='take msa_{i} as sequence sampling from clusters.'
  )
  parser.add_argument(
      '--msa_as_seq_min_alr',
      type=float,
      default=None,
      help='take msa_{i} as sequence with alr <= DATA_MSA_AS_SEQ_MIN_ALR.'
  )

  parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch size.')
  parser.add_argument('--num_workers', type=int, default=1, help='number of workers.')

  parser.add_argument(
      '--model_recycles', type=int, default=0, help='number of recycles in profold2.'
  )
  parser.add_argument(
      '--model_shard_size',
      type=int,
      default=None,
      help='shard size in evoformer model.'
  )

  parser.add_argument(
      '--amp_enabled', action='store_true', help='enable automatic mixed precision.'
  )
  parser.add_argument('--enable_profiler', action='store_true', help='enable profiler.')
  parser.add_argument(
      '--enable_memory_snapshot', action='store_true', help='enable memory snapshot.'
  )


if __name__ == '__main__':
  import argparse
  import hydra

  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

  parser.add_argument('-c', '--config', type=str, default=None, help='config file.')
  parser.add_argument(
      'overrides',
      nargs='*',
      metavar='KEY=VAL',
      help='override configs, see: https://hydra.cc'
  )

  args = parser.parse_args()
  config_dir, config_name = os.path.split(
      os.path.abspath(args.config)
  ) if exists(args.config) else (os.getcwd(), None)

  with hydra.initialize_config_dir(
      version_base=None, config_dir=config_dir, job_name=__file__
  ):
    worker.main(
        make_dataclass('t', [], namespace={
            'Args': Args,
            'run': run
        }), hydra.compose(config_name, args.overrides)
    )
