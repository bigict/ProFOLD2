"""Tools for web task, run
     ```bash
     $python task.py -h
     ```
     for further help.
"""
import sys
import argparse
import base64
from datetime import datetime
import functools
import io
import json
from urllib.parse import urlparse
import logging

import matplotlib.pyplot as plt
import torch
import requests

from web import db
from web.utils import (serving_data,
    serving_log,
    serving_meta,
    serving_pdb,
    serving_svg)
from profold2.utils import timing
import relaxer

logger = logging.getLogger(__file__)

def to_log_file(text, f=sys.stdout):
  print(text, file=f, flush=True)

def to_fasta_str(description, sequence):
  return f'>{description}\n{sequence}'

def do_task(task, uri, args, log_func=print):  # pylint: disable=redefined-outer-name
  db.job_set(job_id=task['job_id'], task_id=task['id'],
             status=db.STATUS_RUNNING, time_run=datetime.now())

  fasta_str = to_fasta_str(task['description'], task['sequence'])

  logger.info('Request: %s@%s begin', task['job_id'], task['id'])
  with timing('requesting serving', log_func, prefix='   '):
    if 'params' not in task:
      task['params'] = {}
    r = requests.post(uri,
                      json=dict(sequence=fasta_str, **task['params']),
                      timeout=7200)
  logger.info('Request: %s@%s end (%s)',
      task['job_id'], task['id'], r.status_code)
  if r.status_code != 200:
    logger.error('Request: %s@%s error: %s',
        task['job_id'], task['id'], r.status_code)
    db.job_set(job_id=task['job_id'], task_id=task['id'],
               status=db.STATUS_ERROR, time_done=datetime.now())
    return False

  results = r.json()
  with open(serving_meta(task['job_id'], task['id']),
      'w', encoding='utf-8') as log:
    print('======================', file=log)
    print(fasta_str, file=log)

    for model_name, result in results.items():
      assert 'pdb' in result and 'headers' in result
      pdb, header = result['pdb'], result['headers']
      metrics = {}

      with io.BytesIO(base64.b64decode(header)) as f:
        header = torch.load(f, map_location='cpu')

      if 'confidence' in header:
        if 'loss' in header['confidence']:
          plddt = header['confidence']['loss'].tolist()
          metrics['pLDDT'] = plddt
          print('-------------', file=log)
          print(f'pLDDT@{model_name}: {plddt}', file=log)
        if 'plddt' in header['confidence']:
          print(header['confidence']['plddt'], file=log)

      if 'distogram' in header:
        logits = torch.squeeze(header['distogram']['logits'])
        logits = torch.argmax(logits, dim=-1)
        plt.matshow(-logits)
        # plt.tight_layout()
        with io.BytesIO() as f:
          plt.savefig(f, format='svg', dpi=100)
          t = io.TextIOWrapper(io.BytesIO(f.getvalue()))
        plt.close()

        svg = t.read()
        print('-------------', file=log)
        print(svg, file=log)
        if args.dump_contact:
          with open(serving_svg(
                  task['job_id'], task['id'], prefix=f'{model_name}_'),
              'w', encoding='utf-8') as f:
            f.write(svg)

      print('-------------', file=log)
      print(f'{pdb}', file=log)
      with open(serving_pdb(
              task['job_id'], task['id'], prefix=f'unrelaxed_{model_name}_'),
          'w', encoding='utf-8') as f:
        f.write(pdb)
      if args.run_relaxer:
        c = argparse.Namespace(
                use_gpu_relax=args.use_gpu_relax,
                pdb_files=[serving_pdb(task['job_id'], task['id'],
                                       prefix=f'unrelaxed_{model_name}_')],
                output=serving_data(task['job_id']),
                prefix='relaxed_{model_name}_')
        retry = 0
        while retry < args.relax_retry:
          try:
            with timing(f'relaxing pdb {model_name}', log_func, prefix='   '):
              relaxer.main(c)
            break
          except Exception as e:  # pylint: disable=broad-except
            print(f'warning: {e}', file=log)
          retry += 1
        if retry >= args.relax_retry:
          db.job_set(job_id=task['job_id'], task_id=task['id'],
                     status=db.STATUS_ERROR, time_done=datetime.now())
          return False

  db.job_set(job_id=task['job_id'], task_id=task['id'],
      metrics=json.dumps(metrics),
      status=db.STATUS_DONE, time_done=datetime.now())
  return True

def do_job(job, uri, args):  # pylint: disable=redefined-outer-name
  with open(serving_log(job['job_id']), 'w', encoding='utf-8') as f:
    log_func = functools.partial(to_log_file, f=f)

    db.job_set(job_id=job['job_id'],
               status=db.STATUS_RUNNING, time_run=datetime.now())
    with timing('job', log_func):
      n = 0
      for task in job['tasks']:
        if args.skip_task_done and task['status'] == db.STATUS_DONE:
          n += 1
          continue

        with timing(f'task: `>{task["description"]}`', log_func):
          if do_task(task, uri, args, log_func=log_func):
            log_func(f'Run task: `>{task["description"]}` succeed.')
            n += 1
          else:
            log_func(f'Run task: `>{task["description"]}` failed.')
      if n == len(job['tasks']):
        db.job_set(job_id=job['job_id'],
                   status=db.STATUS_DONE, time_done=datetime.now())
        log_func('Job has done ...')
      else:
        db.job_set(job_id=job['job_id'],
                   status=db.STATUS_ERROR, time_done=datetime.now())
        log_func('Job has done with errors ...')

def run_jobs(args):  # pylint: disable=redefined-outer-name
  o = urlparse(args.uri)
  app = o.scheme

  o = o._replace(scheme='http')
  uri = o.geturl()

  if args.job_list:
    job_list = [db.job_get(app=app, job_id=job_id) for job_id in args.job_list]
  else:
    job_list = db.job_get(app=app, status=db.STATUS_QUEUING)
  for job in job_list:
    do_job(job, uri, args)

def list_jobs(args):  # pylint: disable=redefined-outer-name
  kwargs = dict(map(lambda x: x.split('=', 1), args.job_kwargs))
  job_list = db.job_get(with_tasks=False, logic_op=args.logic_op, **kwargs)
  for job in job_list:
    print(f'{job["app"]}\t{job["job_id"]}\t{job["status"]}')

def main(args):  # pylint: disable=redefined-outer-name
  logger.info(args)
  if args.cmd == 'run':
    run_jobs(args)
  elif args.cmd == 'list':
    list_jobs(args)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parsers = parser.add_subparsers(dest='cmd', help='commands help')

  run_job = parsers.add_parser('run', help='run jobs')
  run_job.add_argument('job_list', type=str, nargs='*',
      help='job list')
  run_job.add_argument('--uri', type=str,
      default='profold0://127.0.0.1:8080/predictions/profold0_0',
      help='uri')
  run_job.add_argument('--dump_pdb', action='store_true',
      help='dump pdb files')
  run_job.add_argument('--dump_contact', action='store_true',
      help='dump contact images')
  run_job.add_argument('--run_relaxer', action='store_true',
      help='dump relaxed pdb files')
  run_job.add_argument('--use_gpu_relax', action='store_true',
      help='run relax on gpu')
  run_job.add_argument('--relax_retry', type=int, default=2,
      help='try to run relax `n` times')
  run_job.add_argument('--skip_task_done', action='store_true',
      help='skip tasks done')

  list_job = parsers.add_parser('list', help='list jobs')
  list_job.add_argument('--logic_op', choices=['and', 'or'], default='and',
      help='sql where ops, default `and`')
  list_job.add_argument('job_kwargs', type=str, nargs='+',
      help='sql where kwargs: key=value')

  parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
  args = parser.parse_args()

  logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

  main(args)
