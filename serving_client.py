"""Tools for post request to serving, run
     ```bash
     $python serving_client.py -h
     ```
     for further help.
"""
import os
import base64
import io
import json
import re
import logging

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
import requests

from profold2.common import residue_constants
from profold2.data.parsers import parse_fasta
from profold2.utils import exists, timing

logger = logging.getLogger(__file__)

# pylint: disable=line-too-long
# curl http://127.0.0.1:8080/predictions/profold0_0?abc=1 -H "Content-Type: plain/text" -X POST --data-binary @a.fasta
# pylint: enable=line-too-long
def main(args):  # pylint: disable=redefined-outer-name
  def filename_get(desc, model_name, ext, prefix=''):
    pid = re.split('[ |\t]', desc)[0]
    p = os.path.join(args.prefix, pid)
    os.makedirs(p, exist_ok=True)
    return os.path.join(p, f'{prefix}{model_name}.{ext}')

  os.makedirs(args.prefix, exist_ok=True)

  for fasta_file in args.fasta_list:
    logger.info('Request: %s begin', fasta_file)
    with open(fasta_file, 'r') as f:
      fasta_str = f.read()
    with timing(f'Request: {fasta_file}', print):
      r = requests.post(args.uri,
                        json={'sequence':fasta_str, 'fmt':args.fasta_fmt,
                                  'num_recycle':args.model_recycles,
                                  'shard_size':args.model_shard_size},
                        timeout=7200)
    if r.status_code != 200:
      logger.error('Request: %s error: %s', fasta_file, r.status_code)
    else:
      if args.fasta_fmt == 'a4m':
        sequences = fasta_str.splitlines()
        descriptions = [f'{i}' for i in range(len(sequences))]
      else:
        sequences, descriptions = parse_fasta(fasta_str)
      if args.fasta_fmt != 'single':
        sequences, descriptions = sequences[:1], descriptions[:1]
      results = r.json()
      if not isinstance(results, list):
        results = [results]
      if not args.multi_model_format:
        results = [{'model_1':row} for row in results]
      assert len(sequences) == len(descriptions)
      assert len(sequences) == len(results)
      for seq, desc, result in zip(
          sequences, descriptions, results):

        unrelaxed_pdbs, unrelaxed_svgs = {}, {}
        ranking_scores = {}

        for model_name, result in result.items():
          assert 'pdb' in result and 'headers' in result
          pdb, header = result['pdb'], result['headers']

          with io.BytesIO(base64.b64decode(header)) as f:
            header = torch.load(f, map_location='cpu')

          if args.dump_header:
            f = filename_get(desc, model_name, 'pth')
            torch.save(header, f)

          plddt = None
          if 'confidence' in header:
            if 'loss' in header['confidence']:
              plddt = header['confidence']['loss']

          ranking_scores[model_name] = 0
          if exists(plddt):
            ranking_scores[model_name] = plddt.item()

          print('======================')
          print(f'>{desc} pLDDT: {plddt}')
          print(f'{seq}')

          unrelaxed_pdbs[model_name] = pdb
          with open(filename_get(desc, model_name, 'pdb'), 'w') as f:
            f.write(pdb)

          if 'metric' in header:
            print('-------------')
            print(f'{desc} metric:', header['metric']['loss'])
          if 'coevolution' in header and 'logits' in header['coevolution']:
            print('-------------')
            logits = torch.squeeze(header['coevolution']['logits'], dim=0)
            print('.............')
            def yield_aa(logits_msa):
              for i, logits_seq in enumerate(logits_msa):
                for j, logits_aa in enumerate(logits_seq):
                  prob_aa = F.softmax(logits_aa, dim=-1)
                  print(i, j, torch.argmax(logits_aa, dim=-1), prob_aa)
            yield_aa(logits)
            pred = torch.argmax(logits, dim=-1)
            def yield_seq(int_msa):
              for int_seq in int_msa:
                yield ''.join(map(
                    lambda i: residue_constants.restypes_with_x_and_gap[i],
                    int_seq))

            msa_list = list(yield_seq(pred))
            if args.dump_msa:
              with open(filename_get(desc, model_name, 'a4m'), 'w') as f:
                f.write('\n'.join(msa_list))
            print('\n'.join(msa_list))
          if 'distogram' in header:
            logits = torch.squeeze(header['distogram']['logits'], dim=0)
            logits = torch.argmax(logits, dim=-1)
            plt.matshow(-logits)
            # plt.tight_layout()
            with io.BytesIO() as f:
              plt.savefig(f, format='svg', dpi=100)
              t = io.TextIOWrapper(io.BytesIO(f.getvalue()))
            plt.close()

            svg = t.read()
            unrelaxed_svgs[model_name] = svg
            with open(filename_get(desc, model_name, 'svg'), 'w') as f:
              f.write(svg)

        # Rank by model confidence and write out relaxed PDBs in rank order.
        ranked_order = []
        for idx, (model_name, _) in enumerate(
            sorted(ranking_scores.items(), key=lambda x: x[1], reverse=True)):
          ranked_order.append(model_name)
          with open(filename_get(desc, f'{idx}', 'pdb', prefix='ranked_'),
                    'w') as f:
            f.write(unrelaxed_pdbs[model_name])
          with open(filename_get(desc, f'{idx}', 'svg', prefix='ranked_'),
                    'w') as f:
            f.write(unrelaxed_svgs[model_name])

        with open(filename_get(desc, 'ranking_debug', 'json'), 'w') as f:
          f.write(json.dumps(
              {'confidences': ranking_scores, 'order': ranked_order}, indent=4))

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--uri', type=str,
      default='http://127.0.0.1:8080/predictions/profold0',
      help='ipc file to initialize the process group')
  parser.add_argument('-o', '--prefix', type=str, default='.',
      help='prefix of out directory, default=\'.\'')
  parser.add_argument('fasta_list', type=str, nargs='+',
      help='list of fasta files')
  parser.add_argument('--fasta_fmt', type=str, default='single',
      choices=['single', 'a3m', 'a4m'],
      help='format of fasta files')
  parser.add_argument('--model_recycles', type=int, default=0,
      help='number of recycles in model, default=0')
  parser.add_argument('--model_shard_size', type=int, default=None,
      help='shard size in evoformer model, default=None')
  parser.add_argument('--multi_model_format', action='store_true',
      help='dump multi model format pdb files')
  parser.add_argument('--dump_msa', action='store_true', help='dump msa files')
  parser.add_argument('--dump_header', action='store_true', help='dump headers')
  parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
  args = parser.parse_args()

  main(args)
