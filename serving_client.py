"""Tools for post request to serving, run
     ```bash
     $python serving_client.py -h
     ```
     for further help.
"""
import os
import base64
import io
import re
import logging

import matplotlib.pyplot as plt
import torch
import requests

from profold2.data.parsers import parse_fasta

logger = logging.getLogger(__file__)

# pylint: disable=line-too-long
# curl http://127.0.0.1:8080/predictions/profold0_0?abc=1 -H "Content-Type: plain/text" -X POST --data-binary @a.fasta
# pylint: enable=line-too-long
def main(args):  # pylint: disable=redefined-outer-name
  def filename_get(desc, ext):
    pid = re.split('[ |\t]', desc)[0]
    if args.multi_model_format:
      p = os.path.join(args.prefix, desc.split()[0])
      os.makedirs(p, exist_ok=True)
      return os.path.join(p, f'top_1.{ext}')
    return os.path.join(args.prefix, f'{pid}.{ext}')

  os.makedirs(args.prefix, exist_ok=True)

  for fasta_file in args.fasta_list:
    logger.info('Request: %s begin', fasta_file)
    with open(fasta_file, 'r', encoding='utf-8') as f:
      fasta_str = f.read()
    r = requests.post(args.uri,
        data=dict(body=fasta_str, fmt=args.fasta_fmt), timeout=7200)
    if r.status_code != 200:
      logger.error('Request: %s error: %s', fasta_file, r.status_code)
    else:
      logger.info('Request: %s end', fasta_file)
      if args.fasta_fmt == 'a4m':
        sequences = fasta_str.splitlines()
        descriptions = [f'{i}' for i in range(len(sequences))]
      else:
        sequences, descriptions = parse_fasta(fasta_str)
      if args.fasta_fmt != 'single':
        sequences, descriptions = sequences[:1], descriptions[:1]
      results = r.json()
      assert 'pdb' in results and 'headers' in results
      assert len(sequences) == len(descriptions)
      assert len(sequences) == len(results['pdb'])
      assert len(sequences) == len(results['headers'])
      for seq, desc, pdb, header in zip(
          sequences, descriptions, results['pdb'], results['headers']):
        with io.BytesIO(base64.b64decode(header)) as f:
          header = torch.load(f, map_location='cpu')

        plddt = None
        if 'confidence' in header:
          if 'loss' in header['confidence']:
            plddt = header['confidence']['loss']

        print('======================')
        print(f'>{desc} pLDDT: {plddt}')
        print(f'{seq}')

        print('-------------')
        print(f'{pdb}')
        if args.dump_pdb:
          with open(filename_get(desc, 'pdb'), 'w', encoding='utf-8') as f:
            f.write(pdb)

        if 'confidence' in header:
          if 'loss' in header['confidence']:
            plddt = header['confidence']['loss']
            print('-------------')
            print(f'pLDDT: {plddt}')
          if 'plddt' in header['confidence']:
            print(header['confidence']['plddt'])
        if 'metric' in header:
          print('-------------')
          print('metric:', header['metric']['loss'])
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
          print('-------------')
          print(svg)
          if args.dump_contact:
            with open(filename_get(desc, 'svg'), 'w', encoding='utf-8') as f:
              f.write(svg)

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--uri', type=str,
      default='http://127.0.0.1:8080/predictions/profold0_0',
      help='ipc file to initialize the process group')
  parser.add_argument('-o', '--prefix', type=str, default='.',
      help='prefix of out directory, default=\'.\'')
  parser.add_argument('fasta_list', type=str, nargs='+',
      help='list of fasta files')
  parser.add_argument('--fasta_fmt', type=str, default='single',
      choices=['single', 'a3m', 'a4m'],
      help='format of fasta files')
  parser.add_argument('--multi_model_format', action='store_true',
      help='dump multi model format pdb files')
  parser.add_argument('--dump_pdb', action='store_true', help='dump pdb files')
  parser.add_argument('--dump_contact', action='store_true',
      help='dump contact images')
  parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
  args = parser.parse_args()

  main(args)
