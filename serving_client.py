import base64
import io
import matplotlib.pyplot as plt
import logging

import torch
import requests

from profold2.data.parsers import parse_fasta

logger = logging.getLogger(__file__)

# curl http://127.0.0.1:8080/predictions/profold0_0?abc=1 -H "Content-Type: plain/text" -X POST --data-binary @a.fasta
def main(args):  # pylint: disable=redefined-outer-name
  headers = {'Content-Type': 'plain/text'}

  for fasta_file in args.fasta_list:
    logger.info('Request: %s begin', fasta_file)
    with open(fasta_file, 'r') as f:
      fasta_str = f.read()
    r = requests.post(args.uri, data=fasta_str)
    if r.status_code != 200:
      logger.error('Request: %s error: %s', fasta_file, r.status_code)
    else:
      logger.info('Request: %s end', fasta_file)
      sequences, descriptions = parse_fasta(fasta_str)
      results = r.json()
      assert 'pdb' in results and 'headers' in results
      assert len(sequences) == len(descriptions)
      assert len(sequences) == len(results['pdb'])
      assert len(sequences) == len(results['headers'])
      for seq, desc, _, header in zip(sequences, descriptions, results['pdb'], results['headers']):
        with io.BytesIO(base64.b64decode(header)) as f:
          header = torch.load(f, map_location='cpu')
        if 'distogram' in header:
          logits = torch.squeeze(header['distogram']['logits'])
          logits = torch.argmax(logits, dim=-1)
          plt.matshow(-logits)
          plt.tight_layout()
          with io.BytesIO() as f:
            plt.savefig(f, format='svg', dpi=100)
            t = io.TextIOWrapper(io.BytesIO(f.getvalue()))
            print('-------')
            print(desc)
            print(t.read())
          plt.close()

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--uri', type=str,
      default='http://127.0.0.1:8080/predictions/profold0_0',
      help='ipc file to initialize the process group')
  parser.add_argument('-f', '--fasta_list', type=str, nargs='+',
      help='list of GPU IDs')
  parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
  args = parser.parse_args()

  main(args)
