"""Tools for calculate TMscore, run
     ```bash
     $python tmscore.py -h
     ```
     for further help.
"""
import os
import sys
import io
import subprocess

import pandas as pd


def parse_tmscore_output(src, terms: tuple):
  if src[:6] == ' There':
    # Case: There is no common residues in the input structures
    result = dict(zip(terms, (float('nan'),) * len(terms)))
    return result
  result = {}
  start_keyword = dict(tmscore='TM-score    = ',
                       gdt='GDT-TS-score= ',
                       length='Length=  ')
  end_char = dict(tmscore=' ', gdt=' ', length='\n')
  for term in terms:
    assert term in end_char
    key_word = start_keyword[term]
    start_index = src.find(key_word)
    start_index += len(key_word)
    end_index = None
    if term in end_char:
      end_index = src.find(end_char[term], start_index)
    result[term] = float(src[start_index:end_index])
  return result


def parse_deepscore_output(src, terms):
  key_name = dict(tmscore='TMscore', gdt='GDT_TS', length='length')
  length_line = 7  # 0-indexed.
  title_line = 12  # 0-indexed.
  score_line = 13  # 0-indexed.
  lines = src.split('\n')
  titles = lines[title_line].strip('#').split()
  scores = map(float, lines[score_line].split())
  all_scores = dict(zip(titles, scores))
  all_scores['length'] = float(lines[length_line].split('length=')[-1].strip())
  return {k: all_scores[key_name[k]] for k in terms}


def run_scorer(exe_path, predicted_path, gt_path, terms):
  scorer_process = subprocess.run([exe_path, predicted_path, gt_path],
                                  stdout=subprocess.PIPE, check=True)
  f = io.TextIOWrapper(io.BytesIO(scorer_process.stdout))
  if 'Deep' in os.path.basename(exe_path):
    # DeepScore
    return parse_deepscore_output(f.read(), terms)
  # TMScore
  return parse_tmscore_output(f.read(), terms)


def read_pairwise_list(f):
  for line in filter(lambda x: x, map(lambda x: x.strip(), f)):
    pid, model_f, native_f = line.split('\t')
    assert os.path.exists(model_f), model_f
    if os.path.exists(native_f):
      yield pid, model_f, native_f


def main(args):  # pylint: disable=redefined-outer-name
  name_map = {'tmscore': 'TM-score', 'length': 'Length'}
  if args.gdt:
    name_map['gdt'] = 'GDT-TS'

  if args.pairwise_list:
    if args.pairwise_list == '-':
      protein_ids = list(read_pairwise_list(sys.stdin))
    else:
      with open(args.pairwise_list, 'r') as f:
        protein_ids = list(read_pairwise_list(f))
  elif os.path.isdir(args.model_dir):
    assert os.path.isdir(args.native_dir)
    protein_ids = []
    for model_file in os.path.listdir(args.model_dir):
      pdb_file = os.path.basename(model_file)
      protein_id, pdb_ext = os.path.splitext(pdb_file)
      native_file = os.path.join(args.native_dir, pdb_file)
      if pdb_ext == '.pdb' and os.path.exists(native_file):
        protein_ids.append((protein_id, pdb_file, native_file))
  else:
    assert not os.path.isdir(args.native_dir)
    protein_ids = [(f'{args.model_dir}->{args.native_dir}', args.model_dir,
                    args.native_dir)]
  df = pd.DataFrame(index=list(map(lambda x: x[0], protein_ids)),
                    columns=name_map.values(),
                    dtype='float')
  for i, (pid, model_f, native_f) in enumerate(protein_ids, start=1):
    if args.verbose:
      print(f'{i}/{len(protein_ids)}: {pid}')

    scores = run_scorer(args.exe_path, model_f, native_f, terms=name_map.keys())
    for k, v in scores.items():
      df.loc[pid][name_map[k]] = v
      if args.verbose:
        print(f'{pid} {k}={v}')
  print(df.describe())
  if args.output:
    df.to_csv(args.output, sep='\t')


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()

  parser.add_argument('model_dir',
                      nargs='?',
                      help='The dir holding the predicted protein structures.')
  parser.add_argument('native_dir',
                      nargs='?',
                      help='The dir holding the native protein structures.')
  parser.add_argument('-l',
                      '--pairwise_list',
                      default=None,
                      help='The dir holding the native protein structures.')
  parser.add_argument(
      '-o',
      '--output',
      type=str,
      help=
      'The output file (.csv) of scores of each predicted protein structure.')
  parser.add_argument('--gdt',
                      action='store_true',
                      help='Get GDT-TS score as long as TMscore.')
  parser.add_argument('-e',
                      '--exe_path',
                      type=str,
                      default='TMscore',
                      help='The path of binary `TMscore`')
  parser.add_argument('-v',
                      '--verbose',
                      action='store_true',
                      help='Get verbose information')

  args = parser.parse_args()

  main(args)
