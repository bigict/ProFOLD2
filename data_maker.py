"""Tools for make dataset, run
     ```bash
     $python data_maker.py -h
     ```
     for further help.
"""
import os
import argparse
import functools
import gzip
import multiprocessing as mp
from itertools import groupby
import logging

import numpy as np

from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.SeqIO.FastaIO import SimpleFastaParser as FastaParser

from profold2.common import residue_constants

logger = logging.getLogger(__file__)

def lines(f):
  for line in filter(lambda x: len(x)>0, map(lambda x: x.strip(), f)):
    yield line

def parse_fasta(filename, datasource=None):
  def iter_fasta(it):
    for name, seq in it:
      if datasource == 'swissprot':
        name = name.split('|')[1]
      elif datasource in ['rcsb', 'norm']:
        name = name.split()[0]
      yield name, seq
  if filename.endswith('.gz'):
    with gzip.open(filename, 'rt', encoding='utf-8') as f:
      for name, seq in iter_fasta(FastaParser(f)):
        yield name, seq
  with open(filename, 'r', encoding='utf-8') as f:
    for name, seq in iter_fasta(FastaParser(f)):
      yield name, seq

def decompose_pid(pid, datasource=None):
  if datasource != 'norm':
    k = pid.find('_')
    if k != -1:
      return pid[:k], pid[k+1:]
  return pid, None

def compose_pid(pid, chain):
  return f'{pid}_{chain}' if chain else f'{pid}'

def parse_cluster(filename):
  with open(filename, 'r', encoding='utf-8') as f:
    for cid, values in groupby(map(lambda x: x.split(), lines(f)),
                               key=lambda x: x[0]):
      items = set()
      for value in values:
        items |= set(value)
      yield cid, items

def mmcif_get_filename(pid, chain, datasource=None):
  if datasource == 'swissprot':
    return f'AF-{pid}-F1-model_v2.cif.gz'
  elif datasource == 'rcsb':
    return f'{pid.lower()}.cif.gz'
  elif datasource == 'norm':
    assert chain is None
    return f'{pid}.cif.gz'
  del chain

def mmcif_parse(filename):
  if filename.endswith('.gz'):
    with gzip.open(filename, 'rt', encoding='utf-8') as f:
      return MMCIF2Dict(f)
  return MMCIF2Dict(filename)

def mmcif_get_chain(structure, chain_id):
  for model in structure.get_models():
    if chain_id is None or model.has_id(chain_id):
      for chain in model.get_chains():
        if chain_id is None or chain.id == chain_id:
          return chain
  return None

def mmcif_get_coords(
    mmcif_dict, chain, str_seqs, datasource=None, add_plddt=False):
  n = len(str_seqs)
  assert n > 0

  labels = np.zeros((n, 14, 3), dtype=np.float32)
  label_mask = np.zeros((n, 14), dtype=np.bool)
  bfactors = np.zeros((n, 14), dtype=np.float32)

  # two special chars as placeholders in the mmCIF format
  # for item values that cannot be explicitly assigned
  # see: pdbx/mmcif syntax web page
  _unassigned = {'.', '?'}  # pylint: disable=invalid-name

  atom_id_list = mmcif_dict['_atom_site.label_atom_id']
  residue_id_list = mmcif_dict['_atom_site.label_comp_id']
  chain_id_list = mmcif_dict['_atom_site.auth_asym_id']
  x_list = [float(x) for x in mmcif_dict['_atom_site.Cartn_x']]
  y_list = [float(x) for x in mmcif_dict['_atom_site.Cartn_y']]
  z_list = [float(x) for x in mmcif_dict['_atom_site.Cartn_z']]
  icode_list = mmcif_dict['_atom_site.pdbx_PDB_ins_code']
  b_factor_list = mmcif_dict['_atom_site.B_iso_or_equiv']
  fieldname_list = mmcif_dict['_atom_site.group_PDB']

  def _is_unmatch(int_resseq, resname):
    return (int_resseq < 1 or int_resseq > len(str_seqs) or
        (resname != residue_constants.unk_restype and
         str_seqs[int_resseq - 1] != 'X' and
         str_seqs[int_resseq - 1] != resname))
  def _is_match(int_resseq, resname):
    return not _is_unmatch(int_resseq, resname)
  def _get_resseq(seq_id_key, i):
    if 0 <= i < len(mmcif_dict[seq_id_key]):
      return int(mmcif_dict[seq_id_key][i])
    return 0
  def _running_error(int_resseq, resname):
    aa = '-'
    if 1 <= int_resseq <= len(str_seqs):
      aa = str_seqs[int_resseq - 1]
    raise PDBConstructionException(f'{int_resseq} str_seqs={aa} resname={resname}')  # pylint: disable=line-too-long

  def seq_id_key_switch(atom_id_list, delta=0):
    def seq_id_key_match(seq_id_key):
      if not seq_id_key in mmcif_dict:
        return False
      running_state, prev_resseq = 0, None

      for i, _ in atom_id_list:
        if running_state == 0:
          running_state = 1

        int_resseq = int(mmcif_dict[seq_id_key][i])
        resname = residue_constants.restype_3to1.get(
            residue_id_list[i], residue_constants.unk_restype)
        if running_state == 1:
          if prev_resseq == int_resseq:
            continue
          elif _is_unmatch(int_resseq + delta, resname):
            if int_resseq != _get_resseq(seq_id_key, i + 1):
              return False  #_running_error(int_resseq + delta, resname)
            running_state = 2
            prev_resseq = int_resseq
            continue
        elif running_state == 2:
          if prev_resseq != int_resseq:
            return False #_running_error(int_resseq + delta, resname)
          elif _is_match(int_resseq + delta, resname):
            running_state = 1
          else:
            continue
        prev_resseq = int_resseq

      return True

    for seq_id_key in ('_atom_site.auth_seq_id', '_atom_site.label_seq_id'):
      if seq_id_key_match(seq_id_key):
        return seq_id_key
    return None

  atom_id_list = list(filter(
      lambda x: ((chain is None or chain_id_list[x[0]] == chain)
          and fieldname_list[x[0]] == 'ATOM'),
      enumerate(atom_id_list)))

  delta = 0
  if atom_id_list:
    i, _ = atom_id_list[0]
    delta = min(map(lambda x: 1-int(mmcif_dict[x][i]),
        filter(lambda x: x in mmcif_dict,
            ('_atom_site.auth_seq_id', '_atom_site.label_seq_id'))))

  for delta in range(delta, len(str_seqs)):
    seq_id_key = seq_id_key_switch(atom_id_list, delta)
    if seq_id_key is not None:
      break
  if seq_id_key is None:
    seq_id_key, delta = '_atom_site.auth_seq_id', 0

  current_resseq = None
  residue_plddt, residue_n, atom_plddt, atom_n = 0.0, 0, 0.0, 0

  running_state, prev_resseq = 0, 0

  for i, atom_id in atom_id_list:
    if running_state == 0:
      running_state = 1

    icode = icode_list[i]
    if icode in _unassigned:
      icode = ' '
    int_resseq = _get_resseq(seq_id_key, i)

    resname = residue_constants.restype_3to1.get(
        residue_id_list[i], residue_constants.unk_restype)
    # if (int_resseq < 1 or int_resseq > len(str_seqs) or
    #     str_seqs[int_resseq - 1] != resname):
    if running_state == 1:
      if prev_resseq == int_resseq:
        pass
      elif _is_unmatch(int_resseq + delta, resname):
        if int_resseq != _get_resseq(seq_id_key, i + 1):
          _running_error(int_resseq + delta, resname)
        running_state = 2
        prev_resseq = int_resseq
        continue
      # aa = '-'
      # if 1 <= int_resseq <= len(str_seqs):
      #   aa = str_seqs[int_resseq - 1]
      # raise PDBConstructionException(f'{int_resseq} str_seqs={aa} resname={resname}, icode={icode}')  # pylint: disable=line-too-long
    elif running_state == 2:
      if prev_resseq != int_resseq:
        _running_error(int_resseq + delta, resname)
      elif _is_match(int_resseq + delta, resname):
        running_state = 1
        prev_resseq = int_resseq
      else:
        continue

    if residue_id_list[i] in residue_constants.restype_name_to_atom14_names:
      res_atom14_list = residue_constants.restype_name_to_atom14_names[residue_id_list[i]]  # pylint: disable=line-too-long
    else:
      res_atom14_list = residue_constants.restype_name_to_atom14_names[residue_constants.unk_restype]  # pylint: disable=line-too-long
    try:
      atom14idx = res_atom14_list.index(atom_id)
      coord = np.asarray((x_list[i], y_list[i], z_list[i]))
      if np.any(np.isnan(coord)):
        continue
      labels[int_resseq - 1 + delta][atom14idx] = coord
      if np.any(coord != 0):
        # occupancy & B factor
        tempfactor = 0.0
        try:
          tempfactor = float(b_factor_list[i])
          if current_resseq == int_resseq:
            atom_plddt, atom_n = atom_plddt + tempfactor, atom_n + 1
        except ValueError as e:
          raise PDBConstructionException('Invalid or missing B factor') from e
        bfactors[int_resseq - 1 + delta][atom14idx] = tempfactor
        if datasource != 'swissprot' or tempfactor >= 85.0:
          label_mask[int_resseq - 1 + delta][atom14idx] = True
    except ValueError as e:
      logger.debug(e)
    if current_resseq != int_resseq:
      if atom_n > 0:
        residue_plddt = residue_plddt + atom_plddt/atom_n
        residue_n = residue_n + 1
      atom_plddt, atom_n = 0.0, 0
      current_resseq = int_resseq

  if datasource != 'swissprot' or (
      residue_n > 0 and residue_plddt/residue_n > 85.0):
    revision = mmcif_dict.get('_pdbx_audit_revision_history.revision_date')
    if revision:
      revision = min(revision)
    r = dict(coord=labels, coord_mask=label_mask, revision=revision)
    if add_plddt:
      r['bfactor'] = bfactors
    return r
  return None

def process(item, sequences=None, args=None):  # pylint: disable=redefined-outer-name
  cid, pid_list = item
  clu_list = []

  mmcif_fn, mmcif_dict = None, None
  for i, (pid, chain_id) in enumerate(
      map(lambda x: decompose_pid(x, args.datasource), sorted(pid_list))):
    mmcif_filename = os.path.join(
        args.mmcif_dir,
        mmcif_get_filename(pid, chain_id, datasource=args.datasource))
    if os.path.exists(mmcif_filename):
      fid = compose_pid(
          pid.lower() if args.datasource=='rcsb' else pid, chain_id)

      logger.info('(%s) (%d/%d) %s - %s',
          cid, i, len(pid_list), fid, mmcif_filename)
      try:
        if mmcif_fn != mmcif_filename:
          mmcif_fn, mmcif_dict = mmcif_filename, mmcif_parse(mmcif_filename)
        assert mmcif_dict is not None
        coords = mmcif_get_coords(
            mmcif_dict,
            chain_id,
            sequences[fid],
            datasource=args.datasource,
            add_plddt=args.add_plddt)
        if coords:
          clu_list.append(fid)

          with open(
              os.path.join(args.output, 'fasta', f'{fid}.fasta'),
              'w', encoding='utf-8') as fasta:
            print(f'>{fid}', file=fasta)
            print(f'{sequences[fid]}', file=fasta)
          np.savez(os.path.join(args.output, 'npz', fid), **coords)
      except PDBConstructionException as e:
        logger.warning('mmcif_parse: %s {%s}', mmcif_filename, str(e))
      except Exception as e:  # pylint: disable=broad-except
        logger.error('mmcif_parse: %s {%s}', mmcif_filename, str(e))
        # raise Exception('...') from e
  logger.info('result (%s) %d - %d', cid, len(pid_list), len(clu_list))
  return cid, pid_list, clu_list

def main(args):  # pylint: disable=redefined-outer-name
  logger.info('output - %s', args.output)
  os.makedirs(args.output, exist_ok=True)
  for d in ('fasta', 'npz'):
    os.makedirs(os.path.join(args.output, d), exist_ok=True)

  sequences = dict(parse_fasta(args.fasta_file, datasource=args.datasource))

  logger.info('sequences - %s', len(sequences))

  with open(os.path.join(args.output, 'name.idx'), 'w', encoding='utf-8') as f:
    succeed, total = 0, 0
    with mp.Pool(args.processes) as p:
      for cid, pid_list, clu_list in p.imap(
          functools.partial(process, sequences=sequences, args=args),
          parse_cluster(args.clustering_file)):
        del cid
        succeed += len(clu_list)
        total += len(pid_list)
        if clu_list:
          print('\t'.join(clu_list), file=f, flush=True)
    logger.info('succeed: %d, total: %d', succeed, total)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('-p', '--processes', type=int, default=None,
      help='number of worker processes to use, default=None')
  parser.add_argument('-c', '--clustering_file', type=str, default='bc-30.out',
      help='sequnce clusters data, default=\'bc-30.out\'')
  parser.add_argument('-s', '--fasta_file',
      type=str, default='derived_data/pdb_seqres.txt',
      help='fasta file, default=\'./derived_data/pdb_seqres.txt\'')
  parser.add_argument('-m', '--mmcif_dir', type=str, default='mmCIF',
      help='fasta file, default=\'mmCIF\'')
  parser.add_argument('-o', '--output', type=str, default='.',
      help='output dir, default=\'.\'')
  parser.add_argument('-t', '--datasource',
      choices=['swissprot', 'rcsb', 'norm'], default='norm',
      help='data source type: [swissprot, rcsb, norm], default=\'norm\'')
  parser.add_argument('--add_plddt', action='store_true', help='add plddt')
  parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
  args = parser.parse_args()

  logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

  main(args)
