import os
import contextlib
import functools
import logging
import math
from io import BytesIO
import pathlib
import random
import re
import string
import zipfile

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from profold2.common import protein, residue_constants
from profold2.data.parsers import parse_a3m, parse_fasta
from profold2.data.utils import domain_parser
from profold2.utils import default, exists, timing

logger = logging.getLogger(__file__)

def _make_msa_features(sequences, max_msa_depth=None):
    """Constructs a feature dict of MSA features."""
    def parse_a4m(sequences):
        deletion_matrix = []
        for msa_sequence in sequences:
            deletion_vec = []
            deletion_count = 0
            for j in msa_sequence:
                if j.islower():
                    deletion_count += 1
                else:
                    deletion_vec.append(deletion_count)
                    deletion_count = 0
            deletion_matrix.append(deletion_vec)
        # Make the MSA matrix out of aligned (deletion-free) sequences
        deletion_table = str.maketrans('', '', string.ascii_lowercase)
        aligned_sequences = [s.translate(deletion_table) for s in sequences]
        return aligned_sequences, deletion_matrix

    msa_depth = len(sequences)
    if exists(max_msa_depth) and len(sequences) > max_msa_depth:
        sequences = sequences[:1] + list(np.random.choice(
                sequences, size=max_msa_depth - 1, replace=False) if max_msa_depth > 1 else [])
    msa, del_matirx = parse_a4m(sequences)

    int_msa = []
    for sequence in msa:
        int_msa.append([residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE[residue_constants.HHBLITS_AA_TO_ID[res]] for res in sequence])

    return dict(msa=torch.as_tensor(int_msa), str_msa=msa, del_msa=torch.as_tensor(del_matirx), num_msa=msa_depth)

def _parse_seq_index(description, input_sequence, seq_index):
    # description: pid field1 field2 ...
    seq_index_pattern = '(\d+)-(\d+)'
    
    def seq_index_split(text):
        for s in text.split(','):
          r = re.match(seq_index_pattern, s)
          assert r
          yield tuple(map(int, r.group(1, 2)))
    def seq_index_check(positions):
        for i in range(len(positions) - 1):
            p, q = positions[i]
            m, n = positions[i + 1]
            assert p < q and m < n
            assert q < m
        m, n = positions[-1]
        assert m <= n
        assert sum(map(lambda p: p[1] - p[0] + 1, positions)) == len(input_sequence)

    fields = description.split()
    pid = fields[0]
    for f in fields[1:]:
        r = re.match(f'.*:({seq_index_pattern}(,{seq_index_pattern})*)', f)
        if r:
            positions = list(seq_index_split(r.group(1)))
            seq_index_check(positions)
            p, q = positions[0]
            start, gap = p, 0
            for m, n in positions[1:]:
                gap += m - q - 1
                seq_index[m - start - gap: n - start - gap + 1] = torch.arange(m - start, n - start + 1)
                p, q = m, n
            logger.debug('_parse_seq_index: desc=%s, positions=%s', description, positions)
            break

    return seq_index

class ProteinSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, descriptions=None, msa=None):
        self.sequences = sequences
        self.descriptions = descriptions
        self.msa = msa
        assert not exists(self.descriptions) or len(self.sequences) == len(self.descriptions)
        assert not exists(self.msa) or len(self.sequences) == len(self.msa)

    def __getitem__(self, idx):
        input_sequence = self.sequences[idx]
        seq = torch.tensor(residue_constants.sequence_to_onehot(
            sequence=input_sequence,
            mapping=residue_constants.restype_order_with_x,
            map_unknown_to_x=True), dtype=torch.int).argmax(-1)
        residue_index = torch.arange(len(input_sequence), dtype=torch.int)
        str_seq = ''.join(map(lambda a: a if a in residue_constants.restype_order_with_x else residue_constants.restypes_with_x[-1], input_sequence))
        mask = torch.ones(len(input_sequence), dtype=torch.bool)
        if exists(self.descriptions) and exists(self.descriptions[idx]):
            desc = self.descriptions[idx]
            residue_index = _parse_seq_index(desc, input_sequence, residue_index)
        else:
            desc = str(idx)
        ret = dict(pid=desc,
                seq=seq,
                seq_index=residue_index,
                str_seq=str_seq,
                mask=mask)
        if exists(self.msa) and exists(self.msa[idx]):
            ret.update(_make_msa_features(self.msa[idx]))
        return ret

    def __len__(self):
        return len(self.sequences)

    def collate_fn(self, batch):
        fields = ('pid', 'seq', 'seq_index', 'mask', 'str_seq')
        pids, seqs, seqs_idx, masks, str_seqs = list(zip(*[[b[k] for k in fields] for b in batch]))
        lengths = tuple(len(s) for s in str_seqs)
        max_batch_len = max(lengths)

        padded_seqs = pad_for_batch(seqs, max_batch_len, 'seq')
        padded_seqs_idx = pad_for_batch(seqs_idx, max_batch_len, 'seq_index')
        padded_masks = pad_for_batch(masks, max_batch_len, 'msk')

        ret = dict(pid=pids,
                seq=padded_seqs,
                seq_index=padded_seqs_idx,
                mask=padded_masks,
                str_seq=str_seqs)

        fields = ('msa', 'str_msa', 'del_msa', 'num_msa')
        if all(all(field in b for field in fields) for b in batch):
            msas, str_msas, del_msas, num_msa = list(zip(*[[b[k] for k in fields] for b in batch]))

            padded_msas = pad_for_batch(msas, max_batch_len, 'msa')
            ret.update(
                msa=padded_msas,
                str_msa=str_msas,
                del_msa=del_msas,
                num_msa=num_msa)

        return ret

class ProteinStructureDataset(torch.utils.data.Dataset):
    FEAT_PDB = 0x01
    FEAT_MSA = 0x02
    FEAT_ALL = 0xff

    def __init__(self, data_dir, data_idx='name.idx', max_msa_size=128, max_seq_len=None, coord_type='npz', feat_flags=FEAT_ALL&(~FEAT_MSA)):
        super().__init__()

        self.data_dir = pathlib.Path(data_dir)
        if zipfile.is_zipfile(self.data_dir):
            self.data_dir = zipfile.ZipFile(self.data_dir)
        self.max_msa_size = max_msa_size
        self.max_seq_len = max_seq_len
        self.feat_flags = feat_flags
        logger.info('load idx data from: %s', data_idx)
        with self._fileobj(data_idx) as f:
            self.pids = list(map(lambda x: self._ftext(x).strip().split(), f))

        self.mapping = {}
        if self._fstat('mapping.idx'):
            with self._fileobj('mapping.idx') as f:
                for line in filter(lambda x: len(x)>0, map(lambda x: self._ftext(x).strip(), f)):
                    v, k = line.split()
                    self.mapping[k] = v

        self.resolu = {}
        if self._fstat('resolu.idx'):
            with self._fileobj('resolu.idx') as f:
                for line in filter(lambda x: len(x)>0, map(lambda x: self._ftext(x).strip(), f)):
                    k, v = line.split()
                    self.resolu[k] = float(v)

        self.FASTA = 'fasta'

        self.PDB = coord_type
        for t in ('npz', 'coord'):
            if self.PDB is None and self._fstat(f'{t}/'):
                self.PDB = t
                break
        logger.info('load structure data from: %s', self.PDB)
        assert not (feat_flags & ProteinStructureDataset.FEAT_PDB) or (self.PDB is not None)

        self.MSA_LIST = ['BFD30_E-3']

    def __getstate__(self):
        logger.debug('being pickled ...')
        d = self.__dict__
        if isinstance(self.data_dir, zipfile.ZipFile):
            d['data_dir'] = self.data_dir.filename
        return d

    def __setstate__(self, d):
        logger.debug('being unpickled ...')
        if zipfile.is_zipfile(d['data_dir']):
            d['data_dir'] = zipfile.ZipFile(d['data_dir'])
        self.__dict__ = d

    def __getitem__(self, idx):
        with timing(f'ProteinStructureDataset.__getitem__ {idx}', logger.debug):
            pids = self.pids[idx]
            pid = pids[np.random.randint(len(pids))]

            pkey = self.mapping[pid] if pid in self.mapping else pid
            seq_feats = self.get_seq_features(pkey)

            ret = dict(pid=pid, resolu=self.get_resolution(pid), **seq_feats)
            if self.feat_flags & ProteinStructureDataset.FEAT_MSA:
                ret.update(self.get_msa_features_new(pkey))
            if self.feat_flags & ProteinStructureDataset.FEAT_PDB:
                assert self.PDB in ('npz', 'coord')
                if self.PDB == 'npz':
                    ret.update(self.get_structure_label_npz(pid, seq_feats['str_seq']))
                else:
                    ret.update(self.get_structure_label_numpy(pid, seq_feats['str_seq']))
            if 'coord_mask' in ret:
                ret['mask'] = torch.sum(ret['coord_mask'], dim=-1) > 0
        return ret

    def __len__(self):
        return len(self.pids)

    @contextlib.contextmanager
    def _fileobj(self, filename):
        if os.path.isabs(filename):
            with open(filename, mode='rb') as f:
                yield f
        elif isinstance(self.data_dir, zipfile.ZipFile):
            with self.data_dir.open(filename, mode='r') as f:
                yield f
        else:
            with open(self.data_dir / filename, mode='rb') as f:
                yield f

    def _fstat(self, filename):
        if isinstance(self.data_dir, zipfile.ZipFile):
            try:
                self.data_dir.getinfo(filename)
                return True
            except KeyError as e:
                return False
        return (self.data_dir / filename).exists()

    def _ftext(self, line, encoding='utf-8'):
        if isinstance(line, bytes):
            return line.decode(encoding)
        return line

    def get_resolution(self, protein_id):
        pid = protein_id.split('_')
        return self.resolu.get(pid[0], None)

    def get_msa_features_new(self, protein_id):
        k = np.random.randint(len(self.MSA_LIST))
        source = self.MSA_LIST[k]
        with self._fileobj(f'msa/{protein_id}/{source}/{protein_id}.a4m') as f:
            sequences = list(map(lambda x: self._ftext(x).strip(), f))
        return _make_msa_features(sequences, max_msa_depth=self.max_msa_size)

    def get_structure_label_npz(self, protein_id, str_seq):
        if self._fstat(f'{self.PDB}/{protein_id}.npz'):
            with self._fileobj(f'{self.PDB}/{protein_id}.npz') as f:
                structure = np.load(BytesIO(f.read()))
                ret = dict(
                        coord=torch.from_numpy(structure['coord']),
                        coord_mask=torch.from_numpy(structure['coord_mask']))
                if 'bfactor' in structure:
                    ret.update(coord_plddt=torch.from_numpy(structure['bfactor']))
                return ret
        return dict()

    def get_seq_features(self, protein_id):
        """Runs alignment tools on the input sequence and creates features."""
        with self._fileobj(f'{self.FASTA}/{protein_id}.fasta') as f:
            input_fasta_str = self._ftext(f.read())
        input_seqs, input_descs = parse_fasta(input_fasta_str)
        if len(input_seqs) != 1:
            raise ValueError(
                f'More than one input sequence found in {input_fasta_path}.')
        input_sequence = input_seqs[0]
        input_description = input_descs[0]

        residue_index = torch.arange(len(input_sequence), dtype=torch.int)
        residue_index = _parse_seq_index(input_description, input_sequence, residue_index)

        input_sequence = input_sequence[:self.max_seq_len]
        residue_index = residue_index[:self.max_seq_len]

        seq = torch.tensor(residue_constants.sequence_to_onehot(
            sequence=input_sequence,
            mapping=residue_constants.restype_order_with_x,
            map_unknown_to_x=True), dtype=torch.int).argmax(-1)
        #residue_index = torch.arange(len(input_sequence), dtype=torch.int)
        str_seq = ''.join(map(lambda a: a if a in residue_constants.restype_order_with_x else residue_constants.restypes_with_x[-1], input_sequence))
        mask = torch.ones(len(input_sequence), dtype=torch.bool)

        return dict(seq=seq,
                seq_index=residue_index,
                str_seq=str_seq,
                mask=mask)

    def batch_clips_fn(self, batch, min_crop_len=None, max_crop_len=None, crop_probability=0.0, crop_algorithm='random', **kwargs):
        def _random_sampler(b, n, batch):
            def _crop_length(n, crop):
                assert exists(min_crop_len) or exists(max_crop_len)

                if not exists(max_crop_len):
                    assert min_crop_len < n
                    return np.random.randint(min_crop_len, n+1) if crop else n
                elif not exists(min_crop_len):
                    assert max_crop_len < n
                    return max_crop_len
                assert min_crop_len <= max_crop_len and (min_crop_len < n or max_crop_len < n)
                return np.random.randint(min_crop_len, min(n, max_crop_len)+1) if crop else min(max_crop_len, n)

            l = _crop_length(n, np.random.random() < crop_probability)
            logger.debug('min_crop_len=%s, max_crop_len=%s, n=%s, l=%s', min_crop_len, max_crop_len, n, l)
            i, j = 0, l
            if not 'coord_mask' in batch[b] or torch.any(batch[b]['coord_mask']):
                while True:
                    i = np.random.randint(n - l + 1)
                    j = i + l
                    if not 'coord_mask' in batch[b] or torch.any(batch[b]['coord_mask'][i:j]):
                        break
            return dict(i=i, j=j, l=n)

        def _domain_sampler(b, n, batch):
            def _cascade_sampler(weights, width=4):
                if len(weights) <= width:
                    i = torch.multinomial(weights, 1)
                    return i.item()

                p = torch.zeros((width,))
                l, k = len(weights) // width, len(weights) % width
                for i in range(width):
                    v, w = l*i + min(i, k), l*(i+1) + min(i+1, k)
                    p[i] = torch.amax(weights[v:w])
                i = _cascade_sampler(p, width=width)
                v, w = l*i + min(i, k), l*(i+1) + min(i+1, k)
                return v + _cascade_sampler(weights[v:w], width=width)

            def _domain_next(weights, i, ca_coord, ca_mask, min_len=None, max_len=None):
                min_len, max_len = default(min_len, 80), default(max_len, 255)

                direction = np.random.randint(2)
                for _ in range(2):
                    #if direction % 2 == 0 and torch.sum(ca_mask[i:]) >= min_len:
                    if direction % 2 == 0 and i + min_len < n:
                        j = min(len(weights), i + max_len)
                        return j if i + min_len >= j else i + min_len + _cascade_sampler(weights[i+min_len:j])
                    #if direction % 2 == 1 and torch.sum(ca_mask[:i]) >= min_len:
                    if direction % 2 == 1 and i > min_len:
                        j = max(0, i - max_len)
                        return j + _cascade_sampler(weights[j:i-min_len]) if i - min_len > j else j
                    direction += 1
                return None

            assert exists(min_crop_len) or exists(max_crop_len)
            assert 'coord' in batch[b] and 'coord_mask' in batch[b]

            if exists(max_crop_len) and n <= max_crop_len and crop_probability < np.random.random():
                assert not exists(min_crop_len) or min_crop_len < n
                return None

            ca_idx = residue_constants.atom_order['CA']
            ca_coord, ca_coord_mask = batch[b]['coord'][...,ca_idx,:], batch[b]['coord_mask'][...,ca_idx]
            logger.debug('domain_sampler: batch=%d, seq_len=%d', b, n)
            weights = domain_parser(ca_coord, ca_coord_mask, max_len=max_crop_len)
            intra_domain_probability = kwargs.get('intra_domain_probability', 0)
            while True:
                i = _cascade_sampler(weights)
                if np.random.random() < intra_domain_probability:
                    logger.debug('domain_intra: batch=%d, seq_len=%d, i=%d', b, n, i)
                    half = max_crop_len // 2 + max_crop_len % 2
                    if i + 1 < n - i:  # i <= n // 2
                        i = max(0, i - half)
                        j = min(i + max_crop_len, n)
                    else:
                        i = min(n, i + half)
                        j = max(0, i - max_crop_len)
                else:
                    j = _domain_next(weights, i, ca_coord, ca_coord_mask, min_len=min_crop_len, max_len=max_crop_len)
                    logger.debug('domain_next: batch=%d, seq_len=%d, i=%d, j=%s', b, n, i, str(j))
                if j is not None and torch.any(ca_coord_mask[min(i, j): max(i, j)]):
                    break
            return dict(i=min(i, j), j=max(i, j), l=n)

        logger.debug('batch_clips_fn: crop_algorithm=%s', crop_algorithm)
        sampler_list = dict(random=_random_sampler, domain=_domain_sampler)

        assert crop_algorithm in sampler_list

        clips = {}

        for k in range(len(batch)):
            n = len(batch[k]['str_seq'])
            if (exists(max_crop_len) and max_crop_len < n) or (exists(min_crop_len) and min_crop_len < n and crop_probability > 0):
                sampler_fn = sampler_list[crop_algorithm]
                if crop_algorithm == 'domain' and ('coord' not in batch[k] or 'coord_mask' not in batch[k]):
                    sampler_fn = sampler_list['random']
                    logger.debug('batch_clips_fn: crop_algorithm=%s downgrad to: random', crop_algorithm)
                clip = sampler_fn(k, n, batch)
                if clip:
                    clips[k] = clip

        return clips

    def collate_fn(self, batch, min_crop_len=None, max_crop_len=None, crop_probability=0.0, crop_algorithm='random', **kwargs):
        if exists(max_crop_len) and exists(min_crop_len):
            assert max_crop_len >= min_crop_len 

        clips = self.batch_clips_fn(batch,
                min_crop_len=min_crop_len, max_crop_len=max_crop_len, crop_probability=crop_probability, crop_algorithm=crop_algorithm, **kwargs)
        for k, clip in clips.items():
            i, j = clip['i'], clip['j']

            batch[k]['str_seq'] = batch[k]['str_seq'][i:j]
            for field in ('seq', 'seq_index', 'mask', 'coord', 'coord_mask', 'coord_plddt'):
                if field in batch[k]:
                    batch[k][field] = batch[k][field][i:j,...]
            for field in ('str_msa',):
                if field in batch[k]:
                    batch[k][field] = [v[i:j] for v in batch[k][field]]
            for field in ('msa', 'del_msa'):
                if field in batch[k]:
                    batch[k][field] = batch[k][field][:,i:j,...]

        fields = ('pid', 'resolu', 'seq', 'seq_index', 'mask', 'str_seq')
        pids, resolutions, seqs, seqs_idx, masks, str_seqs = list(zip(*[[b[k] for k in fields] for b in batch]))
        lengths = tuple(len(s) for s in str_seqs)
        max_batch_len = max(lengths)

        padded_seqs = pad_for_batch(seqs, max_batch_len, 'seq')
        padded_seqs_idx = pad_for_batch(seqs_idx, max_batch_len, 'seq_index')
        padded_masks = pad_for_batch(masks, max_batch_len, 'msk')

        ret = dict(pid=pids,
                resolution=resolutions,
                seq=padded_seqs,
                seq_index=padded_seqs_idx,
                mask=padded_masks,
                str_seq=str_seqs)

        if self.feat_flags & ProteinStructureDataset.FEAT_PDB and 'coord' in batch[0]:
            # required
            fields = ('coord', 'coord_mask')
            coords, coord_masks = list(zip(*[[b[k] for k in fields] for b in batch]))

            padded_coords = pad_for_batch(coords, max_batch_len, 'crd')
            padded_coord_masks = pad_for_batch(coord_masks, max_batch_len, 'crd_msk')
            ret.update(
                coord=padded_coords,
                coord_mask=padded_coord_masks)
            # optional
            fields = ('coord_plddt',)
            for field in fields:
                if not field in batch[0]:
                    continue
                padded_values = pad_for_batch([b[field] for b in batch], max_batch_len, field)
                ret[field] = padded_values

        if self.feat_flags & ProteinStructureDataset.FEAT_MSA:
            fields = ('msa', 'str_msa', 'del_msa', 'num_msa')
            msas, str_msas, del_msas, num_msa = list(zip(*[[b[k] for k in fields] for b in batch]))

            padded_msas = pad_for_batch(msas, max_batch_len, 'msa')
            padded_dels = pad_for_batch(del_msas, max_batch_len, 'del_msa')
            ret.update(
                msa=padded_msas,
                str_msa=str_msas,
                del_msa=padded_dels,
                num_msa=num_msa)

        if clips:
            ret['clips'] = clips

        return ret

def pad_for_batch(items, batch_length, dtype):
    """Pad a list of items to batch_length using values dependent on the item type.

    Args:
        items: List of items to pad (i.e. sequences or masks represented as arrays of
            numbers, angles, coordinates, pssms).
        batch_length: The integer maximum length of any of the items in the input. All
            items are padded so that their length matches this number.
        dtype: A string ('seq', 'msk', 'crd') reperesenting the type of
            data included in items.

    Returns:
         A padded list of the input items, all independently converted to Torch tensors.
    """
    batch = []
    if dtype == 'seq':
        for seq in items:
            z = torch.ones(batch_length - seq.shape[0], dtype=seq.dtype) * residue_constants.unk_restype_index
            c = torch.cat((seq, z), dim=0)
            batch.append(c)
    elif dtype == 'seq_index':
        for idx in items:
            z = torch.zeros(batch_length - idx.shape[0], dtype=idx.dtype)
            c = torch.cat((idx, z), dim=0)
            batch.append(c)
    elif dtype == 'msk':
        # Mask sequences (1 if present, 0 if absent) are padded with 0s
        for msk in items:
            z = torch.zeros(batch_length - msk.shape[0], dtype=msk.dtype)
            c = torch.cat((msk, z), dim=0)
            batch.append(c)
    elif dtype == 'crd':
        for item in items:
            z = torch.zeros((batch_length - item.shape[0],  item.shape[-2], item.shape[-1]), dtype=item.dtype)
            c = torch.cat((item, z), dim=0)
            batch.append(c)
    elif dtype == 'crd_msk' or dtype == 'coord_plddt':
        for item in items:
            z = torch.zeros((batch_length - item.shape[0],  item.shape[-1]), dtype=item.dtype)
            c = torch.cat((item, z), dim=0)
            batch.append(c)
    elif dtype == 'msa':
        for msa in items:
            z = torch.ones((msa.shape[0], batch_length - msa.shape[1]), dtype=msa.dtype) * residue_constants.HHBLITS_AA_TO_ID['X']
            c = torch.cat((msa, z), dim=1)
            batch.append(c)
    elif dtype == 'del_msa':
        for del_msa in items:
            z = torch.zeros((del_msa.shape[0], batch_length - del_msa.shape[1]), dtype=del_msa.dtype)
            c = torch.cat((del_msa, z), dim=1)
            batch.append(c)
    else:
        raise ValueError('Not implement yet!')
    batch = torch.stack(batch, dim=0)
    return batch

def load(data_dir,
        data_idx='name.idx',
        min_crop_len=None,
        max_crop_len=None,
        crop_probability=0,
        crop_algorithm='random',
        feat_flags=ProteinStructureDataset.FEAT_ALL,
        **kwargs):
    max_msa_size = 128
    if 'max_msa_size' in kwargs:
        max_msa_size = kwargs.pop('max_msa_size')
    dataset = ProteinStructureDataset(data_dir, data_idx=data_idx, max_msa_size=max_msa_size, feat_flags=feat_flags)
    if not 'collate_fn' in kwargs:
        collate_fn_kwargs = {}
        if 'intra_domain_probability' in kwargs:
            collate_fn_kwargs['intra_domain_probability'] = kwargs.pop('intra_domain_probability')
        kwargs['collate_fn'] = functools.partial(
                dataset.collate_fn,
                min_crop_len=min_crop_len,
                max_crop_len=max_crop_len,
                crop_probability=crop_probability,
                crop_algorithm=crop_algorithm,
                **collate_fn_kwargs)
    if 'weights' in kwargs:
        weights = kwargs.pop('weights')
        if weights:
            kwargs['sampler'] = WeightedRandomSampler(weights, num_samples=len(weights))
            if 'shuffle' in kwargs:
                kwargs.pop('shuffle')
    elif 'num_replicas' in kwargs and 'rank' in kwargs:
        num_replicas, rank = kwargs.pop('num_replicas'), kwargs.pop('rank')
        kwargs['sampler'] = DistributedSampler(dataset, num_replicas=num_replicas, rank=rank)
    return torch.utils.data.DataLoader(dataset, **kwargs)
