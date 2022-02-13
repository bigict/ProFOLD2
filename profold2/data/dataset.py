import contextlib
import functools
from io import BytesIO
import logging
import math
import pathlib
import random
import zipfile

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import WeightedRandomSampler

from profold2.common import protein, residue_constants
from profold2.data.parsers import parse_a3m, parse_fasta
from profold2.data.utils import batch_data_crop
from profold2.model.features import FeatureBuilder

logger = logging.getLogger(__file__)

NUM_COORDS_PER_RES = 14

class ProteinStructureDataset(torch.utils.data.Dataset):
    FEAT_PDB = 0x01
    FEAT_MSA = 0x02
    FEAT_ALL = 0xff

    def __init__(self, data_dir, msa_max_size=128, max_seq_len=None, coord_type=None, feat_flags=FEAT_ALL):
        super().__init__()

        self.data_dir = pathlib.Path(data_dir)
        if zipfile.is_zipfile(self.data_dir):
            self.data_dir = zipfile.ZipFile(self.data_dir)
        self.max_msa = msa_max_size
        self.max_seq_len = max_seq_len
        self.feat_flags = feat_flags
        with self._fileobj('name.idx') as f:
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
        pids = self.pids[idx]
        pid = pids[np.random.randint(len(pids))]

        pkey = self.mapping[pid] if pid in self.mapping else pid
        seq_feats = self.get_seq_features(pkey)

        ret = dict(pid=pid, resolu=self.get_resolution(pid), **seq_feats)
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
        if isinstance(self.data_dir, zipfile.ZipFile):
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
        """Constructs a feature dict of MSA features."""
        msa_path = self.data_dir / f'a3ms/{protein_id}.a3m'
        with open(msa_path) as f:
            text = f.read()
        msa, del_matirx = parse_a3m(text)

    def get_msa_features(self, protein_id):
        """Constructs a feature dict of MSA features."""
        msa_path = self.data_dir / f'a3ms/{protein_id}.a3m'
        with open(msa_path) as f:
            text = f.read()
        msa, del_matirx = parse_a3m(text)
        msas = (msa,)
        if not msas:
            raise ValueError('At least one MSA must be provided.')
        deletion_matrices = (del_matirx,)

        int_msa = []
        deletion_matrix = []
        seen_sequences = set()
        for msa_index, msa in enumerate(msas):
            if not msa:
                raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
            for sequence_index, sequence in enumerate(msa):
                if sequence in seen_sequences:
                    continue
                seen_sequences.add(sequence)
                int_msa.append(
                    [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
                deletion_matrix.append(deletion_matrices[msa_index][sequence_index])

        features = {}
        if self.max_msa == 0:
            features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
            features['msa'] = np.array(int_msa, dtype=np.int32)
        else:
            features['deletion_matrix_int'] = np.array(random.choices(deletion_matrix, k=self.max_msa), dtype=np.int32)
            features['msa'] = np.array(random.choices(int_msa, k=self.max_msa), dtype=np.int32)

        msa = torch.tensor(features['msa'], dtype=torch.long)[:, :self.max_seq_len]  # (N, L) 22 possible values.
        msa_one_hot = F.one_hot(msa, 23).float()  # (N, L, 23) <- (N, L)  extra dim for mask flag.
        deletion_matrix_int = torch.tensor(features['deletion_matrix_int'], dtype=torch.float)[:, :self.max_seq_len]  # (N, L)
        cluster_deletion_value = 2 / math.pi * torch.arctan(deletion_matrix_int / 3)  # (N, L)
        cluster_deletion_mean = deletion_matrix_int.mean(axis=0, keepdim=True)  # (1, L) <- (N, L)
        cluster_deletion_mean = cluster_deletion_mean.expand(deletion_matrix_int.shape)  # (N, L) <- (1, L)
        cluster_deletion_mean = 2 / math.pi * torch.arctan(cluster_deletion_mean / 3)  # (N, L)
        cluster_has_deletion = (deletion_matrix_int != 0).float()  # (N, L)
        deletion_features = torch.stack((cluster_deletion_value, cluster_deletion_mean, cluster_has_deletion), dim=2)  # (N, L, 3) <- ...
        result = torch.cat((msa_one_hot, deletion_features), dim=2)  # (N, L, 23 + 3) <- ...
        return result

    def get_structure_label_npz(self, protein_id, str_seq):
        with self._fileobj(f'{self.PDB}/{protein_id}.npz') as f:
            structure = np.load(BytesIO(f.read()))
            return dict(coord=torch.from_numpy(structure['coord']), coord_mask=torch.from_numpy(structure['coord_mask']))
        
    def get_structure_label_numpy(self, protein_id, str_seq):
        labels = np.zeros((len(str_seq), NUM_COORDS_PER_RES, 3), dtype=np.float32)
        label_mask = np.zeros((len(str_seq), NUM_COORDS_PER_RES), dtype=np.bool)

        #input_structure_path = self.data_dir / f"{self.PDB}/{protein_id}.npz"
        with self._fileobj(f'{self.PDB}/{protein_id}.npz') as f:
            structure = np.load(BytesIO(f.read()))

            for atom_name, coords in structure.items():
                for i, res_letter in enumerate(str_seq):
                    res_name = residue_constants.restype_1to3.get(res_letter, residue_constants.unk_restype)
                    res_atom14_list = residue_constants.restype_name_to_atom14_names[res_name]
                    try:
                        atom14idx = residue_constants.restype_name_to_atom14_names[res_name].index(atom_name)
                        if np.any(np.isnan(coords[i])):
                            continue
                        labels[i][atom14idx] = coords[i]
                        if np.any(coords[i]):
                            label_mask[i][atom14idx] = 1
                    except ValueError: pass

        return dict(coord=torch.from_numpy(labels), coord_mask=torch.from_numpy(label_mask))

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

        input_sequence = input_sequence[:self.max_seq_len]

        seq = torch.tensor(residue_constants.sequence_to_onehot(
            sequence=input_sequence,
            mapping=residue_constants.restype_order_with_x,
            map_unknown_to_x=True), dtype=torch.int).argmax(-1)
        #residue_index = torch.arange(len(input_sequence), dtype=torch.int)
        str_seq = ''.join(map(lambda a: a if a in residue_constants.restype_order_with_x else residue_constants.restypes_with_x[-1], input_sequence))
        mask = torch.ones(len(input_sequence), dtype=torch.bool)

        return dict(seq=seq,
                str_seq=str_seq,
                mask=mask)

    def collate_fn(self, batch, max_seq_len=None, feat_builder=None):
        fields = ('pid', 'resolu', 'seq', 'mask', 'str_seq')
        pids, resolutions, seqs, masks, str_seqs = list(zip(*[[b[k] for k in fields] for b in batch]))
        lengths = tuple(len(s) for s in str_seqs)
        max_batch_len = max(lengths)

        padded_seqs = pad_for_batch(seqs, max_batch_len, 'seq')
        padded_masks = pad_for_batch(masks, max_batch_len, 'msk')

        ret = dict(pid=pids,
                resolution=resolutions,
                seq=padded_seqs,
                mask=padded_masks,
                str_seq=str_seqs)

        if self.feat_flags & ProteinStructureDataset.FEAT_PDB:
            fields = ('coord', 'coord_mask')
            coords, coord_masks = list(zip(*[[b[k] for k in fields] for b in batch]))

            padded_coords = pad_for_batch(coords, max_batch_len, 'crd')
            padded_coord_masks = pad_for_batch(coord_masks, max_batch_len, 'crd_msk')
            ret.update(
                coord=padded_coords,
                coord_mask=padded_coord_masks)

        ret = batch_data_crop(ret, max_seq_len=max_seq_len)
        if feat_builder:
            ret = feat_builder.build(ret)

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
    elif dtype == 'msk':
        # Mask sequences (1 if present, 0 if absent) are padded with 0s
        for msk in items:
            z = torch.zeros(batch_length - msk.shape[0], dtype=msk.dtype)
            c = torch.cat((msk, z), dim=0)
            batch.append(c)
    elif dtype == "crd":
        for item in items:
            z = torch.zeros((batch_length - item.shape[0],  item.shape[-2], item.shape[-1]), dtype=item.dtype)
            c = torch.cat((item, z), dim=0)
            batch.append(c)
    elif dtype == "crd_msk":
        for item in items:
            z = torch.zeros((batch_length - item.shape[0],  item.shape[-1]), dtype=item.dtype)
            c = torch.cat((item, z), dim=0)
            batch.append(c)
    else:
        raise ValueError('Not implement yet!')
    batch = torch.stack(batch, dim=0)
    return batch

def load(data_dir, msa_max_size=128, max_seq_len=None, feats=None, is_training=True, feat_flags=ProteinStructureDataset.FEAT_ALL, **kwargs):
    dataset = ProteinStructureDataset(data_dir, msa_max_size, feat_flags=feat_flags)
    if not 'collate_fn' in kwargs:
        kwargs['collate_fn'] = functools.partial(dataset.collate_fn, max_seq_len=max_seq_len,
                feat_builder=FeatureBuilder(feats, is_training=is_training))
    if 'weights' in kwargs:
        weights = kwargs.pop('weights')
        if weights:
            kwargs['sampler'] = WeightedRandomSampler(weights, num_samples=kwargs.get('batch_size'))
    return torch.utils.data.DataLoader(dataset, **kwargs)
