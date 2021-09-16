import os
import random
import math
import pickle

import numpy as np
import torch
from torch.nn import functional as F

from profold2.common import residue_constants
from profold2.data.parsers import parse_a3m,parse_fasta

NUM_COORDS_PER_RES=14

class ProteinStructureDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, msa_max_size=128, max_seq_len=None):
        super().__init__()

        self.data_dir = data_dir
        self.max_msa = msa_max_size
        self.max_seq_len = max_seq_len
        with open(os.path.join(self.data_dir, "pdb_names")) as f:
            self.pids = list(map(str.rstrip, f))

    def __getitem__(self, idx):
        pid = self.pids[idx]

        seq_feats = self.get_seq_features(pid)
        coord_feats = self.get_structure_label(pid, seq_feats['str_seq'])
        return dict(pid=pid, **seq_feats, **coord_feats)

    def __len__(self):
        return len(self.pids)

    def get_msa_features(self, protein_id):
        """Constructs a feature dict of MSA features."""
        msa_path = os.path.join(self.data_dir, f'a3ms/{protein_id}.a3m')
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

    def get_structure_label(self, protein_id, str_seq):
        input_structure_path = os.path.join(self.data_dir, f"pdbs/{protein_id}.pkl")
        with open(input_structure_path, 'rb') as fin:
            structure = pickle.load(fin)

        labels = torch.zeros(len(str_seq), NUM_COORDS_PER_RES, 3)
        label_mask = torch.zeros(len(str_seq), NUM_COORDS_PER_RES)
        for i, (res_coords, res_letter) in enumerate(zip(structure[:self.max_seq_len], str_seq)):
            res_name = residue_constants.restype_1to3.get(res_letter, residue_constants.unk_restype)
            for atom_name, coords in res_coords.items():
                try:
                    atom14idx = residue_constants.restype_name_to_atom14_names[res_name].index(atom_name)
                    labels[i][atom14idx] = torch.from_numpy(coords)
                    label_mask[i][atom14idx] = 1
                except ValueError: pass

        return dict(coord=labels, coord_mask=label_mask)

    def get_seq_features(self, protein_id):
        """Runs alignment tools on the input sequence and creates features."""
        input_fasta_path = os.path.join(self.data_dir, f"fastas/{protein_id}.fasta")
        with open(input_fasta_path) as f:
            input_fasta_str = f.read()
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
            map_unknown_to_x=True)).argmax(-1)
        residue_index = torch.arange(len(input_sequence), dtype=torch.float)
        str_seq = input_sequence
        mask = torch.ones(len(input_sequence))

        return dict(seq=seq,
                residue_index=residue_index,
                str_seq=str_seq,
                mask=mask)

    @staticmethod
    def collate_fn(batch):
        fields = ('pid', 'seq', 'mask', 'str_seq', 'coord', 'coord_mask')
        pids, seqs, masks, str_seqs, coords, coord_masks = list(zip(*[[b[k] for k in fields] for b in batch]))
        lengths = tuple(len(s) for s in str_seqs)
        max_batch_len = max(lengths)

        padded_seqs = pad_for_batch(seqs, max_batch_len, 'seq')
        padded_masks = pad_for_batch(masks, max_batch_len, 'msk')
        padded_coords = pad_for_batch(coords, max_batch_len, 'crd')
        padded_coord_masks = pad_for_batch(coord_masks, max_batch_len, 'crd_msk')

        return dict(pid=pids,
                seq=padded_seqs,
                mask=padded_masks,
                str_seq=str_seqs,
                coord=padded_coords,
                coord_mask=padded_coord_masks)

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
            z = torch.ones(batch_length - seq.shape[0]) * residue_constants.unk_restype_index
            c = torch.cat((seq, z), dim=0)
            batch.append(c)
    elif dtype == 'msk':
        # Mask sequences (1 if present, 0 if absent) are padded with 0s
        for msk in items:
            z = torch.zeros(batch_length - msk.shape[0])
            c = torch.cat((msk, z), dim=0)
            batch.append(c)
    elif dtype == "crd":
        for item in items:
            z = torch.zeros((batch_length - item.shape[0],  NUM_COORDS_PER_RES, item.shape[-1]))
            c = torch.cat((item, z), dim=0)
            batch.append(c)
    elif dtype == "crd_msk":
        for item in items:
            z = torch.zeros((batch_length - item.shape[0],  NUM_COORDS_PER_RES))
            c = torch.cat((item, z), dim=0)
            batch.append(c)
    else:
        raise ValueError('Not implement yet!')
    batch = torch.stack(batch, dim=0)
    return batch

def load(data_dir, msa_max_size=128, max_seq_len=None, **kwargs):
    dataset = ProteinStructureDataset(data_dir, msa_max_size, max_seq_len)
    if not 'collate_fn' in kwargs:
        kwargs['collate_fn'] = ProteinStructureDataset.collate_fn
    return torch.utils.data.DataLoader(dataset, **kwargs)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    PATH_DIR = sys.argv[1]
    db = ProteinStructureDataset(PATH_DIR)
    dataloader = DataLoader(db, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn)
