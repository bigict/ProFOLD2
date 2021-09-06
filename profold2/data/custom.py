import os
import random
import math
import pickle

import numpy as np
import torch
from torch.nn import functional as F

from profold2.common import residue_constants
from profold2.data.parsers import parse_a3m,parse_fasta


class ProteinStructureDataset(torch.utils.data.Dataset):

    atom_map = {'N': 0, 'CA': 1, 'C': 2}

    def __init__(self, work_dir, msa_max_size=128, max_seq_length=500) -> None:
        super().__init__()
        self.dir = work_dir
        self.max_msa = msa_max_size
        self.max_seq_len = max_seq_length
        with open(os.path.join(self.dir, "pdb_names")) as fin:
            self.protein_ids = list(map(str.rstrip, fin.readlines())) 

    def __getitem__(self, idx):
        pid = self.protein_ids[idx]
        return dict(pid=pid, 
                **self.get_seq_features(pid), 
                **self.get_msa_features(pid),
                **self.get_structure_label(pid))

    def __len__(self):
        return len(self.protein_ids)

    def get_msa_features(self, protein_id):
        """Constructs a feature dict of MSA features."""
        msa_path = os.path.join(self.dir, f'a3ms/{protein_id}.a3m')
        with open(msa_path) as fin:
            string = fin.read()
        msa, del_matirx = parse_a3m(string)
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

    def get_structure_label(self, protein_id):
        input_structure_path = os.path.join(self.dir, f"pdbs/{protein_id}.pkl")
        with open(input_structure_path, 'rb') as fin:
            structure = pickle.load(fin)
        
        seq_len = len(structure)
        result = torch.empty(seq_len, 3, 3)
        for (i, residule) in enumerate(structure):
            is_empty = True
            for atom_name, coords in residule.items():
                is_empty = False
                if atom_name in self.atom_map:
                    result[i][self.atom_map[atom_name]] = torch.tensor(coords)
            if is_empty:
                result[i].fill_(float('nan'))

        result = result[:self.max_seq_len]

        v1 = result[:, 2:3, :] - result[:, 1:2, :]  # (L, 1, 3)
        v2 = result[:, 0:1, :] - result[:, 1:2, :]  # (L, 1, 3)
        e1 = v1 / torch.linalg.norm(v1, ord=2, dim=-1, keepdim=True)  # (L, 1, 3)  <<- (L, 1, 1)
        u2 = v2 - e1 * torch.bmm(e1, v2.transpose(1, 2))  # (L, 1, 3) <<- (L, 1, 3), (L, 1, 1)
        e2 = u2 / torch.linalg.norm(u2, ord=2, dim=-1, keepdim=True)  # (L, 1, 3)  <<- (L, 1, 1)
        e3 = torch.cross(e1, e2, dim=2)  # (L, 1, 3)
        result2 = torch.cat((e1, e2, e3), dim=1)  # (L, 3, 3) <- ...
        transition = result[:, 1, :]  # (L, 3)
        return {'coord': result, 'rotation': result2, 'transition': transition}

    def get_seq_features(self, protein_id):
        """Runs alignment tools on the input sequence and creates features."""
        input_fasta_path = os.path.join(self.dir, f"fastas/{protein_id}.fasta")
        with open(input_fasta_path) as f:
            input_fasta_str = f.read()
        input_seqs, input_descs = parse_fasta(input_fasta_str)
        if len(input_seqs) != 1:
            raise ValueError(
                f'More than one input sequence found in {input_fasta_path}.')
        input_sequence = input_seqs[0]
        input_description = input_descs[0]
        num_res = len(input_sequence)
        features = {}
        features['aatype'] = torch.tensor(residue_constants.sequence_to_onehot(
            sequence=input_sequence,
            mapping=residue_constants.restype_order_with_x,
            map_unknown_to_x=True)).float()[:self.max_seq_len]
        # features['between_segment_residues'] = np.zeros((num_res,), dtype=np.int32)
        # features['domain_name'] = np.array([input_description.encode('utf-8')], dtype=np.object_)
        features['residue_index'] = torch.tensor(range(num_res)).float()[:self.max_seq_len]
        # features['seq_length'] = np.array([num_res] * num_res, dtype=np.int32)
        # features['sequence'] = np.array([input_sequence.encode('utf-8')], dtype=np.object_)
        return features

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)
        lengths = [ele[0]['aatype'].size(0) for ele in batch]
        aatype_emb_size = batch[0][0]['aatype'].size(1)
        max_len = max(lengths)
        batch[0][1]  # (N, L, 26)
        max_msa_number = max([ele[1].size(0) for ele in batch])
        aatype = torch.full((max_len, batch_size, aatype_emb_size), float('nan'), dtype=torch.float)
        # (ML, B, E)
        residue_index = torch.full((max_len, batch_size), float('nan'), dtype=torch.float)  # (ML, B)
        msa_feature = torch.full((max_msa_number, max_len, batch_size, 26), float('nan'), dtype=torch.float)
        # (MN, ML, B, 26)
        coord = torch.full((max_len, 3, batch_size, 3), float('nan'), dtype=torch.float)  # (L, 3, B, 3)
        rotation = torch.full((max_len, batch_size, 3, 3), float('nan'), dtype=torch.float)  # (L, B, 3, 3)
        transition = torch.full((max_len, batch_size, 3), float('nan'), dtype=torch.float)  # (L, B, 3)
        for i, (seq_features, single_msa_feature, label) in enumerate(batch):
            single_aatype = seq_features['aatype']
            single_residue_index = seq_features['residue_index']
            single_length = single_aatype.size(0)
            single_coord = label['coord']
            single_rotation = label['rotation']
            single_transition = label['transition']
    
            aatype[:single_length, i, :] = single_aatype
            residue_index[:single_length, i] = single_residue_index
            msa_feature[:single_msa_feature.size(0), :single_length, i, :] = single_msa_feature
            coord[:single_length, :, i, :] = single_coord
            rotation[:single_length, i, :, :] = single_rotation
            transition[:single_length, i, :] = single_transition
        features = {'residue_index': residue_index, 'target_feat': aatype, 'msa_feat': msa_feature}
        label = {'coord': coord, 'rotation': rotation, 'transition': transition}
        return features, label

def load(work_dir, msa_max_size=128, max_seq_length=500, **kwargs):
    dataset = ProteinStructureDataset(work_dir, msa_max_size, max_seq_length)
    if not 'collate_fn' in kwargs:
        kwargs['collate_fn'] = ProteinStructureDataset.collate_fn
    return torch.utils.data.DataLoader(dataset, **kwargs)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    PATH_DIR = sys.argv[1]
    db = ProteinStructureDataset(PATH_DIR)
    dataloader = DataLoader(db, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn)
