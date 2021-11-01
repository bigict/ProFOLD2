import logging
import os

import numpy as np
import torch

from profold2.common import protein, residue_constants

def embedding_get_labels(name, mat):
    if name == 'token':
        return [residue_constants.restypes_with_x[i if i < len(residue_constants.restypes_with_x) else -1]
                for i in range(mat.shape[0])]
    return None

def pdb_save(step, batch, headers, prefix='/tmp'):
    b, N = batch['seq'].shape

    for x, pid in enumerate(batch['pid']):
        str_seq = batch['str_seq'][x]
        #aatype = batch['seq'][x,...].numpy()
        aatype = np.array([residue_constants.restype_order_with_x.get(aa, residue_constants.unk_restype_index) for aa in str_seq])
        features = dict(aatype=aatype, 
                residue_index=np.arange(N))

        p = os.path.join(prefix, '{}_{}_{}.pdb'.format(pid, step, x))
        with open(p, 'w') as f:
            if 'mask' in batch:
                masked_seq_len = torch.sum(batch['mask'][x,...], dim=-1)
            else:
                masked_seq_len = len(str_seq)
            coords = headers['folding']['coords'].detach().cpu()  # (b l c d)
            _, _, num_atoms, _ = coords.shape
            coord_mask = np.asarray([residue_constants.restype_atom14_mask[restype][:num_atoms] for restype in aatype])

            result = dict(structure_module=dict(
                final_atom_mask = coord_mask,
                final_atom_positions = coords[x,...].numpy()))
            prot = protein.from_prediction(features=features, result=result)
            f.write(protein.to_pdb(prot))
            logging.debug('step: {}/{} length: {}/{} PDB save: {}'.format(step, x, masked_seq_len, len(str_seq), pid))

            if 'coord' in batch:
                p = os.path.join(prefix, '{}_{}_{}_gt.pdb'.format(pid, step, x))
                with open(p, 'w') as f:
                    coord_mask = batch['coord_mask'].detach().cpu()
                    coords = batch['coord'].detach().cpu()
                    result = dict(structure_module=dict(
                        final_atom_mask = coord_mask[x,...].numpy(),
                        final_atom_positions = coords[x,...].numpy()))
                    prot = protein.from_prediction(features=features, result=result)
                    f.write(protein.to_pdb(prot))
                    logging.debug('step: {}/{} length: {}/{} PDB save: {} (groundtruth)'.format(step, x, masked_seq_len, len(str_seq), pid))
