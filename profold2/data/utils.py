import logging
import os

import numpy as np

from profold2.common import protein,residue_constants

def save_pdb(step, batch, headers, prefix='/tmp'):
    b, N = batch['seq'].shape

    for x, pid in enumerate(batch['pid']):
        str_seq = batch['str_seq'][x]
        #aatype = batch['seq'][x,...].numpy()
        aatype = np.array([residue_constants.restype_order_with_x.get(aa, residue_constants.unk_restype_index) for aa in str_seq])
        features = dict(aatype=aatype, 
                residue_index=np.arange(N))

        p = os.path.join(prefix, '{}_{}_{}.pdb'.format(pid, step, x))
        with open(p, 'w') as f:
            coord_mask = np.asarray([residue_constants.restype_atom14_mask[restype] for restype in aatype])
            coords = headers['folding']['coords'].detach()

            result = dict(structure_module=dict(
                final_atom_mask = coord_mask,
                final_atom_positions = coords[x,...].numpy()))
            prot = protein.from_prediction(features=features, result=result)
            f.write(protein.to_pdb(prot))
            logging.debug('{}/{} PDB save: {}'.format(step, x, pid))

            if 'coord' in batch:
                p = os.path.join(prefix, '{}_{}_{}_gt.pdb'.format(pid, step, x))
                with open(p, 'w') as f:
                    coord_mask = batch['coord_mask'].detach()
                    coords = batch['coord'].detach()
                    result = dict(structure_module=dict(
                        final_atom_mask = coord_mask[x,...].numpy(),
                        final_atom_positions = coords[x,...].numpy()))
                    prot = protein.from_prediction(features=features, result=result)
                    f.write(protein.to_pdb(prot))
                    logging.debug('{}/{} PDB save: {} (groundtruth)'.format(step, x, pid))
