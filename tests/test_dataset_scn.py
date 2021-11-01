import unittest

import os

import numpy as np
import torch
from profold2.common import protein, residue_constants
from profold2.data import scn

def _to_pdb_str(seq, mask, coords, coord_mask):
    N = seq.shape[0]
    print(N, seq.shape)
    features = dict(aatype=seq.numpy(), residue_index=np.arange(N))
    result = dict(structure_module=dict(
        final_atom_positions = coords.numpy(),
        final_atom_mask=coord_mask.numpy()))
    prot = protein.from_prediction(features=features, result=result)
    return protein.to_pdb(prot)

class TestDataSet(unittest.TestCase):
    def test_scn_dataset(self):
        scn_seq = torch.tensor([[0, 1]])
        mask = scn.cloud_mask(scn_seq, boolean=True)
        result = torch.tensor(
            [[[True, True, True, True, True, False, False, False, False, False, False, False, False, False], \
              [True, True, True, True, True, True, False, False, False, False, False, False, False, False]]])
        self.assertTrue(torch.equal(mask, result))

    def test_scn_loader(self):
        data = scn.load(casp_version=12,
                thinning=30,
                batch_size=1,
                max_seq_len=256,
                dynamic_batching=False)
        os.makedirs('tmp', exist_ok=True)
        for batch in iter(data['train']):
            for x, pid in enumerate(batch['pid']):
                if x in batch['clips'] and pid in ['1O59_1_A', '1LQT_1_A']:
                    print(f'{pid}\t{batch["clips"][x]}\t{batch["str_seq"][x]}')
                    with open(f'tmp/{pid}_{batch["clips"][x]["i"]}_cropped.pdb', 'w') as f:
                        data = _to_pdb_str(batch['seq'][x], batch['mask'][x], batch['coord'][x], batch['coord_mask'][x])
                        f.write(data)
        self.assertTrue(True)

    def test_scn_loader2(self):
        data = scn.load(casp_version=12,
                thinning=30,
                batch_size=1,
                max_seq_len=None,
                dynamic_batching=False)
        os.makedirs('tmp', exist_ok=True)
        for batch in iter(data['train']):
            for x, pid in enumerate(batch['pid']):
                if len(batch['str_seq'][x]) > 256 and pid in ['1O59_1_A', '1LQT_1_A']:
                    print(f'{pid}\t{len(batch["str_seq"][x])}\t{batch["str_seq"][x]}')
                    with open(f'tmp/{pid}_all.pdb', 'w') as f:
                        data = _to_pdb_str(batch['seq'][x], batch['mask'][x], batch['coord'][x], batch['coord_mask'][x])
                        f.write(data)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
