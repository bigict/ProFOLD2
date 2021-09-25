import os
import sys
import argparse
import logging
import resource

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange

# models & data
from profold2 import constants
from profold2.common import protein,residue_constants
from profold2.data import esm,scn,custom
from profold2.utils import Kabsch,TMscore

def main(args):
    # constants
    DEVICE = constants.DEVICE # defaults to cuda if available, else cpu
    
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    # set emebdder model from esm if appropiate - Load ESM-1b model
    if args.features == "esm":
        esm_extractor = esm.ESMEmbeddingExtractor(*esm.ESM_MODEL_PATH)
    
    if args.casp_version > 12:
        test_loader = custom.load(
            data_dir = args.casp_data,
            feat_flags = ~custom.ProteinStructureDataset.FEAT_PDB,
            batch_size = args.batch_size,
            num_workers = 0)
    else:
        # get data
        data = scn.load(
            casp_version = args.casp_version,
            thinning = args.casp_thinning,
            batch_size = args.batch_size,
            num_workers = 0,
            filter_by_resolution = args.filter_by_resolution if args.filter_by_resolution > 0 else False,
            dynamic_batching = False)
        
        print(data.keys())
        test_loader = data[args.casp_data]
    data_cond = lambda x: args.min_protein_len <= x['seq'].shape[1] and x['seq'].shape[1] < args.max_protein_len

    # model
    model = torch.load(os.path.join(args.model))
    model.eval()
    model.to(DEVICE)

    tmscore, n = 0, 0
    # eval loop
    for i, batch in enumerate(filter(data_cond, iter(test_loader))):
        if args.num_batches <= 0:
            break
        if args.pid and (not set(args.pid) & set(batch['pid'])):
            continue

        args.num_batches -= 1

        logging.debug('seq.shape: {}'.format(batch['seq'].shape))
    
        with torch.no_grad():
            # predict - out is (batch, L * 3, 3)
            r = model(batch=batch, num_recycle=args.alphafold2_recycles)
    
            if 'folding' in r.headers:
                assert 'coords' in r.headers['folding']
                if 'coord' in batch:
                    flat_cloud_mask = rearrange(batch['coord_mask'], 'b l c -> b (l c)')

                    # rotate / align
                    coords_aligned, labels_aligned = Kabsch(
                            rearrange(rearrange(r.headers['folding']['coords'], 'b l c d -> b (l c) d')[flat_cloud_mask], 'c d -> d c'), 
                            rearrange(rearrange(batch['coord'], 'b l c d -> b (l c) d')[flat_cloud_mask], 'c d -> d c'))
                    logging.debug('coords_aligned: {}'.format(coords_aligned.shape))
                    logging.debug('labels_aligned: {}'.format(labels_aligned.shape))
    
                    tms = TMscore(rearrange(coords_aligned, 'd l -> () d l'), rearrange(labels_aligned, 'd l -> () d l'),
                            L=torch.sum(batch['mask'], dim=-1))
                    logging.info('{} pid: {} TM-score: {}'.format(i, ','.join(batch['pid']), tms.item()))

                    tmscore, n = tmscore + tms.item(), n + 1

                if args.save_pdb:
                    if not os.path.exists(os.path.join(args.prefix, 'pdbs')):
                        os.makedirs(os.path.join(args.prefix, 'pdbs'))
                    b, N = batch['seq'].shape

                    for x, pid in enumerate(batch['pid']):
                        str_seq = batch['str_seq'][x]
                        #aatype = batch['seq'][x,...].numpy()
                        aatype = np.array([residue_constants.restype_order_with_x.get(aa, residue_constants.unk_restype_index) for aa in str_seq])
                        features = dict(aatype=aatype, 
                                residue_index=np.arange(N))

                        p = os.path.join(args.prefix, 'pdbs', '{}_{}_{}.pdb'.format(pid, i, x))
                        with open(p, 'w') as f:
                            #if 'coord_mask' in batch:
                            #    coord_mask = batch['coord_mask'][x,...].numpy()
                            #else:
                            coord_mask = np.asarray([residue_constants.restype_atom14_mask[restype] for restype in aatype])

                            result = dict(structure_module=dict(
                                final_atom_mask = coord_mask,
                                final_atom_positions = r.headers['folding']['coords'][x,...].numpy()))
                            prot = protein.from_prediction(features=features, result=result)
                            f.write(protein.to_pdb(prot))
                            logging.info('{}/{} PDB save: {}'.format(i, x, pid))

                        if not 'coord' in batch:
                            continue

                        p = os.path.join(args.prefix, 'pdbs', '{}_{}_{}_gt.pdb'.format(pid, i, x))
                        with open(p, 'w') as f:
                            coord_mask = batch['coord_mask'][x,...].numpy()
                            result = dict(structure_module=dict(
                                final_atom_mask = coord_mask,
                                final_atom_positions = batch['coord'][x,...].numpy()))
                            prot = protein.from_prediction(features=features, result=result)
                            f.write(protein.to_pdb(prot))
                            logging.info('{}/{} PDB save: {} gt'.format(i, x, pid))

            else:
                raise ValueError('{} are not implemented yet!'.format(','.join(r.headers.keys())))

    if n > 0:
        logging.info('{} TM-score: {} (average)'.format(n, tmscore/n))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-X', '--model', type=str, default='model.pth', help='model of alphafold2, default=\'model.pth\'')
    parser.add_argument('--pid', type=str, action='append', help='pids to eval, default=\'\'')
    parser.add_argument('-o', '--prefix', type=str, default='.', help='prefix of out directory, default=\'.\'')
    parser.add_argument('-C', '--casp_version', type=int, default=12, help='CASP version, default=12')
    parser.add_argument('-T', '--casp_thinning', type=int, default=30, help='CASP version, default=30')
    parser.add_argument('-k', '--casp_data', type=str, default='test', help='CASP dataset, default=\'test\'')
    parser.add_argument('-F', '--features', type=str, default='esm', help='AA residue features one of [esm,msa], default=esm')
    parser.add_argument('-t', '--threads', type=int, default=0, help='number of threads used for intraop parallelism on CPU., default=0')
    parser.add_argument('-m', '--min_protein_len', type=int, default=0, help='filter out proteins whose length<LEN, default=0')
    parser.add_argument('-M', '--max_protein_len', type=int, default=1024, help='filter out proteins whose length>LEN, default=1024')
    parser.add_argument('-r', '--filter_by_resolution', type=float, default=0, help='filter by resolution<=RES')
    parser.add_argument('-n', '--num_batches', type=int, default=100000, help='number of batches, default=10^5')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch size, default=1')

    parser.add_argument('--alphafold2_recycles', type=int, default=0, help='number of recycles in alphafold2, default=0')

    parser.add_argument('--save_pdb', action='store_true', help='save pdb files')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
    args = parser.parse_args()

    # logging

    if not os.path.exists(args.prefix):
        os.makedirs(os.path.abspath(args.prefix))
        if args.save_pdb and not os.path.exists(os.path.join(args.prefix, 'pdbs')):
            os.makedirs(os.path.abspath(os.path.join(args.prefix, 'pdbs')))
    logging.basicConfig(
            format = '%(asctime)-15s [%(levelname)s] (%(filename)s:%(lineno)d) %(message)s',
            level = logging.DEBUG if args.verbose else logging.INFO,
            handlers = [
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(args.prefix, '{}.log'.format(
                    os.path.splitext(os.path.basename(__file__))[0])))
            ]
        )

    logging.info('-----------------')
    logging.info('Arguments: {}'.format(args))
    logging.info('-----------------')

    main(args)

    logging.info('-----------------')
    logging.info('Resources(myself): {}'.format(resource.getrusage(resource.RUSAGE_SELF)))
    logging.info('Resources(children): {}'.format(resource.getrusage(resource.RUSAGE_CHILDREN)))
    logging.info('-----------------')
