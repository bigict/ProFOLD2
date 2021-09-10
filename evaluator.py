import os
import sys
import argparse
import logging
import resource

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange

# models & data

from profold2 import constants
from profold2.data import esm,scn
from profold2.model import Alphafold2,sidechain
from profold2.utils import *

def main(args):
    # constants
    
    DEVICE = constants.DEVICE # defaults to cuda if available, else cpu
    
    DISTOGRAM_BUCKETS = constants.DISTOGRAM_BUCKETS

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    
    # get data
    data = scn.load(
        casp_version = args.casp_version,
        thinning = 30,
        batch_size = args.batch_size,
        num_workers = 0,
        dynamic_batching = False)
    
    test_loader = data['train']

    # model
    model = torch.load(os.path.join(args.model))
    model.eval()
    model.to(DEVICE)

    # eval loop
    for i, batch in enumerate(iter(test_loader)):
        logging.debug('seq.shape: {}'.format(batch['seq'].shape))
    
        # predict - out is (batch, L * 3, 3)
        r = model(batch=batch)
        print(r.headers)
        sys.exit(0)
    
        # atom mask
        #_, atom_mask, _ = scn_backbone_mask(seq, boolean=True)
        atom_mask = torch.zeros(scn.NUM_COORDS_PER_RES).to(seq.device)
        atom_mask[..., 1] = 1
        cloud_mask = scn.cloud_mask(seq, boolean = True, coords=coords)
        flat_cloud_mask = rearrange(cloud_mask, 'b l c -> b (l c)')
    
        ## build SC container. set SC points to CA and optionally place carbonyl O
        proto_sidechain = sidechain.fold(seq, backbones=backbones, atom_mask=atom_mask,
                                              cloud_mask=cloud_mask, num_coords_per_res=scn.NUM_COORDS_PER_RES)
    
        proto_sidechain = rearrange(proto_sidechain, 'b l c d -> b (l c) d')
    
        # rotate / align
        coords_aligned, labels_aligned = Kabsch(proto_sidechain[flat_cloud_mask], coords[flat_cloud_mask])
        print('coords_aligned', coords_aligned.shape)
        print('labels_aligned', labels_aligned.shape)
    
    
        # chain_mask is all atoms that will be backpropped thru -> existing + trainable 
    
        #chain_mask = (mask * cloud_mask)[cloud_mask]
        #flat_chain_mask = rearrange(chain_mask, 'b l c -> b (l c)')
    
        logging.info('{} TM-score: {}'.format(i, TMscore(rearrange(coords_aligned, 'l d -> () d l'), rearrange(labels_aligned, 'l d -> () d l'))))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--prefix', type=str, default='.', help='prefix of out directory, default=\'.\'')
    parser.add_argument('-C', '--casp-version', type=int, default=12, help='CASP version, default=12')
    parser.add_argument('-F', '--features', type=str, default='esm', help='AA residue features one of [esm,msa], default=esm')
    parser.add_argument('-t', '--threads', type=int, default=0, help='number of threads used for intraop parallelism on CPU., default=0')
    parser.add_argument('-m', '--min_protein_len', type=int, default=50, help='filter out proteins whose length<LEN, default=50')
    parser.add_argument('-M', '--max_protein_len', type=int, default=1024, help='filter out proteins whose length>LEN, default=1024')

    parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch size, default=1')

    parser.add_argument('--model', type=str, default='model.pkl', help='model of alphafold2, default=\'model.pkl\'')

    parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
    args = parser.parse_args()

    # logging

    if not os.path.exists(args.prefix):
        os.makedirs(os.path.abspath(args.prefix))
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
