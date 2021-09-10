import os
import sys
import argparse
import logging
import resource

import torch
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange

# models & data
from profold2 import constants
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
    
    # get data
    data = scn.load(
        casp_version = args.casp_version,
        thinning = args.casp_thinning,
        batch_size = args.batch_size,
        num_workers = 0,
        filter_by_resolution = args.filter_by_resolution if args.filter_by_resolution > 0 else False,
        dynamic_batching = False)
    
    test_loader = data[args.casp_data]
    data_cond = lambda x: args.min_protein_len <= x['seq'].shape[1] and x['seq'].shape[1] < args.max_protein_len

    # model
    model = torch.load(os.path.join(args.model))
    model.eval()
    model.to(DEVICE)

    tmscore, n = 0, 0
    # eval loop
    for i, batch in enumerate(filter(data_cond, iter(test_loader))):
        if i >= args.num_batches:
            break

        logging.debug('seq.shape: {}'.format(batch['seq'].shape))
    
        # predict - out is (batch, L * 3, 3)
        r = model(batch=batch)
    
        if 'folding' in r.headers:
            assert 'coords' in r.headers['folding'] and 'coord' in batch
            flat_cloud_mask = rearrange(batch['coord_mask'], 'b l c -> b (l c)')

            # rotate / align
            coords_aligned, labels_aligned = Kabsch(
                    rearrange(r.headers['folding']['coords'], 'b l c d -> b (l c) d')[flat_cloud_mask], 
                    rearrange(batch['coord'], 'b l c d -> b (l c) d')[flat_cloud_mask])
            logging.debug('coords_aligned: {}'.format(coords_aligned.shape))
            logging.debug('labels_aligned: {}'.format(labels_aligned.shape))
    
            tms = TMscore(rearrange(coords_aligned, 'l d -> () d l'), rearrange(labels_aligned, 'l d -> () d l'))
            logging.info('{} TM-score: {}'.format(i, tms.item()))

            tmscore, n = tmscore + tms.item(), n + 1
        else:
            raise ValueError('{} are not implemented yet!'.format(','.join(r.headers.keys())))

    if n > 0:
        logging.info('{} TM-score: {} (average)'.format(n, tmscore/n))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-X', '--model', type=str, default='model.pkl', help='model of alphafold2, default=\'model.pkl\'')
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
