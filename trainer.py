import os
import sys
import argparse
import logging
import resource

import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from profold2 import constants
from profold2.data import esm,scn,custom
from profold2.model import Alphafold2

def main(args):
    # constants
    DEVICE = constants.DEVICE # defaults to cuda if available, else cpu
    
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    
    # set emebdder model from esm if appropiate - Load ESM-1b model
    if args.features == "esm":
        esm_extractor = esm.ESMEmbeddingExtractor(*esm.ESM_MODEL_PATH)
    
    # helpers
    def cycle(loader, cond = lambda x: True):
        epoch = 0
        while True:
            logging.info('epoch: {}'.format(epoch))

            data_iter = iter(loader)
            for data in data_iter:
                if cond(data):
                    yield data

            epoch += 1
    
    # get data
    data = scn.load(
        casp_version = args.casp_version,
        thinning = args.casp_thinning,
        batch_size = args.batch_size,
        num_workers = 0,
        filter_by_resolution = args.filter_by_resolution if args.filter_by_resolution > 0 else False,
        dynamic_batching = False)
    
    train_loader = data['train']
    data_cond = lambda x: args.min_protein_len <= x['seq'].shape[1] and x['seq'].shape[1] < args.max_protein_len
    dl = cycle(train_loader, data_cond)

    # model
    if args.alphafold2_continue:
        model = torch.load(os.path.join(args.prefix, 'model.pkl'))
        mode.eval()
        model.to(DEVICE)
    else:
        # features
        feats = [('make_pseudo_beta', {}),
                 ('make_esm_embedd', dict(esm_extractor=esm_extractor, repr_layer=esm.ESM_EMBED_LAYER)),
                 ('make_to_device', dict(
                    fields=['seq', 'mask', 'coord', 'coord_mask', 'pseudo_beta', 'pseudo_beta_mask'],
                    device=DEVICE))
                ]

        headers = [('distogram', dict(buckets_first_break=2.3125, buckets_last_break=21.6875,
                            buckets_num=constants.DISTOGRAM_BUCKETS), dict(weight=0.01)),
                       ('folding', dict(structure_module_depth=4, structure_module_heads=4,
                            fape_min=args.alphafold2_fape_min, fape_max=args.alphafold2_fape_max, fape_z=args.alphafold2_fape_z), dict(weight=1.0)),
                       ('tmscore', {}, {})]

        logging.info('Alphafold2.feats: {}'.format(feats))
        logging.info('Alphafold2.headers: {}'.format(headers))

        model = Alphafold2(
            dim = args.alphafold2_dim,
            depth = args.alphafold2_depth,
            heads = 8,
            dim_head = 64,
            predict_angles = False,
            feats = feats,
            headers = headers
        ).to(DEVICE)

    # optimizer 
    optim = Adam(model.parameters(), lr = args.learning_rate)
    
    # tensorboard
    writer = SummaryWriter(os.path.join(args.prefix, 'runs', 'eval'))
    
    # training loop
    for it in range(args.num_batches):
        running_loss = {}
        for jt in range(args.gradient_accumulate_every):
            batch = next(dl)

            seq, mask = batch['seq'], batch['mask']
            logging.debug('seq.shape: {}'.format(seq.shape))
    
            # sequence embedding (msa / esm / attn / or nothing)
            r = model(batch = batch)
    
            if it == 0 and jt == 0 and args.tensorboard_add_graph:
                with SummaryWriter(os.path.join(args.prefix, 'runs', 'network')) as w:
                    w.add_graph(model, (batch,), verbose=True)
   
            # running loss
            running_loss['all'] = running_loss.get('all', 0) + r.loss.item()
            for h, v in r.headers.items():
                if 'loss' in v:
                    running_loss[h] = running_loss.get(h, 0) + v['loss'].item()

            r.loss.backward()
    
        for k, v in running_loss.items():
            v /= (args.batch_size*args.gradient_accumulate_every)
            logging.info('{} loss@{}: {}'.format(it, k, v))
            writer.add_scalar('Loss/train@{}'.format(k), v, it)
    
        optim.step()
        optim.zero_grad()

    writer.close()

    # save model
    torch.save(model, os.path.join(args.prefix, 'model.pkl'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--prefix', type=str, default='.', help='prefix of out directory, default=\'.\'')
    parser.add_argument('-C', '--casp_version', type=int, default=12, help='CASP version, default=12')
    parser.add_argument('-T', '--casp_thinning', type=int, default=30, help='CASP version, default=30')
    parser.add_argument('-F', '--features', type=str, default='esm', help='AA residue features one of [esm,msa], default=esm')
    parser.add_argument('-t', '--threads', type=int, default=0, help='number of threads used for intraop parallelism on CPU., default=0')
    parser.add_argument('-m', '--min_protein_len', type=int, default=50, help='filter out proteins whose length<LEN, default=50')
    parser.add_argument('-M', '--max_protein_len', type=int, default=1024, help='filter out proteins whose length>LEN, default=1024')
    parser.add_argument('-r', '--filter_by_resolution', type=float, default=0, help='filter by resolution<=RES')

    parser.add_argument('-n', '--num_batches', type=int, default=100000, help='number of batches, default=10^5')
    parser.add_argument('-k', '--gradient_accumulate_every', type=int, default=16, help='accumulate grads every k times, default=16')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch size, default=1')
    parser.add_argument('-l', '--learning_rate', type=float, default='3e-4', help='learning rate, default=3e-4')

    parser.add_argument('--alphafold2_dim', type=int, default=256, help='dimension of alphafold2, default=256')
    parser.add_argument('--alphafold2_depth', type=int, default=1, help='depth of alphafold2, default=1')
    parser.add_argument('--alphafold2_fape_min', type=float, default=1e-4, help='minimum of dij in alphafold2, default=1e-4')
    parser.add_argument('--alphafold2_fape_max', type=float, default=10.0, help='maximum of dij in alphafold2, default=10.0')
    parser.add_argument('--alphafold2_fape_z', type=float, default=10.0, help='Z of dij in alphafold2, default=10.0')
    parser.add_argument('--alphafold2_continue', action='store_true', help='load a model and continue to train')

    parser.add_argument('--tensorboard_add_graph', action='store_true', help='call tensorboard.add_graph')
    parser.add_argument('--save_pdb', action='store_true', help='save pdb')
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
