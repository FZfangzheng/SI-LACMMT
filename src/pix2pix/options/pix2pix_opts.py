import argparse
import os

def get_default_pix2pix_args():
    parser ={}
    # model arguments
    parser['model']='pix2pix'
    parser['gpu_ids'] = '-1'
    parser['input_nc'] = 3
    parser['output_nc'] = 3
    parser['ngf'] = 64
    parser['ndf'] = 64
    parser['which_model_netD'] = 'basic'
    parser['which_model_netG'] = 'resnet_9blocks'
    parser['n_layers_D'] = 3
    parser['norm'] = 'instance'
    parser['init_type'] = 'xavier'
    parser['niter'] = 100
    parser['niter_decay'] = 100
    parser['phase'] = 'train'
    parser['beta1'] = 0.5
    parser['lr'] = 0.0002
    parser['no_lsgan'] = False
    parser['lambda_AB'] = 10.0
    parser['lambda_A'] = 10.0
    parser['lambda_B'] = 10.0
    parser['lr_policy'] = 'lambda'
    parser['geometry'] = 'rot'
    parser['lr_decay_iters'] = 50
    parser['identity'] = 0.5
    parser['lambda_gc'] = 2.0
    parser['lambda_G'] = 1.0


    parser['isTrain'] = True if parser['phase'] == 'train' else False
    parser['batchSize'] = 4
    parser['fineSize'] = 256
    parser['no_dropout'] = False
    parser['from_pretrain_model'] = False
    parser['pool_size'] = 50

    args_parser = argparse.Namespace(**parser)
    return args_parser

def get_pix2pix_args(main_args=None):
    pix2pix_def_args=get_default_pix2pix_args()

    pix2pix_def_args.gpu_ids=[]
    for i in range(main_args.gpu):
        pix2pix_def_args.gpu_ids.append(int(i))
    pix2pix_def_args.batchSize=main_args.batchSize
    pix2pix_def_args.fineSize =main_args.fineSize


    return pix2pix_def_args