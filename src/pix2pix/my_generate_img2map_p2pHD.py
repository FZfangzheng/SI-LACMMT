import os
from os import path as osp
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append(osp.join(sys.path[0], '../'))
sys.path.append(osp.join(sys.path[0], '../../'))
# sys.path.append(osp.join(sys.path[0], '../../../'))
import time
import torch
import torch.nn as nn
from src.utils.train_utils import model_accelerate, get_device, mean, get_lr
from src.pix2pixHD.train_config import config
from src.pix2pixHD.networks import get_G, get_D, get_E
from torch.optim import Adam
from src.pix2pixHD.hinge_lr_scheduler import get_hinge_scheduler
from src.utils.logger import ModelSaver, Logger
from src.datasets import get_pix2pix_maps_dataloader
from src.pix2pixHD.utils import get_edges, label_to_one_hot, get_encode_features
from src.utils.visualizer import Visualizer
from tqdm import tqdm
from torchvision import transforms
from src.pix2pixHD.criterion import get_GANLoss, get_VGGLoss, get_DFLoss, get_low_level_loss
from tensorboardX import SummaryWriter
from src.pix2pixHD.utils import from_std_tensor_save_image, create_dir
from src.data.image_folder import make_dataset
import shutil
import numpy as np
from PIL import Image

from src.pix2pixHD.deeplabv3plus.deeplabv3plus import Configuration
from src.pix2pixHD.deeplabv3plus.deeplabv3plus.my_deeplabv3plus_featuremap import deeplabv3plus
import torch.nn.functional as F

from src.pix2pixHD.deeplabv3plus.lovasz_losses import lovasz_softmax

from util.util import tensor2im  # 注意，该函数使用0.5与255恢复可视图像，所以如果是ImageNet标准化的可能会有色差？这里都显示试一下
from src.pix2pixHD.myutils import pred2gray, gray2rgb

from src.pix2pixHD.deeplabv3plus.focal_loss import FocalLoss
from evaluation.fid.fid_score import fid_score
import json

def eval_fid(args, model_G, data_loader):
    floder_name=args.dataroot.split(os.sep)[-2]

    device = get_device(args)
    data_loader = tqdm(data_loader)
    model_G.eval()
    model_G = model_G.to(device)

    label_preds = []
    label_targets = []

    real_dir = osp.join(args.save, floder_name, 'real_result')
    A_dir = osp.join(args.save, floder_name, 'real_source')
    fake_dir=osp.join(args.save, floder_name, 'fake_result')
    create_dir(real_dir)
    create_dir(A_dir)
    create_dir(fake_dir)

    for i, sample in enumerate(data_loader):
        imgs = sample['A'].to(device)
        maps = sample['B'].to(device)

        fakes = model_G(imgs).detach()

        batch_size = imgs.size(0)
        im_name = sample['A_paths']
        for b in range(batch_size):
            file_name = osp.split(im_name[b])[0].split(os.sep)[-1]+osp.split(im_name[b])[-1].split('.')[0]
            real_file = osp.join(real_dir, f'{file_name}.tif')
            A_file = osp.join(A_dir, f'{file_name}.tif')
            fake_file=osp.join(fake_dir, f'{file_name}.tif')

            from_std_tensor_save_image(filename=real_file, data=sample['B'][b].cpu())
            from_std_tensor_save_image(filename=A_file, data=sample['A'][b].cpu())
            from_std_tensor_save_image(filename=fake_file, data=fakes[b].cpu())

    fid = fid_score(real_path=real_dir, fake_path=fake_dir, gpu=str(args.gpu))
    print(f'===> fid score:{fid:.4f}')
    model_G.train()
    return fid

def label_nums(data_loader,label_num=5): # 遍历dataloader，计算其所有图像中分割GT各label的pix总数
    ret=[]
    for i in range(label_num):
        ret.append(0)
    for step, sample in enumerate(data_loader):
        seg=sample["seg"]
        for i in range(label_num):
            ret[i]+=(seg==i).sum().item()
    return ret




def train(args, get_dataloader_func=get_pix2pix_maps_dataloader):
    logger = Logger(save_path=args.save, json_name='img2map')
    epoch_now = len(logger.get_data('G_loss'))

    model_saver = ModelSaver(save_path=args.save,
                             name_list=['G', 'D', 'G_optimizer', 'D_optimizer',
                                        'G_scheduler', 'D_scheduler', 'DLV3P', "DLV3P_global_optimizer",
                                        "DLV3P_backbone_optimizer", "DLV3P_global_scheduler",
                                        "DLV3P_backbone_scheduler",
                                        'best_G', 'best_D', 'best_G_optimizer', 'best_D_optimizer',
                                        'best_G_scheduler', 'best_D_scheduler', 'best_DLV3P',
                                        "best_DLV3P_global_optimizer",
                                        "best_DLV3P_backbone_optimizer", "best_DLV3P_global_scheduler",
                                        "best_DLV3P_backbone_scheduler"
                                        ])
    visualizer = Visualizer(keys=['image', 'encode_feature', 'fake', 'label', 'instance'])
    sw = SummaryWriter(args.tensorboard_path)
    G = get_G(args)
    D = get_D(args)
    model_saver.load('G', G)
    model_saver.load('D', D)

    G_optimizer = Adam(G.parameters(), lr=args.G_lr, betas=(args.beta1, 0.999))
    D_optimizer = Adam(D.parameters(), lr=args.D_lr, betas=(args.beta1, 0.999))

    model_saver.load('G_optimizer', G_optimizer)
    model_saver.load('D_optimizer', D_optimizer)

    G_scheduler = get_hinge_scheduler(args, G_optimizer)
    D_scheduler = get_hinge_scheduler(args, D_optimizer)

    model_saver.load('G_scheduler', G_scheduler)
    model_saver.load('D_scheduler', D_scheduler)

    device = get_device(args)

    GANLoss = get_GANLoss(args)

    if args.use_ganFeat_loss:
        DFLoss = get_DFLoss(args)
    if args.use_vgg_loss:
        VGGLoss = get_VGGLoss(args)
    if args.use_low_level_loss:
        LLLoss = get_low_level_loss(args)


    if epoch_now==args.epochs:
        print('get final models')
        fid = eval_fid(args,model_G=G, data_loader=get_pix2pix_maps_dataloader(args, train=False))
        logger.log(key='fid', data=fid)
        # if iou < logger.get_max(key='FID'):
        #     model_saver.save(f'DLV3P_{iou:.4f}', DLV3P)
        sw.add_scalar('eval/fid', fid, epoch_now)
    else:
        print(f'haven\'t get final models! please check! now model is {epoch_now} epochs, need {args.epochs}.')

if __name__ == '__main__':
    args = config()

    # args.label_nc = 5

    from src.pix2pixHD.myutils import seed_torch

    print(f'\nset seed as {args.seed}!\n')
    seed_torch(args.seed)

    train(args, get_dataloader_func=get_pix2pix_maps_dataloader)

pass
