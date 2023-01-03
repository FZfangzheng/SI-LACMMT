__author__ = "charles"
__email__ = "charleschen2013@163.com"
import os
from os import path as osp
import sys
import time

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
from pix2pixHD.eval_img2map import eval, get_fid
from src.data.image_folder import make_dataset
import shutil
import numpy as np
from PIL import Image
import json

from src.pix2pixHD.myutils import pred2gray, gray2rgb
from evaluation.fid.fid_score import fid_score

from src.pix2pix.options.pix2pix_opts import get_pix2pix_args
from src.pix2pix.models.pix2pix_model import Pix2PixModel



def eval_fid(args, model_G, data_loader):
    device = get_device(args)
    data_loader = tqdm(data_loader)
    model_G.eval()
    model_G = model_G.to(device)

    label_preds = []
    label_targets = []
    create_dir(args.result)
    real_dir = osp.join(args.result, 'real_result')
    A_dir = osp.join(args.result, 'real_source')
    fake_dir=osp.join(args.result, 'fake_result')
    create_dir(real_dir)
    create_dir(A_dir)
    create_dir(fake_dir)

    count_time=0
    loop_time=0
    for i, sample in enumerate(data_loader):
        imgs = sample['A'].to(device)
        maps = sample['B'].to(device)

        start = time.clock()
        fakes = model_G(imgs).detach()
        # 获取结束时间
        loop_time = loop_time + 1
        end = time.clock()
        # 计算运行时间
        runTime = end - start
        print("运行时间：", runTime, "秒")
        count_time = count_time + runTime
        if loop_time == 100:
            print("运行时间100：", count_time, "秒")
            count_time=0

        batch_size = imgs.size(0)
        im_name = sample['A_paths']
        for b in range(batch_size):
            file_name = osp.split(im_name[b])[-1].split('.')[0]
            new_real_dir = osp.join(real_dir, osp.split(im_name[b])[0].split(os.sep)[-1])
            new_A_dir = osp.join(A_dir, osp.split(im_name[b])[0].split(os.sep)[-1])
            new_fake_dir = osp.join(fake_dir, osp.split(im_name[b])[0].split(os.sep)[-1])
            create_dir(new_real_dir)

            create_dir(new_A_dir)
            create_dir(new_fake_dir)
            real_file = osp.join(new_real_dir, f'{file_name}.png')
            A_file = osp.join(new_A_dir, f'{file_name}.png')
            fake_file=osp.join(new_fake_dir, f'{file_name}.png')

            from_std_tensor_save_image(filename=real_file, data=sample['B'][b].cpu())
            from_std_tensor_save_image(filename=A_file, data=sample['A'][b].cpu())
            from_std_tensor_save_image(filename=fake_file, data=fakes[b].cpu())


    fid = fid_score(real_path=real_dir, fake_path=fake_dir, gpu=str(args.gpu))
    print(f'===> fid score:{fid:.4f}')

    model_G.train()
    return fid


def train(args, get_dataloader_func=get_pix2pix_maps_dataloader):
    print(torch.cuda.is_available())
    with open(os.path.join(args.save,'args.json'), 'w') as f:
        json.dump(vars(args), f)
    logger = Logger(save_path=args.save, json_name='pix2pix')
    epoch_now = len(logger.get_data('G_loss'))

    model_saver = ModelSaver(save_path=args.save,
                             name_list=['G', 'D',  'G_optimizer', 'D_optimizer',
                                        'G_scheduler', 'D_scheduler',
                                        'best_G', 'best_D', 'best_G_optimizer', 'best_D_optimizer',
                                        'best_G_scheduler', 'best_D_scheduler'])
    sw = SummaryWriter(args.tensorboard_path)

    pix2pix_args = get_pix2pix_args(args)
    pix2pix_model=Pix2PixModel()
    pix2pix_model.initialize(pix2pix_args)

    G = pix2pix_model.netG
    D = pix2pix_model.netD
    model_saver.load('G', G)
    model_saver.load('D', D)

    # params_G = sum([param.nelement() for param in G.parameters()])
    # params_D = sum([param.nelement() for param in D.parameters()])
    # print(f"{params_G}, {params_D}")
    # print(f"{params_G}")
    # sys.exit(0)  # 测完退出

    G_optimizer = pix2pix_model.optimizer_G
    D_optimizer = pix2pix_model.optimizer_D
    model_saver.load('G_optimizer', G_optimizer)
    model_saver.load('D_optimizer', D_optimizer)

    G_scheduler = pix2pix_model.schedulers[0]
    D_scheduler = pix2pix_model.schedulers[1]
    model_saver.load('G_scheduler', G_scheduler)
    model_saver.load('D_scheduler', D_scheduler)

    device = get_device(args)

    # GANLoss = get_GANLoss(args)
    #
    # if args.use_ganFeat_loss:
    #     DFLoss = get_DFLoss(args)
    # if args.use_vgg_loss:
    #     VGGLoss = get_VGGLoss(args)
    # if args.use_low_level_loss:
    #     LLLoss = get_low_level_loss(args)

    print('Got final models!')
    fid = eval_fid(args, model_G=G, data_loader=get_dataloader_func(args, train=False))
    print('Test finish!')



if __name__ == '__main__':
    args = config()
    assert args.feat_num == 0
    assert args.use_instance == 0

    from src.pix2pixHD.myutils import seed_torch

    print(f'set seed as {args.seed}!')
    seed_torch(args.seed)

    train(args, get_dataloader_func=get_pix2pix_maps_dataloader)

pass
