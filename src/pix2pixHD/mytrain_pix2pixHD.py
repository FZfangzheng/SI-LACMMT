__author__ = "charles"
__email__ = "charleschen2013@163.com"
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
from src.eval.eval_every10epoch import eval_epoch
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
import time

def make_new_dir(newpath):
    if not osp.exists(newpath):
        os.mkdir(newpath)


def eval_fidiou(args,sw, model_G, data_loader,epoch=-1):
    device = get_device(args)
    data_loader = tqdm(data_loader)
    model_G.eval()

    model_G = model_G.to(device)
    create_dir(args.result)
    real_dir = osp.join(args.result, 'real_result')
    A_dir = osp.join(args.result, 'real_source')

    fake_dir=osp.join(args.result, 'fake_result')
    eval_dir = osp.join(args.result, 'eval_result')
    create_dir(real_dir)

    create_dir(A_dir)

    create_dir(fake_dir)
    create_dir(eval_dir)
    count_time=0
    loop_time=0
    for i, sample in enumerate(data_loader):
        inputs, labels = sample['A_seg'], sample['seg'].squeeze(dim=1)
        inputs = inputs.cuda() if args.gpu else inputs
        labels = labels.cuda() if args.gpu else labels
        imgs = sample['A'].to(device)


        imgs_plus=imgs

        start = time.clock()
        fakes = model_G(imgs_plus).detach()
        # 获取结束时间
        loop_time = loop_time + 1
        end = time.clock()
        # 计算运行时间
        runTime = end - start
        count_time = count_time + runTime
        if loop_time == 100:
            print("运行时间100：", count_time, "秒")
            count_time=0

        batch_size = inputs.size(0)
        im_name = sample['A_paths']
        for b in range(batch_size):
            file_name = osp.split(im_name[b])[-1].split('.')[0]
            new_real_dir = osp.join(real_dir, osp.split(im_name[b])[0].split(os.sep)[-1])
            new_A_dir = osp.join(A_dir, osp.split(im_name[b])[0].split(os.sep)[-1])
            new_fake_dir = osp.join(fake_dir, osp.split(im_name[b])[0].split(os.sep)[-1])
            make_new_dir(new_real_dir)

            make_new_dir(new_A_dir)
            make_new_dir(new_fake_dir)
            real_file = osp.join(new_real_dir, f'{file_name}.png')

            A_file = osp.join(new_A_dir, f'{file_name}.png')

            fake_file=osp.join(new_fake_dir, f'{file_name}.png')

            from_std_tensor_save_image(filename=real_file, data=sample['B'][b].cpu())
            from_std_tensor_save_image(filename=A_file, data=sample['A'][b].cpu())
            from_std_tensor_save_image(filename=fake_file, data=fakes[b].cpu())
    real_img_list = make_dataset(real_dir)
    fake_img_list = make_dataset(fake_dir)
    new_fake_dir = osp.join(args.save, 'fake_result_without_layer')
    new_real_dir = osp.join(args.save, 'real_result_without_layer')
    create_dir(new_fake_dir)
    create_dir(new_real_dir)
    for real_img in real_img_list:
        img_name = osp.split(real_img)[-1]
        new_real_img_path = osp.join(new_real_dir, img_name)
        shutil.copy(real_img, new_real_img_path)
    for fake_img in fake_img_list:
        img_name = osp.split(fake_img)[-1]
        new_fake_img_path = osp.join(new_fake_dir, img_name)
        shutil.copy(fake_img, new_fake_img_path)
    real_paths = [osp.join(real_dir,"1"),osp.join(real_dir,"2"),osp.join(real_dir,"3"),osp.join(real_dir,"4")]
    fake_paths = [osp.join(fake_dir, "1"), osp.join(fake_dir, "2"), osp.join(fake_dir, "3"), osp.join(fake_dir, "4")]

    rets = eval_epoch(real_paths,fake_paths,epoch, eval_dir)
    if epoch != -1:
        sw.add_scalar('eval1/kid_mean', rets[0].kid_mean, int(epoch/10))
        sw.add_scalar('eval1/fid', rets[0].fid, int(epoch / 10))
        sw.add_scalar('eval1/kNN', rets[0].kNN, int(epoch / 10))
        sw.add_scalar('eval1/K_MMD', rets[0].K_MMD, int(epoch / 10))
        sw.add_scalar('eval1/WD', rets[0].WD, int(epoch / 10))
        sw.add_scalar('eval1/_IS', rets[0]._IS, int(epoch/10))
        sw.add_scalar('eval1/_MS', rets[0]._MS, int(epoch / 10))
        sw.add_scalar('eval1/_mse_skimage', rets[0]._mse_skimage, int(epoch / 10))
        sw.add_scalar('eval1/_ssim_skimage', rets[0]._ssim_skimage, int(epoch / 10))
        sw.add_scalar('eval1/_ssimrgb_skimage', rets[0]._ssimrgb_skimage, int(epoch / 10))
        sw.add_scalar('eval1/_psnr_skimage', rets[0]._psnr_skimage, int(epoch / 10))
        sw.add_scalar('eval1/_kid_std', rets[0]._kid_std, int(epoch / 10))
        sw.add_scalar('eval1/_fid_inkid_mean', rets[0]._fid_inkid_mean, int(epoch / 10))
        sw.add_scalar('eval1/_fid_inkid_std', rets[0]._fid_inkid_std, int(epoch / 10))
        sw.add_scalar('eval2/kid_mean', rets[1].kid_mean, int(epoch/10))
        sw.add_scalar('eval2/fid', rets[1].fid, int(epoch / 10))
        sw.add_scalar('eval2/kNN', rets[1].kNN, int(epoch / 10))
        sw.add_scalar('eval2/K_MMD', rets[1].K_MMD, int(epoch / 10))
        sw.add_scalar('eval2/WD', rets[1].WD, int(epoch / 10))
        sw.add_scalar('eval2/_IS', rets[1]._IS, int(epoch/10))
        sw.add_scalar('eval2/_MS', rets[1]._MS, int(epoch / 10))
        sw.add_scalar('eval2/_mse_skimage', rets[1]._mse_skimage, int(epoch / 10))
        sw.add_scalar('eval2/_ssim_skimage', rets[1]._ssim_skimage, int(epoch / 10))
        sw.add_scalar('eval2/_ssimrgb_skimage', rets[1]._ssimrgb_skimage, int(epoch / 10))
        sw.add_scalar('eval2/_psnr_skimage', rets[1]._psnr_skimage, int(epoch / 10))
        sw.add_scalar('eval2/_kid_std', rets[1]._kid_std, int(epoch / 10))
        sw.add_scalar('eval2/_fid_inkid_mean', rets[1]._fid_inkid_mean, int(epoch / 10))
        sw.add_scalar('eval2/_fid_inkid_std', rets[1]._fid_inkid_std, int(epoch / 10))
        sw.add_scalar('eval3/kid_mean', rets[2].kid_mean, int(epoch/10))
        sw.add_scalar('eval3/fid', rets[2].fid, int(epoch / 10))
        sw.add_scalar('eval3/kNN', rets[2].kNN, int(epoch / 10))
        sw.add_scalar('eval3/K_MMD', rets[2].K_MMD, int(epoch / 10))
        sw.add_scalar('eval3/WD', rets[2].WD, int(epoch / 10))
        sw.add_scalar('eval3/_IS', rets[2]._IS, int(epoch/10))
        sw.add_scalar('eval3/_MS', rets[2]._MS, int(epoch / 10))
        sw.add_scalar('eval3/_mse_skimage', rets[2]._mse_skimage, int(epoch / 10))
        sw.add_scalar('eval3/_ssim_skimage', rets[2]._ssim_skimage, int(epoch / 10))
        sw.add_scalar('eval3/_ssimrgb_skimage', rets[2]._ssimrgb_skimage, int(epoch / 10))
        sw.add_scalar('eval3/_psnr_skimage', rets[2]._psnr_skimage, int(epoch / 10))
        sw.add_scalar('eval3/_kid_std', rets[2]._kid_std, int(epoch / 10))
        sw.add_scalar('eval3/_fid_inkid_mean', rets[2]._fid_inkid_mean, int(epoch / 10))
        sw.add_scalar('eval3/_fid_inkid_std', rets[2]._fid_inkid_std, int(epoch / 10))


    model_G.train()


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
    with open(os.path.join(args.save,'args.json'), 'w') as f:
        json.dump(vars(args), f)
    logger = Logger(save_path=args.save, json_name='img2map_seg')
    epoch_now = len(logger.get_data('G_loss'))

    model_saver = ModelSaver(save_path=args.save,
                             name_list=['G', 'D', 'G_optimizer', 'D_optimizer',
                                        'G_scheduler', 'D_scheduler'])

    sw = SummaryWriter(args.tensorboard_path)


    G = get_G(args,input_nc=3) # 3+256+1，256为分割网络输出featuremap的通道数
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

    # CE_loss=nn.CrossEntropyLoss(ignore_index=255)
    LVS_loss = lovasz_softmax
    data_loader_focal = get_dataloader_func(args, train=True)
    data_loader_focal = tqdm(data_loader_focal)
    alpha = label_nums(data_loader_focal,label_num=args.label_nc)
    # alpha = [1,1,1,1,1]
    tmp_min = min(alpha)
    assert tmp_min > 0
    for i in range(len(alpha)):
        alpha[i] = tmp_min / alpha[i]
    if args.focal_alpha_revise:
        assert len(args.focal_alpha_revise) == len(alpha)
        for i in range(len(alpha)):
            alpha[i]=alpha[i]*args.focal_alpha_revise[i]
    print(alpha)
    FOCAL_loss = FocalLoss(gamma=2, alpha=alpha)



    print('get final models')
    eval_fidiou(args,sw=sw, model_G=G,data_loader=get_pix2pix_maps_dataloader(args, train=False),epoch=epoch_now)


    




if __name__ == '__main__':
    args = config()

    # args.label_nc = 5

    from src.pix2pixHD.myutils import seed_torch
    print(f'\nset seed as {args.seed}!\n')
    seed_torch(args.seed)

    train(args, get_dataloader_func=get_pix2pix_maps_dataloader)

pass
