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
from src.CreativeGAN.train_config import config
from src.CreativeGAN.networks import get_G, get_D, get_E
from torch.optim import Adam
from src.CreativeGAN.hinge_lr_scheduler import get_hinge_scheduler
from src.utils.logger import ModelSaver, Logger
from src.datasets import get_pix2pix_maps_dataloader
from src.CreativeGAN.utils import get_edges, label_to_one_hot, get_encode_features
from src.utils.visualizer import Visualizer
from tqdm import tqdm
from torchvision import transforms
from src.CreativeGAN.criterion import get_GANLoss, get_VGGLoss, get_DFLoss, get_low_level_loss
from tensorboardX import SummaryWriter
from src.CreativeGAN.utils import from_std_tensor_save_image, create_dir
from src.data.image_folder import make_dataset
import shutil
import numpy as np
from PIL import Image

from src.CreativeGAN.deeplabv3plus.deeplabv3plus import Configuration
from src.CreativeGAN.deeplabv3plus.deeplabv3plus.my_deeplabv3plus_featuremap import deeplabv3plus
import torch.nn.functional as F

from src.CreativeGAN.deeplabv3plus.lovasz_losses import lovasz_softmax

from util.util import tensor2im  # 注意，该函数使用0.5与255恢复可视图像，所以如果是ImageNet标准化的可能会有色差？这里都显示试一下
from src.CreativeGAN.myutils import pred2gray, gray2rgb

from src.CreativeGAN.deeplabv3plus.focal_loss import FocalLoss
from evaluation.fid.fid_score import fid_score
import json
import time

def eval_fidiou(args, model_G,model_seg, data_loader):
    device = get_device(args)
    data_loader = tqdm(data_loader)
    model_G.eval()
    model_seg.eval()
    model_G = model_G.to(device)
    model_seg = model_seg.to(device)

    label_preds = []
    label_targets = []
    create_dir(args.result)
    real_seg_dir = osp.join(args.result, 'real_seg')
    real_dir = osp.join(args.result, 'real_result')
    A_dir = osp.join(args.result, 'real_source')
    seg_dir = osp.join(args.result, 'seg_result')
    fake_dir=osp.join(args.result, 'fake_result')
    create_dir(real_dir)
    create_dir(real_seg_dir)
    create_dir(A_dir)
    create_dir(seg_dir)
    create_dir(fake_dir)
    count_time=0
    loop_time=0
    for i, sample in enumerate(data_loader):
        inputs, labels = sample['A_seg'], sample['seg'].squeeze(dim=1)
        inputs = inputs.cuda() if args.gpu else inputs
        labels = labels.cuda() if args.gpu else labels
        imgs = sample['A'].to(device)
        maps = sample['B'].to(device)

        start = time.clock()
        outputs, feature_map = model_seg(inputs)
        bs, n_class, h, w = outputs.shape
        outs = outputs.data.cpu().numpy()
        features = feature_map.data.cpu().numpy()
        pred = outs.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(bs, h, w)
        target = labels.cpu().numpy().reshape(bs, h, w)
        label_preds.append(pred)
        label_targets.append(target)

        # seg_ret = pred2gray(outputs).unsqueeze(1).type(torch.FloatTensor).to(device)  # bs*1*h*w
        feature_map = feature_map.detach()
        imgs_plus=torch.cat((imgs,feature_map),1)
        fakes = model_G(imgs_plus).detach()
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

        batch_size = inputs.size(0)
        im_name = sample['A_paths']
        for b in range(batch_size):
            file_name = osp.split(im_name[b])[-1].split('.')[0]
            real_file = osp.join(real_dir, f'{file_name}.png')
            real_seg_file = osp.join(real_seg_dir, f'{file_name}.png')
            A_file = osp.join(A_dir, f'{file_name}.png')
            seg_file = osp.join(seg_dir, f'{file_name}.png')
            fake_file=osp.join(fake_dir, f'{file_name}.png')

            from_std_tensor_save_image(filename=real_file, data=sample['B'][b].cpu())
            from_std_tensor_save_image(filename=A_file, data=sample['A'][b].cpu())
            from_std_tensor_save_image(filename=fake_file, data=fakes[b].cpu())
            tmpimg= sample['seg'][b].data.cpu().numpy()
            tmpimg = gray2rgb(tmpimg)
            tmpimg=Image.fromarray(tmpimg)
            tmpimg.save(fp=real_seg_file)

            tmpimg = gray2rgb(pred[b])
            tmpimg = Image.fromarray(tmpimg)
            tmpimg.save(fp=seg_file)

    fid = fid_score(real_path=real_dir, fake_path=fake_dir, gpu=str(args.gpu))
    print(f'===> fid score:{fid:.4f}')
    iou=None
    from src.pix2pixHD.eval_iou import label_accuracy_score
    _,_,iou,_,_=label_accuracy_score(label_targets, label_preds, n_class)
    print(f'===> iou score:{iou:.4f}')

    model_seg.train()
    model_G.train()
    return fid,iou

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
    epoch_now = len(logger.get_data('FOCAL_loss'))

    model_saver = ModelSaver(save_path=args.save,
                             name_list=['G', 'D', 'G_optimizer', 'D_optimizer',
                                        'G_scheduler', 'D_scheduler','DLV3P',"DLV3P_global_optimizer",
                                        "DLV3P_backbone_optimizer","DLV3P_global_scheduler","DLV3P_backbone_scheduler"])

    sw = SummaryWriter(args.tensorboard_path)

    G = get_G(args,input_nc=3+256) # 3+256，256为分割网络输出featuremap的通道数
    D = get_D(args)
    model_saver.load('G', G)
    model_saver.load('D', D)

    cfg=Configuration()
    cfg.MODEL_NUM_CLASSES=args.label_nc
    DLV3P=deeplabv3plus(cfg)
    if args.gpu:
        # DLV3P=nn.DataParallel(DLV3P)
        DLV3P=DLV3P.cuda()
    model_saver.load('DLV3P', DLV3P)

    G_optimizer = Adam(G.parameters(), lr=args.G_lr, betas=(args.beta1, 0.999))
    D_optimizer = Adam(D.parameters(), lr=args.D_lr, betas=(args.beta1, 0.999))

    seg_global_params, seg_backbone_params=DLV3P.get_paras()
    DLV3P_global_optimizer = torch.optim.Adam([{'params': seg_global_params, 'initial_lr': args.seg_lr_global}], lr=args.seg_lr_global,betas=(args.beta1, 0.999))
    DLV3P_backbone_optimizer = torch.optim.Adam([{'params': seg_backbone_params, 'initial_lr': args.seg_lr_backbone}], lr=args.seg_lr_backbone, betas=(args.beta1, 0.999))

    model_saver.load('G_optimizer', G_optimizer)
    model_saver.load('D_optimizer', D_optimizer)
    model_saver.load('DLV3P_global_optimizer', DLV3P_global_optimizer)
    model_saver.load('DLV3P_backbone_optimizer', DLV3P_backbone_optimizer)

    G_scheduler = get_hinge_scheduler(args, G_optimizer)
    D_scheduler = get_hinge_scheduler(args, D_optimizer)
    DLV3P_global_scheduler=torch.optim.lr_scheduler.LambdaLR(DLV3P_global_optimizer, lr_lambda=lambda epoch:(1 - epoch/args.epochs)**0.9,last_epoch=epoch_now)
    DLV3P_backbone_scheduler = torch.optim.lr_scheduler.LambdaLR(DLV3P_backbone_optimizer,lr_lambda=lambda epoch: (1 - epoch / args.epochs) ** 0.9,last_epoch=epoch_now)

    model_saver.load('G_scheduler', G_scheduler)
    model_saver.load('D_scheduler', D_scheduler)
    model_saver.load('DLV3P_global_scheduler', DLV3P_global_scheduler)
    model_saver.load('DLV3P_backbone_scheduler', DLV3P_backbone_scheduler)

    device = get_device(args)

    print('get final models')
    iou = eval_fidiou(args,model_G=G, model_seg=DLV3P,data_loader=get_pix2pix_maps_dataloader(args, train=False))
    logger.log(key='iou', data=iou)
    if iou < logger.get_max(key='FID'):
        model_saver.save(f'DLV3P_{iou:.4f}', DLV3P)
    sw.add_scalar('eval/iou', iou, epoch_now)

  



if __name__ == '__main__':
    args = config()

    # args.label_nc = 5

    from src.pix2pixHD.myutils import seed_torch

    print(f'\nset seed as {args.seed}!\n')
    seed_torch(args.seed)

    train(args, get_dataloader_func=get_pix2pix_maps_dataloader)

pass
