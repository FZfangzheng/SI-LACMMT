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
import math
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
from src.pix2pixHD.deeplabv3plus.deeplabv3plus.my_deeplabv3plus_theta_wid import deeplabv3plus
import torch.nn.functional as F

from src.pix2pixHD.deeplabv3plus.lovasz_losses import lovasz_softmax

from util.util import tensor2im  # 注意，该函数使用0.5与255恢复可视图像，所以如果是ImageNet标准化的可能会有色差？这里都显示试一下  试过了，有很明显的色差和噪声
from src.pix2pixHD.myutils import pred2gray, gray2rgb,visual_theta_rho

from src.pix2pixHD.deeplabv3plus.focal_loss import FocalLoss


def get_theta_loss(road_mask,theta_gt,theta_x,theta_y,device):
    # loss需要在tensor中算
    # theta_gt 其中的值为 -π/2 ~ π/2 对应直线角度为 x轴负方向 ~ y轴正方向 ~ x轴正方向
    road_mask[theta_gt==-2.0]=0 # 扣掉交叉点，交叉点不计算loss
    theta=torch.atan2(theta_y,theta_x) # 求出来的值为0~π，以x轴正方向为0，逆时针为正
    theta_gt=np.pi/2-theta_gt # 使其取值方式与theta一致
    theta_gt=torch.from_numpy(theta_gt).float().to(device)
    loss=torch.min(torch.abs(theta_gt-theta),math.pi-torch.abs(theta_gt-theta))
    road_mask=torch.from_numpy(road_mask).float().to(device)
    # loss=(road_mask*loss).sum()/(road_mask.sum()) if road_mask.sum()>0 else torch.tensor(0)
    loss=(road_mask*loss).sum() #todo loss是否应采用sum？感觉均值的话会因为各图像道路像素不一致导致权重不一
    return loss

def get_wid_loss(road_mask,wid_gt,wid,device):
    # wid_gt 均小于等于20，且均值滤波后应几乎没有等于20的，所以直接除以20；因为应该也没有宽度为0的点，所以不考虑饱和区问题
    wid_gt=wid_gt/20
    wid_gt=torch.from_numpy(wid_gt).float().to(device)
    loss=torch.abs(wid_gt-wid)
    road_mask = torch.from_numpy(road_mask).float().to(device)
    # loss=(road_mask*loss).sum()/(road_mask.sum()) if road_mask.sum()>0 else torch.tensor(0)
    loss=(road_mask*loss).sum()
    return loss

def eval_iou(args, model, data_loader):
    device = get_device(args)
    data_loader = tqdm(data_loader)
    model.eval()
    model = model.to(device)

    label_preds = []
    label_targets = []

    real_seg_dir = osp.join(args.save, 'real_seg')
    real_dir = osp.join(args.save, 'real_result')
    A_dir = osp.join(args.save, 'real_source')
    seg_dir = osp.join(args.save, 'seg_result')
    real_theta_wid_dir=osp.join(args.save, 'real_theta_wid')
    fake_theta_wid_dir = osp.join(args.save, 'fake_theta_wid')
    create_dir(real_dir)
    create_dir(real_seg_dir)
    create_dir(A_dir)
    create_dir(seg_dir)
    create_dir(real_theta_wid_dir)
    create_dir(fake_theta_wid_dir)

    for i, sample in enumerate(data_loader):
        inputs, labels = sample['A_seg'], sample['seg'].squeeze(dim=1)
        bs_test = inputs.shape[1]
        inputs = inputs.cuda() if args.gpu else inputs
        labels = labels.cuda() if args.gpu else labels

        outputs, feature_map = model(inputs)
        theta_x, theta_y, wid = model.get_theta_and_wid()
        bs, n_class, h, w = outputs.shape
        outs = outputs.data.cpu().numpy()
        pred = outs.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(bs, h, w)
        target = labels.cpu().numpy().reshape(bs, h, w)
        label_preds.append(pred)
        label_targets.append(target)

        batch_size = inputs.size(0)
        im_name = sample['A_paths']
        for b in range(batch_size):
            file_name = osp.split(im_name[b])[-1].split('.')[0]
            real_file = osp.join(real_dir, f'{file_name}.tif')
            real_seg_file = osp.join(real_seg_dir, f'{file_name}.tif')
            A_file = osp.join(A_dir, f'{file_name}.tif')
            seg_file = osp.join(seg_dir, f'{file_name}.tif')
            real_theta_wid_file=osp.join(real_theta_wid_dir, f'{file_name}.tif')
            fake_theta_wid_file=osp.join(fake_theta_wid_dir, f'{file_name}.tif')

            from_std_tensor_save_image(filename=real_file, data=sample['B'][b].cpu())
            from_std_tensor_save_image(filename=A_file, data=sample['A'][b].cpu())
            tmpimg= sample['seg'][b].data.cpu().numpy()
            tmpimg = gray2rgb(tmpimg)
            tmpimg=Image.fromarray(tmpimg)
            tmpimg.save(fp=real_seg_file)

            tmpimg = gray2rgb(pred[b])
            tmpimg = Image.fromarray(tmpimg)
            tmpimg.save(fp=seg_file)

            seg_vis = labels[0].data.cpu().numpy()
            road_mask = (seg_vis == 1)
            seg_theta = sample['theta'][0].data.cpu().numpy()
            seg_wid = sample['wid'][0].data.cpu().numpy()
            tmpimg = visual_theta_rho(seg_vis, road_mask, seg_theta, None, seg_wid)
            tmpimg = Image.fromarray(tmpimg)
            tmpimg.save(fp=real_theta_wid_file)

            seg_vis = pred2gray(outputs)[0].data.cpu().numpy()
            road_mask = (seg_vis == 1)
            seg_theta = torch.atan2(theta_y, theta_x)[0].data.cpu().numpy()  # 注意角度含义不一致
            seg_theta = 90 - np.rad2deg(seg_theta).astype(np.int64)
            seg_theta[seg_theta >= 90] = 89
            seg_theta[seg_theta < -90] = -90
            seg_theta = np.deg2rad(seg_theta)
            seg_wid = wid[0].data.cpu().numpy().astype(np.uint8)
            tmpimg = visual_theta_rho(seg_vis, road_mask, seg_theta, None, seg_wid)
            tmpimg = Image.fromarray(tmpimg)
            tmpimg.save(fp=fake_theta_wid_file)

    iou=None
    from src.pix2pixHD.eval_iou import label_accuracy_score
    _,_,iou,_,_=label_accuracy_score(label_targets, label_preds, n_class)

    model.train()
    return iou

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
    logger = Logger(save_path=args.save, json_name='img2map_seg')
    epoch_now = len(logger.get_data('FOCAL_loss'))

    model_saver = ModelSaver(save_path=args.save,
                             name_list=['DLV3P',"DLV3P_global_optimizer",
                                        "DLV3P_backbone_optimizer","DLV3P_global_scheduler","DLV3P_backbone_scheduler"])

    sw = SummaryWriter(args.tensorboard_path)


    cfg=Configuration()
    cfg.MODEL_NUM_CLASSES=args.label_nc
    DLV3P=deeplabv3plus(cfg)
    # DLV3P = nn.DataParallel(DLV3P)
    if args.gpu:
        DLV3P=DLV3P.cuda()
    model_saver.load('DLV3P', DLV3P)

    seg_global_params, seg_backbone_params=DLV3P.get_paras()
    DLV3P_global_optimizer = torch.optim.Adam([{'params': seg_global_params, 'initial_lr': args.seg_lr_global}], lr=args.seg_lr_global,betas=(args.beta1, 0.999))
    DLV3P_backbone_optimizer = torch.optim.Adam([{'params': seg_backbone_params, 'initial_lr': args.seg_lr_backbone}], lr=args.seg_lr_backbone, betas=(args.beta1, 0.999))

    model_saver.load('DLV3P_global_optimizer', DLV3P_global_optimizer)
    model_saver.load('DLV3P_backbone_optimizer', DLV3P_backbone_optimizer)

    DLV3P_global_scheduler=torch.optim.lr_scheduler.LambdaLR(DLV3P_global_optimizer, lr_lambda=lambda epoch:(1 - epoch/args.epochs)**0.9,last_epoch=epoch_now)
    DLV3P_backbone_scheduler = torch.optim.lr_scheduler.LambdaLR(DLV3P_backbone_optimizer,lr_lambda=lambda epoch: (1 - epoch / args.epochs) ** 0.9,last_epoch=epoch_now)

    model_saver.load('DLV3P_global_scheduler', DLV3P_global_scheduler)
    model_saver.load('DLV3P_backbone_scheduler', DLV3P_backbone_scheduler)

    device = get_device(args)

    # CE_loss=nn.CrossEntropyLoss(ignore_index=255)
    LVS_loss = lovasz_softmax

    data_loader_focal = get_dataloader_func(args, train=True,flag=1)
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
    #

    if epoch_now==args.epochs:
        iou = eval_iou(args, model=DLV3P,data_loader=get_pix2pix_maps_dataloader(args, train=False))
        logger.log(key='iou', data=iou)
        if iou < logger.get_max(key='FID'):
            model_saver.save(f'DLV3P_{iou:.4f}', DLV3P)
        sw.add_scalar('eval/iou', iou, epoch_now)

    for epoch in range(epoch_now, args.epochs):
        # CE_loss_list = []
        LVS_loss_list=[]
        FOCAL_loss_list=[]

        data_loader = get_dataloader_func(args, train=True,flag=1)
        data_loader = tqdm(data_loader)

        for step, sample in enumerate(data_loader):
            # 先训练deeplabv3+
            imgs = sample['A_seg'].to(device)  # (shape: (batch_size, 3, img_h, img_w))
            label_imgs = sample['seg'].type(torch.LongTensor).to(device)  # (shape: (batch_size, img_h, img_w))
            # print(label_imgs)
            # print(label_imgs.max())
            # print(label_imgs.min())

            imgs_show=sample['A'].to(device) # (shape: (batch_size, 3, img_h, img_w))
            maps_show= sample['B'].to(device)  # (shape: (batch_size, 3, img_h, img_w))

            outputs,feature_map = DLV3P(imgs)  # (shape: (batch_size, num_classes, img_h, img_w))
            theta_x,theta_y,wid=DLV3P.get_theta_and_wid()
            # feature_map=feature_map.detach()

            # compute the loss:
            # ce_loss = CE_loss(outputs, label_imgs)
            # ce_loss_value = ce_loss.data.cpu().numpy()
            soft_outputs = torch.nn.functional.softmax(outputs, dim=1)
            lvs_loss = LVS_loss(soft_outputs, label_imgs, ignore=255)
            lvs_loss_value = lvs_loss.data.cpu().numpy()
            focal_loss=FOCAL_loss(outputs,label_imgs)
            focal_loss_value=focal_loss.data.cpu().numpy()

            road_mask=label_imgs.data.cpu().numpy()
            road_mask=(road_mask==1)
            theta_loss=get_theta_loss(road_mask,sample['theta'].numpy(),theta_x,theta_y,device)
            theta_loss_value=theta_loss.data.cpu().numpy()
            wid_loss=get_wid_loss(road_mask,sample['wid'].numpy(),wid,device)
            wid_loss_value = wid_loss.data.cpu().numpy()

            a_loss = args.a_loss if args.a_loss else [1, 1, 1, 1]
            seg_loss = a_loss[0]*focal_loss+a_loss[1]*lvs_loss+a_loss[2]*theta_loss+a_loss[3]*wid_loss

            # optimization step:
            DLV3P_global_optimizer.zero_grad()  # (reset gradients)
            DLV3P_backbone_optimizer.zero_grad()
            seg_loss.backward()  # (compute gradients)
            DLV3P_global_optimizer.step()  # (perform optimization step)
            DLV3P_backbone_optimizer.step()


            data_loader.write(f'Epochs:{epoch} | FOCAL_loss:{a_loss[0]*focal_loss_value:.6f}|LVS_loss:{a_loss[1]*lvs_loss_value:.6f} '
                              f'theta_loss:{a_loss[2]*theta_loss_value:.6f}|wid_loss:{a_loss[3]*wid_loss_value:.6f}'
                              f'| lr_global:{get_lr(DLV3P_global_optimizer):.8f}| lr_backbone:{get_lr(DLV3P_backbone_optimizer):.8f}')

            # CE_loss_list.append(ce_loss_value)
            LVS_loss_list.append(lvs_loss_value)
            FOCAL_loss_list.append(focal_loss_value)

            # tensorboard log
            if args.tensorboard_log and step % args.tensorboard_log == 0:  # defalut is 5
                total_steps = epoch * len(data_loader) + step
                # sw.add_scalar('Loss/CE', ce_loss_value, total_steps)
                sw.add_scalar('Loss/focal', focal_loss_value*a_loss[0], total_steps)
                sw.add_scalar('Loss/LVS', lvs_loss_value*a_loss[1], total_steps)
                sw.add_scalar('Loss/theta', theta_loss_value*a_loss[2], total_steps)
                sw.add_scalar('Loss/wid', wid_loss_value*a_loss[3], total_steps)
                sw.add_scalar('LR/global_seg', get_lr(DLV3P_global_optimizer), total_steps)
                sw.add_scalar('LR/backbone_seg', get_lr(DLV3P_backbone_optimizer), total_steps)


                sw.add_image('img/realA', tensor2im(imgs_show.data), total_steps, dataformats='HWC')
                tmpsegmap = pred2gray(outputs)
                tmpsegmap = tmpsegmap[0].data.numpy()
                tmpsegmap = gray2rgb(tmpsegmap)
                sw.add_image('img/fake_segB', tmpsegmap, total_steps, dataformats='HWC')
                tmpsegmap = label_imgs[0].data.cpu().numpy()
                tmpsegmap = gray2rgb(tmpsegmap)
                sw.add_image('img/real_segB', tmpsegmap, total_steps, dataformats='HWC')
                # sw.add_image('img2/realA_imgnet', tensor2im(imgs.data), total_steps, dataformats='HWC')
                sw.add_image('img/realB', tensor2im(maps_show.data), total_steps, dataformats='HWC')

                seg_vis=label_imgs[0].data.cpu().numpy()
                road_mask=(seg_vis==1)
                seg_theta=sample['theta'][0].data.cpu().numpy()
                seg_wid=sample['wid'][0].data.cpu().numpy()
                tmpimg=visual_theta_rho(seg_vis,road_mask,seg_theta,None,seg_wid)
                sw.add_image('img2/real_theta_wid', tmpimg, total_steps, dataformats='HWC')

                seg_vis = pred2gray(outputs)[0].data.cpu().numpy()
                road_mask = (seg_vis == 1)
                seg_theta = torch.atan2(theta_y,theta_x)[0].data.cpu().numpy() # 注意角度含义不一致
                seg_theta =90-np.rad2deg(seg_theta).astype(np.int64)
                seg_theta[seg_theta>=90]=89
                seg_theta[seg_theta<-90]=-90
                seg_theta=np.deg2rad(seg_theta)
                seg_wid = wid[0].data.cpu().numpy().astype(np.uint8)
                tmpimg = visual_theta_rho(seg_vis, road_mask, seg_theta,None,seg_wid)
                sw.add_image('img2/fake_theta_wid', tmpimg, total_steps, dataformats='HWC')


        DLV3P_global_scheduler.step()
        DLV3P_backbone_scheduler.step()
        if epoch % 10 == 0 or epoch == (args.epochs-1):
            import copy
            args2=copy.deepcopy(args)
            args2.batch_size=args.batch_size_eval
            iou = eval_iou(args, model=DLV3P,data_loader=get_pix2pix_maps_dataloader(args, train=False,flag=1))
            logger.log(key='iou', data=iou)
            if iou < logger.get_max(key='FID'):
                model_saver.save(f'DLV3P_{iou:.4f}', DLV3P)
            sw.add_scalar('eval/iou', iou, epoch)

        # logger.log(key='CE_loss', data=sum(CE_loss_list) / float(len(CE_loss_list)))
        logger.log(key='LVS_loss', data=sum(LVS_loss_list) / float(len(LVS_loss_list)))
        logger.log(key='FOCAL_loss', data=sum(FOCAL_loss_list) / float(len(FOCAL_loss_list)))
        logger.save_log()
        # logger.visualize()


        model_saver.save('DLV3P', DLV3P)

        model_saver.save('DLV3P_global_optimizer', DLV3P_global_optimizer)
        model_saver.save('DLV3P_backbone_optimizer', DLV3P_backbone_optimizer)

        model_saver.save('DLV3P_global_scheduler', DLV3P_global_scheduler)
        model_saver.save('DLV3P_backbone_scheduler', DLV3P_backbone_scheduler)



if __name__ == '__main__':
    # a=np.array([1,2,3])
    # b = np.array([3, 2, 1])
    # c=np.fmin(a,b)
    # a=torch.tensor(0.0)
    # b=torch.tensor(-1.0)
    # c=torch.atan2(a,b)
    # a=torch.tensor([5,8,8,8])
    # b=torch.tensor([2,-1,2,10])
    # c=torch.min(a,b)

    args = config()

    # args.label_nc = 5

    train(args, get_dataloader_func=get_pix2pix_maps_dataloader)

pass
