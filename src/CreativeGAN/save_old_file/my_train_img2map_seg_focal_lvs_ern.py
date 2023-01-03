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
import cv2


from src.pix2pixHD.deeplabv3plus.deeplabv3plus import Configuration
# from src.pix2pixHD.deeplabv3plus.deeplabv3plus.deeplabv3plus_seg import deeplabv3plus
from src.pix2pixHD.deeplabv3plus.deeplabv3plus.my_deeplabv3plus_ern import deeplabv3plus
import torch.nn.functional as F

from src.pix2pixHD.deeplabv3plus.lovasz_losses import lovasz_softmax

from util.util import tensor2im  # 注意，该函数使用0.5与255恢复可视图像，所以如果是ImageNet标准化的可能会有色差？这里都显示试一下
from src.pix2pixHD.myutils import pred2gray, gray2rgb

from src.pix2pixHD.deeplabv3plus.focal_loss import FocalLoss

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
    edge_dir = osp.join(args.save, 'edge_result')
    en_edge_dir=osp.join(args.save, 'en_edge_result')
    de_edge_dir = osp.join(args.save, 'de_edge_result')
    create_dir(real_dir)
    create_dir(real_seg_dir)
    create_dir(A_dir)
    create_dir(seg_dir)
    create_dir(edge_dir)
    create_dir(en_edge_dir)
    create_dir(de_edge_dir)

    for i, sample in enumerate(data_loader):
        inputs, labels = sample['A_seg'], sample['seg'].squeeze(dim=1)
        bs_test = inputs.shape[1]
        inputs = inputs.cuda() if args.gpu else inputs
        labels = labels.cuda() if args.gpu else labels

        outputs, feature_map = model(inputs)
        bs, n_class, h, w = outputs.shape
        edge=get_edge_from_tensor(labels.type(torch.Tensor)).cpu()
        en_edge, de_edge = model.get_edges()
        en_edge=en_edge.data.cpu().numpy()
        en_edge=en_edge.transpose(0, 2, 3, 1).reshape(-1, 2).argmax(axis=1).reshape(bs, h, w)
        de_edge = de_edge.data.cpu().numpy()
        de_edge = de_edge.transpose(0, 2, 3, 1).reshape(-1, 2).argmax(axis=1).reshape(bs, h, w)
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
            edge_file=osp.join(edge_dir, f'{file_name}.tif')
            en_edge_file=osp.join(en_edge_dir, f'{file_name}.tif')
            de_edge_file = osp.join(de_edge_dir, f'{file_name}.tif')

            from_std_tensor_save_image(filename=real_file, data=sample['B'][b].cpu())
            from_std_tensor_save_image(filename=A_file, data=sample['A'][b].cpu())
            tmpimg= sample['seg'][b].data.cpu().numpy()
            tmpimg = gray2rgb(tmpimg,3,label_list=[[239,238,236],[255,255,255],[170,218,255]])
            tmpimg=Image.fromarray(tmpimg)
            tmpimg.save(fp=real_seg_file)

            tmpimg = gray2rgb(pred[b],3,label_list=[[239,238,236],[255,255,255],[170,218,255]])
            tmpimg = Image.fromarray(tmpimg)
            tmpimg.save(fp=seg_file)

            tmpimg=gray2rgb(edge[b],2,label_list=[[0,0,0],[255,255,255]])
            tmpimg = Image.fromarray(tmpimg)
            tmpimg.save(fp=edge_file)
            tmpimg = gray2rgb(en_edge[b], 2, label_list=[[0, 0, 0], [255, 255, 255]])
            tmpimg = Image.fromarray(tmpimg)
            tmpimg.save(fp=en_edge_file)
            tmpimg = gray2rgb(de_edge[b], 2, label_list=[[0, 0, 0], [255, 255, 255]])
            tmpimg = Image.fromarray(tmpimg)
            tmpimg.save(fp=de_edge_file)

    iou=None
    from src.pix2pixHD.eval_iou import label_accuracy_score
    _,_,iou,_,_=label_accuracy_score(label_targets, label_preds, n_class)

    model.train()
    return iou

def label_nums(data_loader,label_num=5): # 遍历dataloader，计算其所有图像中分割GT各label的pix总数
    ret=[]
    edge_num=[0,0]
    for i in range(label_num):
        ret.append(0)
    for step, sample in enumerate(data_loader):
        seg=sample["seg"]
        for i in range(label_num):
            ret[i]+=(seg==i).sum().item()
        edge=get_edge_from_tensor(seg)
        edge_num[0]+=(edge==0).sum().item()
        edge_num[1] += (edge == 1).sum().item()
    return ret,edge_num

def get_edge_from_tensor(seg):
    seg_np = seg.numpy()
    edge_np = []
    for i in range(seg_np.shape[0]):
        edge_np.append(np.expand_dims(sobel_edge(seg_np[i]), axis=0))
    edge_np = np.concatenate(edge_np, axis=0)
    edge_np[edge_np != 0] = 1
    edge_tensor=torch.from_numpy(edge_np)
    return edge_tensor

def sobel_edge(img):# 接收seg 也就是label图
    sobelx = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=0)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(img, cv2.CV_64F, dx=0, dy=1)
    sobely = cv2.convertScaleAbs(sobely)
    result = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    result=np.array(result)
    return result

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

    data_loader_focal = get_dataloader_func(args, train=True)
    data_loader_focal = tqdm(data_loader_focal)
    alpha,alpha_edge = label_nums(data_loader_focal,label_num=args.label_nc)
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

    tmp_min = min(alpha_edge)
    assert tmp_min > 0
    for i in range(len(alpha_edge)):
        alpha_edge[i] = tmp_min / alpha_edge[i]
    print(alpha_edge)
    ERN_e_loss=FocalLoss(gamma=0,alpha=alpha_edge)
    ERN_d_loss =FocalLoss(gamma=0,alpha=alpha_edge)

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
        ERN_e_loss_list=[]
        ERN_d_loss_list=[]

        data_loader = get_dataloader_func(args, train=True)
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
            en_edge,de_edge=DLV3P.get_edges()
            # feature_map=feature_map.detach()

            # compute the loss:
            # ce_loss = CE_loss(outputs, label_imgs)
            # ce_loss_value = ce_loss.data.cpu().numpy()
            soft_outputs = torch.nn.functional.softmax(outputs, dim=1)
            lvs_loss = LVS_loss(soft_outputs, label_imgs, ignore=255)
            lvs_loss_value = lvs_loss.data.cpu().numpy()
            focal_loss=FOCAL_loss(outputs,label_imgs)
            focal_loss_value=focal_loss.data.cpu().numpy()
            edge=get_edge_from_tensor(label_imgs.type(torch.Tensor)).type(torch.LongTensor).to(device)
            ern_e_loss=ERN_e_loss(en_edge,edge)
            ern_e_loss_value=ern_e_loss.data.cpu().numpy()
            ern_d_loss = ERN_d_loss(de_edge, edge)
            ern_d_loss_value = ern_d_loss.data.cpu().numpy()

            a_loss=args.a_loss if args.a_loss else [1,1,1,1]
            seg_loss = a_loss[0]*focal_loss+a_loss[1]*lvs_loss+a_loss[2]*ern_e_loss+a_loss[3]*ern_d_loss

            # optimization step:
            DLV3P_global_optimizer.zero_grad()  # (reset gradients)
            DLV3P_backbone_optimizer.zero_grad()
            seg_loss.backward()  # (compute gradients)
            DLV3P_global_optimizer.step()  # (perform optimization step)
            DLV3P_backbone_optimizer.step()


            data_loader.write(f'Epochs:{epoch} | FOCAL_loss:{focal_loss_value*a_loss[0]:.6f}|LVS_loss:{lvs_loss_value*a_loss[1]:.6f} '
                              f'| ERN_e_loss:{ern_e_loss_value*a_loss[2]:.6f}| ERN_d_loss:{ern_d_loss_value*a_loss[3]:.6f}'
                              f'| lr_global:{get_lr(DLV3P_global_optimizer):.8f}| lr_backbone:{get_lr(DLV3P_backbone_optimizer):.8f}')

            # CE_loss_list.append(ce_loss_value)
            LVS_loss_list.append(lvs_loss_value)
            FOCAL_loss_list.append(focal_loss_value)
            ERN_e_loss_list.append(ern_e_loss_value)
            ERN_d_loss_list.append(ern_d_loss_value)

            # tensorboard log
            if args.tensorboard_log and step % args.tensorboard_log == 0:  # defalut is 5
                total_steps = epoch * len(data_loader) + step
                # sw.add_scalar('Loss/CE', ce_loss_value, total_steps)
                sw.add_scalar('Loss/LVS', lvs_loss_value, total_steps)
                sw.add_scalar('Loss/focal', focal_loss_value, total_steps)
                sw.add_scalar('Loss/ERN_e', ern_e_loss_value, total_steps)
                sw.add_scalar('Loss/ERN_d', ern_d_loss_value, total_steps)
                sw.add_scalar('LR/global_seg', get_lr(DLV3P_global_optimizer), total_steps)
                sw.add_scalar('LR/backbone_seg', get_lr(DLV3P_backbone_optimizer), total_steps)


                sw.add_image('img2/realA', tensor2im(imgs_show.data), total_steps, dataformats='HWC')
                tmpsegmap = pred2gray(outputs)
                tmpsegmap = tmpsegmap[0].data.numpy()
                tmpsegmap = gray2rgb(tmpsegmap)
                sw.add_image('img2/fake_segB', tmpsegmap, total_steps, dataformats='HWC')
                tmpsegmap = label_imgs[0].data.cpu().numpy()
                tmpsegmap = gray2rgb(tmpsegmap)
                sw.add_image('img2/real_segB', tmpsegmap, total_steps, dataformats='HWC')
                tmpedgemap=pred2gray(en_edge)
                tmpedgemap=tmpedgemap[0].data.cpu().numpy()
                tmpedgemap=gray2rgb(tmpedgemap,2,label_list=[[0,0,0],[255,255,255]])
                sw.add_image('img2/en_edge', tmpedgemap, total_steps, dataformats='HWC')
                tmpedgemap = pred2gray(de_edge)
                tmpedgemap = tmpedgemap[0].data.cpu().numpy()
                tmpedgemap = gray2rgb(tmpedgemap, 2, label_list=[[0, 0, 0], [255, 255, 255]])
                sw.add_image('img2/de_edge', tmpedgemap, total_steps, dataformats='HWC')
                tmpedgemap = edge[0].type(torch.Tensor).data.cpu().numpy()
                tmpedgemap = gray2rgb(tmpedgemap, 2, label_list=[[0, 0, 0], [255, 255, 255]])
                sw.add_image('img2/GT_edge', tmpedgemap, total_steps, dataformats='HWC')
                # sw.add_image('img2/realA_imgnet', tensor2im(imgs.data), total_steps, dataformats='HWC')
                sw.add_image('img2/realB', tensor2im(maps_show.data), total_steps, dataformats='HWC')

        DLV3P_global_scheduler.step()
        DLV3P_backbone_scheduler.step()
        if epoch % 10 == 0 or epoch == (args.epochs-1):
            import copy
            args2=copy.deepcopy(args)
            args2.batch_size=args.batch_size_eval
            iou = eval_iou(args2, model=DLV3P,data_loader=get_pix2pix_maps_dataloader(args2, train=False))
            logger.log(key='iou', data=iou)
            if iou < logger.get_max(key='FID'):
                model_saver.save(f'DLV3P_{iou:.4f}', DLV3P)
            sw.add_scalar('eval/iou', iou, epoch)

        # logger.log(key='CE_loss', data=sum(CE_loss_list) / float(len(CE_loss_list)))
        logger.log(key='LVS_loss', data=sum(LVS_loss_list) / float(len(LVS_loss_list)))
        logger.log(key='FOCAL_loss', data=sum(FOCAL_loss_list) / float(len(FOCAL_loss_list)))
        logger.log(key='ERN_e_loss', data=sum(ERN_e_loss_list) / float(len(ERN_e_loss_list)))
        logger.log(key='ERN_d_loss', data=sum(ERN_d_loss_list) / float(len(ERN_d_loss_list)))
        logger.save_log()
        # logger.visualize()


        model_saver.save('DLV3P', DLV3P)

        model_saver.save('DLV3P_global_optimizer', DLV3P_global_optimizer)
        model_saver.save('DLV3P_backbone_optimizer', DLV3P_backbone_optimizer)

        model_saver.save('DLV3P_global_scheduler', DLV3P_global_scheduler)
        model_saver.save('DLV3P_backbone_scheduler', DLV3P_backbone_scheduler)



if __name__ == '__main__':
    args = config()

    # args.label_nc = 5

    train(args, get_dataloader_func=get_pix2pix_maps_dataloader)

pass
