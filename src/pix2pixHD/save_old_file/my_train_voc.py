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
from src.dataset.voc.dataset import prepare_for_train_dataloader,prepare_for_val_dataloader
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

# from src.pix2pixHD.deeplabv3.model.deeplabv3 import DeepLabV3
from src.pix2pixHD.deeplabv3plus.deeplabv3plus import Configuration
from src.pix2pixHD.deeplabv3plus.deeplabv3plus.deeplabv3plus_seg import deeplabv3plus
import torch.nn.functional as F


def eval_voc(args, model, data_loader):
    device = get_device(args)
    data_loader = tqdm(data_loader)
    model.eval()
    model = model.to(device)

    label_preds = []
    label_targets = []

    for i, sample in enumerate(data_loader):
        inputs, labels = sample['image'], sample['label'].squeeze(dim=1)
        bs_test = inputs.shape[1]
        inputs = inputs.cuda() if args.gpu else inputs
        labels = labels.cuda() if args.gpu else labels

        outputs, feature_map = model(inputs)
        bs, n_class, h, w = outputs.shape
        outs = outputs.data.cpu().numpy()
        pred = outs.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(bs, h, w)
        target = labels.cpu().numpy().reshape(bs, h, w)
        label_preds.append(pred)
        label_targets.append(target)

    iou=None
    from src.pix2pixHD.eval_iou import label_accuracy_score
    _,_,iou,_,_=label_accuracy_score(label_targets, label_preds, n_class)

    model.train()
    return iou


def train(args, data_loader):
    logger = Logger(save_path=args.save, json_name='voc_seg')
    epoch_now = len(logger.get_data('CE_loss'))

    model_saver = ModelSaver(save_path=args.save,
                             name_list=['DLV3P',"DLV3P_global_optimizer",
                                        "DLV3P_backbone_optimizer","DLV3P_global_scheduler","DLV3P_backbone_scheduler"])

    sw = SummaryWriter(args.tensorboard_path)


    cfg=Configuration()
    cfg.MODEL_NUM_CLASSES=21
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

    DLV3P_loss=nn.CrossEntropyLoss(ignore_index=255)

    for epoch in range(epoch_now, args.epochs):
        CE_loss_list = []

        data_loader = tqdm(data_loader)

        for step, sample in enumerate(data_loader):
            # 先训练deeplabv3+
            imgs = sample['image'].to(device)  # (shape: (batch_size, 3, img_h, img_w))
            label_imgs = sample['label'].squeeze(dim=1).type(torch.LongTensor).to(device)  # (shape: (batch_size, img_h, img_w))

            outputs,feature_map = DLV3P(imgs)  # (shape: (batch_size, num_classes, img_h, img_w))
            # feature_map=feature_map.detach()

            # compute the loss:
            seg_loss = DLV3P_loss(outputs, label_imgs)
            seg_loss_value = seg_loss.data.cpu().numpy()

            # optimization step:
            DLV3P_global_optimizer.zero_grad()  # (reset gradients)
            DLV3P_backbone_optimizer.zero_grad()
            seg_loss.backward()  # (compute gradients)
            DLV3P_global_optimizer.step()  # (perform optimization step)
            DLV3P_backbone_optimizer.step()


            data_loader.write(f'Epochs:{epoch} | CEloss:{seg_loss_value:.6f}'
                              f'| lr_global:{get_lr(DLV3P_global_optimizer):.8f}| lr_backbone:{get_lr(DLV3P_backbone_optimizer):.8f}')

            CE_loss_list.append(seg_loss_value)


            # tensorboard log
            if args.tensorboard_log and step % args.tensorboard_log == 0:
                total_steps = epoch * len(data_loader) + step
                sw.add_scalar('Loss/CE', seg_loss_value, total_steps)

                sw.add_scalar('LR/global_seg', get_lr(DLV3P_global_optimizer), total_steps)
                sw.add_scalar('LR/backbone_seg', get_lr(DLV3P_backbone_optimizer), total_steps)

        DLV3P_global_scheduler.step()
        DLV3P_backbone_scheduler.step()
        if epoch % 10 == 0 or epoch == (args.epochs-1):
            iou = eval_voc(args, model=DLV3P,data_loader=prepare_for_val_dataloader(args.dataroot,bs_val=2))
            logger.log(key='iou', data=iou)
            if iou < logger.get_max(key='FID'):
                model_saver.save(f'DLV3P_{iou:.4f}', DLV3P)
            sw.add_scalar('fid/iou', iou, epoch)

        logger.log(key='CE_loss', data=sum(CE_loss_list) / float(len(CE_loss_list)))
        logger.save_log()
        # logger.visualize()


        model_saver.save('DLV3P', DLV3P)

        model_saver.save('DLV3P_global_optimizer', DLV3P_global_optimizer)
        model_saver.save('DLV3P_backbone_optimizer', DLV3P_backbone_optimizer)

        model_saver.save('DLV3P_global_scheduler', DLV3P_global_scheduler)
        model_saver.save('DLV3P_backbone_scheduler', DLV3P_backbone_scheduler)



if __name__ == '__main__':
    args = config()

    args.label_nc = 21

    train(args, data_loader=prepare_for_train_dataloader(args.dataroot,bs_train=args.batch_size))

pass
