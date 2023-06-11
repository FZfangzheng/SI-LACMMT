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
import cv2
import torch.nn as nn
from src.utils.train_utils import model_accelerate, get_device, mean, get_lr
from src.pix2pixHD.train_config import config
from src.pix2pixHD.networks import get_G, get_D, get_E, get_C, get_G_consis_nolowlevel
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
from src.eval.eval_every10epoch import eval_epoch
from src.pix2pixHD.deeplabv3plus.deeplabv3plus import Configuration
from src.pix2pixHD.deeplabv3plus.deeplabv3plus.my_deeplabv3plus_featuremap import deeplabv3plus
import torch.nn.functional as F

from src.pix2pixHD.deeplabv3plus.lovasz_losses import lovasz_softmax

from util.util import tensor2im,tensor2im_seg  # 注意，该函数使用0.5与255恢复可视图像，所以如果是ImageNet标准化的可能会有色差？这里都显示试一下
from src.pix2pixHD.myutils import pred2gray, gray2rgb
from src.eval.eval_every10epoch import eval_epoch
from src.pix2pixHD.deeplabv3plus.focal_loss import FocalLoss
from evaluation.fid.fid_score import fid_score
import json
from src.SingleMap.swin_transformer.segmentors.encoder_decoder import EncoderDecoder as Swin
import itertools
def make_new_dir(newpath):
    if not osp.exists(newpath):
        os.mkdir(newpath)


def eval_fidiou(args, sw, model_G,model_seg,model_G_consis, data_loader,epoch=-1):
    device = get_device(args)
    data_loader = tqdm(data_loader)
    model_G.eval()
    model_seg.eval()
    model_G_consis.eval()
    model_G = model_G.to(device)
    model_seg = model_seg.to(device)
    model_G_consis = model_G_consis.to(device)

    label_preds = []
    label_targets = []

    real_seg_dir = osp.join(args.save, 'real_seg')
    real_dir = osp.join(args.save, 'real_result')
    A_dir = osp.join(args.save, 'real_source')
    seg_dir = osp.join(args.save, 'seg_result')
    fake_dir=osp.join(args.save, 'fake_result')
    fake1_dir = osp.join(args.save, 'fake1_result')
    fakec_dir = osp.join(args.save, 'fakec_result')
    eval_dir = osp.join(args.save, 'eval_result')
    create_dir(eval_dir)
    create_dir(real_dir)
    create_dir(real_seg_dir)
    create_dir(A_dir)
    create_dir(seg_dir)
    create_dir(fake_dir)
    create_dir(fake1_dir)
    create_dir(fakec_dir)

    for i, sample in enumerate(data_loader):
        inputs, labels = sample['A_seg'], sample['seg'].squeeze(dim=1)
        inputs = inputs.cuda() if args.gpu else inputs
        labels = labels.cuda() if args.gpu else labels
        imgs = sample['A'].to(device)

        outputs, feature_map = model_seg.forward_test(inputs)
        bs, n_class, h, w = outputs.shape
        outs = outputs.data.cpu().numpy()
        pred = outs.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(bs, h, w)
        target = labels.cpu().numpy().reshape(bs, h, w)
        label_preds.append(pred)
        label_targets.append(target)


        feature_map = feature_map.detach()
        imgs_plus=torch.cat((imgs,feature_map),1)
        fakes = model_G(imgs_plus).detach()

        # img_c = sample['C'].to(device)
        # img_c_seg = sample['C_seg'].to(device)
        # domain_label_c = sample['domain_c'].to(device)
        # domain_label_onehot_c = label2onehot(domain_label_c).to(device)
        # domain_c = domain_label_onehot_c.view(domain_label_onehot_c.size(0), domain_label_onehot_c.size(1), 1, 1)
        # domain_c = domain_c.repeat(1, 1, img_c.size(2), img_c.size(3))
        # domain_c = domain_c.type(torch.FloatTensor)
        # domain_c = domain_c.to(device)


        # outputs_c, feature_map_c = model_seg.forward_test(img_c_seg)
        # imgs_plus_c = torch.cat((img_c, feature_map_c, domain_c), 1)  # bs*(3+256)*h*w
        # fakes_c = model_G(imgs_plus_c).detach()
        fake_refined = model_G_consis(fakes).detach()
        # fake_refined = model_G_consis(fakes, fakes_c).detach()




        batch_size = inputs.size(0)
        im_name = sample['A_paths']
        for b in range(batch_size):
            file_name = osp.split(im_name[b])[-1].split('.')[0]
            new_real_dir = osp.join(real_dir, osp.split(im_name[b])[0].split(os.sep)[-1])
            new_real_seg_dir = osp.join(real_seg_dir, osp.split(im_name[b])[0].split(os.sep)[-1])
            new_A_dir = osp.join(A_dir, osp.split(im_name[b])[0].split(os.sep)[-1])
            new_seg_dir = osp.join(seg_dir, osp.split(im_name[b])[0].split(os.sep)[-1])
            new_fake_dir = osp.join(fake_dir, osp.split(im_name[b])[0].split(os.sep)[-1])
            new_fake1_dir = osp.join(fake1_dir, osp.split(im_name[b])[0].split(os.sep)[-1])
            new_fakec_dir = osp.join(fakec_dir, osp.split(im_name[b])[0].split(os.sep)[-1])
            make_new_dir(new_real_dir)
            make_new_dir(new_real_seg_dir)
            make_new_dir(new_A_dir)
            make_new_dir(new_seg_dir)
            make_new_dir(new_fake_dir)
            make_new_dir(new_fake1_dir)
            make_new_dir(new_fakec_dir)
            real_file = osp.join(new_real_dir, f'{file_name}.png')
            real_seg_file = osp.join(new_real_seg_dir, f'{file_name}.png')
            A_file = osp.join(new_A_dir, f'{file_name}.png')
            seg_file = osp.join(new_seg_dir, f'{file_name}.png')
            fake_file=osp.join(new_fake_dir, f'{file_name}.png')
            fake1_file = osp.join(new_fake1_dir, f'{file_name}.png')
            fakec_file = osp.join(new_fakec_dir, f'{file_name}.png')

            from_std_tensor_save_image(filename=real_file, data=sample['B'][b].cpu())
            from_std_tensor_save_image(filename=A_file, data=sample['A'][b].cpu())
            from_std_tensor_save_image(filename=fake_file, data=fake_refined[b].cpu())
            from_std_tensor_save_image(filename=fake1_file, data=fakes[b].cpu())
            # from_std_tensor_save_image(filename=fakec_file, data=fakes_c[b].cpu())

            tmpimg= sample['seg'][b].data.cpu().numpy()
            tmpimg = gray2rgb(tmpimg)
            tmpimg=Image.fromarray(tmpimg)
            tmpimg.save(fp=real_seg_file)

            tmpimg = gray2rgb(pred[b])
            tmpimg = Image.fromarray(tmpimg)
            tmpimg.save(fp=seg_file)
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
    real_paths = [osp.join(real_dir,"15"),osp.join(real_dir,"16"),osp.join(real_dir,"17"),osp.join(real_dir,"18")]
    fake_paths = [osp.join(fake_dir, "15"), osp.join(fake_dir, "16"), osp.join(fake_dir, "17"), osp.join(fake_dir, "18")]
    # real_paths = [osp.join(real_dir,"18")]
    # fake_paths = [osp.join(fake_dir, "18")]

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

    model_seg.train()
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

def label2onehot(labels, dim=4):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out


def classification_loss(logit, target, dataset='RaFD'):
    """Compute binary or softmax cross entropy loss."""
    if dataset == 'CelebA':
        return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
    elif dataset == 'RaFD':
        return F.cross_entropy(logit, target)



def adjust_learning_rate(optimizer, cur_iter, max_iters, lr_pow, set_lr, warmup_steps):

    warm_lr = 1e-6/4
    if cur_iter < warmup_steps:
        linear_step = set_lr - warm_lr
        lr = warm_lr + linear_step * (cur_iter / warmup_steps)
    else:
        scale_running_lr = ((1. - float(cur_iter) / max_iters) ** lr_pow)
        lr = set_lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def train(args, get_dataloader_func=get_pix2pix_maps_dataloader):
    with open(os.path.join(args.save,'args.json'), 'w') as f:
        json.dump(vars(args), f)
    logger = Logger(save_path=args.save, json_name='img2map_seg')
    epoch_now = len(logger.get_data('D_loss'))

    model_saver = ModelSaver(save_path=args.save,
                             name_list=['G', 'D','C','G_consis', 'G_optimizer', 'D_optimizer','C_optimizer','swin_optimizer',
                                        'G_scheduler', 'D_scheduler','C_scheduler', 'Swin_Transformer'])

    sw = SummaryWriter(args.tensorboard_path)


    G = get_G(args,input_nc=3+512) # 3+256+1，256为分割网络输出featuremap的通道数
    D = get_D(args,input_nc=6)

    G_consis = get_G_consis_nolowlevel(args,input_nc=3)
    C = get_C(args)
    model_saver.load('G', G)
    model_saver.load('G_consis', G_consis)
    model_saver.load('D', D)
    model_saver.load('C', C)

    cfg=Configuration()
    cfg.MODEL_NUM_CLASSES=args.label_nc

    swin_transformer = Swin('swin_base_patch4_window7_224.pth')
    print(swin_transformer)
    # DLV3P = deeplabv3plus(cfg)
    if args.gpu:
        # DLV3P=nn.DataParallel(DLV3P)
        swin_transformer=swin_transformer.cuda()
    model_saver.load('Swin_Transformer', swin_transformer)

    G_optimizer = Adam(itertools.chain(G.parameters(), G_consis.parameters()), lr=args.G_lr, betas=(args.beta1, 0.999))
    D_optimizer = Adam(D.parameters(), lr=args.D_lr, betas=(args.beta1, 0.999))
    C_optimizer = Adam(C.parameters(), lr=args.C_lr, betas=(args.beta1, 0.999))

    #获取优化参数
    nodecay_params = []
    other_params = []
    for pname, p in swin_transformer.named_parameters():
        if 'absolute_pos_embed' in pname or 'relative_position_bias_table' in pname or 'norm' in pname:
            # print(pname)
            nodecay_params += [p]
        else:
            other_params += [p]
    optim_p = [{'params': other_params},
              {'params': nodecay_params, 'weight_decay': 0. }]

    #add to optim
    swin_optimizer = torch.optim.AdamW(optim_p,lr=0.00006/4,betas=(0.9, 0.999),weight_decay=0.01)

    model_saver.load('G_optimizer', G_optimizer)
    model_saver.load('D_optimizer', D_optimizer)
    model_saver.load('C_optimizer', C_optimizer)
    model_saver.load('swin_optimizer', swin_optimizer)


    G_scheduler = get_hinge_scheduler(args, G_optimizer)
    D_scheduler = get_hinge_scheduler(args, D_optimizer)
    C_scheduler = get_hinge_scheduler(args, C_optimizer)

    model_saver.load('G_scheduler', G_scheduler)
    model_saver.load('D_scheduler', D_scheduler)
    model_saver.load('C_scheduler', C_scheduler)


    device = get_device(args)

    GANLoss = get_GANLoss(args)
    if args.use_ganFeat_loss:
        DFLoss = get_DFLoss(args)
    if args.use_vgg_loss:
        VGGLoss = get_VGGLoss(args)
    if args.use_low_level_loss:
        LLLoss = get_low_level_loss(args)


    data_loader = get_dataloader_func(args, train=True, flag=0)
    datasize = len(data_loader)
    max_iters = datasize * args.epochs

    if epoch_now==args.epochs or args.epochs==-1:
        print('get final models')
        eval_fidiou(args,sw=sw,model_G=G,model_G_consis=G_consis, model_seg=swin_transformer,data_loader=get_pix2pix_maps_dataloader(args, train=False),epoch=epoch_now)


    total_steps = datasize*epoch_now
    for epoch in range(epoch_now, args.epochs):
        G_loss_list = []
        D_loss_list = []


        data_loader = get_dataloader_func(args, train=True, flag=0)


        data_loader = tqdm(data_loader)

        for step, sample in enumerate(data_loader):


            #########stage1
            total_steps = total_steps + 1
            # 先训练deeplabv3+
            imgs_seg = sample['A_seg'].to(device)  # (shape: (batch_size, 3, img_h, img_w))
            label_imgs = sample['seg'].type(torch.LongTensor).to(device)  # (shape: (batch_size, img_h, img_w))
            label_imgs_unsq = torch.unsqueeze(label_imgs, 1)

            cur_iter = total_steps


            swin_lr = adjust_learning_rate(swin_optimizer, cur_iter, max_iters, lr_pow=1, set_lr=0.00006 / 2,
                                           warmup_steps=1500)

            swinloss, outputs, feature_map = swin_transformer.forward_train(imgs_seg, label_imgs_unsq)

            seg_loss = swinloss['decode.loss_seg'] + swinloss['aux.loss_seg']

            # 然后训练GAN
            imgs = sample['A'].to(device)
            maps = sample['B'].to(device)

            #

            imgs_plus_old = torch.cat((imgs,feature_map),1) # bs*(3+256)*h*w
            # id_layer = torch.from_numpy(np.array(id_layer_np)).to(device)

            imgs_plus = imgs_plus_old
            # train the Discriminator
            # D_optimizer.zero_grad()

            reals_maps = torch.cat([imgs.float(), maps.float()], dim=1)

            fakes = G(imgs_plus)
            fakes_detach = fakes.detach()
            fakes_maps = torch.cat([imgs.float(), fakes_detach.float()], dim=1)



            D_real_outs = D(reals_maps)
            D_real_loss = GANLoss(D_real_outs, True)

            D_fake_outs = D(fakes_maps)
            D_fake_loss = GANLoss(D_fake_outs, False)


            #########stage2

            fake_refined = G_consis(fakes)
            fake_refined_detach = fake_refined.detach()
            fakes_maps_refined = torch.cat([imgs.float(), fake_refined_detach.float()], dim=1)


            D_fake_outs_refined = D(fakes_maps_refined)
            D_fake_loss_refined = GANLoss(D_fake_outs_refined, False)
            D_loss = 0.5 * (5*D_real_loss + D_fake_loss + 4*D_fake_loss_refined)
            # D_loss = 0.5 * (D_real_loss + D_fake_loss)
            D_loss = D_loss.mean()
            D_real_loss_item = D_real_loss.item()
            D_fake_loss_item = D_fake_loss.item()
            D_fake_loss_refined_item = D_fake_loss_refined.item()


            #backward D
            D_optimizer.zero_grad()
            D_loss.backward()
            D_loss = D_loss.item()
            D_optimizer.step()




            ############################################

            fakes_maps1 = torch.cat([imgs.float(), fakes.float()], dim=1)
            D_fake_outs1 = D(fakes_maps1)
            gan_loss = GANLoss(D_fake_outs1, True)
            G_loss = 0
            G_loss += gan_loss
            gan_loss = gan_loss.mean().item()
            if args.use_vgg_loss:
                vgg_loss = VGGLoss(fakes, maps)
                G_loss += args.lambda_feat_vgg * vgg_loss
                vgg_loss = vgg_loss.mean().item()
            else:
                vgg_loss = 0.

            if args.use_ganFeat_loss:
                df_loss = DFLoss(D_fake_outs1, D_real_outs)
                G_loss += args.lambda_feat_df * df_loss
                df_loss = df_loss.mean().item()
            else:
                df_loss = 0.

            if args.use_low_level_loss:
                ll_loss = LLLoss(fakes, maps)
                G_loss += args.lambda_feat_ll * ll_loss
                ll_loss = ll_loss.mean().item()
            else:
                ll_loss = 0.

            G_loss = G_loss.mean()
            G_seg_loss = G_loss + seg_loss
            # G_loss = G_loss.item()
            seg_loss = seg_loss.item()

            fakes_maps_refined2 = torch.cat([imgs.float(), fake_refined.float()], dim=1)
            D_fake_outs_refined2 = D(fakes_maps_refined2)
            gan_loss_refined = GANLoss(D_fake_outs_refined2, True)
            gan_loss_r = gan_loss_refined.item()
            G_loss_refined = 0
            G_loss_refined += gan_loss_refined

            if args.use_vgg_loss:
                vgg_loss_r = VGGLoss(fake_refined, maps)
                G_loss_refined += args.lambda_feat_vgg * vgg_loss_r
                vgg_loss_r = vgg_loss_r.mean().item()
            else:
                vgg_loss_r = 0.

            if args.use_ganFeat_loss:
                df_loss_r = DFLoss(D_fake_outs_refined2, D_real_outs)
                G_loss_refined += args.lambda_feat_df * df_loss_r
                df_loss_r = df_loss_r.mean().item()
            else:
                df_loss_r = 0.

            if args.use_low_level_loss:
                ll_loss_r = LLLoss(fake_refined, maps)
                G_loss_refined += args.lambda_feat_ll * ll_loss_r
                ll_loss_r = ll_loss_r.mean().item()
            else:
                ll_loss_r = 0.

            G_seg_loss_all = G_seg_loss+G_loss_refined
            # G_seg_loss_all = G_seg_loss
            # G_seg_loss_all = G_loss_refined
            G_seg_loss_all_itme = G_seg_loss_all.item()
            G_seg_loss_item = G_seg_loss.item()
            G_loss_refined_item = G_loss_refined.item()

            # backward G
            G_optimizer.zero_grad()
            # if epoch >= args.start_train_C_epoch:
            #     C_optimizer.zero_grad()
            swin_optimizer.zero_grad()
            G_seg_loss_all.backward()
            # if epoch >= args.start_train_C_epoch:
            #     C_optimizer.step()
            G_optimizer.step()
            swin_optimizer.step()


            data_loader.write(f'Epochs:{epoch}  | Dloss:{D_loss:.6f} | Gloss:{G_loss:.6f}'
                              f'| GANloss:{gan_loss:.6f} | VGGloss:{vgg_loss:.6f} | DFloss:{df_loss:.6f} '
                              f'| LLloss:{ll_loss:.6f} | lr_gan:{get_lr(G_optimizer):.8f}')


            G_loss_list.append(G_loss)
            D_loss_list.append(D_loss)


            decode_loss_seg = swinloss['decode.loss_seg'].item()
            decode_acc_seg = swinloss['decode.acc_seg'].item()
            aux_loss_seg = swinloss['aux.loss_seg'].item()
            aux_acc_seg = swinloss['aux.acc_seg'].item()
            # tensorboard log
            if args.tensorboard_log and step % args.tensorboard_log == 0:  # defalut is 5
                # total_steps = epoch * len(data_loader) + step
                sw.add_scalar('Loss_swin/decode.loss_seg', decode_loss_seg, total_steps)
                sw.add_scalar('Loss_swin/aux.loss_seg', aux_loss_seg, total_steps)
                sw.add_scalar('Loss_swin/decode.acc_seg', decode_acc_seg, total_steps)
                sw.add_scalar('Loss_swin/aux.acc_seg', aux_acc_seg, total_steps)


                sw.add_scalar('Loss1/G_seg_loss_item', G_seg_loss_item, total_steps)
                sw.add_scalar('Loss1/G_loss_refined_item', G_loss_refined_item, total_steps)

                sw.add_scalar('Loss_all/G', G_seg_loss_all_itme, total_steps)
                sw.add_scalar('Loss_all/D', D_loss, total_steps)
                sw.add_scalar('Loss/D_real_loss_item', D_real_loss_item, total_steps)
                sw.add_scalar('Loss/D_fake_loss_item', D_fake_loss_item, total_steps)
                sw.add_scalar('Loss/D_fake_loss_refined_item', D_fake_loss_refined_item, total_steps)
                sw.add_scalar('Loss/seg', seg_loss, total_steps)
                sw.add_scalar('Loss/gan', gan_loss, total_steps)
                sw.add_scalar('Loss/vgg', vgg_loss, total_steps)
                sw.add_scalar('Loss/df', df_loss, total_steps)
                sw.add_scalar('Loss/ll', ll_loss, total_steps)
                sw.add_scalar('Loss/gan_r', gan_loss_r, total_steps)
                sw.add_scalar('Loss/vgg_r', vgg_loss_r, total_steps)
                sw.add_scalar('Loss/df_r', df_loss_r, total_steps)
                sw.add_scalar('Loss/ll_r', ll_loss_r, total_steps)
                sw.add_scalar('LR/G', get_lr(G_optimizer), total_steps)
                sw.add_scalar('LR/D', get_lr(D_optimizer), total_steps)
                sw.add_scalar('LR/swin', swin_lr, total_steps)


                sw.add_image('img2/realA', tensor2im(imgs.data), total_steps, dataformats='HWC')
                sw.add_image('img2/fakeB', tensor2im(fakes.data), total_steps, dataformats='HWC')
                sw.add_image('img2/realB', tensor2im(maps.data), total_steps, dataformats='HWC')
                # sw.add_image('img2/fakeC', tensor2im(fakes_c.data), total_steps, dataformats='HWC')
                sw.add_image('img2/fake_refined', tensor2im(fake_refined.data), total_steps, dataformats='HWC')

                tmpsegmap = pred2gray(outputs)
                tmpsegmap = tmpsegmap[0].data.numpy()
                tmpsegmap = gray2rgb(tmpsegmap)
                sw.add_image('img2/fake_segB', tmpsegmap, total_steps, dataformats='HWC')
                tmpsegmap = label_imgs[0].data.cpu().numpy()
                tmpsegmap = gray2rgb(tmpsegmap)
                sw.add_image('img2/real_segB', tmpsegmap, total_steps, dataformats='HWC')



        D_scheduler.step(epoch)
        G_scheduler.step(epoch)
        C_scheduler.step(epoch)



        logger.log(key='D_loss', data=sum(D_loss_list) / float(len(D_loss_list)))
        # logger.log(key='G_loss', data=sum(G_loss_list) / float(len(G_loss_list)))

        logger.save_log()


        model_saver.save('G', G)
        model_saver.save('G_consis', G_consis)
        model_saver.save('D', D)
        model_saver.save('C', C)
        # model_saver.save('DLV3P', DLV3P)
        model_saver.save('Swin_Transformer', swin_transformer)

        model_saver.save('G_optimizer', G_optimizer)
        model_saver.save('D_optimizer', D_optimizer)
        model_saver.save('C_optimizer', C_optimizer)
        model_saver.save('swin_optimizer', swin_optimizer)
        # model_saver.save('DLV3P_global_optimizer', DLV3P_global_optimizer)
        # model_saver.save('DLV3P_backbone_optimizer', DLV3P_backbone_optimizer)

        model_saver.save('G_scheduler', G_scheduler)
        model_saver.save('D_scheduler', D_scheduler)
        model_saver.save('C_scheduler', C_scheduler)
        # model_saver.save('DLV3P_global_scheduler', DLV3P_global_scheduler)
        # model_saver.save('DLV3P_backbone_scheduler', DLV3P_backbone_scheduler)
        if epoch >=args.epochs-5:
            model_saver.save('G_'+str(epoch), G)
            model_saver.save('G_consis' + str(epoch), G_consis)
            model_saver.save('Swin_Transformer_'+str(epoch), swin_transformer)
        if epoch == (args.epochs-1) or (epoch%100==0 and epoch!=0):
            import copy
            args2=copy.deepcopy(args)
            args2.batch_size=args.batch_size_eval
            eval_fidiou(args,sw=sw,model_G=G, model_seg=swin_transformer,model_G_consis=G_consis,data_loader=get_pix2pix_maps_dataloader(args2, train=False),epoch=epoch)



if __name__ == '__main__':
    args = config()

    # args.label_nc = 5

    from src.pix2pixHD.myutils import seed_torch
    print(f'\nset seed as {args.seed}!\n')
    seed_torch(args.seed)

    train(args, get_dataloader_func=get_pix2pix_maps_dataloader)

pass
