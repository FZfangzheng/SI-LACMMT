
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
from src.pix2pixHD.networks import get_G, get_D, get_E, get_C, get_G_consis_inputfpn
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
from src.LACMMT.swin_transformer.segmentors.encoder_decoder import EncoderDecoder as Swin
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
    create_dir(args.result)
    real_seg_dir = osp.join(args.result, 'real_seg')
    real_dir = osp.join(args.result, 'real_result')
    A_dir = osp.join(args.result, 'real_source')
    seg_dir = osp.join(args.result, 'seg_result')
    fake_dir=osp.join(args.result, 'fake_result')
    fake1_dir = osp.join(args.result, 'fake1_result')
    fakec_dir = osp.join(args.result, 'fakec_result')
    eval_dir = osp.join(args.result, 'eval_result')
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
        domain_label = sample['domain'].to(device)
        domain_label_onehot = label2onehot(domain_label).to(device)
        domain = domain_label_onehot.view(domain_label_onehot.size(0), domain_label_onehot.size(1), 1, 1)
        domain = domain.repeat(1, 1, imgs.size(2), imgs.size(3))
        domain = domain.type(torch.FloatTensor)
        domain = domain.to(device)
        outputs, feature_map = model_seg.forward_test(inputs)
        bs, n_class, h, w = outputs.shape
        outs = outputs.data.cpu().numpy()
        pred = outs.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(bs, h, w)
        target = labels.cpu().numpy().reshape(bs, h, w)
        label_preds.append(pred)
        label_targets.append(target)


        feature_map = feature_map.detach()
        imgs_plus=torch.cat((imgs,feature_map,domain),1)
        fakes = model_G(imgs_plus).detach()

        img_c = sample['C'].to(device)
        img_c_seg = sample['C_seg'].to(device)
        domain_label_c = sample['domain_c'].to(device)
        domain_label_onehot_c = label2onehot(domain_label_c).to(device)
        domain_c = domain_label_onehot_c.view(domain_label_onehot_c.size(0), domain_label_onehot_c.size(1), 1, 1)
        domain_c = domain_c.repeat(1, 1, img_c.size(2), img_c.size(3))
        domain_c = domain_c.type(torch.FloatTensor)
        domain_c = domain_c.to(device)


        outputs_c, feature_map_c = model_seg.forward_test(img_c_seg)
        imgs_plus_c = torch.cat((img_c, feature_map_c, domain_c), 1)  # bs*(3+256)*h*w
        fakes_c = model_G(imgs_plus_c).detach()
        fake_refined = model_G_consis(fakes, fakes_c).detach()




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
            from_std_tensor_save_image(filename=fakec_file, data=fakes_c[b].cpu())

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

def test(args, get_dataloader_func=get_pix2pix_maps_dataloader):
    with open(os.path.join(args.save,'args.json'), 'w') as f:
        json.dump(vars(args), f)
    logger = Logger(save_path=args.save, json_name='img2map_seg')
    epoch_now = len(logger.get_data('D_loss'))

    model_saver = ModelSaver(save_path=args.save,
                             name_list=['G', 'D','C','G_consis', 'G_optimizer', 'D_optimizer','C_optimizer','swin_optimizer',
                                        'G_scheduler', 'D_scheduler','C_scheduler', 'Swin_Transformer'])

    sw = SummaryWriter(args.tensorboard_path)


    G = get_G(args,input_nc=3+512+4) # 3+256+1，256为分割网络输出featuremap的通道数
    D = get_D(args,input_nc=6)

    # G_consis = get_G_consis_nolowlevel(args,input_nc=3)
    G_consis = get_G_consis_inputfpn(args, input_nc=3)
    C = get_C(args)
    print("Now  C jiegou")
    print(C)
    model_saver.load('G', G)
    model_saver.load('G_consis', G_consis)
    model_saver.load('D', D)
    model_saver.load('C', C)

    cfg=Configuration()
    cfg.MODEL_NUM_CLASSES=args.label_nc

    swin_transformer = Swin('./src/LACMMT/swin_base_patch4_window7_224.pth')
    print("Now  swin_transformer jiegou")
    print(swin_transformer)
    # DLV3P = deeplabv3plus(cfg)
    if args.gpu:
        # DLV3P=nn.DataParallel(DLV3P)
        swin_transformer=swin_transformer.cuda()
    model_saver.load('Swin_Transformer', swin_transformer)


    data_loader = get_dataloader_func(args, train=True, flag=14)
    datasize = len(data_loader)
    max_iters = datasize * args.epochs


    eval_fidiou(args,sw=sw,model_G=G,model_G_consis=G_consis, model_seg=swin_transformer,data_loader=get_pix2pix_maps_dataloader(args, train=False, flag=14),epoch=epoch_now)


   


if __name__ == '__main__':
    args = config()

    # args.label_nc = 5

    from src.pix2pixHD.myutils import seed_torch
    print(f'\nset seed as {args.seed}!\n')
    seed_torch(args.seed)

    test(args, get_dataloader_func=get_pix2pix_maps_dataloader)

pass
