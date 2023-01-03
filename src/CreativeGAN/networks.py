import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from src.utils.train_utils import get_device, model_accelerate
import math
import torch.nn.functional as F
from torch.nn import init
###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class UnetGenerator_a(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(UnetGenerator_a, self).__init__()
        # self.deconvolution_1 = torch.nn.ConvTranspose2d(1024, 104, kernel_size=2, stride=2)
        self.convolution_1 = torch.nn.Conv2d(1024, 104, kernel_size=3, stride=1,padding=1)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 1))
        self.pool2 = nn.AvgPool2d(kernel_size=(4, 4))
        self.pool3 = nn.AvgPool2d(kernel_size=(9, 9))

        self.model_attention = nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=1, padding=0,
                                         bias=nn.InstanceNorm2d)
        self.model_image = nn.Conv2d(input_nc, output_nc * 3, kernel_size=3, stride=1, padding=1,
                                     bias=nn.InstanceNorm2d)
        self.conv330 = nn.Conv2d(330, 110, kernel_size=3, stride=1, padding=1, bias=nn.InstanceNorm2d)
        self.conv440 = nn.Conv2d(440, 110, kernel_size=3, stride=1, padding=1, bias=nn.InstanceNorm2d)

        self.convolution_for_attention = torch.nn.Conv2d(10, 1, 1, stride=1, padding=0)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, feature_combine, image_combine):
        output_feature = self.relu(self.convolution_1(feature_combine))  # feature_combine: 136, output_feature: 104
        feature_image_combine = torch.cat((output_feature, image_combine),
                                          1)  # image_combine: 6; feature_image_combine: 110

        pool_feature1 = self.pool1(feature_image_combine)
        pool_feature2 = self.pool2(feature_image_combine)
        pool_feature3 = self.pool3(feature_image_combine)

        # change all (256, 256) to (64, 64) if your image size is 64*64
        pool_feature1_up = F.upsample(input=pool_feature1, size=(256, 256), mode='bilinear', align_corners=True)
        pool_feature2_up = F.upsample(input=pool_feature2, size=(256, 256), mode='bilinear', align_corners=True)
        pool_feature3_up = F.upsample(input=pool_feature3, size=(256, 256), mode='bilinear', align_corners=True)

        f1 = feature_image_combine * pool_feature1_up
        f2 = feature_image_combine * pool_feature2_up
        f3 = feature_image_combine * pool_feature3_up

        feature_image_combine = torch.cat((f1, f2, f3, feature_image_combine), 1)  # feature_image_combine: 440
        feature_image_combine = self.conv440(feature_image_combine)  # feature_image_combine: 110

        attention = self.model_attention(feature_image_combine)  # attention: 10
        image = self.model_image(feature_image_combine)  # image: 30

        softmax_ = torch.nn.Softmax(dim=1)
        attention = softmax_(attention)

        attention1_ = attention[:, 0:1, :, :]
        attention2_ = attention[:, 1:2, :, :]
        attention3_ = attention[:, 2:3, :, :]
        attention4_ = attention[:, 3:4, :, :]
        attention5_ = attention[:, 4:5, :, :]
        attention6_ = attention[:, 5:6, :, :]
        attention7_ = attention[:, 6:7, :, :]
        attention8_ = attention[:, 7:8, :, :]
        attention9_ = attention[:, 8:9, :, :]
        attention10_ = attention[:, 9:10, :, :]

        attention1 = attention1_.repeat(1, 3, 1, 1)
        attention2 = attention2_.repeat(1, 3, 1, 1)
        attention3 = attention3_.repeat(1, 3, 1, 1)
        attention4 = attention4_.repeat(1, 3, 1, 1)
        attention5 = attention5_.repeat(1, 3, 1, 1)
        attention6 = attention6_.repeat(1, 3, 1, 1)
        attention7 = attention7_.repeat(1, 3, 1, 1)
        attention8 = attention8_.repeat(1, 3, 1, 1)
        attention9 = attention9_.repeat(1, 3, 1, 1)
        attention10 = attention10_.repeat(1, 3, 1, 1)

        image = self.tanh(image)

        image1 = image[:, 0:3, :, :]
        image2 = image[:, 3:6, :, :]
        image3 = image[:, 6:9, :, :]
        image4 = image[:, 9:12, :, :]
        image5 = image[:, 12:15, :, :]
        image6 = image[:, 15:18, :, :]
        image7 = image[:, 18:21, :, :]
        image8 = image[:, 21:24, :, :]
        image9 = image[:, 24:27, :, :]
        image10 = image[:, 27:30, :, :]

        output1 = image1 * attention1
        output2 = image2 * attention2
        output3 = image3 * attention3
        output4 = image4 * attention4
        output5 = image5 * attention5
        output6 = image6 * attention6
        output7 = image7 * attention7
        output8 = image8 * attention8
        output9 = image9 * attention9
        output10 = image10 * attention10

        output11 = output1 + output2 + output3 + output4 + output5 + output6 + output7 + output8 + output9 + output10

        ##uncertainty map generation
        # sigmoid_ = torch.nn.Sigmoid()
        # uncertainty = self.convolution_for_attention(attention)
        #
        # uncertainty = sigmoid_(uncertainty)
        # uncertainty_map = uncertainty.repeat(1, 3, 1, 1)

        return output11

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net

def define_Ga(input_nc, output_nc, ngf, which_model_netG='unet_256', norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG == 'unet_128':
        netG = UnetGenerator_a(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator_a(input_nc, output_nc)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, init_gain, gpu_ids)


def get_G_consis(args, input_nc=None):
    if input_nc is None:
        #input_nc = args.label_nc
        input_nc = args.input_nc
        if args.use_instance:
            input_nc += 1
        if args.feat_num > 0:
            input_nc += args.feat_num
    if args.if_mutil_layer:
        input_nc +=1
    norm_layer = get_norm_layer(norm_type=args.norm)


    netG = LocalEnhancer_consis(input_nc, args.output_nc, args.ngf, args.n_downsample_global, args.n_blocks_global,args.n_local_enhancers, args.n_blocks_local, norm_layer)
    print(netG)
    netG.apply(weights_init)
    netG = nn.DataParallel(netG).to(get_device(args))

    return netG

def get_G_consis_v3(args, input_nc=None):
    if input_nc is None:
        #input_nc = args.label_nc
        input_nc = args.input_nc
        if args.use_instance:
            input_nc += 1
        if args.feat_num > 0:
            input_nc += args.feat_num
    if args.if_mutil_layer:
        input_nc +=1
    norm_layer = get_norm_layer(norm_type=args.norm)


    netG = LocalEnhancer_consis_v3(input_nc, args.output_nc, args.ngf, args.n_downsample_global, args.n_blocks_global,args.n_local_enhancers, args.n_blocks_local, norm_layer)
    print(netG)
    netG.apply(weights_init)
    netG = nn.DataParallel(netG).to(get_device(args))

    return netG

def get_G_consis_inputfpn(args, input_nc=None):
    if input_nc is None:
        #input_nc = args.label_nc
        input_nc = args.input_nc
        if args.use_instance:
            input_nc += 1
        if args.feat_num > 0:
            input_nc += args.feat_num
    if args.if_mutil_layer:
        input_nc +=1
    norm_layer = get_norm_layer(norm_type=args.norm)


    # netG = LocalEnhancer_consis_nolowlevel(input_nc, args.output_nc, args.ngf, args.n_downsample_global, args.n_blocks_global,args.n_local_enhancers, args.n_blocks_local, norm_layer)
    netG = LocalEnhancer_consis_inputfpn(input_nc, args.output_nc, args.ngf, args.n_downsample_global, args.n_blocks_global,args.n_local_enhancers, args.n_blocks_local, norm_layer)
    print(netG)
    netG.apply(weights_init)
    netG = nn.DataParallel(netG).to(get_device(args))

    return netG

def get_G_consis_inputfpn_v2(args, input_nc=None):
    if input_nc is None:
        #input_nc = args.label_nc
        input_nc = args.input_nc
        if args.use_instance:
            input_nc += 1
        if args.feat_num > 0:
            input_nc += args.feat_num
    if args.if_mutil_layer:
        input_nc +=1
    norm_layer = get_norm_layer(norm_type=args.norm)


    # netG = LocalEnhancer_consis_nolowlevel(input_nc, args.output_nc, args.ngf, args.n_downsample_global, args.n_blocks_global,args.n_local_enhancers, args.n_blocks_local, norm_layer)
    netG = LocalEnhancer_consis_inputfpn_v2(input_nc, args.output_nc, args.ngf, args.n_downsample_global, args.n_blocks_global,args.n_local_enhancers, args.n_blocks_local, norm_layer)
    print(netG)
    netG.apply(weights_init)
    netG = nn.DataParallel(netG).to(get_device(args))

    return netG

def get_G_consis_inputcat(args, input_nc=None):
    if input_nc is None:
        #input_nc = args.label_nc
        input_nc = args.input_nc
        if args.use_instance:
            input_nc += 1
        if args.feat_num > 0:
            input_nc += args.feat_num
    if args.if_mutil_layer:
        input_nc +=1
    norm_layer = get_norm_layer(norm_type=args.norm)


    # netG = LocalEnhancer_consis_nolowlevel(input_nc, args.output_nc, args.ngf, args.n_downsample_global, args.n_blocks_global,args.n_local_enhancers, args.n_blocks_local, norm_layer)
    netG = LocalEnhancer_consis_inputcat(input_nc, args.output_nc, args.ngf, args.n_downsample_global, args.n_blocks_global,args.n_local_enhancers, args.n_blocks_local, norm_layer)
    print(netG)
    netG.apply(weights_init)
    netG = nn.DataParallel(netG).to(get_device(args))

    return netG


def get_G_consis_nolowlevel(args, input_nc=None):
    if input_nc is None:
        #input_nc = args.label_nc
        input_nc = args.input_nc
        if args.use_instance:
            input_nc += 1
        if args.feat_num > 0:
            input_nc += args.feat_num
    if args.if_mutil_layer:
        input_nc +=1
    norm_layer = get_norm_layer(norm_type=args.norm)


    netG = LocalEnhancer_consis_nolowlevel(input_nc, args.output_nc, args.ngf, args.n_downsample_global, args.n_blocks_global,args.n_local_enhancers, args.n_blocks_local, norm_layer)
    print(netG)
    netG.apply(weights_init)
    netG = nn.DataParallel(netG).to(get_device(args))

    return netG


def get_G_consis_concat(args, input_nc=None):
    if input_nc is None:
        #input_nc = args.label_nc
        input_nc = args.input_nc
        if args.use_instance:
            input_nc += 1
        if args.feat_num > 0:
            input_nc += args.feat_num
    if args.if_mutil_layer:
        input_nc +=1
    norm_layer = get_norm_layer(norm_type=args.norm)


    netG = LocalEnhancer_consis_concat(input_nc, args.output_nc, args.ngf, args.n_downsample_global, args.n_blocks_global,args.n_local_enhancers, args.n_blocks_local, norm_layer)
    print(netG)
    netG.apply(weights_init)
    netG = nn.DataParallel(netG).to(get_device(args))

    return netG


def get_G(args, input_nc=None):
    if input_nc is None:
        #input_nc = args.label_nc
        input_nc = args.input_nc
        if args.use_instance:
            input_nc += 1
        if args.feat_num > 0:
            input_nc += args.feat_num
    if args.if_mutil_layer:
        input_nc +=1
    norm_layer = get_norm_layer(norm_type=args.norm)
    if args.netG == 'global':
        netG = GlobalGenerator(input_nc, args.output_nc, args.ngf, args.n_downsample_global, args.n_blocks_global,
                               norm_layer)
    elif args.netG == 'local':
        netG = LocalEnhancer(input_nc, args.output_nc, args.ngf, args.n_downsample_global, args.n_blocks_global,
                             args.n_local_enhancers, args.n_blocks_local, norm_layer)
    elif args.netG == 'local_swin':
        netG = LocalEnhancer_Swin(input_nc, args.output_nc, args.ngf, args.n_downsample_global, args.n_blocks_global,
                             args.n_local_enhancers, args.n_blocks_local, norm_layer)
    elif args.netG == 'local_swin_leakrelu':
        netG = LocalEnhancer_Swin_leakrelu(input_nc, args.output_nc, args.ngf, args.n_downsample_global, args.n_blocks_global,
                             args.n_local_enhancers, args.n_blocks_local, norm_layer)
        # C是Local入口，A是Global入口，输出更像C而不是A，应该换下
    elif args.netG == 'consistency_G':
        netG = LocalEnhancer_consistency(input_nc, args.output_nc, args.ngf, args.n_downsample_global, args.n_blocks_global,
                             args.n_local_enhancers, args.n_blocks_local, norm_layer)
        # 三口输入，CD是Global入口，A是Local入口
    elif args.netG == 'consistency_G_v2':
        netG = LocalEnhancer_consistency_v2(input_nc, args.output_nc, args.ngf, args.n_downsample_global, args.n_blocks_global,
                             args.n_local_enhancers, args.n_blocks_local, norm_layer)
        #only15
    elif args.netG == 'consistency_G_v3':
        netG = LocalEnhancer_consistency_v3(input_nc, args.output_nc, args.ngf, args.n_downsample_global, args.n_blocks_global,
                             args.n_local_enhancers, args.n_blocks_local, norm_layer)
    elif args.netG == 'encoder':
        netG = Encoder(input_nc, args.output_nc, args.ngf, args.n_downsample_global, norm_layer)
    else:
        raise ('generator not implemented!')
    # print(netG)
    netG.apply(weights_init)
    netG = nn.DataParallel(netG).to(get_device(args))
    # netG = netG.cuda()
    return netG



def get_E(args):
    norm_layer = get_norm_layer(norm_type=args.norm)
    netE = Encoder(args.output_nc, args.feat_num, args.ngf, args.n_downsample_global, norm_layer)
    netE.apply(weights_init)
    print(netE)
    netE = nn.DataParallel(netE).to(get_device(args))
    return netE


def get_D(args, input_nc=None):
    if input_nc is None:
        #input_nc = args.label_nc + args.output_nc
        input_nc = args.input_nc + args.output_nc
        if args.use_instance:
            input_nc += 1
    if args.if_mutil_layer:
        input_nc +=1
    norm_layer = get_norm_layer(norm_type=args.norm)
    if args.netD == 'originD':
        netD = MultiscaleDiscriminator(input_nc, args.ndf, args.n_layers_D, norm_layer, args.use_lsgan, args.num_D,
                                   args.use_ganFeat_loss)
    elif args.netD == 'classD':
        netD = MultiscaleDiscriminator_CLASS(input_nc, args.ndf, args.n_layers_D, norm_layer, args.use_lsgan, args.num_D,
                                       args.use_ganFeat_loss)
    elif args.netD == 'classD_starganv1':
        netD = MultiscaleDiscriminator_CLASS_starganv1(input_nc, args.ndf, args.n_layers_D, norm_layer, args.use_lsgan, args.num_D,
                                       args.use_ganFeat_loss)
    else:
        raise ('discriminator not implemented!')
    # print(netD)
    netD.apply(weights_init)
    netD = nn.DataParallel(netD).to(get_device(args))
    # netD = netD.cuda()
    return netD

def get_C(args):
    netC = Classifier()
    netC.apply(weights_init)
    netC = nn.DataParallel(netC).to(get_device(args))
    return netC

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# module
##############################################################################
class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        # if self.upsample:
        #     x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        # if self.upsample:
        #     x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):

        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out

##############################################################################
# Generator
##############################################################################
class LocalEnhancer_Swin_leakrelu(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(LocalEnhancer_Swin_leakrelu, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####
        ngf_global = ngf * (2 ** n_local_enhancers)
        model_global_1 = GlobalGenerator_leak(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                       norm_layer).model_1
        model_global_2 = GlobalGenerator_leak(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                         norm_layer).model_2
        model_global_2 = [model_global_2[i] for i in
                        range(len(model_global_2) - 3)]  # get rid of final convolution layers
        # self.model = nn.Sequential(*model_global)
        self.model_1 = nn.Sequential(*model_global_1)
        self.model_2 = nn.Sequential(*model_global_2)
        # self.pre2= nn.Sequential(nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=1, padding=0),
        #                           norm_layer(ngf_global* 2), nn.ReLU(True))
        # self.mix2 = nn.Sequential(nn.Conv2d(ngf_global * 4, ngf_global * 2, kernel_size=3, padding=1),
        #                           norm_layer(ngf_global* 2), nn.ReLU(True),
        #                           nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=3, padding=1),
        #                           norm_layer(ngf_global * 2), nn.ReLU(True),
        #                           nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=3, padding=1),
        #                           norm_layer(ngf_global* 2), nn.ReLU(True))

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers + 1):
            ### downsample
            ngf_global = ngf * (2 ** (n_local_enhancers - n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                                norm_layer(ngf_global), nn.LeakyReLU(0.2),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                                norm_layer(ngf_global * 2), nn.LeakyReLU(0.2)]
            ### AdainResBlk
            # model_adainblk = []
            # for i in range(num_adainblk):
            #
            #     model_adainblk += [AdainResBlk(ngf_global * 2, ngf_global * 2, style_dim=64, w_hpf=0, upsample=False)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [
                nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(ngf_global), nn.LeakyReLU(0.2)]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                   nn.Tanh()]

            setattr(self, 'model' + str(n) + '_1', nn.Sequential(*model_downsample))#为对象self添加属性
            setattr(self, 'model' + str(n) + '_2', nn.Sequential(*model_upsample))

            # setattr(self, 'model' + str(n) + '_adain', nn.Sequential(*model_adainblk))
        self.num_adainblk = 6
        self.adain_blk = nn.ModuleList()
        for i in range(self.num_adainblk):
            self.adain_blk.insert(0, AdainResBlk(ngf_global * 2, ngf_global * 2, style_dim=64, w_hpf=0, upsample=False))
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        # self.pre3= nn.Sequential(nn.Conv2d(ngf_global * 4, ngf_global*2, kernel_size=1, padding=0),
        #                           norm_layer(ngf_global* 2), nn.ReLU(True))
        # self.mix3=nn.Sequential(nn.Conv2d(ngf_global*4, ngf_global*2, kernel_size=3, padding=1),
        #                         norm_layer(ngf_global*2), nn.ReLU(True),
        #                         nn.Conv2d(ngf_global*2, ngf_global*2, kernel_size=3, padding=1),
        #                         norm_layer(ngf_global*2), nn.ReLU(True),
        #                         nn.Conv2d(ngf_global*2, ngf_global*2, kernel_size=3, padding=1),
        #                         norm_layer(ngf_global*2), nn.ReLU(True))

    def forward(self, input,input2=None):  #input2: 小尺寸的高层特征   input3:大尺寸的低层特征
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        # output_prev = self.model(input_downsampled[-1])
        output_prev= self.model_1(input_downsampled[-1])
        output_prev=self.model_2(output_prev)
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = getattr(self, 'model' + str(n_local_enhancers) + '_1')
            model_upsample = getattr(self, 'model' + str(n_local_enhancers) + '_2')
            # model_adain = getattr(self, 'model' + str(n_local_enhancers) + '_adain')
            input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]
            tmp=model_downsample(input_i) + output_prev

            for block in self.adain_blk:

                tmp = block(tmp, input2)

            output_prev = model_upsample(tmp)

        return output_prev


class LocalEnhancer_Swin(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(LocalEnhancer_Swin, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####
        ngf_global = ngf * (2 ** n_local_enhancers)
        model_global_1 = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                       norm_layer).model_1
        model_global_2 = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                         norm_layer).model_2
        model_global_2 = [model_global_2[i] for i in
                        range(len(model_global_2) - 3)]  # get rid of final convolution layers
        # self.model = nn.Sequential(*model_global)
        self.model_1 = nn.Sequential(*model_global_1)
        self.model_2 = nn.Sequential(*model_global_2)
        self.pre2= nn.Sequential(nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=1, padding=0),
                                  norm_layer(ngf_global* 2), nn.ReLU(True))
        self.mix2 = nn.Sequential(nn.Conv2d(ngf_global * 4, ngf_global * 2, kernel_size=3, padding=1),
                                  norm_layer(ngf_global* 2), nn.ReLU(True),
                                  nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=3, padding=1),
                                  norm_layer(ngf_global * 2), nn.ReLU(True),
                                  nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=3, padding=1),
                                  norm_layer(ngf_global* 2), nn.ReLU(True))

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers + 1):
            ### downsample
            ngf_global = ngf * (2 ** (n_local_enhancers - n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### AdainResBlk
            # model_adainblk = []
            # for i in range(num_adainblk):
            #
            #     model_adainblk += [AdainResBlk(ngf_global * 2, ngf_global * 2, style_dim=64, w_hpf=0, upsample=False)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [
                nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(ngf_global), nn.ReLU(True)]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                   nn.Tanh()]

            setattr(self, 'model' + str(n) + '_1', nn.Sequential(*model_downsample))#为对象self添加属性
            setattr(self, 'model' + str(n) + '_2', nn.Sequential(*model_upsample))

            # setattr(self, 'model' + str(n) + '_adain', nn.Sequential(*model_adainblk))
        self.num_adainblk = 6
        self.adain_blk = nn.ModuleList()
        for i in range(self.num_adainblk):
            self.adain_blk.insert(0, AdainResBlk(ngf_global * 2, ngf_global * 2, style_dim=64, w_hpf=0, upsample=False))
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        self.pre3= nn.Sequential(nn.Conv2d(ngf_global * 4, ngf_global*2, kernel_size=1, padding=0),
                                  norm_layer(ngf_global* 2), nn.ReLU(True))
        self.mix3=nn.Sequential(nn.Conv2d(ngf_global*4, ngf_global*2, kernel_size=3, padding=1),
                                norm_layer(ngf_global*2), nn.ReLU(True),
                                nn.Conv2d(ngf_global*2, ngf_global*2, kernel_size=3, padding=1),
                                norm_layer(ngf_global*2), nn.ReLU(True),
                                nn.Conv2d(ngf_global*2, ngf_global*2, kernel_size=3, padding=1),
                                norm_layer(ngf_global*2), nn.ReLU(True))

    def forward(self, input,input2=None):  #input2: 小尺寸的高层特征   input3:大尺寸的低层特征
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        # output_prev = self.model(input_downsampled[-1])
        output_prev= self.model_1(input_downsampled[-1])
        output_prev=self.model_2(output_prev)
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = getattr(self, 'model' + str(n_local_enhancers) + '_1')
            model_upsample = getattr(self, 'model' + str(n_local_enhancers) + '_2')
            # model_adain = getattr(self, 'model' + str(n_local_enhancers) + '_adain')
            input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]
            tmp=model_downsample(input_i) + output_prev
            # print(input2)
            # print('tmp.shape', tmp.shape)
            # print('input2.shape', input2.shape)
            # print(model_adain)
            # print(model_upsample)
            # tmp = model_adain((tmp,input2))
            # num=0
            for block in self.adain_blk:
                # num+=1
                # print('Num:',num)
                # print('tmp.shape', tmp.shape)
                # print('input2.shape', input2.shape)
                tmp = block(tmp, input2)

            # print('after adain tmp.shape', tmp.shape)
            output_prev = model_upsample(tmp)

        # print('output_prev.shape',output_prev.shape)
        return output_prev

class LocalEnhancer_consistency_v2(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(LocalEnhancer_consistency_v2, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####

        ngf_global = ngf * (2 ** n_local_enhancers)
        model_global_1d = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                       norm_layer).model_1
        model_global_2d = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                         norm_layer).model_2
        model_global_2d = [model_global_2d[i] for i in
                        range(len(model_global_2d) - 3)]  # get rid of final convolution layers
        # self.model = nn.Sequential(*model_global)
        self.model_1d = nn.Sequential(*model_global_1d)
        self.model_2d = nn.Sequential(*model_global_2d)


        ngf_global = ngf * (2 ** n_local_enhancers)
        model_global_1c = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                       norm_layer).model_1
        model_global_2c = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                         norm_layer).model_2
        model_global_2c = [model_global_2c[i] for i in
                        range(len(model_global_2c) - 3)]  # get rid of final convolution layers
        # self.model = nn.Sequential(*model_global)
        self.model_1c = nn.Sequential(*model_global_1c)
        self.model_2c = nn.Sequential(*model_global_2c)
        self.pre2= nn.Sequential(nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=1, padding=0),
                                  norm_layer(ngf_global* 2), nn.ReLU(True))
        self.mix2 = nn.Sequential(nn.Conv2d(ngf_global * 4, ngf_global * 2, kernel_size=3, padding=1),
                                  norm_layer(ngf_global* 2), nn.ReLU(True),
                                  nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=3, padding=1),
                                  norm_layer(ngf_global * 2), nn.ReLU(True),
                                  nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=3, padding=1),
                                  norm_layer(ngf_global* 2), nn.ReLU(True))

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers + 1):
            ### downsample
            ngf_global = ngf * (2 ** (n_local_enhancers - n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(3, ngf_global, kernel_size=7, padding=0),
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=1, padding=1),
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            model_upsample += [
                nn.Conv2d(ngf_global * 2, ngf_global, kernel_size=3, stride=1, padding=1),
                norm_layer(ngf_global), nn.ReLU(True)]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                   nn.Tanh()]

            setattr(self, 'model' + str(n) + '_1', nn.Sequential(*model_downsample))#为对象self添加属性
            setattr(self, 'model' + str(n) + '_2', nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        self.pre3= nn.Sequential(nn.Conv2d(ngf_global * 4, ngf_global*2, kernel_size=1, padding=0),
                                  norm_layer(ngf_global* 2), nn.ReLU(True))
        self.mix3=nn.Sequential(nn.Conv2d(ngf_global*4, ngf_global*2, kernel_size=3, padding=1),
                                norm_layer(ngf_global*2), nn.ReLU(True),
                                nn.Conv2d(ngf_global*2, ngf_global*2, kernel_size=3, padding=1),
                                norm_layer(ngf_global*2), nn.ReLU(True),
                                nn.Conv2d(ngf_global*2, ngf_global*2, kernel_size=3, padding=1),
                                norm_layer(ngf_global*2), nn.ReLU(True))

    def forward(self, input, input2, input3):  #input2: 小尺寸的高层特征   input3:大尺寸的低层特征
        ### create input pyramid
        # input2是本层级， input1是大尺度的高层级，input2是小尺度的低层级
        input_downsampled = [input]
        input_downsampled.append(input2)
        input_downsampled.append(input3)
        # for i in range(self.n_local_enhancers):
        #     input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        # output_prev = self.model(input_downsampled[-1])
        output_prevc= self.model_1c(input_downsampled[1])
        output_prevc=self.model_2c(output_prevc)

        output_prevd= self.model_1d(input_downsampled[2])
        output_prevd=self.model_2d(output_prevd)
        ### build up one layer at a time

        model_downsample = getattr(self, 'model' + str(1) + '_1')
        model_upsample = getattr(self, 'model' + str(1) + '_2')
        input_i = input_downsampled[0]
        tmp=model_downsample(input_i) + output_prevc + output_prevd
        output_prev = model_upsample(tmp)

        return output_prev

class LocalEnhancer_consistency_v3(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(LocalEnhancer_consistency_v3, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####
        ngf_global = ngf * (2 ** n_local_enhancers)
        model_global_1 = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                       norm_layer).model_1
        model_global_2 = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                         norm_layer).model_2
        model_global_2 = [model_global_2[i] for i in
                        range(len(model_global_2) - 3)]  # get rid of final convolution layers
        # self.model = nn.Sequential(*model_global)
        self.model_1 = nn.Sequential(*model_global_1)
        self.model_2 = nn.Sequential(*model_global_2)
        self.pre2= nn.Sequential(nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=1, padding=0),
                                  norm_layer(ngf_global* 2), nn.ReLU(True))
        self.mix2 = nn.Sequential(nn.Conv2d(ngf_global * 4, ngf_global * 2, kernel_size=3, padding=1),
                                  norm_layer(ngf_global* 2), nn.ReLU(True),
                                  nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=3, padding=1),
                                  norm_layer(ngf_global * 2), nn.ReLU(True),
                                  nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=3, padding=1),
                                  norm_layer(ngf_global* 2), nn.ReLU(True))

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers + 1):
            ### downsample
            ngf_global = ngf * (2 ** (n_local_enhancers - n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(3, ngf_global, kernel_size=7, padding=0),
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=1, padding=1),
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            # model_upsample += [
            #     nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
            #     norm_layer(ngf_global), nn.ReLU(True)]
            # 输出保持256*256
            model_upsample += [
                nn.Conv2d(ngf_global * 2, ngf_global, kernel_size=3, stride=1, padding=1),
                norm_layer(ngf_global), nn.ReLU(True)]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                   nn.Tanh()]

            setattr(self, 'model' + str(n) + '_1', nn.Sequential(*model_downsample))#为对象self添加属性
            setattr(self, 'model' + str(n) + '_2', nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        self.pre3= nn.Sequential(nn.Conv2d(ngf_global * 4, ngf_global*2, kernel_size=1, padding=0),
                                  norm_layer(ngf_global* 2), nn.ReLU(True))
        self.mix3=nn.Sequential(nn.Conv2d(ngf_global*4, ngf_global*2, kernel_size=3, padding=1),
                                norm_layer(ngf_global*2), nn.ReLU(True),
                                nn.Conv2d(ngf_global*2, ngf_global*2, kernel_size=3, padding=1),
                                norm_layer(ngf_global*2), nn.ReLU(True),
                                nn.Conv2d(ngf_global*2, ngf_global*2, kernel_size=3, padding=1),
                                norm_layer(ngf_global*2), nn.ReLU(True))

    def forward(self, input, input2=None):  #input2: 小尺寸的高层特征   input3:大尺寸的低层特征
        ### create input pyramid
        #input本层，input2低层相同尺度
        input_downsampled = [input]
        input_downsampled.append(input2)
        # for i in range(self.n_local_enhancers):
        #     input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        # output_prev = self.model(input_downsampled[-1])
        # print(input_downsampled[-1].shape)
        output_prev= self.model_1(input_downsampled[-1])

        output_prev=self.model_2(output_prev)
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = getattr(self, 'model' + str(n_local_enhancers) + '_1')
            model_upsample = getattr(self, 'model' + str(n_local_enhancers) + '_2')
            input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]
            d = model_downsample(input_i)
            # print(d.shape)
            # print(output_prev.shape)
            tmp= d + output_prev
            output_prev = model_upsample(tmp)

        return output_prev



class LocalEnhancer_consistency(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(LocalEnhancer_consistency, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####
        ngf_global = ngf * (2 ** n_local_enhancers)
        model_global_1 = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                       norm_layer).model_1
        model_global_2 = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                         norm_layer).model_2
        model_global_2 = [model_global_2[i] for i in
                        range(len(model_global_2) - 3)]  # get rid of final convolution layers
        # self.model = nn.Sequential(*model_global)
        self.model_1 = nn.Sequential(*model_global_1)
        self.model_2 = nn.Sequential(*model_global_2)
        self.pre2= nn.Sequential(nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=1, padding=0),
                                  norm_layer(ngf_global* 2), nn.ReLU(True))
        self.mix2 = nn.Sequential(nn.Conv2d(ngf_global * 4, ngf_global * 2, kernel_size=3, padding=1),
                                  norm_layer(ngf_global* 2), nn.ReLU(True),
                                  nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=3, padding=1),
                                  norm_layer(ngf_global * 2), nn.ReLU(True),
                                  nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=3, padding=1),
                                  norm_layer(ngf_global* 2), nn.ReLU(True))

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers + 1):
            ### downsample
            ngf_global = ngf * (2 ** (n_local_enhancers - n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(3, ngf_global, kernel_size=7, padding=0),
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            # model_upsample += [
            #     nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
            #     norm_layer(ngf_global), nn.ReLU(True)]
            # 输出保持256*256
            model_upsample += [
                nn.Conv2d(ngf_global * 2, ngf_global, kernel_size=3, stride=1, padding=1),
                norm_layer(ngf_global), nn.ReLU(True)]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                   nn.Tanh()]

            setattr(self, 'model' + str(n) + '_1', nn.Sequential(*model_downsample))#为对象self添加属性
            setattr(self, 'model' + str(n) + '_2', nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        self.pre3= nn.Sequential(nn.Conv2d(ngf_global * 4, ngf_global*2, kernel_size=1, padding=0),
                                  norm_layer(ngf_global* 2), nn.ReLU(True))
        self.mix3=nn.Sequential(nn.Conv2d(ngf_global*4, ngf_global*2, kernel_size=3, padding=1),
                                norm_layer(ngf_global*2), nn.ReLU(True),
                                nn.Conv2d(ngf_global*2, ngf_global*2, kernel_size=3, padding=1),
                                norm_layer(ngf_global*2), nn.ReLU(True),
                                nn.Conv2d(ngf_global*2, ngf_global*2, kernel_size=3, padding=1),
                                norm_layer(ngf_global*2), nn.ReLU(True))

    def forward(self, input, input2=None):  #input2: 小尺寸的高层特征   input3:大尺寸的低层特征
        ### create input pyramid
        # input2是本层级， input1是大尺度的高层级
        input_downsampled = [input]
        input_downsampled.append(input2)
        # for i in range(self.n_local_enhancers):
        #     input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        # output_prev = self.model(input_downsampled[-1])
        output_prev= self.model_1(input_downsampled[0])

        output_prev=self.model_2(output_prev)
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = getattr(self, 'model' + str(n_local_enhancers) + '_1')
            model_upsample = getattr(self, 'model' + str(n_local_enhancers) + '_2')
            input_i = input_downsampled[n_local_enhancers]
            tmp=model_downsample(input_i) + output_prev
            output_prev = model_upsample(tmp)

        return output_prev



class LocalEnhancer_consis_concat(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(LocalEnhancer_consis_concat, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####
        ngf_global = ngf * (2 ** n_local_enhancers)
        model_global_1 = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                       norm_layer).model_1
        model_global_2 = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                         norm_layer).model_2
        model_global_2 = [model_global_2[i] for i in
                        range(len(model_global_2) - 3)]  # get rid of final convolution layers
        # self.model = nn.Sequential(*model_global)
        self.model_1 = nn.Sequential(*model_global_1)
        self.model_2 = nn.Sequential(*model_global_2)
        self.pre2= nn.Sequential(nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=1, padding=0),
                                  norm_layer(ngf_global* 2), nn.ReLU(True))
        self.mix2 = nn.Sequential(nn.Conv2d(ngf_global * 4, ngf_global * 2, kernel_size=3, padding=1),
                                  norm_layer(ngf_global* 2), nn.ReLU(True),
                                  nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=3, padding=1),
                                  norm_layer(ngf_global * 2), nn.ReLU(True),
                                  nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=3, padding=1),
                                  norm_layer(ngf_global* 2), nn.ReLU(True))

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers + 1):
            ### downsample
            ngf_global = ngf * (2 ** (n_local_enhancers - n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 4, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [
                nn.Conv2d(ngf_global * 4, ngf_global * 2, kernel_size=3, stride=1, padding=1),
                norm_layer(ngf_global * 2), nn.ReLU(True),
                nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(ngf_global), nn.ReLU(True)]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                   nn.Tanh()]

            setattr(self, 'model' + str(n) + '_1', nn.Sequential(*model_downsample))#为对象self添加属性
            setattr(self, 'model' + str(n) + '_2', nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        self.pre3= nn.Sequential(nn.Conv2d(ngf_global * 4, ngf_global*2, kernel_size=1, padding=0),
                                  norm_layer(ngf_global* 2), nn.ReLU(True))
        self.mix3=nn.Sequential(nn.Conv2d(ngf_global*4, ngf_global*2, kernel_size=3, padding=1),
                                norm_layer(ngf_global*2), nn.ReLU(True),
                                nn.Conv2d(ngf_global*2, ngf_global*2, kernel_size=3, padding=1),
                                norm_layer(ngf_global*2), nn.ReLU(True),
                                nn.Conv2d(ngf_global*2, ngf_global*2, kernel_size=3, padding=1),
                                norm_layer(ngf_global*2), nn.ReLU(True))

    def forward(self, input,input2):  #input2: 小尺寸的高层特征   input3:大尺寸的低层特征
        ### create input pyramid
        input_downsampled = [input]
        input_downsampled.append(input2)


        ### output at coarest level

        output_prev= self.model_1(input_downsampled[-1])

        output_prev=self.model_2(output_prev)
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = getattr(self, 'model' + str(n_local_enhancers) + '_1')
            model_upsample = getattr(self, 'model' + str(n_local_enhancers) + '_2')
            input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]
            md_t = model_downsample(input_i)
            tmp= torch.cat((md_t, output_prev),dim=1)
            output_prev = model_upsample(tmp)

        return output_prev


class LocalEnhancer_consis_inputfpn(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(LocalEnhancer_consis_inputfpn, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####
        ngf_global = ngf * (2 ** (n_local_enhancers-1))
        model_fpn1 = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                            norm_layer(ngf_global), nn.ReLU(True)]
        setattr(self, 'model_fpn1', nn.Sequential(*model_fpn1))

        model_fpn2 = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                            norm_layer(ngf_global), nn.ReLU(True)]
        setattr(self, 'model_fpn2', nn.Sequential(*model_fpn2))

        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv_fpn = nn.Conv2d(ngf_global, ngf_global, kernel_size=1, stride=1, padding=0)
        self.conv_norm = norm_layer(ngf_global)
        self.conv_relu = nn.ReLU()


        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers + 1):
            ### downsample
            ngf_global = ngf * (2 ** (n_local_enhancers - n))
            model_downsample = [nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [
                nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(ngf_global), nn.ReLU(True)]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                   nn.Tanh()]

            setattr(self, 'model' + str(n) + '_1', nn.Sequential(*model_downsample))#为对象self添加属性
            setattr(self, 'model' + str(n) + '_2', nn.Sequential(*model_upsample))





    def forward(self, input,input2):  #input2: 小尺寸的高层特征   input3:大尺寸的低层特征

        x1 = self.model_fpn1(input)

        x2 = self.model_fpn2(input2)

        x2 = self.upsample2(x2)

        x1 = self.conv_fpn(x1)

        x_out = x1+x2
        x_out = self.conv_norm(x_out)
        x_out = self.conv_relu(x_out)



        input_downsampled = [x_out]

        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = getattr(self, 'model' + str(n_local_enhancers) + '_1')
            model_upsample = getattr(self, 'model' + str(n_local_enhancers) + '_2')
            input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]
            tmp=model_downsample(input_i)
            output_prev = model_upsample(tmp)

        return output_prev

class LocalEnhancer_consis_inputfpn_v2(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(LocalEnhancer_consis_inputfpn_v2, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####
        ngf_global = ngf * (2 ** (n_local_enhancers-1))
        model_fpn1 = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                            norm_layer(ngf_global), nn.ReLU(True)]
        setattr(self, 'model_fpn1', nn.Sequential(*model_fpn1))

        model_fpn2 = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                            norm_layer(ngf_global), nn.ReLU(True)]
        setattr(self, 'model_fpn2', nn.Sequential(*model_fpn2))

        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv_fpn = nn.Conv2d(ngf_global, ngf_global, kernel_size=1, stride=1, padding=0)
        self.conv_norm = norm_layer(ngf_global)
        self.conv_relu = nn.ReLU()


        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers + 1):
            ### downsample
            ngf_global = ngf * (2 ** (n_local_enhancers - n))
            model_downsample = [nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            model_downsample += [nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [
                nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(ngf_global), nn.ReLU(True)]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                   nn.Tanh()]

            setattr(self, 'model' + str(n) + '_1', nn.Sequential(*model_downsample))#为对象self添加属性
            setattr(self, 'model' + str(n) + '_2', nn.Sequential(*model_upsample))





    def forward(self, input,input2):  #input2: 小尺寸的高层特征   input3:大尺寸的低层特征

        x1 = self.model_fpn1(input)

        x2 = self.model_fpn2(input2)

        x2 = self.upsample2(x2)

        x1 = self.conv_fpn(x1)

        x_out = x1+x2
        x_out = self.conv_norm(x_out)
        x_out = self.conv_relu(x_out)



        input_downsampled = [x_out]

        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = getattr(self, 'model' + str(n_local_enhancers) + '_1')
            model_upsample = getattr(self, 'model' + str(n_local_enhancers) + '_2')
            input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]
            tmp=model_downsample(input_i)
            output_prev = model_upsample(tmp)

        return output_prev

class LocalEnhancer_consis_inputcat(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(LocalEnhancer_consis_inputcat, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####
        ngf_global = ngf * (2 ** n_local_enhancers)
        model_global_1 = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                       norm_layer).model_1
        model_global_2 = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                         norm_layer).model_2
        model_global_2 = [model_global_2[i] for i in
                        range(len(model_global_2) - 3)]  # get rid of final convolution layers
        # self.model = nn.Sequential(*model_global)
        self.model_1 = nn.Sequential(*model_global_1)
        self.model_2 = nn.Sequential(*model_global_2)
        self.pre2= nn.Sequential(nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=1, padding=0),
                                  norm_layer(ngf_global* 2), nn.ReLU(True))
        self.mix2 = nn.Sequential(nn.Conv2d(ngf_global * 4, ngf_global * 2, kernel_size=3, padding=1),
                                  norm_layer(ngf_global* 2), nn.ReLU(True),
                                  nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=3, padding=1),
                                  norm_layer(ngf_global * 2), nn.ReLU(True),
                                  nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=3, padding=1),
                                  norm_layer(ngf_global* 2), nn.ReLU(True))

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers + 1):
            ### downsample
            ngf_global = ngf * (2 ** (n_local_enhancers - n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [
                nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(ngf_global), nn.ReLU(True)]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                   nn.Tanh()]

            setattr(self, 'model' + str(n) + '_1', nn.Sequential(*model_downsample))#为对象self添加属性
            setattr(self, 'model' + str(n) + '_2', nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        self.pre3= nn.Sequential(nn.Conv2d(ngf_global * 4, ngf_global*2, kernel_size=1, padding=0),
                                  norm_layer(ngf_global* 2), nn.ReLU(True))
        self.mix3=nn.Sequential(nn.Conv2d(ngf_global*4, ngf_global*2, kernel_size=3, padding=1),
                                norm_layer(ngf_global*2), nn.ReLU(True),
                                nn.Conv2d(ngf_global*2, ngf_global*2, kernel_size=3, padding=1),
                                norm_layer(ngf_global*2), nn.ReLU(True),
                                nn.Conv2d(ngf_global*2, ngf_global*2, kernel_size=3, padding=1),
                                norm_layer(ngf_global*2), nn.ReLU(True))

    def forward(self, input,input2):  #input2: 小尺寸的高层特征   input3:大尺寸的低层特征
        input2 = F.interpolate(input2, scale_factor=2)

        input = torch.cat((input,input2),dim=1)
        input_downsampled = [input]

        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = getattr(self, 'model' + str(n_local_enhancers) + '_1')
            model_upsample = getattr(self, 'model' + str(n_local_enhancers) + '_2')
            input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]
            tmp=model_downsample(input_i)
            output_prev = model_upsample(tmp)

        return output_prev




class LocalEnhancer_consis_nolowlevel(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(LocalEnhancer_consis_nolowlevel, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####
        ngf_global = ngf * (2 ** n_local_enhancers)
        model_global_1 = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                       norm_layer).model_1
        model_global_2 = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                         norm_layer).model_2
        model_global_2 = [model_global_2[i] for i in
                        range(len(model_global_2) - 3)]  # get rid of final convolution layers
        # self.model = nn.Sequential(*model_global)
        self.model_1 = nn.Sequential(*model_global_1)
        self.model_2 = nn.Sequential(*model_global_2)
        self.pre2= nn.Sequential(nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=1, padding=0),
                                  norm_layer(ngf_global* 2), nn.ReLU(True))
        self.mix2 = nn.Sequential(nn.Conv2d(ngf_global * 4, ngf_global * 2, kernel_size=3, padding=1),
                                  norm_layer(ngf_global* 2), nn.ReLU(True),
                                  nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=3, padding=1),
                                  norm_layer(ngf_global * 2), nn.ReLU(True),
                                  nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=3, padding=1),
                                  norm_layer(ngf_global* 2), nn.ReLU(True))

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers + 1):
            ### downsample
            ngf_global = ngf * (2 ** (n_local_enhancers - n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [
                nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(ngf_global), nn.ReLU(True)]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                   nn.Tanh()]

            setattr(self, 'model' + str(n) + '_1', nn.Sequential(*model_downsample))#为对象self添加属性
            setattr(self, 'model' + str(n) + '_2', nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        self.pre3= nn.Sequential(nn.Conv2d(ngf_global * 4, ngf_global*2, kernel_size=1, padding=0),
                                  norm_layer(ngf_global* 2), nn.ReLU(True))
        self.mix3=nn.Sequential(nn.Conv2d(ngf_global*4, ngf_global*2, kernel_size=3, padding=1),
                                norm_layer(ngf_global*2), nn.ReLU(True),
                                nn.Conv2d(ngf_global*2, ngf_global*2, kernel_size=3, padding=1),
                                norm_layer(ngf_global*2), nn.ReLU(True),
                                nn.Conv2d(ngf_global*2, ngf_global*2, kernel_size=3, padding=1),
                                norm_layer(ngf_global*2), nn.ReLU(True))

    def forward(self, input,input2=None):  #input2: 小尺寸的高层特征   input3:大尺寸的低层特征
        ### create input pyramid
        input_downsampled = [input]
        # input_downsampled.append(input2)


        ### output at coarest level

        # output_prev= self.model_1(input_downsampled[-1])
        #
        # output_prev=self.model_2(output_prev)
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = getattr(self, 'model' + str(n_local_enhancers) + '_1')
            model_upsample = getattr(self, 'model' + str(n_local_enhancers) + '_2')
            input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]
            tmp=model_downsample(input_i)
            output_prev = model_upsample(tmp)

        return output_prev


class LocalEnhancer_consis_v3(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(LocalEnhancer_consis_v3, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####
        ngf_global = ngf
        model_downsample1 = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                            norm_layer(ngf_global), nn.ReLU(True),
                            nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                            norm_layer(ngf_global * 2), nn.ReLU(True)]
        setattr(self, 'model1_down', nn.Sequential(*model_downsample1))  # 为对象self添加属性
        model_downsample2 = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                            norm_layer(ngf_global), nn.ReLU(True),
                            nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=1, padding=1),
                            norm_layer(ngf_global * 2), nn.ReLU(True)]
        setattr(self, 'model2_down', nn.Sequential(*model_downsample2))  # 为对象self添加属性
        self.model1_res1 = ResnetBlock(ngf_global * 4, padding_type=padding_type, norm_layer=norm_layer)
        self.model1_res2 = ResnetBlock(ngf_global * 4, padding_type=padding_type, norm_layer=norm_layer)
        self.model1_res3 = ResnetBlock(ngf_global * 4, padding_type=padding_type, norm_layer=norm_layer)

        self.model2_res1 = ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)
        self.model2_res2 = ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)
        self.model2_res3 = ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)
        model_upsample1 = [
            nn.ConvTranspose2d(ngf_global * 4, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(ngf_global), nn.ReLU(True)]
        setattr(self, 'model1_up', nn.Sequential(*model_upsample1))
        model_upsample2 = [
            nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(ngf_global), nn.ReLU(True)]
        setattr(self, 'model2_up', nn.Sequential(*model_upsample2))
        model_out = [nn.ReflectionPad2d(1), nn.Conv2d(ngf*2, ngf, kernel_size=3, padding=0), nn.ReLU(True),
                     nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),nn.Tanh()]
        setattr(self, 'model_out', nn.Sequential(*model_out))





    def forward(self, input,input2):  #input2: 小尺寸的高层特征   input3:大尺寸的低层特征
        ### create input pyramid
        t1 = self.model1_down(input)
        t2 = self.model2_down(input2)
        t1 = torch.cat((t1, t2), 1)

        t1 = self.model1_res1(t1)
        t1 = self.model1_res2(t1)
        t1 = self.model1_res3(t1)
        t1 = self.model1_up(t1)
        t2 = self.model2_res1(t2)
        t2 = self.model2_res2(t2)
        t2 = self.model2_res3(t2)
        t2 = self.model2_up(t2)

        t1 = torch.cat((t1, t2), 1)
        output_prev = self.model_out(t1)

        return output_prev


class LocalEnhancer_consis(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(LocalEnhancer_consis, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####
        ngf_global = ngf * (2 ** n_local_enhancers)
        model_global_1 = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                       norm_layer).model_1
        model_global_2 = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                         norm_layer).model_2
        model_global_2 = [model_global_2[i] for i in
                        range(len(model_global_2) - 3)]  # get rid of final convolution layers
        # self.model = nn.Sequential(*model_global)
        self.model_1 = nn.Sequential(*model_global_1)
        self.model_2 = nn.Sequential(*model_global_2)
        self.pre2= nn.Sequential(nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=1, padding=0),
                                  norm_layer(ngf_global* 2), nn.ReLU(True))
        self.mix2 = nn.Sequential(nn.Conv2d(ngf_global * 4, ngf_global * 2, kernel_size=3, padding=1),
                                  norm_layer(ngf_global* 2), nn.ReLU(True),
                                  nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=3, padding=1),
                                  norm_layer(ngf_global * 2), nn.ReLU(True),
                                  nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=3, padding=1),
                                  norm_layer(ngf_global* 2), nn.ReLU(True))

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers + 1):
            ### downsample
            ngf_global = ngf * (2 ** (n_local_enhancers - n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [
                nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(ngf_global), nn.ReLU(True)]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                   nn.Tanh()]

            setattr(self, 'model' + str(n) + '_1', nn.Sequential(*model_downsample))#为对象self添加属性
            setattr(self, 'model' + str(n) + '_2', nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        self.pre3= nn.Sequential(nn.Conv2d(ngf_global * 4, ngf_global*2, kernel_size=1, padding=0),
                                  norm_layer(ngf_global* 2), nn.ReLU(True))
        self.mix3=nn.Sequential(nn.Conv2d(ngf_global*4, ngf_global*2, kernel_size=3, padding=1),
                                norm_layer(ngf_global*2), nn.ReLU(True),
                                nn.Conv2d(ngf_global*2, ngf_global*2, kernel_size=3, padding=1),
                                norm_layer(ngf_global*2), nn.ReLU(True),
                                nn.Conv2d(ngf_global*2, ngf_global*2, kernel_size=3, padding=1),
                                norm_layer(ngf_global*2), nn.ReLU(True))

    def forward(self, input,input2):  #input2: 小尺寸的高层特征   input3:大尺寸的低层特征
        ### create input pyramid
        input_downsampled = [input]
        input_downsampled.append(input2)


        ### output at coarest level

        output_prev= self.model_1(input_downsampled[-1])

        output_prev=self.model_2(output_prev)
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = getattr(self, 'model' + str(n_local_enhancers) + '_1')
            model_upsample = getattr(self, 'model' + str(n_local_enhancers) + '_2')
            input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]
            tmp=model_downsample(input_i) + output_prev
            output_prev = model_upsample(tmp)

        return output_prev


class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####           
        ngf_global = ngf * (2 ** n_local_enhancers)
        model_global_1 = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                       norm_layer).model_1
        model_global_2 = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                         norm_layer).model_2
        model_global_2 = [model_global_2[i] for i in
                        range(len(model_global_2) - 3)]  # get rid of final convolution layers
        # self.model = nn.Sequential(*model_global)
        self.model_1 = nn.Sequential(*model_global_1)
        self.model_2 = nn.Sequential(*model_global_2)
        self.pre2= nn.Sequential(nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=1, padding=0),
                                  norm_layer(ngf_global* 2), nn.ReLU(True))
        self.mix2 = nn.Sequential(nn.Conv2d(ngf_global * 4, ngf_global * 2, kernel_size=3, padding=1),
                                  norm_layer(ngf_global* 2), nn.ReLU(True),
                                  nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=3, padding=1),
                                  norm_layer(ngf_global * 2), nn.ReLU(True),
                                  nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=3, padding=1),
                                  norm_layer(ngf_global* 2), nn.ReLU(True))

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers + 1):
            ### downsample            
            ngf_global = ngf * (2 ** (n_local_enhancers - n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [
                nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(ngf_global), nn.ReLU(True)]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                   nn.Tanh()]

            setattr(self, 'model' + str(n) + '_1', nn.Sequential(*model_downsample))#为对象self添加属性
            setattr(self, 'model' + str(n) + '_2', nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        self.pre3= nn.Sequential(nn.Conv2d(ngf_global * 4, ngf_global*2, kernel_size=1, padding=0),
                                  norm_layer(ngf_global* 2), nn.ReLU(True))
        self.mix3=nn.Sequential(nn.Conv2d(ngf_global*4, ngf_global*2, kernel_size=3, padding=1),
                                norm_layer(ngf_global*2), nn.ReLU(True),
                                nn.Conv2d(ngf_global*2, ngf_global*2, kernel_size=3, padding=1),
                                norm_layer(ngf_global*2), nn.ReLU(True),
                                nn.Conv2d(ngf_global*2, ngf_global*2, kernel_size=3, padding=1),
                                norm_layer(ngf_global*2), nn.ReLU(True))

    def forward(self, input,input2=None,input3=None):  #input2: 小尺寸的高层特征   input3:大尺寸的低层特征
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        # output_prev = self.model(input_downsampled[-1])
        output_prev= self.model_1(input_downsampled[-1])
        # if not (input2 is None):
        #     input2=self.pre2(input2)
        #     output_prev=self.mix2(torch.cat((output_prev,input2),dim=1))
        output_prev=self.model_2(output_prev)
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = getattr(self, 'model' + str(n_local_enhancers) + '_1')
            model_upsample = getattr(self, 'model' + str(n_local_enhancers) + '_2')
            input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]
            tmp=model_downsample(input_i) + output_prev
            # if not (input3 is None):
            #     input3=self.pre3(input3)
            #     tmp=self.mix3(torch.cat((tmp,input3),dim=1))
            output_prev = model_upsample(tmp)
            # output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev


class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True) #将会改变输入的原数据

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample         
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model_1 = nn.Sequential(*model[:7])
        self.model_2 = nn.Sequential(*model[7:])

        self.mix2=nn.Sequential(nn.Conv2d(ngf*4, ngf*2, kernel_size=3, padding=1),
                                norm_layer(ngf*2), nn.ReLU(True),
                                nn.Conv2d(ngf*2, ngf*2, kernel_size=3, padding=1),
                                norm_layer(ngf*2), nn.ReLU(True),
                                nn.Conv2d(ngf*2, ngf*2, kernel_size=3, padding=1),
                                norm_layer(ngf*2), nn.ReLU(True))

    def forward(self, input ,input2=None):
        out=self.model_1(input)
        if input2!=None:
            out=self.mix2(torch.cat((out,input2),dim=1))
        out=self.model_2(out)
        return out

    # Define a resnet block

class GlobalGenerator_leak(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGenerator_leak, self).__init__()
        activation = nn.LeakyReLU(0.2) #将会改变输入的原数据

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model_1 = nn.Sequential(*model[:7])
        self.model_2 = nn.Sequential(*model[7:])

        self.mix2=nn.Sequential(nn.Conv2d(ngf*4, ngf*2, kernel_size=3, padding=1),
                                norm_layer(ngf*2), nn.ReLU(True),
                                nn.Conv2d(ngf*2, ngf*2, kernel_size=3, padding=1),
                                norm_layer(ngf*2), nn.ReLU(True),
                                nn.Conv2d(ngf*2, ngf*2, kernel_size=3, padding=1),
                                norm_layer(ngf*2), nn.ReLU(True))

    def forward(self, input ,input2=None):
        out=self.model_1(input)
        if input2!=None:
            out=self.mix2(torch.cat((out,input2),dim=1))
        out=self.model_2(out)
        return out

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()
        self.output_nc = output_nc

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf), nn.ReLU(True)]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample         
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b + 1] == int(i)).nonzero()  # n x 4
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:, 0] + b, indices[:, 1] + j, indices[:, 2], indices[:, 3]]
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)
                    outputs_mean[indices[:, 0] + b, indices[:, 1] + j, indices[:, 2], indices[:, 3]] = mean_feat
        return outputs_mean


class Classifier(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=256, conv_dim=64, c_dim=4, repeat_num=7):
        super(Classifier, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        # out_src = self.conv1(h)
        out_cls = self.conv2(h)
        out_cls_view = out_cls.view(out_cls.size(0), out_cls.size(1))

        return out_cls_view



class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


class MultiscaleDiscriminator_CLASS_starganv1(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator_CLASS_starganv1, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.class_head1 = nn.Conv2d(256, 256, kernel_size=4, bias=False)
        self.class_head2 = nn.Conv2d(256, 256, stride=2, kernel_size=3, bias=False)
        self.class_head3 = nn.Conv2d(256, 256, stride=2, kernel_size=3, bias=False)
        self.class_head4 = nn.Conv2d(256, 256, stride=2, kernel_size=3, bias=False)
        self.class_head5 = nn.Conv2d(256, 256, stride=2, kernel_size=1, bias=False)
        self.class_head6 = nn.Conv2d(256, 4, 1,1,0)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))

            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        cls_input = result[0][-3]
        # print('cls_input.shape', cls_input.shape)
        cls_input = self.class_head1(cls_input)
        cls_input = self.class_head2(cls_input)
        cls_input = self.class_head3(cls_input)
        cls_input = self.class_head4(cls_input)
        cls_input = self.class_head5(cls_input)
        cls_out = self.class_head6(cls_input)

        cls_out = cls_out.view(cls_out.size(0), -1)  # (batch, num_domains)
        # idx = torch.LongTensor(range(y.size(0))).to(y.device)
        # cls_out = cls_out[idx, y]

        return result, cls_out


class MultiscaleDiscriminator_CLASS(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator_CLASS, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.class_head1 = nn.Conv2d(256, 256, kernel_size=4, bias=False)
        self.class_head2 = nn.Conv2d(256, 256, stride=2, kernel_size=3, bias=False)
        self.class_head3 = nn.Conv2d(256, 256, stride=2, kernel_size=3, bias=False)
        self.class_head4 = nn.Conv2d(256, 256, stride=2, kernel_size=3, bias=False)
        self.class_head5 = nn.Conv2d(256, 256, stride=2, kernel_size=1, bias=False)
        self.class_head6 = nn.Conv2d(256, 4, 1,1,0)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input, y):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))

            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        cls_input = result[0][-3]
        # print('cls_input.shape', cls_input.shape)
        cls_input = self.class_head1(cls_input)
        cls_input = self.class_head2(cls_input)
        cls_input = self.class_head3(cls_input)
        cls_input = self.class_head4(cls_input)
        cls_input = self.class_head5(cls_input)
        cls_out = self.class_head6(cls_input)

        cls_out = cls_out.view(cls_out.size(0), -1)  # (batch, num_domains)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        cls_out = cls_out[idx, y]

        return result, cls_out

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        self.if_use_sigmoid = use_sigmoid

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


def test_model(model_func=Encoder):
    model = model_func(21, 3)
    x = torch.rand(3, 21, 32, 32)
    y = model(x)
    # print(y)
    try:
        print(y.shape)
    except:
        pass

    try:
        print(model)
    except:
        pass

    try:
        for i in y:
            for j in i:
                print(j.shape)
    except:
        pass


if __name__ == '__main__':
    from src.pix2pixHD.train_config import config

    args = config()
    # model = get_G(args, 21)
    model = get_D(args, 21)
    model = get_G(args, 21)
    x = torch.rand(3, 21, 32, 32)
    y = model(x)
    # print(y)
    try:
        print(y.shape)
    except:
        pass

    try:
        print(model)
    except:
        pass

    try:
        for i in y:
            for j in i:
                print(j.shape)
    except:
        pass
