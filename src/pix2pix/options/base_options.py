import argparse
import os
from util import util
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--name', required=True, help='name of program')
        self.parser.add_argument('--developed',  action='store_true',  help='developed or produced')
        self.parser.add_argument('--from_pretrain_model',  action='store_true', help='if true, select a pretrain model and copy that')
        self.parser.add_argument('--pretrain_model_root',help='path for pretrain model (if needed)')
        self.parser.add_argument('--program_path', required=True,help='path for program (should be an empty floder)')
        self.parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, testA, testB, etc)')
        self.parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
        self.parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--nThreads', default=0, type=int, help='# threads for loading data')# 本来是16，后多进程有bug
        self.parser.add_argument('--model', type=str, default='pix2pix',
                                 help='chooses which model to use. pix2pix, edge1pix2pix')

        self.parser.add_argument('--use_tdf_data', action='store_true', help='use tdf data rather img data, not suppose more than one threads')
        self.parser.add_argument('--dataroot_2',type=str,default=None, help='if use tdf data,dataroot will be root of tdf_A,and this is root of tdf_B')
        self.parser.add_argument('--tdf_level', type=int, default=None, help='if use tdf data, this is level to use')
        self.parser.add_argument('--test_rate', type=float, default=0.3, help='if use tdf data, how many imgs will be devided into testset')
        self.parser.add_argument('--no_random_sample', action='store_true', help='sample from tdf by default order (row and cow)')


        # 以下参数仅供开发使用
        self.parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str, default='resnet_9blocks', help='selects model to use for netG')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--dataset_mode', type=str, default='aligned', help='chooses how datasets are loaded. [aligned | single]')
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        #self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        #self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        #self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        self.opt.program_path=os.path.abspath(self.opt.program_path)
        self.opt.dataroot=os.path.abspath(self.opt.dataroot)

        args = vars(self.opt) #转化为dict

        if self.opt.developed:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')


        return self.opt