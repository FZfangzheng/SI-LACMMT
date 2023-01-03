import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from tensorboardX import SummaryWriter


class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    def initialize(self, opt,G_inputnc_plus=0):
        assert isinstance(G_inputnc_plus,int)
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc+ G_inputnc_plus,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)

        # load/define networks
        self.netG = networks.define_G(opt.input_nc + G_inputnc_plus, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        # if opt.from_pretrain_model:
        #     assert os.path.isdir(opt.pretrain_model_root),"预训练模型选择有误[%s]"%opt.pretrain_model_root
        #     self.load_network(self.netG, 'netG', opt.which_epoch,load_path=opt.pretrain_model_root)
        #     if self.isTrain:
        #         self.load_network(self.netD, 'netD', opt.which_epoch,load_path=opt.pretrain_model_root)
        #     print("load pretrain model from [%s]"%opt.pretrain_model_root)
        # elif not self.isTrain or opt.continue_train:
        #     self.load_network(self.netG, 'netG', opt.which_epoch)
        #     if self.isTrain:
        #         self.load_network(self.netD, 'netD', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            # if opt.continue_train:
            #     self.load_optimizer(self.optimizer_G, 'optimizerG', opt.which_epoch)
            #     self.load_optimizer(self.optimizer_D, 'optimizerD', opt.which_epoch)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        # if opt.developed:
        #     print('---------- Networks initialized -------------')
        #     networks.print_network(self.netG)
        #     if self.isTrain:
        #         networks.print_network(self.netD)
        #     print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.path_A=input['A_paths' if AtoB else 'B_paths']
        if ('B' if AtoB else 'A') in input:
            input_B = input['B' if AtoB else 'A']
            self.input_B.resize_(input_B.size()).copy_(input_B)
            self.path_B = input['B_paths' if AtoB else 'A_paths']
        else:
            self.input_B=None
            self.path_B=None
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG(self.real_A)
        self.real_B = Variable(self.input_B)

    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

    def generate(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG(self.real_A)
        self.real_B = None

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1).data)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        # return OrderedDict([('G_GAN', self.loss_G_GAN.data.item()),
        #                     ('G_L1', self.loss_G_L1.data.item()),
        #                     ('D_real', self.loss_D_real.data.item()),
        #                     ('D_fake', self.loss_D_fake.data.item())
        #                     ])
        return OrderedDict([('G_loss', self.loss_G_GAN.data.item()+self.loss_G_L1.data.item()),
                            ('D_loss', (self.loss_D_real.data.item()+self.loss_D_fake.data.item())*0.5),
                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        if isinstance(self.real_B, self.Tensor):
            real_B = util.tensor2im(self.real_B.data)
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])
        else:
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])

    def get_current_paths(self):
        path_A=self.path_A
        if isinstance(self.path_B,str):
            path_B=self.path_B
            return OrderedDict([('path_A', path_A), ('path_B', path_B)])
        else:
            return OrderedDict([('path_A', path_A)])

    def save(self, label):
        self.save_network(self.netG, 'netG', label, self.gpu_ids)
        self.save_optimizer(self.optimizer_G, 'optimizerG', label)
        self.save_network(self.netD, 'netD', label, self.gpu_ids)
        self.save_optimizer(self.optimizer_D, 'optimizerD', label)

    def show_graphs(self,writer):# show graphs in temsorboardX
        if self.opt.gpu_ids:
            torch_device = torch.device("cuda")
        else:
            torch_device = torch.device("cpu")
        # writer.add_graph(self.netG, torch.rand([1, 3, 256, 256]).to(torch_device))
        # writer2 = SummaryWriter(os.path.join(self.opt.program_path, 'visdom/netD'))
        # writer2.add_graph(self.netD, torch.rand([1, 6, 256, 256]).to(torch_device))
