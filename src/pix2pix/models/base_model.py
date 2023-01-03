import os
import torch


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor   # tensor 默认为浮点型tensor
        # self.save_dir = os.path.join(opt.program_path, "modeldata")

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def generate(self):
        pass


    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def get_current_paths(self):
        pass

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename_tmp = '%s_%s_tmp.pth' % (epoch_label, network_label)
        save_path_tmp = os.path.join(self.save_dir, save_filename_tmp)
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        torch.save(network.cpu().state_dict(), save_path_tmp)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])
        save_filename = '%s_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        if os.path.isfile(save_path):
            os.remove(save_path)
        os.rename(save_path_tmp,save_path)

    def save_optimizer(self, optimizer, optimizer_label, epoch_label):
        save_filename_tmp = '%s_%s_tmp.pth' % (epoch_label, optimizer_label)
        save_path_tmp = os.path.join(self.save_dir, save_filename_tmp)
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        torch.save(optimizer.state_dict(), save_path_tmp)
        save_filename= '%s_%s.pth' % (epoch_label, optimizer_label)
        save_path = os.path.join(self.save_dir, save_filename)
        if os.path.isfile(save_path):
            os.remove(save_path)
        os.rename(save_path_tmp,save_path)


    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label,load_path=''):
        save_filename = '%s_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir if load_path=='' else load_path, save_filename)
        network.load_state_dict(torch.load(save_path))

    def load_optimizer(self, optimizer, optimizer_label, epoch_label):
        save_filename = '%s_%s.pth' % (epoch_label, optimizer_label)
        save_path = os.path.join(self.save_dir, save_filename)
        optimizer.load_state_dict(torch.load(save_path))

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
        return lr