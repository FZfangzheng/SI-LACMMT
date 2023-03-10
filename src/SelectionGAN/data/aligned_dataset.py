import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image

class AlignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase+"A")
        self.dir_B = os.path.join(opt.dataroot, opt.phase+"B")
        # self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.data_len = len(self.A_paths)
        # self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]

        CD_index = random.randint(1, self.data_len) - 1
        C_path = self.A_paths[CD_index]
        D_path = self.B_paths[CD_index]

        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')
        C = Image.open(C_path).convert('RGB')
        D = Image.open(D_path).convert('RGB')


        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        C = transforms.ToTensor()(C)
        D = transforms.ToTensor()(D)


        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)
        C = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(C)
        D = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(D)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        return {'A': A, 'B': B, 'C': C, 'D': D,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset'
# class AlignedDataset(BaseDataset):
#     @staticmethod
#     def modify_commandline_options(parser, is_train):
#         return parser
#
#     def initialize(self, opt):
#         self.opt = opt
#         self.root = opt.dataroot
#         self.dir_AB = os.path.join(opt.dataroot, opt.phase)
#         self.AB_paths = sorted(make_dataset(self.dir_AB))
#         assert(opt.resize_or_crop == 'resize_and_crop')
#
#     def __getitem__(self, index):
#         AB_path = self.AB_paths[index]
#         ABCD = Image.open(AB_path).convert('RGB')
#         w, h = ABCD.size
#         w2 = int(w / 4)
#         A = ABCD.crop((0, 0, w2, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
#         B = ABCD.crop((w2, 0, w2+w2, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
#         C = ABCD.crop((w2+w2, 0, w2+w2+w2, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
#         D = ABCD.crop((w2+w2+w2, 0, w, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
#
#         A = transforms.ToTensor()(A)
#         B = transforms.ToTensor()(B)
#         C = transforms.ToTensor()(C)
#         D = transforms.ToTensor()(D)
#         w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
#         h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
#
#         A = A[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
#         B = B[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
#         C = C[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
#         D = D[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
#
#         A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
#         B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)
#         C = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(C)
#         D = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(D)
#
#         if self.opt.which_direction == 'BtoA':
#             input_nc = self.opt.output_nc
#             output_nc = self.opt.input_nc
#         else:
#             input_nc = self.opt.input_nc
#             output_nc = self.opt.output_nc
#
#         if (not self.opt.no_flip) and random.random() < 0.5:
#             idx = [i for i in range(A.size(2) - 1, -1, -1)]
#             idx = torch.LongTensor(idx)
#             A = A.index_select(2, idx)
#             B = B.index_select(2, idx)
#
#         if input_nc == 1:  # RGB to gray
#             tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
#             A = tmp.unsqueeze(0)
#
#         if output_nc == 1:  # RGB to gray
#             tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
#             B = tmp.unsqueeze(0)
#
#         return {'A': A, 'B': B, 'C': C, 'D': D,
#                 'A_paths': AB_path, 'B_paths': AB_path}
#
#     def __len__(self):
#         return len(self.AB_paths)
#
#     def name(self):
#         return 'AlignedDataset'
