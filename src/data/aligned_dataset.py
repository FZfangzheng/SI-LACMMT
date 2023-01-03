import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
from util.my_util import get_inner_path
import cv2
import numpy as np
import math
import random
from sklearn.decomposition import PCA
import torch.nn.functional as F
from tqdm import tqdm


def make_align_img(dir_A,dir_B,dir_AB):
    print("Align data floder creating!")
    num=0
    imgs_A=sorted(make_dataset(dir_A))
    imgs_B=sorted(make_dataset(dir_B))
    imgs_A_=[]
    imgs_B_=[]
    for img_A in imgs_A:
        imgs_A_.append(os.path.splitext(img_A)[0])
    for img_B in imgs_B:
        imgs_B_.append(os.path.splitext(img_B)[0])
    for i in range(len(imgs_A)):
        img_A=imgs_A[i]
        img_inner = get_inner_path(img_A, dir_A)
        if get_inner_path(imgs_A_[i], dir_A) == get_inner_path(imgs_B_[i],dir_B):
            photo_A = cv2.imread(img_A)
            print(img_A)
            photo_B = cv2.imread(imgs_B[i])
            print(imgs_B[i])
            if photo_A.shape == photo_B.shape:
                photo_AB = np.concatenate([photo_A, photo_B], 1)
                img_AB = os.path.join(dir_AB, os.path.splitext(img_inner)[0]+'.png')
                if not os.path.isdir(os.path.split(img_AB)[0]):
                    os.makedirs(os.path.split(img_AB)[0])
                cv2.imwrite(img_AB, photo_AB)
                num += 1
    # for img_A in tqdm(imgs_A):
    #     img_inner=get_inner_path(img_A,dir_A)
    #     if os.path.join(dir_B,img_inner) in imgs_B:
    #         photo_A=cv2.imread(img_A)
    #         photo_B=cv2.imread(os.path.join(dir_B,img_inner))
    #         if photo_A.shape==photo_B.shape:
    #             photo_AB=np.concatenate([photo_A, photo_B], 1)
    #             img_AB=os.path.join(dir_AB,img_inner)
    #             if not os.path.isdir(os.path.split(img_AB)[0]):
    #                 os.makedirs(os.path.split(img_AB)[0])
    #             cv2.imwrite(img_AB, photo_AB)
    #             num+=1
    print("Align data floder created! %d img was processed"%num)
class AlignedDataset_avg(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        assert (os.path.isdir(self.dir_AB) or (os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'))),"dataset path not exsit! [%s]"%self.dir_AB
        if (not os.path.isdir(self.dir_AB)) and os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'):#to do:数据集AB自动合并功能尚不支持英文路径
            os.makedirs(self.dir_AB)
            dir_A=self.dir_AB+'A'
            dir_B=self.dir_AB+'B'
            make_align_img(dir_A,dir_B,self.dir_AB)

        self.AB_paths = sorted(make_dataset(self.dir_AB))  #make_dataset:返回一个所有图像格式文件（含路径）的list
        #统计层级个数
        self.AB_layer1=[]
        self.AB_layer2=[]
        self.AB_layer3=[]
        self.AB_layer4=[]

        for img_path in self.AB_paths:
            img_name = os.path.split(img_path)[-1]
            layer_num = img_name.split("_")[0]
            if layer_num==1:
                self.AB_layer1.append(img_path)
            if layer_num==2:
                self.AB_layer2.append(img_path)
            if layer_num==1:
                self.AB_layer3.append(img_path)
            if layer_num==1:
                self.AB_layer4.append(img_path)
        print("统计层级比例")
        assert (len(self.AB_paths)!=0),"dataset floder is empty! [%s]"%self.dir_AB

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合

        #deeplabv3+相关
        self.dir_segs=os.path.join(opt.dataroot, opt.phase+"_seg")
        from src.util.make_seg_img import labelpixels
        flag=0
        if os.path.isdir(self.dir_segs):
            segs = make_dataset(self.dir_segs)
            if len(self.AB_paths) == len(segs):
                flag=1
            else:
                import shutil
                shutil.rmtree(self.dir_segs)
        if flag==0:
            os.makedirs(self.dir_segs)
            for map in self.AB_paths:
                map_np_AB = np.array(Image.open(map).convert("RGB"))
                map_np=map_np_AB[:,map_np_AB.shape[1]//2:,:]
                seg_np = labelpixels(map_np).astype(np.uint8)
                seg_pil = Image.fromarray(seg_np)
                seg_path = os.path.join(self.dir_segs, get_inner_path(map, self.dir_AB))
                seg_pil.save(seg_path)
        self.seg_paths=sorted(make_dataset(self.dir_segs))

    def __getitem__(self, index):

        new_index = random.randint(1,4)
        if new_index == 1:
            layer1_amount = len(self.AB_layer1)
            random_select_id = random.randint(1,layer1_amount)-1
            AB_path = self.AB_layer1[random_select_id]
        if new_index == 2:
            layer2_amount = len(self.AB_layer2)
            random_select_id = random.randint(1,layer2_amount)-1
            AB_path = self.AB_layer2[random_select_id]
        if new_index == 3:
            layer3_amount = len(self.AB_layer3)
            random_select_id = random.randint(1,layer3_amount)-1
            AB_path = self.AB_layer3[random_select_id]
        if new_index == 4:
            layer4_amount = len(self.AB_layer4)
            random_select_id = random.randint(1,layer4_amount)-1
            AB_path = self.AB_layer4[random_select_id]



        # AB_path = self.AB_paths[index]
        img_name = os.path.split(AB_path)[-1]
        AB = Image.open(AB_path).convert('RGB')
        AB = AB.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
        AB = self.transform(AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = AB[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]

        # deeplabv3相关，取label图，之后需将label图与A，B同步处理
        seg_path = os.path.join(self.dir_segs,str(new_index),img_name)
        # seg_path = self.seg_paths[index]
        seg = Image.open(seg_path)
        seg = np.asarray(seg)
        seg = torch.from_numpy(seg)  # H W
        seg = seg[h_offset:h_offset + self.opt.fineSize,
              w_offset:w_offset + self.opt.fineSize]

        if False: #self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        # self.opt.no_flip=False
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)  # 64bit带符号整型，why？因为index_select要求下标为longtensor类型
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            seg=seg.index_select(1,idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)



        # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
        toseg_transform_list = [transforms.Normalize((0, 0, 0),(2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                transforms.Normalize((-0.5,-0.5,-0.5),(1,1,1)),
                                transforms.Normalize((0.485,0.456,0.406),((0.229,0.224,0.225)))] # 恢复标准化前的数值，并换一组数据标准化
        toseg_transform = transforms.Compose(toseg_transform_list)
        A_seg=toseg_transform(A)
        B_seg=toseg_transform(B)

        return {'A': A, 'B': B, 'seg':seg,'A_seg':A_seg,'B_seg':B_seg,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset_avg'

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        assert (os.path.isdir(self.dir_AB) or (os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'))),"dataset path not exsit! [%s]"%self.dir_AB
        if (not os.path.isdir(self.dir_AB)) and os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'):#to do:数据集AB自动合并功能尚不支持英文路径
            os.makedirs(self.dir_AB)
            dir_A=self.dir_AB+'A'
            dir_B=self.dir_AB+'B'
            make_align_img(dir_A,dir_B,self.dir_AB)

        self.AB_paths = sorted(make_dataset(self.dir_AB))  #make_dataset:返回一个所有图像格式文件（含路径）的list
        assert (len(self.AB_paths)!=0),"dataset floder is empty! [%s]"%self.dir_AB
        print(opt.resize_or_crop)
        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合

        #deeplabv3+相关
        self.dir_segs=os.path.join(opt.dataroot, opt.phase+"_seg")
        from src.util.make_seg_img import labelpixels
        flag=0
        if os.path.isdir(self.dir_segs):
            segs = make_dataset(self.dir_segs)
            if len(self.AB_paths) == len(segs):
                flag=1
            else:
                import shutil
                shutil.rmtree(self.dir_segs)
        if flag==0:
            os.makedirs(self.dir_segs)
            for map in self.AB_paths:
                map_np_AB = np.array(Image.open(map).convert("RGB"))
                map_np=map_np_AB[:,map_np_AB.shape[1]//2:,:]
                seg_np = labelpixels(map_np).astype(np.uint8)
                seg_pil = Image.fromarray(seg_np)
                seg_path = os.path.join(self.dir_segs, get_inner_path(map, self.dir_AB))
                seg_pil.save(seg_path)
        self.seg_paths=sorted(make_dataset(self.dir_segs))

    def __getitem__(self, index):
        # print(len(self.AB_paths))
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        AB = AB.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
        AB = self.transform(AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = AB[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]

        # deeplabv3相关，取label图，之后需将label图与A，B同步处理
        seg_path = self.seg_paths[index]
        seg = Image.open(seg_path)
        seg = np.asarray(seg)
        seg = torch.from_numpy(seg)  # H W
        seg = seg[h_offset:h_offset + self.opt.fineSize,
              w_offset:w_offset + self.opt.fineSize]

        if False: #self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        # self.opt.no_flip=False
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)  # 64bit带符号整型，why？因为index_select要求下标为longtensor类型
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            seg=seg.index_select(1,idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)



        # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
        toseg_transform_list = [transforms.Normalize((0, 0, 0),(2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                transforms.Normalize((-0.5,-0.5,-0.5),(1,1,1)),
                                transforms.Normalize((0.485,0.456,0.406),((0.229,0.224,0.225)))] # 恢复标准化前的数值，并换一组数据标准化
        toseg_transform = transforms.Compose(toseg_transform_list)
        A_seg=toseg_transform(A)
        B_seg=toseg_transform(B)
        # print(AB_path)

        return {'A': A, 'B': B, 'seg':seg,'A_seg':A_seg,'B_seg':B_seg,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'


class AlignedDataset_frac_1_3(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        assert (os.path.isdir(self.dir_AB) or (os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'))),"dataset path not exsit! [%s]"%self.dir_AB
        if (not os.path.isdir(self.dir_AB)) and os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'):#to do:数据集AB自动合并功能尚不支持英文路径
            os.makedirs(self.dir_AB)
            dir_A=self.dir_AB+'A'
            dir_B=self.dir_AB+'B'
            make_align_img(dir_A,dir_B,self.dir_AB)

        self.AB_paths = sorted(make_dataset(self.dir_AB))  #make_dataset:返回一个所有图像格式文件（含路径）的list
        assert (len(self.AB_paths)!=0),"dataset floder is empty! [%s]"%self.dir_AB

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合

        #deeplabv3+相关
        self.dir_segs=os.path.join(opt.dataroot, opt.phase+"_seg")
        from src.util.make_seg_img import labelpixels
        flag=0
        if os.path.isdir(self.dir_segs):
            segs = make_dataset(self.dir_segs)
            if len(self.AB_paths) == len(segs):
                flag=1
            else:
                import shutil
                shutil.rmtree(self.dir_segs)
        if flag==0:
            os.makedirs(self.dir_segs)
            for map in self.AB_paths:
                map_np_AB = np.array(Image.open(map).convert("RGB"))
                map_np=map_np_AB[:,map_np_AB.shape[1]//2:,:]
                seg_np = labelpixels(map_np).astype(np.uint8)
                seg_pil = Image.fromarray(seg_np)
                seg_path = os.path.join(self.dir_segs, get_inner_path(map, self.dir_AB))
                seg_pil.save(seg_path)
        self.seg_paths=sorted(make_dataset(self.dir_segs))

    def __getitem__(self, index):
        index = random.randint(0, int(len(self.AB_paths)-1))

        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        AB = AB.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
        AB = self.transform(AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = AB[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]

        # deeplabv3相关，取label图，之后需将label图与A，B同步处理
        seg_path = self.seg_paths[index]
        seg = Image.open(seg_path)
        seg = np.asarray(seg)
        seg = torch.from_numpy(seg)  # H W
        seg = seg[h_offset:h_offset + self.opt.fineSize,
              w_offset:w_offset + self.opt.fineSize]

        if False: #self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        # self.opt.no_flip=False
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)  # 64bit带符号整型，why？因为index_select要求下标为longtensor类型
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            seg=seg.index_select(1,idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)



        # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
        toseg_transform_list = [transforms.Normalize((0, 0, 0),(2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                transforms.Normalize((-0.5,-0.5,-0.5),(1,1,1)),
                                transforms.Normalize((0.485,0.456,0.406),((0.229,0.224,0.225)))] # 恢复标准化前的数值，并换一组数据标准化
        toseg_transform = transforms.Compose(toseg_transform_list)
        A_seg=toseg_transform(A)
        B_seg=toseg_transform(B)

        return {'A': A, 'B': B, 'seg':seg,'A_seg':A_seg,'B_seg':B_seg,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return int(len(self.AB_paths)/4)

    def name(self):
        return 'AlignedDataset_frac_1_3'


class AlignedDataset_pca_frac_1_4(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        assert (os.path.isdir(self.dir_AB) or (os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'))),"dataset path not exsit! [%s]"%self.dir_AB
        if (not os.path.isdir(self.dir_AB)) and os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'):#to do:数据集AB自动合并功能尚不支持英文路径
            os.makedirs(self.dir_AB)
            dir_A=self.dir_AB+'A'
            dir_B=self.dir_AB+'B'
            make_align_img(dir_A,dir_B,self.dir_AB)

        self.AB_paths = sorted(make_dataset(self.dir_AB))  #make_dataset:返回一个所有图像格式文件（含路径）的list
        assert (len(self.AB_paths)!=0),"dataset floder is empty! [%s]"%self.dir_AB

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合

        #deeplabv3+相关
        self.dir_segs=os.path.join(opt.dataroot, opt.phase+"_seg")
        from src.util.make_seg_img import labelpixels
        flag=0
        if os.path.isdir(self.dir_segs):
            segs = make_dataset(self.dir_segs)
            if len(self.AB_paths) == len(segs):
                flag=1
            else:
                import shutil
                shutil.rmtree(self.dir_segs)
        if flag==0:
            os.makedirs(self.dir_segs)
            for map in self.AB_paths:
                map_np_AB = np.array(Image.open(map).convert("RGB"))
                map_np=map_np_AB[:,map_np_AB.shape[1]//2:,:]
                seg_np = labelpixels(map_np).astype(np.uint8)
                seg_pil = Image.fromarray(seg_np)
                seg_path = os.path.join(self.dir_segs, get_inner_path(map, self.dir_AB))
                seg_pil.save(seg_path)
        self.seg_paths=sorted(make_dataset(self.dir_segs))

    def __getitem__(self, index):
        index = random.randint(0, int(len(self.AB_paths)-1))

        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        AB = AB.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
        AB = self.transform(AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = AB[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]

        # deeplabv3相关，取label图，之后需将label图与A，B同步处理
        seg_path = self.seg_paths[index]
        seg = Image.open(seg_path)
        seg = np.asarray(seg)
        seg = torch.from_numpy(seg)  # H W
        seg = seg[h_offset:h_offset + self.opt.fineSize,
              w_offset:w_offset + self.opt.fineSize]

        if False: #self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        # self.opt.no_flip=False
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)  # 64bit带符号整型，why？因为index_select要求下标为longtensor类型
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            seg=seg.index_select(1,idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)



        # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
        toseg_transform_list = [transforms.Normalize((0, 0, 0),(2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                transforms.Normalize((-0.5,-0.5,-0.5),(1,1,1)),
                                transforms.Normalize((0.485,0.456,0.406),((0.229,0.224,0.225)))] # 恢复标准化前的数值，并换一组数据标准化
        toseg_transform = transforms.Compose(toseg_transform_list)
        A_seg=toseg_transform(A)
        B_seg=toseg_transform(B)
        # 15-18 ---> 0-3
        domain = int(os.path.split(AB_path)[0].split(os.sep)[-1])-15
        label_domain = np.zeros(4)
        label_domain[domain] = 1
        label_domain = torch.from_numpy(label_domain)

        ##pca deal
        A_numpy = np.array(A)
        # print('A_numpy.shape',A_numpy.shape)
        A_c, A_h, A_w = A_numpy.shape
        Rimg = A_numpy[0]
        Gimg = A_numpy[1]
        Bimg = A_numpy[2]
        # print(R.shape)
        block_size = 16
        R_Blocks = []
        G_Blocks = []
        B_Blocks = []
        for i in range(int(A_h / block_size)):
            for j in range(int(A_w / block_size)):
                R_block = Rimg[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                R_block_flatten = R_block.flatten()
                # print('R_block_flatten.shape',R_block_flatten.shape)
                R_Blocks.append(R_block_flatten)

                G_block = Gimg[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                G_block_flatten = G_block.flatten()
                G_Blocks.append(G_block_flatten)

                B_block = Bimg[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                B_block_flatten = B_block.flatten()
                B_Blocks.append(B_block_flatten)

        R_Blocks = np.array(R_Blocks)
        G_Blocks = np.array(G_Blocks)
        B_Blocks = np.array(B_Blocks)
        # print('B_Blocks.shape',B_Blocks.shape)

        pca_R = PCA(n_components=256)
        pca_R.fit(R_Blocks)
        out_r = pca_R.transform(R_Blocks)
        # out_r = torch.from_numpy(out_r)

        pca_G = PCA(n_components=256)
        pca_G.fit(G_Blocks)
        out_g = pca_G.transform(G_Blocks)
        # out_g = torch.from_numpy(out_g)

        pca_B = PCA(n_components=256)
        pca_B.fit(B_Blocks)
        out_b = pca_B.transform(B_Blocks)

        pca_out = [out_r, out_g, out_b]
        pca_out = np.array(pca_out)
        pca_out = torch.from_numpy(pca_out)

        # pca_out = torch.cat((out_r, out_g, out_b))

        return {'A': A, 'B': B, 'seg': seg, 'A_seg': A_seg, 'B_seg': B_seg,
                'A_paths': AB_path, 'B_paths': AB_path, 'domain': label_domain, 'pca': pca_out}

    def __len__(self):
        return int(len(self.AB_paths)/4)

    def name(self):
        return 'AlignedDataset_pca_frac_1_4'


class AlignedDataset_pca(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        assert (os.path.isdir(self.dir_AB) or (os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'))),"dataset path not exsit! [%s]"%self.dir_AB
        if (not os.path.isdir(self.dir_AB)) and os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'):#to do:数据集AB自动合并功能尚不支持英文路径
            os.makedirs(self.dir_AB)
            dir_A=self.dir_AB+'A'
            dir_B=self.dir_AB+'B'
            make_align_img(dir_A,dir_B,self.dir_AB)

        self.AB_paths = sorted(make_dataset(self.dir_AB))  #make_dataset:返回一个所有图像格式文件（含路径）的list
        assert (len(self.AB_paths)!=0),"dataset floder is empty! [%s]"%self.dir_AB

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合

        #deeplabv3+相关
        self.dir_segs=os.path.join(opt.dataroot, opt.phase+"_seg")
        from src.util.make_seg_img import labelpixels
        flag=0
        if os.path.isdir(self.dir_segs):
            segs = make_dataset(self.dir_segs)
            if len(self.AB_paths) == len(segs):
                flag=1
            else:
                import shutil
                shutil.rmtree(self.dir_segs)
        if flag==0:
            os.makedirs(self.dir_segs)
            for map in self.AB_paths:
                map_np_AB = np.array(Image.open(map).convert("RGB"))
                map_np=map_np_AB[:,map_np_AB.shape[1]//2:,:]
                seg_np = labelpixels(map_np).astype(np.uint8)
                seg_pil = Image.fromarray(seg_np)
                seg_path = os.path.join(self.dir_segs, get_inner_path(map, self.dir_AB))
                seg_pil.save(seg_path)
        self.seg_paths=sorted(make_dataset(self.dir_segs))

    def __getitem__(self, index):
        # index = random.randint(0, int(len(self.AB_paths)-1))

        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        AB = AB.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
        AB = self.transform(AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = AB[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]

        # deeplabv3相关，取label图，之后需将label图与A，B同步处理
        seg_path = self.seg_paths[index]
        seg = Image.open(seg_path)
        seg = np.asarray(seg)
        seg = torch.from_numpy(seg)  # H W
        seg = seg[h_offset:h_offset + self.opt.fineSize,
              w_offset:w_offset + self.opt.fineSize]

        if False: #self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        # self.opt.no_flip=False
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)  # 64bit带符号整型，why？因为index_select要求下标为longtensor类型
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            seg=seg.index_select(1,idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)



        # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
        toseg_transform_list = [transforms.Normalize((0, 0, 0),(2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                transforms.Normalize((-0.5,-0.5,-0.5),(1,1,1)),
                                transforms.Normalize((0.485,0.456,0.406),((0.229,0.224,0.225)))] # 恢复标准化前的数值，并换一组数据标准化
        toseg_transform = transforms.Compose(toseg_transform_list)
        A_seg=toseg_transform(A)
        B_seg=toseg_transform(B)
        # 15-18 ---> 0-3
        domain = int(os.path.split(AB_path)[0].split(os.sep)[-1])-15
        label_domain = np.zeros(4)
        label_domain[domain] = 1
        label_domain = torch.from_numpy(label_domain)

        ##pca deal
        A_numpy = np.array(A)
        # print('A_numpy.shape',A_numpy.shape)
        A_c, A_h, A_w = A_numpy.shape
        Rimg = A_numpy[0]
        Gimg = A_numpy[1]
        Bimg = A_numpy[2]
        # print(R.shape)
        block_size = 16
        R_Blocks = []
        G_Blocks = []
        B_Blocks = []
        for i in range(int(A_h / block_size)):
            for j in range(int(A_w / block_size)):
                R_block = Rimg[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                R_block_flatten = R_block.flatten()
                # print('R_block_flatten.shape',R_block_flatten.shape)
                R_Blocks.append(R_block_flatten)

                G_block = Gimg[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                G_block_flatten = G_block.flatten()
                G_Blocks.append(G_block_flatten)

                B_block = Bimg[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                B_block_flatten = B_block.flatten()
                B_Blocks.append(B_block_flatten)

        R_Blocks = np.array(R_Blocks)
        G_Blocks = np.array(G_Blocks)
        B_Blocks = np.array(B_Blocks)
        # print('B_Blocks.shape',B_Blocks.shape)

        pca_R = PCA(n_components=256)
        pca_R.fit(R_Blocks)
        out_r = pca_R.transform(R_Blocks)
        # out_r = torch.from_numpy(out_r)

        pca_G = PCA(n_components=256)
        pca_G.fit(G_Blocks)
        out_g = pca_G.transform(G_Blocks)
        # out_g = torch.from_numpy(out_g)

        pca_B = PCA(n_components=256)
        pca_B.fit(B_Blocks)
        out_b = pca_B.transform(B_Blocks)

        pca_out = [out_r, out_g, out_b]
        pca_out = np.array(pca_out)
        pca_out = torch.from_numpy(pca_out)

        # pca_out = torch.cat((out_r, out_g, out_b))

        return {'A': A, 'B': B, 'seg': seg, 'A_seg': A_seg, 'B_seg': B_seg,
                'A_paths': AB_path, 'B_paths': AB_path, 'domain': label_domain, 'pca': pca_out}

    def __len__(self):
        # return int(len(self.AB_paths)/4)
        return len(self.AB_paths)
    def name(self):
        return 'AlignedDataset_pca'

class AlignedDataset_starganv1_2_frac_1_4(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        assert (os.path.isdir(self.dir_AB) or (os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'))),"dataset path not exsit! [%s]"%self.dir_AB
        if (not os.path.isdir(self.dir_AB)) and os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'):#to do:数据集AB自动合并功能尚不支持英文路径
            os.makedirs(self.dir_AB)
            dir_A=self.dir_AB+'A'
            dir_B=self.dir_AB+'B'
            make_align_img(dir_A,dir_B,self.dir_AB)

        self.AB_paths = sorted(make_dataset(self.dir_AB))  #make_dataset:返回一个所有图像格式文件（含路径）的list
        assert (len(self.AB_paths)!=0),"dataset floder is empty! [%s]"%self.dir_AB

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合

        #deeplabv3+相关
        self.dir_segs=os.path.join(opt.dataroot, opt.phase+"_seg")
        from src.util.make_seg_img import labelpixels
        flag=0
        if os.path.isdir(self.dir_segs):
            segs = make_dataset(self.dir_segs)
            if len(self.AB_paths) == len(segs):
                flag=1
            else:
                import shutil
                shutil.rmtree(self.dir_segs)
        if flag==0:
            os.makedirs(self.dir_segs)
            for map in self.AB_paths:
                map_np_AB = np.array(Image.open(map).convert("RGB"))
                map_np=map_np_AB[:,map_np_AB.shape[1]//2:,:]
                seg_np = labelpixels(map_np).astype(np.uint8)
                seg_pil = Image.fromarray(seg_np)
                seg_path = os.path.join(self.dir_segs, get_inner_path(map, self.dir_AB))
                seg_pil.save(seg_path)
        self.seg_paths=sorted(make_dataset(self.dir_segs))

    def __getitem__(self, index):
        index = random.randint(0, int(len(self.AB_paths)-1))

        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        AB = AB.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
        AB = self.transform(AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = AB[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]

        # deeplabv3相关，取label图，之后需将label图与A，B同步处理
        seg_path = self.seg_paths[index]
        seg = Image.open(seg_path)
        seg = np.asarray(seg)
        seg = torch.from_numpy(seg)  # H W
        seg = seg[h_offset:h_offset + self.opt.fineSize,
              w_offset:w_offset + self.opt.fineSize]

        if False: #self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        # self.opt.no_flip=False
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)  # 64bit带符号整型，why？因为index_select要求下标为longtensor类型
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            seg=seg.index_select(1,idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)



        # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
        toseg_transform_list = [transforms.Normalize((0, 0, 0),(2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                transforms.Normalize((-0.5,-0.5,-0.5),(1,1,1)),
                                transforms.Normalize((0.485,0.456,0.406),((0.229,0.224,0.225)))] # 恢复标准化前的数值，并换一组数据标准化
        toseg_transform = transforms.Compose(toseg_transform_list)
        A_seg=toseg_transform(A)
        B_seg=toseg_transform(B)
        # 15-18 ---> 0-3
        domain = int(os.path.split(AB_path)[0].split(os.sep)[-1])-15
        domain = np.array(domain)
        domain = torch.from_numpy(domain)
        # label_domain = np.zeros(4)
        # label_domain[domain] = 1
        # label_domain = torch.from_numpy(label_domain)
        return {'A': A, 'B': B, 'seg':seg,'A_seg':A_seg,'B_seg':B_seg,
                'A_paths': AB_path, 'B_paths': AB_path, 'domain':domain}

    def __len__(self):
        return int(len(self.AB_paths)/4)

    def name(self):
        return 'AlignedDataset_starganv1_2_frac_1_4'


class AlignedDataset_downsample_frac_1_4(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.dir_A = self.dir_AB+'A'
        self.dir_B = self.dir_AB+'B'
        self.dir_C = self.dir_AB+'C'

        # self.AB_paths = sorted(make_dataset(self.dir_AB))  #make_dataset:返回一个所有图像格式文件（含路径）的list
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.C_paths = sorted(make_dataset(self.dir_C))


        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合

        #deeplabv3+相关
        self.dir_segs=os.path.join(opt.dataroot, opt.phase+"_seg")
        self.seg_paths=sorted(make_dataset(self.dir_segs))

    def __getitem__(self, index):
        index = random.randint(0, int(len(self.A_paths)-1))
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        C_path = self.C_paths[index]

        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')
        C = Image.open(C_path).convert('RGB')

        A = self.transform(A)
        B = self.transform(B)
        C = self.transform(C)


        # deeplabv3相关，取label图，之后需将label图与A，B同步处理
        seg_path = self.seg_paths[index]
        seg = Image.open(seg_path)
        seg = np.asarray(seg)
        seg = torch.from_numpy(seg)  # H W


        # self.opt.no_flip=False
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)  # 64bit带符号整型，why？因为index_select要求下标为longtensor类型
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            seg=seg.index_select(1,idx)

            idx_c = [i for i in range(C.size(2) - 1, -1, -1)]
            idx_c = torch.LongTensor(idx_c)
            C = C.index_select(2, idx_c)


        # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
        toseg_transform_list = [transforms.Normalize((0, 0, 0),(2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                transforms.Normalize((-0.5,-0.5,-0.5),(1,1,1)),
                                transforms.Normalize((0.485,0.456,0.406),((0.229,0.224,0.225)))] # 恢复标准化前的数值，并换一组数据标准化
        toseg_transform = transforms.Compose(toseg_transform_list)
        A_seg=toseg_transform(A)
        C_seg = toseg_transform(C)
        # 15-18 ---> 0-3
        domain = int(os.path.split(A_path)[0].split(os.sep)[-1])-15
        domain = np.array(domain)
        domain = torch.from_numpy(domain)

        domain_c = int(os.path.split(C_path)[0].split(os.sep)[-1])-16
        if domain_c<0:
            domain_c=0
        domain_c = np.array(domain_c)
        domain_c = torch.from_numpy(domain_c)


        return {'A': A, 'B': B,'C':C, 'seg':seg,'A_seg':A_seg,'C_seg':C_seg,'A_paths': A_path, 'B_paths': B_path, 'domain':domain,'domain_c':domain_c}

    def __len__(self):
        return int(len(self.A_paths)/4)
        # return len(self.A_paths)
    def name(self):
        return 'AlignedDataset_downsample_frac_1_4'


class AlignedDataset_downsample(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.dir_A = self.dir_AB+'A'
        self.dir_B = self.dir_AB+'B'
        self.dir_C = self.dir_AB+'C'

        # self.AB_paths = sorted(make_dataset(self.dir_AB))  #make_dataset:返回一个所有图像格式文件（含路径）的list
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.C_paths = sorted(make_dataset(self.dir_C))


        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合

        #deeplabv3+相关
        self.dir_segs=os.path.join(opt.dataroot, opt.phase+"_seg")
        self.seg_paths=sorted(make_dataset(self.dir_segs))

    def __getitem__(self, index):
        # index = random.randint(0, int(len(self.AB_paths)-1))
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        C_path = self.C_paths[index]

        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')
        C = Image.open(C_path).convert('RGB')

        A = self.transform(A)
        B = self.transform(B)
        C = self.transform(C)


        # deeplabv3相关，取label图，之后需将label图与A，B同步处理
        seg_path = self.seg_paths[index]
        seg = Image.open(seg_path)
        seg = np.asarray(seg)
        seg = torch.from_numpy(seg)  # H W


        # self.opt.no_flip=False
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)  # 64bit带符号整型，why？因为index_select要求下标为longtensor类型
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            seg=seg.index_select(1,idx)

            idx_c = [i for i in range(C.size(2) - 1, -1, -1)]
            idx_c = torch.LongTensor(idx_c)
            C = C.index_select(2, idx_c)


        # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
        toseg_transform_list = [transforms.Normalize((0, 0, 0),(2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                transforms.Normalize((-0.5,-0.5,-0.5),(1,1,1)),
                                transforms.Normalize((0.485,0.456,0.406),((0.229,0.224,0.225)))] # 恢复标准化前的数值，并换一组数据标准化
        toseg_transform = transforms.Compose(toseg_transform_list)
        A_seg=toseg_transform(A)
        C_seg = toseg_transform(C)
        # 15-18 ---> 0-3
        domain = int(os.path.split(A_path)[0].split(os.sep)[-1])-15
        domain = np.array(domain)
        domain = torch.from_numpy(domain)

        domain_c = int(os.path.split(C_path)[0].split(os.sep)[-1])-14
        if domain_c>3:
            domain_c=3
        domain_c = np.array(domain_c)
        domain_c = torch.from_numpy(domain_c)


        return {'A': A, 'B': B,'C':C, 'seg':seg,'A_seg':A_seg,'C_seg':C_seg,'A_paths': A_path, 'B_paths': B_path, 'domain':domain,'domain_c':domain_c}

    def __len__(self):
        # return int(len(self.AB_paths)/4)
        return len(self.A_paths)
    def name(self):
        return 'AlignedDataset_downsample'

class AlignedDataset_downsample_v18(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.dir_A = self.dir_AB+'A'
        self.dir_B = self.dir_AB+'B'
        self.dir_C = self.dir_AB+'C'

        # self.AB_paths = sorted(make_dataset(self.dir_AB))  #make_dataset:返回一个所有图像格式文件（含路径）的list
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.C_paths = sorted(make_dataset(self.dir_C))


        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合

        #deeplabv3+相关
        self.dir_segs=os.path.join(opt.dataroot, opt.phase+"_seg")
        self.seg_paths=sorted(make_dataset(self.dir_segs))

    def __getitem__(self, index):
        # index = random.randint(0, int(len(self.AB_paths)-1))
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        C_path = self.C_paths[index]

        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')
        C = Image.open(C_path).convert('RGB')

        A = self.transform(A)
        B = self.transform(B)
        C = self.transform(C)


        # deeplabv3相关，取label图，之后需将label图与A，B同步处理
        seg_path = self.seg_paths[index]
        seg = Image.open(seg_path)
        seg = np.asarray(seg)
        seg = torch.from_numpy(seg)  # H W


        # self.opt.no_flip=False
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)  # 64bit带符号整型，why？因为index_select要求下标为longtensor类型
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            seg=seg.index_select(1,idx)

            idx_c = [i for i in range(C.size(2) - 1, -1, -1)]
            idx_c = torch.LongTensor(idx_c)
            C = C.index_select(2, idx_c)


        # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
        toseg_transform_list = [transforms.Normalize((0, 0, 0),(2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                transforms.Normalize((-0.5,-0.5,-0.5),(1,1,1)),
                                transforms.Normalize((0.485,0.456,0.406),((0.229,0.224,0.225)))] # 恢复标准化前的数值，并换一组数据标准化
        toseg_transform = transforms.Compose(toseg_transform_list)
        A_seg=toseg_transform(A)
        C_seg = toseg_transform(C)
        # 15-18 ---> 0-3
        domain = int(os.path.split(A_path)[0].split(os.sep)[-1])
        domain = np.array(domain)
        domain = torch.from_numpy(domain)

        domain_c = int(os.path.split(C_path)[0].split(os.sep)[-1])+1
        if domain_c>3:
            domain_c=3
        domain_c = np.array(domain_c)
        domain_c = torch.from_numpy(domain_c)


        return {'A': A, 'B': B,'C':C, 'seg':seg,'A_seg':A_seg,'C_seg':C_seg,'A_paths': A_path, 'B_paths': B_path, 'domain':domain,'domain_c':domain_c}

    def __len__(self):
        # return int(len(self.AB_paths)/4)
        return len(self.A_paths)
    def name(self):
        return 'AlignedDataset_downsample_v18'

class AlignedDataset_upsample(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.dir_A = self.dir_AB+'A'
        self.dir_B = self.dir_AB+'B'
        self.dir_C = self.dir_AB+'C'

        # self.AB_paths = sorted(make_dataset(self.dir_AB))  #make_dataset:返回一个所有图像格式文件（含路径）的list
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.C_paths = sorted(make_dataset(self.dir_C))


        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合

        #deeplabv3+相关
        self.dir_segs=os.path.join(opt.dataroot, opt.phase+"_seg")
        self.seg_paths=sorted(make_dataset(self.dir_segs))

    def __getitem__(self, index):
        # index = random.randint(0, int(len(self.AB_paths)-1))
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        C_path = self.C_paths[index]

        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')
        C = Image.open(C_path).convert('RGB')

        A = self.transform(A)
        B = self.transform(B)
        C = self.transform(C)


        # deeplabv3相关，取label图，之后需将label图与A，B同步处理
        seg_path = self.seg_paths[index]
        seg = Image.open(seg_path)
        seg = np.asarray(seg)
        seg = torch.from_numpy(seg)  # H W


        # self.opt.no_flip=False
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)  # 64bit带符号整型，why？因为index_select要求下标为longtensor类型
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            seg=seg.index_select(1,idx)

            idx_c = [i for i in range(C.size(2) - 1, -1, -1)]
            idx_c = torch.LongTensor(idx_c)
            C = C.index_select(2, idx_c)


        # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
        toseg_transform_list = [transforms.Normalize((0, 0, 0),(2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                transforms.Normalize((-0.5,-0.5,-0.5),(1,1,1)),
                                transforms.Normalize((0.485,0.456,0.406),((0.229,0.224,0.225)))] # 恢复标准化前的数值，并换一组数据标准化
        toseg_transform = transforms.Compose(toseg_transform_list)
        A_seg=toseg_transform(A)
        C_seg = toseg_transform(C)
        # 15-18 ---> 0-3
        domain = int(os.path.split(A_path)[0].split(os.sep)[-1])-15
        domain = np.array(domain)
        domain = torch.from_numpy(domain)

        domain_c = int(os.path.split(C_path)[0].split(os.sep)[-1])-16
        if domain_c<0:
            domain_c=0
        domain_c = np.array(domain_c)
        domain_c = torch.from_numpy(domain_c)


        return {'A': A, 'B': B,'C':C, 'seg':seg,'A_seg':A_seg,'C_seg':C_seg,'A_paths': A_path, 'B_paths': B_path, 'domain':domain,'domain_c':domain_c}

    def __len__(self):
        # return int(len(self.AB_paths)/4)
        return len(self.A_paths)
    def name(self):
        return 'AlignedDataset_upsample'


class AlignedDataset_downsample_eval(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.dir_A = self.dir_AB+'A'
        self.dir_B = self.dir_AB+'B'
        self.dir_C = self.dir_AB+'C'

        # self.AB_paths = sorted(make_dataset(self.dir_AB))  #make_dataset:返回一个所有图像格式文件（含路径）的list
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.C_paths = sorted(make_dataset(self.dir_C))


        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合

        #deeplabv3+相关
        self.dir_segs=os.path.join(opt.dataroot, opt.phase+"_seg")
        self.seg_paths=sorted(make_dataset(self.dir_segs))

    def __getitem__(self, index):
        # index = random.randint(0, int(len(self.AB_paths)-1))
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        C_path = self.C_paths[index]

        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')
        C = Image.open(C_path).convert('RGB')

        A = self.transform(A)
        B = self.transform(B)
        C = self.transform(C)


        # deeplabv3相关，取label图，之后需将label图与A，B同步处理
        seg_path = self.seg_paths[index]
        seg = Image.open(seg_path)
        seg = np.asarray(seg)
        seg = torch.from_numpy(seg)  # H W


        # self.opt.no_flip=False
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)  # 64bit带符号整型，why？因为index_select要求下标为longtensor类型
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            seg=seg.index_select(1,idx)

            idx_c = [i for i in range(C.size(2) - 1, -1, -1)]
            idx_c = torch.LongTensor(idx_c)
            C = C.index_select(2, idx_c)


        # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
        toseg_transform_list = [transforms.Normalize((0, 0, 0),(2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                transforms.Normalize((-0.5,-0.5,-0.5),(1,1,1)),
                                transforms.Normalize((0.485,0.456,0.406),((0.229,0.224,0.225)))] # 恢复标准化前的数值，并换一组数据标准化
        toseg_transform = transforms.Compose(toseg_transform_list)
        A_seg=toseg_transform(A)
        C_seg = toseg_transform(C)
        # 15-18 ---> 0-3
        domain = int(os.path.split(A_path)[0].split(os.sep)[-1])-15
        domain = np.array(domain)
        domain = torch.from_numpy(domain)

        domain_c = int(os.path.split(C_path)[0].split(os.sep)[-1])-16
        if domain_c<0:
            domain_c=0
        domain_c = np.array(domain_c)
        domain_c = torch.from_numpy(domain_c)


        return {'A': C, 'B': B,'C':A, 'seg':seg,'A_seg':C_seg,'C_seg':A_seg,'A_paths': A_path, 'B_paths': B_path, 'domain':domain,'domain_c':domain_c}

    def __len__(self):
        # return int(len(self.AB_paths)/4)
        return len(self.A_paths)
    def name(self):
        return 'AlignedDataset_downsample_eval'



class AlignedDataset_starganv1_2(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        assert (os.path.isdir(self.dir_AB) or (os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'))),"dataset path not exsit! [%s]"%self.dir_AB
        if (not os.path.isdir(self.dir_AB)) and os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'):#to do:数据集AB自动合并功能尚不支持英文路径
            os.makedirs(self.dir_AB)
            dir_A=self.dir_AB+'A'
            dir_B=self.dir_AB+'B'
            make_align_img(dir_A,dir_B,self.dir_AB)

        self.AB_paths = sorted(make_dataset(self.dir_AB))  #make_dataset:返回一个所有图像格式文件（含路径）的list
        assert (len(self.AB_paths)!=0),"dataset floder is empty! [%s]"%self.dir_AB

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合

        #deeplabv3+相关
        self.dir_segs=os.path.join(opt.dataroot, opt.phase+"_seg")
        from src.util.make_seg_img import labelpixels
        flag=0
        if os.path.isdir(self.dir_segs):
            segs = make_dataset(self.dir_segs)
            if len(self.AB_paths) == len(segs):
                flag=1
            else:
                import shutil
                shutil.rmtree(self.dir_segs)
        if flag==0:
            os.makedirs(self.dir_segs)
            for map in self.AB_paths:
                map_np_AB = np.array(Image.open(map).convert("RGB"))
                map_np=map_np_AB[:,map_np_AB.shape[1]//2:,:]
                seg_np = labelpixels(map_np).astype(np.uint8)
                seg_pil = Image.fromarray(seg_np)
                seg_path = os.path.join(self.dir_segs, get_inner_path(map, self.dir_AB))
                seg_pil.save(seg_path)
        self.seg_paths=sorted(make_dataset(self.dir_segs))

    def __getitem__(self, index):
        # index = random.randint(0, int(len(self.AB_paths)-1))

        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        AB = AB.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
        AB = self.transform(AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = AB[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]

        # deeplabv3相关，取label图，之后需将label图与A，B同步处理
        seg_path = self.seg_paths[index]
        seg = Image.open(seg_path)
        seg = np.asarray(seg)
        seg = torch.from_numpy(seg)  # H W
        seg = seg[h_offset:h_offset + self.opt.fineSize,
              w_offset:w_offset + self.opt.fineSize]

        if False: #self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        # self.opt.no_flip=False
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)  # 64bit带符号整型，why？因为index_select要求下标为longtensor类型
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            seg=seg.index_select(1,idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)



        # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
        toseg_transform_list = [transforms.Normalize((0, 0, 0),(2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                transforms.Normalize((-0.5,-0.5,-0.5),(1,1,1)),
                                transforms.Normalize((0.485,0.456,0.406),((0.229,0.224,0.225)))] # 恢复标准化前的数值，并换一组数据标准化
        toseg_transform = transforms.Compose(toseg_transform_list)
        A_seg=toseg_transform(A)
        B_seg=toseg_transform(B)
        # 15-18 ---> 0-3
        domain = int(os.path.split(AB_path)[0].split(os.sep)[-1])-15
        domain = np.array(domain)
        domain = torch.from_numpy(domain)
        # label_domain = np.zeros(4)
        # label_domain[domain] = 1
        # label_domain = torch.from_numpy(label_domain)

        return {'A': A, 'B': B, 'seg':seg,'A_seg':A_seg,'B_seg':B_seg,
                'A_paths': AB_path, 'B_paths': AB_path, 'domain':domain}

    def __len__(self):
        # return int(len(self.AB_paths)/4)
        return len(self.AB_paths)
    def name(self):
        return 'AlignedDataset_starganv1_2'



class AlignedDataset_starganv1_frac_1_4(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        assert (os.path.isdir(self.dir_AB) or (os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'))),"dataset path not exsit! [%s]"%self.dir_AB
        if (not os.path.isdir(self.dir_AB)) and os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'):#to do:数据集AB自动合并功能尚不支持英文路径
            os.makedirs(self.dir_AB)
            dir_A=self.dir_AB+'A'
            dir_B=self.dir_AB+'B'
            make_align_img(dir_A,dir_B,self.dir_AB)

        self.AB_paths = sorted(make_dataset(self.dir_AB))  #make_dataset:返回一个所有图像格式文件（含路径）的list
        assert (len(self.AB_paths)!=0),"dataset floder is empty! [%s]"%self.dir_AB

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合

        #deeplabv3+相关
        self.dir_segs=os.path.join(opt.dataroot, opt.phase+"_seg")
        from src.util.make_seg_img import labelpixels
        flag=0
        if os.path.isdir(self.dir_segs):
            segs = make_dataset(self.dir_segs)
            if len(self.AB_paths) == len(segs):
                flag=1
            else:
                import shutil
                shutil.rmtree(self.dir_segs)
        if flag==0:
            os.makedirs(self.dir_segs)
            for map in self.AB_paths:
                map_np_AB = np.array(Image.open(map).convert("RGB"))
                map_np=map_np_AB[:,map_np_AB.shape[1]//2:,:]
                seg_np = labelpixels(map_np).astype(np.uint8)
                seg_pil = Image.fromarray(seg_np)
                seg_path = os.path.join(self.dir_segs, get_inner_path(map, self.dir_AB))
                seg_pil.save(seg_path)
        self.seg_paths=sorted(make_dataset(self.dir_segs))

    def __getitem__(self, index):
        index = random.randint(0, int(len(self.AB_paths)-1))

        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        AB = AB.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
        AB = self.transform(AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = AB[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]

        # deeplabv3相关，取label图，之后需将label图与A，B同步处理
        seg_path = self.seg_paths[index]
        seg = Image.open(seg_path)
        seg = np.asarray(seg)
        seg = torch.from_numpy(seg)  # H W
        seg = seg[h_offset:h_offset + self.opt.fineSize,
              w_offset:w_offset + self.opt.fineSize]

        if False: #self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        # self.opt.no_flip=False
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)  # 64bit带符号整型，why？因为index_select要求下标为longtensor类型
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            seg=seg.index_select(1,idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)



        # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
        toseg_transform_list = [transforms.Normalize((0, 0, 0),(2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                transforms.Normalize((-0.5,-0.5,-0.5),(1,1,1)),
                                transforms.Normalize((0.485,0.456,0.406),((0.229,0.224,0.225)))] # 恢复标准化前的数值，并换一组数据标准化
        toseg_transform = transforms.Compose(toseg_transform_list)
        A_seg=toseg_transform(A)
        B_seg=toseg_transform(B)
        # 15-18 ---> 0-3
        domain = int(os.path.split(AB_path)[0].split(os.sep)[-1])-15
        label_domain = np.zeros(4)
        label_domain[domain] = 1
        label_domain = torch.from_numpy(label_domain)
        return {'A': A, 'B': B, 'seg':seg,'A_seg':A_seg,'B_seg':B_seg,
                'A_paths': AB_path, 'B_paths': AB_path, 'domain':label_domain}

    def __len__(self):
        return int(len(self.AB_paths)/4)

    def name(self):
        return 'AlignedDataset_starganv1_frac_1_4'


class AlignedDataset_starganv1(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        assert (os.path.isdir(self.dir_AB) or (os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'))),"dataset path not exsit! [%s]"%self.dir_AB
        if (not os.path.isdir(self.dir_AB)) and os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'):#to do:数据集AB自动合并功能尚不支持英文路径
            os.makedirs(self.dir_AB)
            dir_A=self.dir_AB+'A'
            dir_B=self.dir_AB+'B'
            make_align_img(dir_A,dir_B,self.dir_AB)

        self.AB_paths = sorted(make_dataset(self.dir_AB))  #make_dataset:返回一个所有图像格式文件（含路径）的list
        assert (len(self.AB_paths)!=0),"dataset floder is empty! [%s]"%self.dir_AB

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合

        #deeplabv3+相关
        self.dir_segs=os.path.join(opt.dataroot, opt.phase+"_seg")
        from src.util.make_seg_img import labelpixels
        flag=0
        if os.path.isdir(self.dir_segs):
            segs = make_dataset(self.dir_segs)
            if len(self.AB_paths) == len(segs):
                flag=1
            else:
                import shutil
                shutil.rmtree(self.dir_segs)
        if flag==0:
            os.makedirs(self.dir_segs)
            for map in self.AB_paths:
                map_np_AB = np.array(Image.open(map).convert("RGB"))
                map_np=map_np_AB[:,map_np_AB.shape[1]//2:,:]
                seg_np = labelpixels(map_np).astype(np.uint8)
                seg_pil = Image.fromarray(seg_np)
                seg_path = os.path.join(self.dir_segs, get_inner_path(map, self.dir_AB))
                seg_pil.save(seg_path)
        self.seg_paths=sorted(make_dataset(self.dir_segs))

    def __getitem__(self, index):
        # index = random.randint(0, int(len(self.AB_paths)-1))

        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        AB = AB.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
        AB = self.transform(AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = AB[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]

        # deeplabv3相关，取label图，之后需将label图与A，B同步处理
        seg_path = self.seg_paths[index]
        seg = Image.open(seg_path)
        seg = np.asarray(seg)
        seg = torch.from_numpy(seg)  # H W
        seg = seg[h_offset:h_offset + self.opt.fineSize,
              w_offset:w_offset + self.opt.fineSize]

        if False: #self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        # self.opt.no_flip=False
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)  # 64bit带符号整型，why？因为index_select要求下标为longtensor类型
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            seg=seg.index_select(1,idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)



        # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
        toseg_transform_list = [transforms.Normalize((0, 0, 0),(2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                transforms.Normalize((-0.5,-0.5,-0.5),(1,1,1)),
                                transforms.Normalize((0.485,0.456,0.406),((0.229,0.224,0.225)))] # 恢复标准化前的数值，并换一组数据标准化
        toseg_transform = transforms.Compose(toseg_transform_list)
        A_seg=toseg_transform(A)
        B_seg=toseg_transform(B)
        # 15-18 ---> 0-3
        domain = int(os.path.split(AB_path)[0].split(os.sep)[-1])-15
        label_domain = np.zeros(4)
        label_domain[domain] = 1
        label_domain = torch.from_numpy(label_domain)

        return {'A': A, 'B': B, 'seg':seg,'A_seg':A_seg,'B_seg':B_seg,
                'A_paths': AB_path, 'B_paths': AB_path, 'domain':label_domain}

    def __len__(self):
        # return int(len(self.AB_paths)/4)
        return len(self.AB_paths)
    def name(self):
        return 'AlignedDataset_starganv1'



class AlignedDataset_stargan_frac_1_4(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        assert (os.path.isdir(self.dir_AB) or (os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'))),"dataset path not exsit! [%s]"%self.dir_AB
        if (not os.path.isdir(self.dir_AB)) and os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'):#to do:数据集AB自动合并功能尚不支持英文路径
            os.makedirs(self.dir_AB)
            dir_A=self.dir_AB+'A'
            dir_B=self.dir_AB+'B'
            make_align_img(dir_A,dir_B,self.dir_AB)

        self.AB_paths = sorted(make_dataset(self.dir_AB))  #make_dataset:返回一个所有图像格式文件（含路径）的list
        assert (len(self.AB_paths)!=0),"dataset floder is empty! [%s]"%self.dir_AB

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合

        #deeplabv3+相关
        self.dir_segs=os.path.join(opt.dataroot, opt.phase+"_seg")
        from src.util.make_seg_img import labelpixels
        flag=0
        if os.path.isdir(self.dir_segs):
            segs = make_dataset(self.dir_segs)
            if len(self.AB_paths) == len(segs):
                flag=1
            else:
                import shutil
                shutil.rmtree(self.dir_segs)
        if flag==0:
            os.makedirs(self.dir_segs)
            for map in self.AB_paths:
                map_np_AB = np.array(Image.open(map).convert("RGB"))
                map_np=map_np_AB[:,map_np_AB.shape[1]//2:,:]
                seg_np = labelpixels(map_np).astype(np.uint8)
                seg_pil = Image.fromarray(seg_np)
                seg_path = os.path.join(self.dir_segs, get_inner_path(map, self.dir_AB))
                seg_pil.save(seg_path)
        self.seg_paths=sorted(make_dataset(self.dir_segs))

    def __getitem__(self, index):
        index = random.randint(0, int(len(self.AB_paths)-1))

        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        AB = AB.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
        AB = self.transform(AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = AB[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]

        # deeplabv3相关，取label图，之后需将label图与A，B同步处理
        seg_path = self.seg_paths[index]
        seg = Image.open(seg_path)
        seg = np.asarray(seg)
        seg = torch.from_numpy(seg)  # H W
        seg = seg[h_offset:h_offset + self.opt.fineSize,
              w_offset:w_offset + self.opt.fineSize]

        if False: #self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        # self.opt.no_flip=False
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)  # 64bit带符号整型，why？因为index_select要求下标为longtensor类型
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            seg=seg.index_select(1,idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)



        # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
        toseg_transform_list = [transforms.Normalize((0, 0, 0),(2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                transforms.Normalize((-0.5,-0.5,-0.5),(1,1,1)),
                                transforms.Normalize((0.485,0.456,0.406),((0.229,0.224,0.225)))] # 恢复标准化前的数值，并换一组数据标准化
        toseg_transform = transforms.Compose(toseg_transform_list)
        A_seg=toseg_transform(A)
        B_seg=toseg_transform(B)
        # 15-18 ---> 0-3
        domain = int(os.path.split(AB_path)[0].split(os.sep)[-1])-15

        return {'A': A, 'B': B, 'seg':seg,'A_seg':A_seg,'B_seg':B_seg,
                'A_paths': AB_path, 'B_paths': AB_path, 'domain':domain}

    def __len__(self):
        return int(len(self.AB_paths)/4)

    def name(self):
        return 'AlignedDataset_stargan_frac_1_4'
class AlignedDataset_stargan_frac(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        assert (os.path.isdir(self.dir_AB) or (os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'))),"dataset path not exsit! [%s]"%self.dir_AB
        if (not os.path.isdir(self.dir_AB)) and os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'):#to do:数据集AB自动合并功能尚不支持英文路径
            os.makedirs(self.dir_AB)
            dir_A=self.dir_AB+'A'
            dir_B=self.dir_AB+'B'
            make_align_img(dir_A,dir_B,self.dir_AB)

        self.AB_paths = sorted(make_dataset(self.dir_AB))  #make_dataset:返回一个所有图像格式文件（含路径）的list
        assert (len(self.AB_paths)!=0),"dataset floder is empty! [%s]"%self.dir_AB

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合

        #deeplabv3+相关
        self.dir_segs=os.path.join(opt.dataroot, opt.phase+"_seg")
        from src.util.make_seg_img import labelpixels
        flag=0
        if os.path.isdir(self.dir_segs):
            segs = make_dataset(self.dir_segs)
            if len(self.AB_paths) == len(segs):
                flag=1
            else:
                import shutil
                shutil.rmtree(self.dir_segs)
        if flag==0:
            os.makedirs(self.dir_segs)
            for map in self.AB_paths:
                map_np_AB = np.array(Image.open(map).convert("RGB"))
                map_np=map_np_AB[:,map_np_AB.shape[1]//2:,:]
                seg_np = labelpixels(map_np).astype(np.uint8)
                seg_pil = Image.fromarray(seg_np)
                seg_path = os.path.join(self.dir_segs, get_inner_path(map, self.dir_AB))
                seg_pil.save(seg_path)
        self.seg_paths=sorted(make_dataset(self.dir_segs))

    def __getitem__(self, index):
        # index = random.randint(0, int(len(self.AB_paths)-1))

        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        AB = AB.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
        AB = self.transform(AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = AB[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]

        # deeplabv3相关，取label图，之后需将label图与A，B同步处理
        seg_path = self.seg_paths[index]
        seg = Image.open(seg_path)
        seg = np.asarray(seg)
        seg = torch.from_numpy(seg)  # H W
        seg = seg[h_offset:h_offset + self.opt.fineSize,
              w_offset:w_offset + self.opt.fineSize]

        if False: #self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        # self.opt.no_flip=False
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)  # 64bit带符号整型，why？因为index_select要求下标为longtensor类型
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            seg=seg.index_select(1,idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)



        # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
        toseg_transform_list = [transforms.Normalize((0, 0, 0),(2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                transforms.Normalize((-0.5,-0.5,-0.5),(1,1,1)),
                                transforms.Normalize((0.485,0.456,0.406),((0.229,0.224,0.225)))] # 恢复标准化前的数值，并换一组数据标准化
        toseg_transform = transforms.Compose(toseg_transform_list)
        A_seg=toseg_transform(A)
        B_seg=toseg_transform(B)
        # 15-18 ---> 0-3
        domain = int(os.path.split(AB_path)[0].split(os.sep)[-1])-15

        return {'A': A, 'B': B, 'seg':seg,'A_seg':A_seg,'B_seg':B_seg,
                'A_paths': AB_path, 'B_paths': AB_path, 'domain':domain}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset_stargan_frac'

class AlignedDataset_Inter1(BaseDataset):
    def initialize(self, opt, id_layer):
        self.size = opt.loadSize
        self.opt = opt
        self.id_layer = id_layer
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase, "A", str(id_layer))
        self.dir_B = os.path.join(opt.dataroot, opt.phase, "B", str(id_layer+1))
        self.dir_C = os.path.join(opt.dataroot, opt.phase, "C", str(id_layer))
        self.dir_RS = os.path.join(opt.dataroot, opt.phase, "RS", str(id_layer))
        self.dir_Seg = os.path.join(opt.dataroot, opt.phase, "Seg", str(id_layer))


        self.A_paths = sorted(make_dataset(self.dir_A))
        self.C_paths = sorted(make_dataset(self.dir_C))
        if os.path.exists(self.dir_RS):
            self.RS_paths = sorted(make_dataset(self.dir_RS))
            self.Seg_paths = sorted(make_dataset(self.dir_Seg))
        else:
            self.RS_paths = []
            self.Seg_paths = []

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合


    def __getitem__(self, index):

        A_path = self.A_paths[index]
        C_path = self.C_paths[index]
        if self.RS_paths:
            RS_path = self.RS_paths[index]
            RS = Image.open(RS_path).convert('RGB')
            RS = self.transform(RS)

            seg_path = self.Seg_paths[index]
            seg = Image.open(seg_path)

            seg = np.asarray(seg)
            # print(np.max(seg))
            # print(seg_path)
            seg = torch.from_numpy(seg)  # H W
            # print(seg.shape)

            # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
            toseg_transform_list = [transforms.Normalize((0, 0, 0), (2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                    transforms.Normalize((-0.5, -0.5, -0.5), (1, 1, 1)),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                         ((0.229, 0.224, 0.225)))]  # 恢复标准化前的数值，并换一组数据标准化
            toseg_transform = transforms.Compose(toseg_transform_list)
            RS_seg = toseg_transform(RS)  # 送往分割

        A = Image.open(A_path).convert('RGB')
        C = Image.open(C_path).convert('RGB')

        A = self.transform(A)
        C = self.transform(C)



        img_name = os.path.split(A_path)[1]
        img_name_split = img_name.split("_")
        index_x = int(img_name_split[1])
        index_y = int(img_name_split[2].split(".")[0])

        B1_path = os.path.join(self.dir_B,
                               str(self.id_layer + 1) + "_" + str(index_x * 2 - 1) + "_" + str(index_y * 2 - 1)+".png")
        B2_path = os.path.join(self.dir_B,
                               str(self.id_layer + 1) + "_" + str(index_x * 2 - 1) + "_" + str(index_y * 2)+".png")
        B3_path = os.path.join(self.dir_B,
                               str(self.id_layer + 1) + "_" + str(index_x * 2) + "_" + str(index_y * 2 - 1)+".png")
        B4_path = os.path.join(self.dir_B,
                               str(self.id_layer + 1) + "_" + str(index_x * 2) + "_" + str(index_y * 2)+".png")
        B1 = Image.open(B1_path).convert('RGB')
        B2 = Image.open(B2_path).convert('RGB')
        B3 = Image.open(B3_path).convert('RGB')
        B4 = Image.open(B4_path).convert('RGB')

        B = Image.new('RGB', (2 * self.size, 2 * self.size))  # 创建一个新图
        B.paste(B1, (0*self.size, 0*self.size))
        B.paste(B2, (0*self.size, 1*self.size))
        B.paste(B3, (1*self.size, 0*self.size))
        B.paste(B4, (1*self.size, 1*self.size))

        B = B.resize((self.size, self.size), Image.ANTIALIAS)
        # B.save(os.path.join(self.root, self.opt.phase, "tmp_B",
        #                     str(self.id_layer) + "_" + str(index_x) + "_" + str(index_y)) + ".png")

        B = self.transform(B)
        if self.RS_paths:
            return {'A': A, 'B': B, 'C':C, 'A_path': A_path, 'C_path': C_path, 'RS':RS_seg, 'seg':seg}
        else:
            return {'A': A, 'B': B, 'C': C, 'A_path': A_path, 'C_path': C_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset_Inter1'


class AlignedDataset_new_Inter1_all(BaseDataset):
    # fixed iter
    def initialize(self, opt, id_layer):
        self.size = opt.loadSize
        self.opt = opt
        self.id_layer = id_layer
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase, "A", str(id_layer))
        self.dir_B = os.path.join(opt.dataroot, opt.phase, "B", str(id_layer + 1))
        self.dir_C = os.path.join(opt.dataroot, opt.phase, "C", str(id_layer))
        self.dir_RS = os.path.join(opt.dataroot, opt.phase, "RS", str(id_layer))
        self.dir_Seg = os.path.join(opt.dataroot, opt.phase, "Seg", str(id_layer))

        self.A_paths = sorted(make_dataset(self.dir_A))
        self.C_paths = sorted(make_dataset(self.dir_C))
        self.data_len = len(self.A_paths)
        if os.path.exists(self.dir_RS):
            self.RS_paths = sorted(make_dataset(self.dir_RS))
            self.Seg_paths = sorted(make_dataset(self.dir_Seg))
        else:
            self.RS_paths = []
            self.Seg_paths = []

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合

    def __getitem__(self, index):
        # new_index = random.randint(1, self.data_len) - 1
        new_index = index
        A_path = self.A_paths[new_index]
        C_path = self.C_paths[new_index]
        if self.RS_paths:
            RS_path = self.RS_paths[new_index]
            RS = Image.open(RS_path).convert('RGB')
            RS = self.transform(RS)

            seg_path = self.Seg_paths[new_index]
            seg = Image.open(seg_path)
            seg = np.asarray(seg)
            seg = torch.from_numpy(seg)  # H W
            _mask = [seg==i for i in range(5)]
            # print(seg.shape)

            # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
            toseg_transform_list = [transforms.Normalize((0, 0, 0), (2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                    transforms.Normalize((-0.5, -0.5, -0.5), (1, 1, 1)),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                         ((0.229, 0.224, 0.225)))]  # 恢复标准化前的数值，并换一组数据标准化
            toseg_transform = transforms.Compose(toseg_transform_list)
            RS_seg = toseg_transform(RS)  # 送往分割

        A = Image.open(A_path).convert('RGB')
        C = Image.open(C_path).convert('RGB')

        A = self.transform(A)
        C = self.transform(C)
        if self.id_layer == 4:
            B = A
        else:
            img_name = os.path.split(A_path)[1]
            img_name_split = img_name.split("_")
            index_x = int(img_name_split[1])
            index_y = int(img_name_split[2].split(".")[0])

            B1_path = os.path.join(self.dir_B,
                                   str(self.id_layer + 1) + "_" + str(index_x * 2 - 1) + "_" + str(
                                       index_y * 2 - 1) + ".png")
            B2_path = os.path.join(self.dir_B,
                                   str(self.id_layer + 1) + "_" + str(index_x * 2 - 1) + "_" + str(
                                       index_y * 2) + ".png")
            B3_path = os.path.join(self.dir_B,
                                   str(self.id_layer + 1) + "_" + str(index_x * 2) + "_" + str(
                                       index_y * 2 - 1) + ".png")
            B4_path = os.path.join(self.dir_B,
                                   str(self.id_layer + 1) + "_" + str(index_x * 2) + "_" + str(index_y * 2) + ".png")
            B1 = Image.open(B1_path).convert('RGB')
            B2 = Image.open(B2_path).convert('RGB')
            B3 = Image.open(B3_path).convert('RGB')
            B4 = Image.open(B4_path).convert('RGB')
            B = Image.new('RGB', (2 * self.size, 2 * self.size))  # 创建一个新图
            B.paste(B1, (0 * self.size, 0 * self.size))
            B.paste(B2, (0 * self.size, 1 * self.size))
            B.paste(B3, (1 * self.size, 0 * self.size))
            B.paste(B4, (1 * self.size, 1 * self.size))

            B = B.resize((self.size, self.size), Image.ANTIALIAS)
            # B.save(os.path.join(self.root, self.opt.phase, "tmp_B",
            #                     str(self.id_layer) + "_" + str(index_x) + "_" + str(index_y)) + ".png")

            B = self.transform(B)
        if self.RS_paths:
            return {'A': A, 'B': B, 'C': C, 'A_path': A_path, 'C_path': C_path, 'RS': RS_seg, 'seg': seg}
        else:
            return {'A': A, 'B': B, 'C': C, 'A_path': A_path, 'C_path': C_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset_new_Inter1_all'
class AlignedDataset_new_Inter1_for_loss(BaseDataset):
    # fixed iter
    def initialize(self, opt, id_layer):
        self.size = opt.loadSize
        self.opt = opt
        self.id_layer = id_layer
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase, "A", str(id_layer))
        self.dir_B = os.path.join(opt.dataroot, opt.phase, "B", str(id_layer + 1))
        self.dir_C = os.path.join(opt.dataroot, opt.phase, "C", str(id_layer))
        self.dir_RS = os.path.join(opt.dataroot, opt.phase, "RS", str(id_layer))
        self.dir_Seg = os.path.join(opt.dataroot, opt.phase, "Seg", str(id_layer))

        self.A_paths = sorted(make_dataset(self.dir_A))
        self.C_paths = sorted(make_dataset(self.dir_C))
        self.data_len = len(self.A_paths)
        if os.path.exists(self.dir_RS):
            self.RS_paths = sorted(make_dataset(self.dir_RS))
            self.Seg_paths = sorted(make_dataset(self.dir_Seg))
        else:
            self.RS_paths = []
            self.Seg_paths = []

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合

    def __getitem__(self, index):
        # new_index = random.randint(1, self.data_len) - 1
        new_index = index
        A_path = self.A_paths[new_index]
        C_path = self.C_paths[new_index]
        if self.RS_paths:
            RS_path = self.RS_paths[new_index]
            RS = Image.open(RS_path).convert('RGB')
            RS = self.transform(RS)

            seg_path = self.Seg_paths[new_index]
            seg = Image.open(seg_path)
            seg = np.asarray(seg)
            seg = torch.from_numpy(seg)  # H W
            # print(seg.shape)

            # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
            toseg_transform_list = [transforms.Normalize((0, 0, 0), (2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                    transforms.Normalize((-0.5, -0.5, -0.5), (1, 1, 1)),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                         ((0.229, 0.224, 0.225)))]  # 恢复标准化前的数值，并换一组数据标准化
            toseg_transform = transforms.Compose(toseg_transform_list)
            RS_seg = toseg_transform(RS)  # 送往分割

        A = Image.open(A_path).convert('RGB')
        C = Image.open(C_path).convert('RGB')

        A = self.transform(A)
        C = self.transform(C)

        if self.RS_paths:
            return {'A': A, 'C': C, 'A_path': A_path, 'C_path': C_path, 'RS': RS_seg, 'seg': seg}
        else:
            return {'A': A, 'C': C, 'A_path': A_path, 'C_path': C_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset_new_Inter1_for_loss'

class AlignedDataset_new_Inter1(BaseDataset):
    # fixed iter
    def initialize(self, opt, id_layer):
        self.size = opt.loadSize
        self.opt = opt
        self.id_layer = id_layer
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase, "A", str(id_layer))
        self.dir_B = os.path.join(opt.dataroot, opt.phase, "B", str(id_layer+1))
        self.dir_C = os.path.join(opt.dataroot, opt.phase, "C", str(id_layer))
        self.dir_RS = os.path.join(opt.dataroot, opt.phase, "RS", str(id_layer))
        self.dir_Seg = os.path.join(opt.dataroot, opt.phase, "Seg", str(id_layer))


        self.A_paths = sorted(make_dataset(self.dir_A))
        self.C_paths = sorted(make_dataset(self.dir_C))
        self.data_len = len(self.A_paths)
        if os.path.exists(self.dir_RS):
            self.RS_paths = sorted(make_dataset(self.dir_RS))
            self.Seg_paths = sorted(make_dataset(self.dir_Seg))
        else:
            self.RS_paths = []
            self.Seg_paths = []

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合


    def __getitem__(self, index):
        new_index = random.randint(1, self.data_len)-1
        A_path = self.A_paths[new_index]
        C_path = self.C_paths[new_index]
        if self.RS_paths:
            RS_path = self.RS_paths[new_index]
            RS = Image.open(RS_path).convert('RGB')
            RS = self.transform(RS)

            seg_path = self.Seg_paths[new_index]
            seg = Image.open(seg_path)
            seg = np.asarray(seg)
            seg = torch.from_numpy(seg)  # H W
            # print(seg.shape)

            # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
            toseg_transform_list = [transforms.Normalize((0, 0, 0), (2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                    transforms.Normalize((-0.5, -0.5, -0.5), (1, 1, 1)),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                         ((0.229, 0.224, 0.225)))]  # 恢复标准化前的数值，并换一组数据标准化
            toseg_transform = transforms.Compose(toseg_transform_list)
            RS_seg = toseg_transform(RS)  # 送往分割

        A = Image.open(A_path).convert('RGB')
        C = Image.open(C_path).convert('RGB')

        A = self.transform(A)
        C = self.transform(C)
        if self.id_layer == 4:
            B=A
        else:
            img_name = os.path.split(A_path)[1]
            img_name_split = img_name.split("_")
            index_x = int(img_name_split[1])
            index_y = int(img_name_split[2].split(".")[0])

            B1_path = os.path.join(self.dir_B,
                                   str(self.id_layer + 1) + "_" + str(index_x * 2 - 1) + "_" + str(index_y * 2 - 1)+".png")
            B2_path = os.path.join(self.dir_B,
                                   str(self.id_layer + 1) + "_" + str(index_x * 2 - 1) + "_" + str(index_y * 2)+".png")
            B3_path = os.path.join(self.dir_B,
                                   str(self.id_layer + 1) + "_" + str(index_x * 2) + "_" + str(index_y * 2 - 1)+".png")
            B4_path = os.path.join(self.dir_B,
                                   str(self.id_layer + 1) + "_" + str(index_x * 2) + "_" + str(index_y * 2)+".png")
            B1 = Image.open(B1_path).convert('RGB')
            B2 = Image.open(B2_path).convert('RGB')
            B3 = Image.open(B3_path).convert('RGB')
            B4 = Image.open(B4_path).convert('RGB')
            B = Image.new('RGB', (2 * self.size, 2 * self.size))  # 创建一个新图
            B.paste(B1, (0*self.size, 0*self.size))
            B.paste(B2, (0*self.size, 1*self.size))
            B.paste(B3, (1*self.size, 0*self.size))
            B.paste(B4, (1*self.size, 1*self.size))

            B = B.resize((self.size, self.size), Image.ANTIALIAS)
            # B.save(os.path.join(self.root, self.opt.phase, "tmp_B",
            #                     str(self.id_layer) + "_" + str(index_x) + "_" + str(index_y)) + ".png")

            B = self.transform(B)
        if self.RS_paths:
            return {'A': A, 'B': B, 'C':C, 'A_path': A_path, 'C_path': C_path, 'RS':RS_seg, 'seg':seg}
        else:
            return {'A': A, 'B': B, 'C': C, 'A_path': A_path, 'C_path': C_path}

    def __len__(self):
        # fake_size = len(self.A_paths)
        fake_size = int(len(self.A_paths)/3)
        return fake_size

    def name(self):
        return 'AlignedDataset_new_Inter1'
class AlignedDataset_Inter1_1(BaseDataset):
    def initialize(self, opt, id_layer):
        self.size = opt.loadSize
        self.opt = opt
        self.id_layer = id_layer
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase, "B1", str(id_layer))
        self.dir_B = os.path.join(opt.dataroot, opt.phase, "B", str(id_layer+1))
        self.dir_C = os.path.join(opt.dataroot, opt.phase, "C", str(id_layer))
        self.dir_RS = os.path.join(opt.dataroot, opt.phase, "RS", str(id_layer))
        self.dir_Seg = os.path.join(opt.dataroot, opt.phase, "Seg", str(id_layer))


        self.A_paths = sorted(make_dataset(self.dir_A))
        self.C_paths = sorted(make_dataset(self.dir_C))
        if os.path.exists(self.dir_RS):
            self.RS_paths = sorted(make_dataset(self.dir_RS))
            self.Seg_paths = sorted(make_dataset(self.dir_Seg))
        else:
            self.RS_paths = []
            self.Seg_paths = []

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合


    def __getitem__(self, index):

        A_path = self.A_paths[index]
        C_path = self.C_paths[index]
        if self.RS_paths:
            RS_path = self.RS_paths[index]
            RS = Image.open(RS_path).convert('RGB')
            RS = self.transform(RS)

            seg_path = self.Seg_paths[index]
            seg = Image.open(seg_path)

            seg = np.asarray(seg)
            # print(np.max(seg))
            # print(seg_path)
            seg = torch.from_numpy(seg)  # H W
            # print(seg.shape)

            # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
            toseg_transform_list = [transforms.Normalize((0, 0, 0), (2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                    transforms.Normalize((-0.5, -0.5, -0.5), (1, 1, 1)),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                         ((0.229, 0.224, 0.225)))]  # 恢复标准化前的数值，并换一组数据标准化
            toseg_transform = transforms.Compose(toseg_transform_list)
            RS_seg = toseg_transform(RS)  # 送往分割

        A = Image.open(A_path).convert('RGB')
        C = Image.open(C_path).convert('RGB')

        A = self.transform(A)
        C = self.transform(C)



        img_name = os.path.split(A_path)[1]
        img_name_split = img_name.split("_")
        index_x = int(img_name_split[1])
        index_y = int(img_name_split[2].split(".")[0])

        B1_path = os.path.join(self.dir_B,
                               str(self.id_layer + 1) + "_" + str(index_x * 2 - 1) + "_" + str(index_y * 2 - 1)+".png")
        B2_path = os.path.join(self.dir_B,
                               str(self.id_layer + 1) + "_" + str(index_x * 2 - 1) + "_" + str(index_y * 2)+".png")
        B3_path = os.path.join(self.dir_B,
                               str(self.id_layer + 1) + "_" + str(index_x * 2) + "_" + str(index_y * 2 - 1)+".png")
        B4_path = os.path.join(self.dir_B,
                               str(self.id_layer + 1) + "_" + str(index_x * 2) + "_" + str(index_y * 2)+".png")
        B1 = Image.open(B1_path).convert('RGB')
        B2 = Image.open(B2_path).convert('RGB')
        B3 = Image.open(B3_path).convert('RGB')
        B4 = Image.open(B4_path).convert('RGB')

        B = Image.new('RGB', (2 * self.size, 2 * self.size))  # 创建一个新图
        B.paste(B1, (0*self.size, 0*self.size))
        B.paste(B2, (0*self.size, 1*self.size))
        B.paste(B3, (1*self.size, 0*self.size))
        B.paste(B4, (1*self.size, 1*self.size))

        B = B.resize((self.size, self.size), Image.ANTIALIAS)
        B.save(os.path.join(self.root, self.opt.phase, "tmp_B",
                            str(self.id_layer) + "_" + str(index_x) + "_" + str(index_y)) + ".png")

        B = self.transform(B)
        if self.RS_paths:
            return {'A': A, 'B': B, 'C':C, 'A_path': A_path, 'C_path': C_path, 'RS':RS_seg, 'seg':seg}
        else:
            return {'A': A, 'B': B, 'C': C, 'A_path': A_path, 'C_path': C_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset_Inter1_1'
class AlignedDataset_Inter2_2(BaseDataset):
    def initialize(self, opt, id_layer):
        self.size = opt.loadSize
        self.opt = opt
        self.id_layer = id_layer
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase, "A", str(id_layer))
        self.dir_B = os.path.join(opt.dataroot, opt.phase, "B1", str(id_layer-1))
        self.dir_C = os.path.join(opt.dataroot, opt.phase, "C", str(id_layer))


        self.A_paths = sorted(make_dataset(self.dir_A))
        self.C_paths = sorted(make_dataset(self.dir_C))

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合


    def __getitem__(self, index):

        A_path = self.A_paths[index]
        C_path = self.C_paths[index]
        A = Image.open(A_path).convert('RGB')
        C = Image.open(C_path).convert('RGB')
        A = self.transform(A)
        C = self.transform(C)

        img_name = os.path.split(A_path)[1]
        img_name_split = img_name.split("_")
        index_x = int(img_name_split[1])
        index_y = int(img_name_split[2].split(".")[0])

        B_all_path = os.path.join(self.dir_B,
                               str(self.id_layer - 1) + "_" + str(math.ceil(index_x/2)) + "_" + str(math.ceil(index_y/2))+".png")
        if index_x%2==0:
            ix = 1
        else:
            ix = 0
        if index_y%2==0:
            iy = 1
        else:
            iy = 0

        B_all = Image.open(B_all_path).convert('RGB')
        B_all = B_all.resize((self.size*2, self.size*2), Image.ANTIALIAS)
        B = B_all.crop((ix*self.size, iy*self.size, ix*self.size+self.size, iy*self.size+self.size))

        # B = B.resize((self.size, self.size), Image.ANTIALIAS)
        B.save(os.path.join(self.root, self.opt.phase, "tmp_B",
                            str(self.id_layer) + "_" + str(index_x) + "_" + str(index_y)) + ".png")

        B = self.transform(B)

        return {'A': A, 'B': B, 'C':C, 'A_path': A_path, 'C_path': C_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset_Inter2_2'
class AlignedDataset_Inter2(BaseDataset):
    def initialize(self, opt, id_layer):
        self.size = opt.loadSize
        self.opt = opt
        self.id_layer = id_layer
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase, "B", str(id_layer))
        self.dir_B = os.path.join(opt.dataroot, opt.phase, "B1", str(id_layer-1))
        self.dir_C = os.path.join(opt.dataroot, opt.phase, "C", str(id_layer))


        self.A_paths = sorted(make_dataset(self.dir_A))
        self.C_paths = sorted(make_dataset(self.dir_C))

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合


    def __getitem__(self, index):

        A_path = self.A_paths[index]
        C_path = self.C_paths[index]
        A = Image.open(A_path).convert('RGB')
        C = Image.open(C_path).convert('RGB')
        A = self.transform(A)
        C = self.transform(C)

        img_name = os.path.split(A_path)[1]
        img_name_split = img_name.split("_")
        index_x = int(img_name_split[1])
        index_y = int(img_name_split[2].split(".")[0])

        B_all_path = os.path.join(self.dir_B,
                               str(self.id_layer - 1) + "_" + str(math.ceil(index_x/2)) + "_" + str(math.ceil(index_y/2))+".png")
        if index_x%2==0:
            ix = 1
        else:
            ix = 0
        if index_y%2==0:
            iy = 1
        else:
            iy = 0

        B_all = Image.open(B_all_path).convert('RGB')
        B_all = B_all.resize((self.size*2, self.size*2), Image.ANTIALIAS)
        B = B_all.crop((ix*self.size, iy*self.size, ix*self.size+self.size, iy*self.size+self.size))

        # B = B.resize((self.size, self.size), Image.ANTIALIAS)
        B.save(os.path.join(self.root, self.opt.phase, "tmp_B",
                            str(self.id_layer) + "_" + str(index_x) + "_" + str(index_y)) + ".png")

        B = self.transform(B)

        return {'A': A, 'B': B, 'C':C, 'A_path': A_path, 'C_path': C_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset_Inter2'

class AlignedDataset_Inter3(BaseDataset):
    def initialize(self, opt, id_layer):
        self.size = opt.loadSize
        self.opt = opt
        self.id_layer = id_layer
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase, "A", str(id_layer))
        self.dir_B = os.path.join(opt.dataroot, opt.phase, "B", str(id_layer+1))
        self.dir_C = os.path.join(opt.dataroot, opt.phase, "C", str(id_layer))


        self.A_paths = sorted(make_dataset(self.dir_A))
        self.C_paths = sorted(make_dataset(self.dir_C))

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合


    def __getitem__(self, index):

        A_path = self.A_paths[index]
        C_path = self.C_paths[index]
        A = Image.open(A_path).convert('RGB')
        C = Image.open(C_path).convert('RGB')
        A = self.transform(A)
        C = self.transform(C)

        img_name = os.path.split(A_path)[1]
        img_name_split = img_name.split("_")
        index_x = int(img_name_split[1])
        index_y = int(img_name_split[2].split(".")[0])

        B1_path = os.path.join(self.dir_B,
                               str(self.id_layer + 1) + "_" + str(index_x * 2 - 1) + "_" + str(index_y * 2 - 1)+".png")
        B2_path = os.path.join(self.dir_B,
                               str(self.id_layer + 1) + "_" + str(index_x * 2 - 1) + "_" + str(index_y * 2)+".png")
        B3_path = os.path.join(self.dir_B,
                               str(self.id_layer + 1) + "_" + str(index_x * 2) + "_" + str(index_y * 2 - 1)+".png")
        B4_path = os.path.join(self.dir_B,
                               str(self.id_layer + 1) + "_" + str(index_x * 2) + "_" + str(index_y * 2)+".png")
        B1 = Image.open(B1_path).convert('RGB')
        B2 = Image.open(B2_path).convert('RGB')
        B3 = Image.open(B3_path).convert('RGB')
        B4 = Image.open(B4_path).convert('RGB')

        B = Image.new('RGB', (2 * self.size, 2 * self.size))  # 创建一个新图
        B.paste(B1, (0*self.size, 0*self.size))
        B.paste(B2, (0*self.size, 1*self.size))
        B.paste(B3, (1*self.size, 0*self.size))
        B.paste(B4, (1*self.size, 1*self.size))

        B = B.resize((self.size, self.size), Image.ANTIALIAS)
        B.save(os.path.join(self.root, self.opt.phase, "tmp_B",
                            str(self.id_layer) + "_" + str(index_x) + "_" + str(index_y)) + ".png")

        B = self.transform(B)

        return {'A': A, 'B': B, 'C':C, 'A_path': A_path, 'C_path': C_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset_Inter3'

class AlignedDataset_Inter4(BaseDataset):
    def initialize(self, opt, id_layer):
        self.size = opt.loadSize
        self.opt = opt
        self.id_layer = id_layer
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase, "A", str(id_layer))
        self.dir_B = os.path.join(opt.dataroot, opt.phase, "B", str(id_layer+1))
        self.dir_C = os.path.join(opt.dataroot, opt.phase, "C", str(id_layer))
        self.dir_Bounday = os.path.join(opt.dataroot, opt.phase, "Boundary", str(id_layer))
        self.dir_Seg = os.path.join(opt.dataroot, opt.phase, "Seg", str(id_layer))


        self.A_paths = sorted(make_dataset(self.dir_A))
        self.C_paths = sorted(make_dataset(self.dir_C))
        if os.path.exists(self.dir_Bounday):
            self.Bounday_paths = sorted(make_dataset(self.dir_Bounday))
            self.Seg_paths = sorted(make_dataset(self.dir_Seg))

        else:
            self.Bounday_paths = []
            self.Seg_paths = []
            print("no boundary")

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std
        transform_list2 = [transforms.ToTensor()]
        self.transform2 = transforms.Compose(transform_list2)  # 串联组合
        self.transform = transforms.Compose(transform_list)  # 串联组合


    def __getitem__(self, index):

        A_path = self.A_paths[index]
        C_path = self.C_paths[index]
        if self.Bounday_paths:
            Seg_path = self.Seg_paths[index]
            Seg = Image.open(Seg_path).convert('RGB')
            seg = self.transform(Seg)

            bounday_path = self.Bounday_paths[index]
            bounday = Image.open(bounday_path)

            bounday = np.asarray(bounday)
            bounday_tf = self.transform2(bounday)


        A = Image.open(A_path).convert('RGB')
        C = Image.open(C_path).convert('RGB')

        A = self.transform(A)
        C = self.transform(C)



        img_name = os.path.split(A_path)[1]
        img_name_split = img_name.split("_")
        index_x = int(img_name_split[1])
        index_y = int(img_name_split[2].split(".")[0])

        B1_path = os.path.join(self.dir_B,
                               str(self.id_layer + 1) + "_" + str(index_x * 2 - 1) + "_" + str(index_y * 2 - 1)+".png")
        B2_path = os.path.join(self.dir_B,
                               str(self.id_layer + 1) + "_" + str(index_x * 2 - 1) + "_" + str(index_y * 2)+".png")
        B3_path = os.path.join(self.dir_B,
                               str(self.id_layer + 1) + "_" + str(index_x * 2) + "_" + str(index_y * 2 - 1)+".png")
        B4_path = os.path.join(self.dir_B,
                               str(self.id_layer + 1) + "_" + str(index_x * 2) + "_" + str(index_y * 2)+".png")
        B1 = Image.open(B1_path).convert('RGB')
        B2 = Image.open(B2_path).convert('RGB')
        B3 = Image.open(B3_path).convert('RGB')
        B4 = Image.open(B4_path).convert('RGB')

        B = Image.new('RGB', (2 * self.size, 2 * self.size))  # 创建一个新图
        B.paste(B1, (0*self.size, 0*self.size))
        B.paste(B2, (0*self.size, 1*self.size))
        B.paste(B3, (1*self.size, 0*self.size))
        B.paste(B4, (1*self.size, 1*self.size))

        B = B.resize((self.size, self.size), Image.ANTIALIAS)
        B.save(os.path.join(self.root, self.opt.phase, "tmp_B",
                            str(self.id_layer) + "_" + str(index_x) + "_" + str(index_y)) + ".png")

        B = self.transform(B)
        if self.Bounday_paths:
            return {'A': A, 'B': B, 'C':C, 'A_path': A_path, 'C_path': C_path, 'Boundary':bounday_tf, 'seg':seg}
        else:
            return {'A': A, 'B': B, 'C': C, 'A_path': A_path, 'C_path': C_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset_Inter4'


class AlignedDataset_myconsistency_v1(BaseDataset):
    # fixed iter
    def initialize(self, opt):
        self.size = opt.loadSize
        self.opt = opt

        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase, "A")
        self.dir_B = os.path.join(opt.dataroot, opt.phase, "RS")
        self.dir_C = os.path.join(opt.dataroot, opt.phase, "C")
        self.dir_RS = os.path.join(opt.dataroot, opt.phase, "RS")
        self.dir_Seg = os.path.join(opt.dataroot, opt.phase, "Seg")

        self.A_paths = sorted(make_dataset(self.dir_A))
        self.C_paths = sorted(make_dataset(self.dir_C))
        self.data_len = len(self.A_paths)
        if os.path.exists(self.dir_RS):
            self.RS_paths = sorted(make_dataset(self.dir_RS))
            self.Seg_paths = sorted(make_dataset(self.dir_Seg))
        else:
            self.RS_paths = []
            self.Seg_paths = []

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合

    def __getitem__(self, index):
        # new_index = random.randint(1, self.data_len) - 1
        new_index = index
        A_path = self.A_paths[new_index]
        C_path = self.C_paths[new_index]

        id_layer = int(os.path.split(A_path)[-1].split(".")[0].split("_")[0])
        if self.RS_paths:
            RS_path = self.RS_paths[new_index]
            RS_img = Image.open(RS_path).convert('RGB')
            RS = self.transform(RS_img)

            seg_path = self.Seg_paths[new_index]
            seg = Image.open(seg_path)
            seg = np.asarray(seg)
            seg = torch.from_numpy(seg)  # H W
            # print(seg.shape)

            # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
            toseg_transform_list = [transforms.Normalize((0, 0, 0), (2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                    transforms.Normalize((-0.5, -0.5, -0.5), (1, 1, 1)),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                         ((0.229, 0.224, 0.225)))]  # 恢复标准化前的数值，并换一组数据标准化
            toseg_transform = transforms.Compose(toseg_transform_list)
            RS_seg = toseg_transform(RS)  # 送往分割

        A = Image.open(A_path).convert('RGB')
        C = Image.open(C_path).convert('RGB')

        A = self.transform(A)
        C = self.transform(C)
        if id_layer == 18:
            B = RS_img
            B = B.resize((self.size*2, self.size*2), Image.ANTIALIAS)
            B1 = B.crop((0, 0, self.size, self.size))
            B2 = B.crop((self.size, 0, 2*self.size, self.size))
            B3 = B.crop((0, self.size, self.size, 2*self.size))
            B4 = B.crop((self.size, self.size, 2*self.size, 2*self.size))

            # B1 = B[:, 0:self.size, 0:self.size]
            # B2 = B[:, 0:self.size, self.size:self.size * 2]
            # B3 = B[:, self.size:self.size*2, 0:self.size]
            # B4 = B[:, self.size:self.size*2, self.size:self.size * 2]
        else:
            img_name = os.path.split(A_path)[1]
            img_name_split = img_name.split("_")
            index_x = int(img_name_split[1])
            index_y = int(img_name_split[2].split(".")[0])

            B1_path = os.path.join(self.dir_B,str(id_layer + 1),
                                   str(id_layer + 1) + "_" + str(index_x * 2 - 1) + "_" + str(
                                       index_y * 2 - 1) + ".jpg")
            B2_path = os.path.join(self.dir_B,str(id_layer + 1),
                                   str(id_layer + 1) + "_" + str(index_x * 2 - 1) + "_" + str(
                                       index_y * 2) + ".jpg")
            B3_path = os.path.join(self.dir_B,str(id_layer + 1),
                                   str(id_layer + 1) + "_" + str(index_x * 2) + "_" + str(
                                       index_y * 2 - 1) + ".jpg")
            B4_path = os.path.join(self.dir_B,str(id_layer + 1),
                                   str(id_layer + 1) + "_" + str(index_x * 2) + "_" + str(index_y * 2) + ".jpg")
            B1 = Image.open(B1_path).convert('RGB')
            B2 = Image.open(B2_path).convert('RGB')
            B3 = Image.open(B3_path).convert('RGB')
            B4 = Image.open(B4_path).convert('RGB')
            # B = Image.new('RGB', (2 * self.size, 2 * self.size))  # 创建一个新图
            # B.paste(B1, (0 * self.size, 0 * self.size))
            # B.paste(B2, (0 * self.size, 1 * self.size))
            # B.paste(B3, (1 * self.size, 0 * self.size))
            # B.paste(B4, (1 * self.size, 1 * self.size))

            # B = B.resize((self.size, self.size), Image.ANTIALIAS)
            # B.save(os.path.join(self.root, self.opt.phase, "tmp_B",
            #                     str(self.id_layer) + "_" + str(index_x) + "_" + str(index_y)) + ".png")

        B1 = self.transform(B1)
        B2 = self.transform(B2)
        B3 = self.transform(B3)
        B4 = self.transform(B4)
        if self.RS_paths:
            return {'A': A, 'B1': B1, 'B2': B2, 'B3': B3, 'B4': B4, 'C': C, 'A_path': A_path, 'C_path': C_path, 'RS': RS_seg, 'seg': seg}
        else:
            return {'A': A, 'B': B, 'C': C, 'A_path': A_path, 'C_path': C_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset_myconsistency_v1'

class AlignedDataset_myconsistency_v2_1_4(BaseDataset):
    def initialize(self, opt):
        self.size = 256
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        assert (os.path.isdir(self.dir_AB) or (os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'))),"dataset path not exsit! [%s]"%self.dir_AB
        if (not os.path.isdir(self.dir_AB)) and os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'):#to do:数据集AB自动合并功能尚不支持英文路径
            os.makedirs(self.dir_AB)
            dir_A=self.dir_AB+'A'
            dir_B=self.dir_AB+'B'
            make_align_img(dir_A,dir_B,self.dir_AB)
        self.dir_A = self.dir_AB+'A'
        self.dir_B = self.dir_AB+'B'
        self.AB_paths = sorted(make_dataset(self.dir_AB))  #make_dataset:返回一个所有图像格式文件（含路径）的list
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        assert (len(self.AB_paths)!=0),"dataset floder is empty! [%s]"%self.dir_AB

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合

        #deeplabv3+相关
        self.dir_segs=os.path.join(opt.dataroot, opt.phase+"_seg")
        self.dir_rs = os.path.join(opt.dataroot, opt.phase+"_rs")
        from src.util.make_seg_img import labelpixels
        flag=0
        if os.path.isdir(self.dir_segs):
            segs = make_dataset(self.dir_segs)
            if len(self.AB_paths) == len(segs):
                flag=1
            else:
                import shutil
                shutil.rmtree(self.dir_segs)
        if flag==0:
            os.makedirs(self.dir_segs)
            for map in self.AB_paths:
                map_np_AB = np.array(Image.open(map).convert("RGB"))
                map_np=map_np_AB[:,map_np_AB.shape[1]//2:,:]
                seg_np = labelpixels(map_np).astype(np.uint8)
                seg_pil = Image.fromarray(seg_np)
                seg_path = os.path.join(self.dir_segs, get_inner_path(map, self.dir_AB))
                seg_pil.save(seg_path)
        self.seg_paths=sorted(make_dataset(self.dir_segs))
        self.rs_paths = sorted(make_dataset(self.dir_rs))

    def __getitem__(self, index):
        index = random.randint(0, int(len(self.AB_paths)-1))

        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        rs_path = self.rs_paths[index]

        img_name = os.path.split(A_path)[-1]
        id_layer = int(os.path.split(A_path)[-1].split(".")[0].split("_")[0])
        img_x = int(os.path.split(A_path)[-1].split(".")[0].split("_")[1])
        img_y = int(os.path.split(A_path)[-1].split(".")[0].split("_")[2])


        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')
        rs = Image.open(rs_path).convert('RGB')

        if id_layer == 18:
            C = A.resize((512, 512))
        else:
            C1_path = os.path.join(self.dir_A, str(id_layer + 1), str(id_layer + 1) + "_" + str(img_x * 2) + "_" + str(img_y * 2) + ".png")
            C2_path = os.path.join(self.dir_A, str(id_layer + 1), str(id_layer + 1) + "_" + str(img_x * 2) + "_" + str(img_y * 2 + 1) + ".png")
            C3_path = os.path.join(self.dir_A, str(id_layer + 1), str(id_layer + 1) + "_" + str(img_x * 2 + 1) + "_" + str(img_y * 2) + ".png")
            C4_path = os.path.join(self.dir_A, str(id_layer + 1), str(id_layer + 1) + "_" + str(img_x * 2 + 1) + "_" + str(img_y * 2 + 1) + ".png")
            C1 = Image.open(C1_path).convert('RGB')
            C2 = Image.open(C2_path).convert('RGB')
            C3 = Image.open(C3_path).convert('RGB')
            C4 = Image.open(C4_path).convert('RGB')
            C = Image.new('RGB', (2 * 256, 2 * 256))  # 创建一个新图
            C.paste(C1, (0 * 256, 0 * 256))
            C.paste(C2, (0 * 256, 1 * 256))
            C.paste(C3, (1 * 256, 0 * 256))
            C.paste(C4, (1 * 256, 1 * 256))

        A = self.transform(A)
        B = self.transform(B)
        C = self.transform(C)
        rs = self.transform(rs)


        # deeplabv3相关，取label图，之后需将label图与A，B同步处理

        seg_path = self.seg_paths[index]

        seg = Image.open(seg_path)
        seg = np.asarray(seg)
        seg = torch.from_numpy(seg)  # H W


        # self.opt.no_flip=False
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)  # 64bit带符号整型，why？因为index_select要求下标为longtensor类型
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            rs = rs.index_select(2, idx)
            seg=seg.index_select(1, idx)

            idx_c = [i for i in range(C.size(2) - 1, -1, -1)]
            idx_c = torch.LongTensor(idx_c)
            C = C.index_select(2, idx_c)


        # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
        toseg_transform_list = [transforms.Normalize((0, 0, 0),(2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                transforms.Normalize((-0.5,-0.5,-0.5),(1,1,1)),
                                transforms.Normalize((0.485,0.456,0.406),((0.229,0.224,0.225)))] # 恢复标准化前的数值，并换一组数据标准化
        toseg_transform = transforms.Compose(toseg_transform_list)
        rs_seg=toseg_transform(rs)
        # B_seg=toseg_transform(B)
        # 15-18 ---> 0-3
        domain = int(os.path.split(A_path)[0].split(os.sep)[-1])-15
        domain = np.array(domain)
        domain = torch.from_numpy(domain)
        return {'A': A, 'B': B,'C':C, 'seg':seg,'rs_seg': rs_seg, 'A_paths': A_path, 'B_paths': B_path, 'domain':domain}


    def __len__(self):
        return int(len(self.AB_paths)/4)

    def name(self):
        return 'AlignedDataset_myconsistency_v2_1_4'


class AlignedDataset_myconsistency_v7(BaseDataset):
    def initialize(self, opt, layerid):
        self.size = 256
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        assert (os.path.isdir(self.dir_AB) or (os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'))),"dataset path not exsit! [%s]"%self.dir_AB
        if (not os.path.isdir(self.dir_AB)) and os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'):#to do:数据集AB自动合并功能尚不支持英文路径
            os.makedirs(self.dir_AB)
            dir_A=self.dir_AB+'A'
            dir_B=self.dir_AB+'B'
            make_align_img(dir_A,dir_B,self.dir_AB)
        self.dir_A = os.path.join(self.dir_AB+'A',str(layerid))
        self.dir_B = os.path.join(self.dir_AB+'B',str(layerid))
        self.dir_C = os.path.join(self.dir_AB+'C',str(layerid))
        # self.dir_D = os.path.join(self.dir_AB+'D',str(layerid))
        self.AB_paths = sorted(make_dataset(self.dir_AB))  #make_dataset:返回一个所有图像格式文件（含路径）的list
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.C_paths = sorted(make_dataset(self.dir_C))
        # self.D_paths = sorted(make_dataset(self.dir_D))
        assert (len(self.AB_paths)!=0),"dataset floder is empty! [%s]"%self.dir_AB

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合

        #deeplabv3+相关
        self.dir_segs=os.path.join(opt.dataroot, opt.phase+"_seg",str(layerid))
        self.dir_rs = os.path.join(opt.dataroot, opt.phase+"_rs",str(layerid))
        from src.util.make_seg_img import labelpixels
        flag=0
        if os.path.isdir(self.dir_segs):
            segs = make_dataset(self.dir_segs)
            if len(self.A_paths) == len(segs):
                flag=1
            else:
                import shutil
                shutil.rmtree(self.dir_segs)
        if flag==0:
            os.makedirs(self.dir_segs)
            for map in self.AB_paths:
                map_np_AB = np.array(Image.open(map).convert("RGB"))
                map_np=map_np_AB[:,map_np_AB.shape[1]//2:,:]
                seg_np = labelpixels(map_np).astype(np.uint8)
                seg_pil = Image.fromarray(seg_np)
                seg_path = os.path.join(self.dir_segs, get_inner_path(map, self.dir_AB))
                seg_pil.save(seg_path)
        self.seg_paths=sorted(make_dataset(self.dir_segs))
        self.rs_paths = sorted(make_dataset(self.dir_rs))

    def __getitem__(self, index):
        # index = random.randint(0, int(len(self.AB_paths)-1))

        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        C_path = self.C_paths[index]
        # D_path = self.D_paths[index]
        rs_path = self.rs_paths[index]


        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')
        C = Image.open(C_path).convert('RGB')
        # D = Image.open(D_path).convert('RGB')
        rs = Image.open(rs_path).convert('RGB')


        A = self.transform(A)
        B = self.transform(B)
        C = self.transform(C)
        # D = self.transform(D)
        rs = self.transform(rs)


        # deeplabv3相关，取label图，之后需将label图与A，B同步处理

        seg_path = self.seg_paths[index]

        seg = Image.open(seg_path)
        seg = np.asarray(seg)
        seg = torch.from_numpy(seg)  # H W


        # self.opt.no_flip=False
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)  # 64bit带符号整型，why？因为index_select要求下标为longtensor类型
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

            # D = D.index_select(2, idx)
            rs = rs.index_select(2, idx)
            seg=seg.index_select(1, idx)

            idx_c = [i for i in range(C.size(2) - 1, -1, -1)]
            idx_c = torch.LongTensor(idx_c)  # 64bit带符号整型，why？因为index_select要求下标为longtensor类型
            C = C.index_select(2, idx_c)




        # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
        toseg_transform_list = [transforms.Normalize((0, 0, 0),(2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                transforms.Normalize((-0.5,-0.5,-0.5),(1,1,1)),
                                transforms.Normalize((0.485,0.456,0.406),((0.229,0.224,0.225)))] # 恢复标准化前的数值，并换一组数据标准化
        toseg_transform = transforms.Compose(toseg_transform_list)
        rs_seg=toseg_transform(rs)
        # B_seg=toseg_transform(B)
        # 15-18 ---> 0-3
        domain = int(os.path.split(A_path)[0].split(os.sep)[-1])-15
        domain = np.array(domain)
        domain = torch.from_numpy(domain)
        return {'A': A, 'B': B, 'C':C,'seg':seg,'rs_seg': rs_seg, 'A_paths': A_path, 'B_paths': B_path, 'domain':domain,'rs':rs}


    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset_myconsistency_v7'


class AlignedDataset_myconsistency_v6(BaseDataset):
    def initialize(self, opt, layerid):
        self.size = 256
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        assert (os.path.isdir(self.dir_AB) or (os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'))),"dataset path not exsit! [%s]"%self.dir_AB
        if (not os.path.isdir(self.dir_AB)) and os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'):#to do:数据集AB自动合并功能尚不支持英文路径
            os.makedirs(self.dir_AB)
            dir_A=self.dir_AB+'A'
            dir_B=self.dir_AB+'B'
            make_align_img(dir_A,dir_B,self.dir_AB)
        self.dir_A = os.path.join(self.dir_AB+'A',str(layerid))
        self.dir_B = os.path.join(self.dir_AB+'B',str(layerid))
        # self.dir_C = self.dir_AB+'C'
        # self.dir_D = os.path.join(self.dir_AB+'D',str(layerid))
        self.AB_paths = sorted(make_dataset(self.dir_AB))  #make_dataset:返回一个所有图像格式文件（含路径）的list
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        # self.C_paths = sorted(make_dataset(self.dir_C))
        # self.D_paths = sorted(make_dataset(self.dir_D))
        assert (len(self.AB_paths)!=0),"dataset floder is empty! [%s]"%self.dir_AB

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合

        #deeplabv3+相关
        self.dir_segs=os.path.join(opt.dataroot, opt.phase+"_seg",str(layerid))
        self.dir_rs = os.path.join(opt.dataroot, opt.phase+"_rs",str(layerid))
        from src.util.make_seg_img import labelpixels
        flag=0
        if os.path.isdir(self.dir_segs):
            segs = make_dataset(self.dir_segs)
            if len(self.A_paths) == len(segs):
                flag=1
            else:
                import shutil
                shutil.rmtree(self.dir_segs)
        if flag==0:
            os.makedirs(self.dir_segs)
            for map in self.AB_paths:
                map_np_AB = np.array(Image.open(map).convert("RGB"))
                map_np=map_np_AB[:,map_np_AB.shape[1]//2:,:]
                seg_np = labelpixels(map_np).astype(np.uint8)
                seg_pil = Image.fromarray(seg_np)
                seg_path = os.path.join(self.dir_segs, get_inner_path(map, self.dir_AB))
                seg_pil.save(seg_path)
        self.seg_paths=sorted(make_dataset(self.dir_segs))
        self.rs_paths = sorted(make_dataset(self.dir_rs))

    def __getitem__(self, index):
        # index = random.randint(0, int(len(self.AB_paths)-1))

        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        # C_path = self.C_paths[index]
        # D_path = self.D_paths[index]
        rs_path = self.rs_paths[index]


        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')
        # C = Image.open(C_path).convert('RGB')
        # D = Image.open(D_path).convert('RGB')
        rs = Image.open(rs_path).convert('RGB')


        A = self.transform(A)
        B = self.transform(B)
        # C = self.transform(C)
        # D = self.transform(D)
        rs = self.transform(rs)


        # deeplabv3相关，取label图，之后需将label图与A，B同步处理

        seg_path = self.seg_paths[index]

        seg = Image.open(seg_path)
        seg = np.asarray(seg)
        seg = torch.from_numpy(seg)  # H W


        # self.opt.no_flip=False
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)  # 64bit带符号整型，why？因为index_select要求下标为longtensor类型
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            # C = C.index_select(2, idx)
            # D = D.index_select(2, idx)
            rs = rs.index_select(2, idx)
            seg=seg.index_select(1, idx)




        # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
        toseg_transform_list = [transforms.Normalize((0, 0, 0),(2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                transforms.Normalize((-0.5,-0.5,-0.5),(1,1,1)),
                                transforms.Normalize((0.485,0.456,0.406),((0.229,0.224,0.225)))] # 恢复标准化前的数值，并换一组数据标准化
        toseg_transform = transforms.Compose(toseg_transform_list)
        rs_seg=toseg_transform(rs)
        # B_seg=toseg_transform(B)
        # 15-18 ---> 0-3
        domain = int(os.path.split(A_path)[0].split(os.sep)[-1])-15
        domain = np.array(domain)
        domain = torch.from_numpy(domain)
        return {'A': A, 'B': B, 'seg':seg,'rs_seg': rs_seg, 'A_paths': A_path, 'B_paths': B_path, 'domain':domain,'rs':rs}


    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset_myconsistency_v6'



class AlignedDataset_myconsistency_v5(BaseDataset):
    def initialize(self, opt, layerid):
        self.size = 256
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        assert (os.path.isdir(self.dir_AB) or (os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'))),"dataset path not exsit! [%s]"%self.dir_AB
        if (not os.path.isdir(self.dir_AB)) and os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'):#to do:数据集AB自动合并功能尚不支持英文路径
            os.makedirs(self.dir_AB)
            dir_A=self.dir_AB+'A'
            dir_B=self.dir_AB+'B'
            make_align_img(dir_A,dir_B,self.dir_AB)
        self.dir_A = os.path.join(self.dir_AB+'A',str(layerid))
        self.dir_B = os.path.join(self.dir_AB+'B',str(layerid))
        # self.dir_C = self.dir_AB+'C'
        self.dir_D = os.path.join(self.dir_AB+'D',str(layerid))
        self.AB_paths = sorted(make_dataset(self.dir_AB))  #make_dataset:返回一个所有图像格式文件（含路径）的list
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        # self.C_paths = sorted(make_dataset(self.dir_C))
        self.D_paths = sorted(make_dataset(self.dir_D))
        assert (len(self.AB_paths)!=0),"dataset floder is empty! [%s]"%self.dir_AB

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合

        #deeplabv3+相关
        self.dir_segs=os.path.join(opt.dataroot, opt.phase+"_seg",str(layerid))
        self.dir_rs = os.path.join(opt.dataroot, opt.phase+"_rs",str(layerid))
        from src.util.make_seg_img import labelpixels
        flag=0
        if os.path.isdir(self.dir_segs):
            segs = make_dataset(self.dir_segs)
            if len(self.A_paths) == len(segs):
                flag=1
            else:
                import shutil
                shutil.rmtree(self.dir_segs)
        if flag==0:
            os.makedirs(self.dir_segs)
            for map in self.AB_paths:
                map_np_AB = np.array(Image.open(map).convert("RGB"))
                map_np=map_np_AB[:,map_np_AB.shape[1]//2:,:]
                seg_np = labelpixels(map_np).astype(np.uint8)
                seg_pil = Image.fromarray(seg_np)
                seg_path = os.path.join(self.dir_segs, get_inner_path(map, self.dir_AB))
                seg_pil.save(seg_path)
        self.seg_paths=sorted(make_dataset(self.dir_segs))
        self.rs_paths = sorted(make_dataset(self.dir_rs))

    def __getitem__(self, index):
        # index = random.randint(0, int(len(self.AB_paths)-1))

        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        # C_path = self.C_paths[index]
        D_path = self.D_paths[index]
        rs_path = self.rs_paths[index]


        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')
        # C = Image.open(C_path).convert('RGB')
        D = Image.open(D_path).convert('RGB')
        rs = Image.open(rs_path).convert('RGB')


        A = self.transform(A)
        B = self.transform(B)
        # C = self.transform(C)
        D = self.transform(D)
        rs = self.transform(rs)


        # deeplabv3相关，取label图，之后需将label图与A，B同步处理

        seg_path = self.seg_paths[index]

        seg = Image.open(seg_path)
        seg = np.asarray(seg)
        seg = torch.from_numpy(seg)  # H W


        # self.opt.no_flip=False
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)  # 64bit带符号整型，why？因为index_select要求下标为longtensor类型
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            # C = C.index_select(2, idx)
            D = D.index_select(2, idx)
            rs = rs.index_select(2, idx)
            seg=seg.index_select(1, idx)




        # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
        toseg_transform_list = [transforms.Normalize((0, 0, 0),(2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                transforms.Normalize((-0.5,-0.5,-0.5),(1,1,1)),
                                transforms.Normalize((0.485,0.456,0.406),((0.229,0.224,0.225)))] # 恢复标准化前的数值，并换一组数据标准化
        toseg_transform = transforms.Compose(toseg_transform_list)
        rs_seg=toseg_transform(rs)
        # B_seg=toseg_transform(B)
        # 15-18 ---> 0-3
        domain = int(os.path.split(A_path)[0].split(os.sep)[-1])-15
        domain = np.array(domain)
        domain = torch.from_numpy(domain)
        return {'A': A, 'B': B,'D':D, 'seg':seg,'rs_seg': rs_seg, 'A_paths': A_path, 'B_paths': B_path, 'domain':domain,'rs':rs}


    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset_myconsistency_v5'



class AlignedDataset_myconsistency_v4(BaseDataset):
    def initialize(self, opt):
        self.size = 256
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        assert (os.path.isdir(self.dir_AB) or (os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'))),"dataset path not exsit! [%s]"%self.dir_AB
        if (not os.path.isdir(self.dir_AB)) and os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'):#to do:数据集AB自动合并功能尚不支持英文路径
            os.makedirs(self.dir_AB)
            dir_A=self.dir_AB+'A'
            dir_B=self.dir_AB+'B'
            make_align_img(dir_A,dir_B,self.dir_AB)
        self.dir_A = self.dir_AB+'A'
        self.dir_B = self.dir_AB+'B'
        self.dir_C = self.dir_AB+'C'
        # self.dir_D = self.dir_AB+'D'
        self.AB_paths = sorted(make_dataset(self.dir_AB))  #make_dataset:返回一个所有图像格式文件（含路径）的list
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.C_paths = sorted(make_dataset(self.dir_C))
        # self.D_paths = sorted(make_dataset(self.dir_D))
        assert (len(self.AB_paths)!=0),"dataset floder is empty! [%s]"%self.dir_AB

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合

        #deeplabv3+相关
        self.dir_segs=os.path.join(opt.dataroot, opt.phase+"_seg")
        self.dir_rs = os.path.join(opt.dataroot, opt.phase+"_rs")
        from src.util.make_seg_img import labelpixels
        flag=0
        if os.path.isdir(self.dir_segs):
            segs = make_dataset(self.dir_segs)
            if len(self.AB_paths) == len(segs):
                flag=1
            else:
                import shutil
                shutil.rmtree(self.dir_segs)
        if flag==0:
            os.makedirs(self.dir_segs)
            for map in self.AB_paths:
                map_np_AB = np.array(Image.open(map).convert("RGB"))
                map_np=map_np_AB[:,map_np_AB.shape[1]//2:,:]
                seg_np = labelpixels(map_np).astype(np.uint8)
                seg_pil = Image.fromarray(seg_np)
                seg_path = os.path.join(self.dir_segs, get_inner_path(map, self.dir_AB))
                seg_pil.save(seg_path)
        self.seg_paths=sorted(make_dataset(self.dir_segs))
        self.rs_paths = sorted(make_dataset(self.dir_rs))

    def __getitem__(self, index):
        # index = random.randint(0, int(len(self.AB_paths)-1))

        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        C_path = self.C_paths[index]
        # D_path = self.D_paths[index]
        rs_path = self.rs_paths[index]


        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')
        C = Image.open(C_path).convert('RGB')
        # D = Image.open(D_path).convert('RGB')
        rs = Image.open(rs_path).convert('RGB')


        A = self.transform(A)
        B = self.transform(B)
        C = self.transform(C)
        # D = self.transform(D)
        rs = self.transform(rs)


        # deeplabv3相关，取label图，之后需将label图与A，B同步处理

        seg_path = self.seg_paths[index]

        seg = Image.open(seg_path)
        seg = np.asarray(seg)
        seg = torch.from_numpy(seg)  # H W


        # self.opt.no_flip=False
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)  # 64bit带符号整型，why？因为index_select要求下标为longtensor类型
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            C = C.index_select(2, idx)
            # D = D.index_select(2, idx)
            rs = rs.index_select(2, idx)
            seg=seg.index_select(1, idx)




        # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
        toseg_transform_list = [transforms.Normalize((0, 0, 0),(2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                transforms.Normalize((-0.5,-0.5,-0.5),(1,1,1)),
                                transforms.Normalize((0.485,0.456,0.406),((0.229,0.224,0.225)))] # 恢复标准化前的数值，并换一组数据标准化
        toseg_transform = transforms.Compose(toseg_transform_list)
        rs_seg=toseg_transform(rs)
        # B_seg=toseg_transform(B)
        # 15-18 ---> 0-3
        domain = int(os.path.split(A_path)[0].split(os.sep)[-1])-15
        domain = np.array(domain)
        domain = torch.from_numpy(domain)
        return {'A': A, 'B': B,'C':C, 'seg':seg,'rs_seg': rs_seg, 'A_paths': A_path, 'B_paths': B_path, 'domain':domain,'rs':rs}


    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset_myconsistency_v4'


class AlignedDataset_myconsistency_v3(BaseDataset):
    def initialize(self, opt):
        self.size = 256
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        assert (os.path.isdir(self.dir_AB) or (os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'))),"dataset path not exsit! [%s]"%self.dir_AB
        if (not os.path.isdir(self.dir_AB)) and os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'):#to do:数据集AB自动合并功能尚不支持英文路径
            os.makedirs(self.dir_AB)
            dir_A=self.dir_AB+'A'
            dir_B=self.dir_AB+'B'
            make_align_img(dir_A,dir_B,self.dir_AB)
        self.dir_A = self.dir_AB+'A'
        self.dir_B = self.dir_AB+'B'
        self.dir_C = self.dir_AB+'C'
        self.dir_D = self.dir_AB+'D'
        self.AB_paths = sorted(make_dataset(self.dir_AB))  #make_dataset:返回一个所有图像格式文件（含路径）的list
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.C_paths = sorted(make_dataset(self.dir_C))
        self.D_paths = sorted(make_dataset(self.dir_D))
        assert (len(self.AB_paths)!=0),"dataset floder is empty! [%s]"%self.dir_AB

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合

        #deeplabv3+相关
        self.dir_segs=os.path.join(opt.dataroot, opt.phase+"_seg")
        self.dir_rs = os.path.join(opt.dataroot, opt.phase+"_rs")
        from src.util.make_seg_img import labelpixels
        flag=0
        if os.path.isdir(self.dir_segs):
            segs = make_dataset(self.dir_segs)
            if len(self.AB_paths) == len(segs):
                flag=1
            else:
                import shutil
                shutil.rmtree(self.dir_segs)
        if flag==0:
            os.makedirs(self.dir_segs)
            for map in self.AB_paths:
                map_np_AB = np.array(Image.open(map).convert("RGB"))
                map_np=map_np_AB[:,map_np_AB.shape[1]//2:,:]
                seg_np = labelpixels(map_np).astype(np.uint8)
                seg_pil = Image.fromarray(seg_np)
                seg_path = os.path.join(self.dir_segs, get_inner_path(map, self.dir_AB))
                seg_pil.save(seg_path)
        self.seg_paths=sorted(make_dataset(self.dir_segs))
        self.rs_paths = sorted(make_dataset(self.dir_rs))

    def __getitem__(self, index):
        # index = random.randint(0, int(len(self.AB_paths)-1))

        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        C_path = self.C_paths[index]
        D_path = self.D_paths[index]
        rs_path = self.rs_paths[index]


        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')
        C = Image.open(C_path).convert('RGB')
        D = Image.open(D_path).convert('RGB')
        rs = Image.open(rs_path).convert('RGB')


        A = self.transform(A)
        B = self.transform(B)
        C = self.transform(C)
        D = self.transform(D)
        rs = self.transform(rs)


        # deeplabv3相关，取label图，之后需将label图与A，B同步处理

        seg_path = self.seg_paths[index]

        seg = Image.open(seg_path)
        seg = np.asarray(seg)
        seg = torch.from_numpy(seg)  # H W


        # self.opt.no_flip=False
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)  # 64bit带符号整型，why？因为index_select要求下标为longtensor类型
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            C = C.index_select(2, idx)
            D = D.index_select(2, idx)
            rs = rs.index_select(2, idx)
            seg=seg.index_select(1, idx)




        # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
        toseg_transform_list = [transforms.Normalize((0, 0, 0),(2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                transforms.Normalize((-0.5,-0.5,-0.5),(1,1,1)),
                                transforms.Normalize((0.485,0.456,0.406),((0.229,0.224,0.225)))] # 恢复标准化前的数值，并换一组数据标准化
        toseg_transform = transforms.Compose(toseg_transform_list)
        rs_seg=toseg_transform(rs)
        # B_seg=toseg_transform(B)
        # 15-18 ---> 0-3
        domain = int(os.path.split(A_path)[0].split(os.sep)[-1])-15
        domain = np.array(domain)
        domain = torch.from_numpy(domain)
        return {'A': A, 'B': B,'C':C,'D':D, 'seg':seg,'rs_seg': rs_seg, 'A_paths': A_path, 'B_paths': B_path, 'domain':domain,'rs':rs}


    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset_myconsistency_v3'



class AlignedDataset_myconsistency_v2(BaseDataset):
    def initialize(self, opt):
        self.size = 256
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        assert (os.path.isdir(self.dir_AB) or (os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'))),"dataset path not exsit! [%s]"%self.dir_AB
        if (not os.path.isdir(self.dir_AB)) and os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'):#to do:数据集AB自动合并功能尚不支持英文路径
            os.makedirs(self.dir_AB)
            dir_A=self.dir_AB+'A'
            dir_B=self.dir_AB+'B'
            make_align_img(dir_A,dir_B,self.dir_AB)
        self.dir_A = self.dir_AB+'A'
        self.dir_B = self.dir_AB+'B'
        self.AB_paths = sorted(make_dataset(self.dir_AB))  #make_dataset:返回一个所有图像格式文件（含路径）的list
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        assert (len(self.AB_paths)!=0),"dataset floder is empty! [%s]"%self.dir_AB

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合

        #deeplabv3+相关
        self.dir_segs=os.path.join(opt.dataroot, opt.phase+"_seg")
        self.dir_rs = os.path.join(opt.dataroot, opt.phase+"_rs")
        from src.util.make_seg_img import labelpixels
        flag=0
        if os.path.isdir(self.dir_segs):
            segs = make_dataset(self.dir_segs)
            if len(self.AB_paths) == len(segs):
                flag=1
            else:
                import shutil
                shutil.rmtree(self.dir_segs)
        if flag==0:
            os.makedirs(self.dir_segs)
            for map in self.AB_paths:
                map_np_AB = np.array(Image.open(map).convert("RGB"))
                map_np=map_np_AB[:,map_np_AB.shape[1]//2:,:]
                seg_np = labelpixels(map_np).astype(np.uint8)
                seg_pil = Image.fromarray(seg_np)
                seg_path = os.path.join(self.dir_segs, get_inner_path(map, self.dir_AB))
                seg_pil.save(seg_path)
        self.seg_paths=sorted(make_dataset(self.dir_segs))
        self.rs_paths = sorted(make_dataset(self.dir_rs))

    def __getitem__(self, index):
        # index = random.randint(0, int(len(self.AB_paths)-1))

        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        rs_path = self.rs_paths[index]

        img_name = os.path.split(A_path)[-1]
        id_layer = int(os.path.split(A_path)[-1].split(".")[0].split("_")[0])
        img_x = int(os.path.split(A_path)[-1].split(".")[0].split("_")[1])
        img_y = int(os.path.split(A_path)[-1].split(".")[0].split("_")[2])


        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')
        rs = Image.open(rs_path).convert('RGB')

        if id_layer == 18:
            C = A.resize((512, 512))
        else:
            C1_path = os.path.join(self.dir_A, str(id_layer + 1), str(id_layer + 1) + "_" + str(img_x * 2) + "_" + str(img_y * 2) + ".png")
            C2_path = os.path.join(self.dir_A, str(id_layer + 1), str(id_layer + 1) + "_" + str(img_x * 2) + "_" + str(img_y * 2 + 1) + ".png")
            C3_path = os.path.join(self.dir_A, str(id_layer + 1), str(id_layer + 1) + "_" + str(img_x * 2 + 1) + "_" + str(img_y * 2) + ".png")
            C4_path = os.path.join(self.dir_A, str(id_layer + 1), str(id_layer + 1) + "_" + str(img_x * 2 + 1) + "_" + str(img_y * 2 + 1) + ".png")
            C1 = Image.open(C1_path).convert('RGB')
            C2 = Image.open(C2_path).convert('RGB')
            C3 = Image.open(C3_path).convert('RGB')
            C4 = Image.open(C4_path).convert('RGB')
            C = Image.new('RGB', (2 * 256, 2 * 256))  # 创建一个新图
            C.paste(C1, (0 * 256, 0 * 256))
            C.paste(C2, (0 * 256, 1 * 256))
            C.paste(C3, (1 * 256, 0 * 256))
            C.paste(C4, (1 * 256, 1 * 256))

        A = self.transform(A)
        B = self.transform(B)
        C = self.transform(C)
        rs = self.transform(rs)


        # deeplabv3相关，取label图，之后需将label图与A，B同步处理

        seg_path = self.seg_paths[index]

        seg = Image.open(seg_path)
        seg = np.asarray(seg)
        seg = torch.from_numpy(seg)  # H W


        # self.opt.no_flip=False
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)  # 64bit带符号整型，why？因为index_select要求下标为longtensor类型
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            rs = rs.index_select(2, idx)
            seg=seg.index_select(1, idx)

            idx_c = [i for i in range(C.size(2) - 1, -1, -1)]
            idx_c = torch.LongTensor(idx_c)
            C = C.index_select(2, idx_c)


        # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
        toseg_transform_list = [transforms.Normalize((0, 0, 0),(2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                transforms.Normalize((-0.5,-0.5,-0.5),(1,1,1)),
                                transforms.Normalize((0.485,0.456,0.406),((0.229,0.224,0.225)))] # 恢复标准化前的数值，并换一组数据标准化
        toseg_transform = transforms.Compose(toseg_transform_list)
        rs_seg=toseg_transform(rs)
        # B_seg=toseg_transform(B)
        # 15-18 ---> 0-3
        domain = int(os.path.split(A_path)[0].split(os.sep)[-1])-15
        domain = np.array(domain)
        domain = torch.from_numpy(domain)
        return {'A': A, 'B': B,'C':C, 'seg':seg,'rs_seg': rs_seg, 'A_paths': A_path, 'B_paths': B_path, 'domain':domain,'rs':rs}


    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset_myconsistency_v2'


class AlignedDataset_starganv1_2_myconsistency_1_4(BaseDataset):
    def initialize(self, opt):
        self.size = 256
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        assert (os.path.isdir(self.dir_AB) or (os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'))),"dataset path not exsit! [%s]"%self.dir_AB
        if (not os.path.isdir(self.dir_AB)) and os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'):#to do:数据集AB自动合并功能尚不支持英文路径
            os.makedirs(self.dir_AB)
            dir_A=self.dir_AB+'A'
            dir_B=self.dir_AB+'B'
            make_align_img(dir_A,dir_B,self.dir_AB)
        self.dir_A = self.dir_AB+'A'
        self.dir_B = self.dir_AB+'B'
        self.AB_paths = sorted(make_dataset(self.dir_AB))  #make_dataset:返回一个所有图像格式文件（含路径）的list
        assert (len(self.AB_paths)!=0),"dataset floder is empty! [%s]"%self.dir_AB

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合

        #deeplabv3+相关
        self.dir_segs=os.path.join(opt.dataroot, opt.phase+"_seg")
        from src.util.make_seg_img import labelpixels
        flag=0
        if os.path.isdir(self.dir_segs):
            segs = make_dataset(self.dir_segs)
            if len(self.AB_paths) == len(segs):
                flag=1
            else:
                import shutil
                shutil.rmtree(self.dir_segs)
        if flag==0:
            os.makedirs(self.dir_segs)
            for map in self.AB_paths:
                map_np_AB = np.array(Image.open(map).convert("RGB"))
                map_np=map_np_AB[:,map_np_AB.shape[1]//2:,:]
                seg_np = labelpixels(map_np).astype(np.uint8)
                seg_pil = Image.fromarray(seg_np)
                seg_path = os.path.join(self.dir_segs, get_inner_path(map, self.dir_AB))
                seg_pil.save(seg_path)
        self.seg_paths=sorted(make_dataset(self.dir_segs))

    def __getitem__(self, index):
        index = random.randint(0, int(len(self.AB_paths)-1))

        AB_path = self.AB_paths[index]
        img_name = os.path.split(AB_path)[-1]
        id_layer = int(os.path.split(AB_path)[-1].split(".")[0].split("_")[0])
        AB = Image.open(AB_path).convert('RGB')
        AB = AB.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
        AB = self.transform(AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = AB[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]

        # deeplabv3相关，取label图，之后需将label图与A，B同步处理
        seg_path = self.seg_paths[index]
        seg = Image.open(seg_path)
        seg = np.asarray(seg)
        seg = torch.from_numpy(seg)  # H W
        seg = seg[h_offset:h_offset + self.opt.fineSize,
              w_offset:w_offset + self.opt.fineSize]

        if False: #self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        # self.opt.no_flip=False
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)  # 64bit带符号整型，why？因为index_select要求下标为longtensor类型
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            seg=seg.index_select(1,idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)



        # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
        toseg_transform_list = [transforms.Normalize((0, 0, 0),(2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                transforms.Normalize((-0.5,-0.5,-0.5),(1,1,1)),
                                transforms.Normalize((0.485,0.456,0.406),((0.229,0.224,0.225)))] # 恢复标准化前的数值，并换一组数据标准化
        toseg_transform = transforms.Compose(toseg_transform_list)
        A_seg=toseg_transform(A)
        B_seg=toseg_transform(B)
        # 15-18 ---> 0-3
        domain = int(os.path.split(AB_path)[0].split(os.sep)[-1])-15
        domain = np.array(domain)
        domain = torch.from_numpy(domain)
        # 四张对应RS
        # print('A.shape',A.shape)
        # print('A_seg.shape', A_seg.shape)
        # A_seg_big = F.interpolate(A_seg, size=[512, 512], mode='bilinear')
        # A_seg1 = A_seg_big[:, 0:256, 0:256]
        # A_seg2 = A_seg_big[:, 0:256, 256:512]
        # A_seg3 = A_seg_big[:, 256:512, 0:256]
        # A_seg4 = A_seg_big[:, 256:512, 256:512]
        # B = AB[:, h_offset:h_offset + self.opt.fineSize,
        #        w + w_offset:w + w_offset + self.opt.fineSize]
        # img_name_split = img_name.split("_")
        # index_x = int(img_name_split[1])
        # index_y = int(img_name_split[2].split(".")[0])
        # A1_path = os.path.join(self.dir_A, str(id_layer + 1),
        #                        str(id_layer + 1) + "_" + str(index_x * 2 - 1) + "_" + str(
        #                            index_y * 2 - 1) + ".jpg")
        # A2_path = os.path.join(self.dir_A, str(id_layer + 1),
        #                        str(id_layer + 1) + "_" + str(index_x * 2 - 1) + "_" + str(
        #                            index_y * 2) + ".jpg")
        # A3_path = os.path.join(self.dir_A, str(id_layer + 1),
        #                        str(id_layer + 1) + "_" + str(index_x * 2) + "_" + str(
        #                            index_y * 2 - 1) + ".jpg")
        # A4_path = os.path.join(self.dir_A, str(id_layer + 1),
        #                        str(id_layer + 1) + "_" + str(index_x * 2) + "_" + str(index_y * 2) + ".jpg")
        # if id_layer == 18 or (not os.path.exists(A1_path) or not os.path.exists(A2_path) or not os.path.exists(A3_path) or not os.path.exists(A4_path)):
        #     A_path = os.path.join(self.dir_A, str(id_layer), img_name.split('.')[0]+'.jpg')
        #     A_1234 = Image.open(A_path).convert('RGB')
        #     A_1234 = A_1234.resize((self.size*2, self.size*2), Image.ANTIALIAS)
        #     A1 = A_1234.crop((0, 0, self.size, self.size))
        #     A2 = A_1234.crop((self.size, 0, 2*self.size, self.size))
        #     A3 = A_1234.crop((0, self.size, self.size, 2*self.size))
        #     A4 = A_1234.crop((self.size, self.size, 2*self.size, 2*self.size))
        #
        # else:
        #     # img_name = os.path.split(A_path)[1]
        #
        #
        #     A1 = Image.open(A1_path).convert('RGB')
        #     A2 = Image.open(A2_path).convert('RGB')
        #     A3 = Image.open(A3_path).convert('RGB')
        #     A4 = Image.open(A4_path).convert('RGB')
        #
        # A1 = self.transform(A1)
        # A2 = self.transform(A2)
        # A3 = self.transform(A3)
        # A4 = self.transform(A4)
        # A_seg1=toseg_transform(A1)
        # A_seg2=toseg_transform(A2)
        # A_seg3=toseg_transform(A3)
        # A_seg4=toseg_transform(A4)

        # return {'A': A, 'B': B, 'seg':seg,'A_seg':A_seg,'B_seg':B_seg,
        #         'A_paths': AB_path, 'B_paths': AB_path, 'domain':domain, 'A_seg1':A_seg1, 'A_seg2':A_seg2, 'A_seg3':A_seg3, 'A_seg4':A_seg4}
        return {'A': A, 'B': B, 'seg':seg,'A_seg':A_seg,'B_seg':B_seg,
                'A_paths': AB_path, 'B_paths': AB_path, 'domain':domain}

    def __len__(self):
        return int(len(self.AB_paths)/4)

    def name(self):
        return 'AlignedDataset_starganv1_2_myconsistency_1_4'


class AlignedDataset_starganv1_2_myconsistency(BaseDataset):
    def initialize(self, opt):
        self.size=256
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        assert (os.path.isdir(self.dir_AB) or (os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'))),"dataset path not exsit! [%s]"%self.dir_AB
        if (not os.path.isdir(self.dir_AB)) and os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'):#to do:数据集AB自动合并功能尚不支持英文路径
            os.makedirs(self.dir_AB)
            dir_A=self.dir_AB+'A'
            dir_B=self.dir_AB+'B'
            make_align_img(dir_A,dir_B,self.dir_AB)
        self.dir_A = self.dir_AB+'A'
        self.dir_B = self.dir_AB+'B'
        self.AB_paths = sorted(make_dataset(self.dir_AB))  #make_dataset:返回一个所有图像格式文件（含路径）的list
        assert (len(self.AB_paths)!=0),"dataset floder is empty! [%s]"%self.dir_AB

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合

        #deeplabv3+相关
        self.dir_segs=os.path.join(opt.dataroot, opt.phase+"_seg")
        from src.util.make_seg_img import labelpixels
        flag=0
        if os.path.isdir(self.dir_segs):
            segs = make_dataset(self.dir_segs)
            if len(self.AB_paths) == len(segs):
                flag=1
            else:
                import shutil
                shutil.rmtree(self.dir_segs)
        if flag==0:
            os.makedirs(self.dir_segs)
            for map in self.AB_paths:
                map_np_AB = np.array(Image.open(map).convert("RGB"))
                map_np=map_np_AB[:,map_np_AB.shape[1]//2:,:]
                seg_np = labelpixels(map_np).astype(np.uint8)
                seg_pil = Image.fromarray(seg_np)
                seg_path = os.path.join(self.dir_segs, get_inner_path(map, self.dir_AB))
                seg_pil.save(seg_path)
        self.seg_paths=sorted(make_dataset(self.dir_segs))

    def __getitem__(self, index):
        # index = random.randint(0, int(len(self.AB_paths)-1))

        AB_path = self.AB_paths[index]
        img_name = os.path.split(AB_path)[-1]
        id_layer = int(os.path.split(AB_path)[-1].split(".")[0].split("_")[0])
        AB = Image.open(AB_path).convert('RGB')
        AB = AB.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
        AB = self.transform(AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = AB[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]

        # deeplabv3相关，取label图，之后需将label图与A，B同步处理
        seg_path = self.seg_paths[index]
        seg = Image.open(seg_path)
        seg = np.asarray(seg)
        seg = torch.from_numpy(seg)  # H W
        seg = seg[h_offset:h_offset + self.opt.fineSize,
              w_offset:w_offset + self.opt.fineSize]

        if False: #self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        # self.opt.no_flip=False
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)  # 64bit带符号整型，why？因为index_select要求下标为longtensor类型
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            seg=seg.index_select(1,idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)



        # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
        toseg_transform_list = [transforms.Normalize((0, 0, 0),(2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                transforms.Normalize((-0.5,-0.5,-0.5),(1,1,1)),
                                transforms.Normalize((0.485,0.456,0.406),((0.229,0.224,0.225)))] # 恢复标准化前的数值，并换一组数据标准化
        toseg_transform = transforms.Compose(toseg_transform_list)
        A_seg=toseg_transform(A)
        B_seg=toseg_transform(B)
        # 15-18 ---> 0-3
        domain = int(os.path.split(AB_path)[0].split(os.sep)[-1])-15
        domain = np.array(domain)
        domain = torch.from_numpy(domain)
        # 四张对应RS
        # img_name_split = img_name.split("_")
        # index_x = int(img_name_split[1])
        # index_y = int(img_name_split[2].split(".")[0])
        # A1_path = os.path.join(self.dir_A, str(id_layer + 1),
        #                        str(id_layer + 1) + "_" + str(index_x * 2 - 1) + "_" + str(
        #                            index_y * 2 - 1) + ".jpg")
        # A2_path = os.path.join(self.dir_A, str(id_layer + 1),
        #                        str(id_layer + 1) + "_" + str(index_x * 2 - 1) + "_" + str(
        #                            index_y * 2) + ".jpg")
        # A3_path = os.path.join(self.dir_A, str(id_layer + 1),
        #                        str(id_layer + 1) + "_" + str(index_x * 2) + "_" + str(
        #                            index_y * 2 - 1) + ".jpg")
        # A4_path = os.path.join(self.dir_A, str(id_layer + 1),
        #                        str(id_layer + 1) + "_" + str(index_x * 2) + "_" + str(index_y * 2) + ".jpg")
        # if id_layer == 18 or (not os.path.exists(A1_path) or not os.path.exists(A2_path) or not os.path.exists(A3_path) or not os.path.exists(A4_path)):
        #     A_path = os.path.join(self.dir_A, str(id_layer), img_name.split('.')[0]+'.jpg')
        #     A_1234 = Image.open(A_path).convert('RGB')
        #     A_1234 = A_1234.resize((self.size*2, self.size*2), Image.ANTIALIAS)
        #     A1 = A_1234.crop((0, 0, self.size, self.size))
        #     A2 = A_1234.crop((self.size, 0, 2*self.size, self.size))
        #     A3 = A_1234.crop((0, self.size, self.size, 2*self.size))
        #     A4 = A_1234.crop((self.size, self.size, 2*self.size, 2*self.size))
        #
        # else:
        #     # img_name = os.path.split(A_path)[1]
        #
        #
        #     A1 = Image.open(A1_path).convert('RGB')
        #     A2 = Image.open(A2_path).convert('RGB')
        #     A3 = Image.open(A3_path).convert('RGB')
        #     A4 = Image.open(A4_path).convert('RGB')
        #
        # A1 = self.transform(A1)
        # A2 = self.transform(A2)
        # A3 = self.transform(A3)
        # A4 = self.transform(A4)
        # A_seg1=toseg_transform(A1)
        # A_seg2=toseg_transform(A2)
        # A_seg3=toseg_transform(A3)
        # A_seg4=toseg_transform(A4)

        return {'A': A, 'B': B, 'seg':seg,'A_seg':A_seg,'B_seg':B_seg,
                'A_paths': AB_path, 'B_paths': AB_path, 'domain':domain}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset_starganv1_2_myconsistency'


