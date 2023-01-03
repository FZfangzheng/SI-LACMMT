import math
import os
import timeit
import math

import numpy as np
import ot
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.models as models
import pdb
from tqdm import tqdm

from scipy.stats import entropy
from numpy.linalg import norm
from scipy import linalg


'''注意：fid的实现与paper中用TensorFlow实现的有些许偏差，仅供参考'''
'''注意：因vgg的conv空间维数过多导致计算fid时程序卡死，vgg版的fid对conv结果进行了平均池化，由512*7*7池化至512（与官方代码不同）'''

class eval_memo():
    def __init__(self,len,conv_models=['inception_v3'],gpu='-1',needinception=False,needmode=False,needwasserstein=False):
        if gpu == '-1':
            gpu = ''
        #os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        self.gpu = gpu
        self.cuda = gpu != ''
        self.len = len
        self.conv_models=conv_models
        self.needinception=needinception
        self.needmode=needmode
        self.needwasserstein=needwasserstein

        self.numA = 0
        self.numB = 0
        self.predA = {}
        self.predB = {}
        self.score= {}
        self.featrue_makers={}
        for conv_model in conv_models:
            self.featrue_makers[conv_model]=ConvNetFeatureSaver(model=conv_model,cuda=self.cuda)
            self.predA[conv_model] = {'pixl': [], 'conv': [], 'logit': [], 'smax': []}
            self.predB[conv_model] = {'pixl': [], 'conv': [], 'logit': [], 'smax': []}

    def add_imgA(self,imgs):
        # imgs :bn*h*w*c  取值0-255，长宽未知，ndarray
        for conv_model in self.conv_models:
            pixl,conv,logit,smax=self.featrue_makers[conv_model].make_from_ndarray(imgs)
            self.predA[conv_model]['pixl'].append(pixl)
            self.predA[conv_model]['conv'].append(conv)
            self.predA[conv_model]['logit'].append(logit)
            self.predA[conv_model]['smax'].append(smax)
        self.numA+=imgs.shape[0]

    def add_imgB(self,imgs):
        # imgs :bn*h*w*c  取值0-255，长宽未知，ndarray
        for conv_model in self.conv_models:
            pixl, conv, logit, smax = self.featrue_makers[conv_model].make_from_ndarray(imgs)
            self.predB[conv_model]['pixl'].append(pixl)
            self.predB[conv_model]['conv'].append(conv)
            self.predB[conv_model]['logit'].append(logit)
            self.predB[conv_model]['smax'].append(smax)
        self.numB += imgs.shape[0]

    def get_score(self):
        print(self.numA)
        print(self.numB)
        print(self.len)
        assert self.numA==self.len and self.numB==self.len
        for conv_model in self.conv_models:
            self.score[conv_model]={}
            self.score[conv_model]['mmd'] = {}
            self.score[conv_model]['knn']={}
            if self.needwasserstein:
                self.score[conv_model]['wasserstein']={}
            for i in ['pixl', 'conv', 'logit', 'smax']:
                print('compute score in space: ' + i)
                self.predA[conv_model][i]=torch.cat(self.predA[conv_model][i],0)
                self.predB[conv_model][i] = torch.cat(self.predB[conv_model][i], 0)

                Mxx = distance(self.predA[conv_model][i], self.predA[conv_model][i], False)
                Mxy = distance(self.predA[conv_model][i], self.predB[conv_model][i], False)
                Myy = distance(self.predB[conv_model][i], self.predB[conv_model][i], False)

                self.score[conv_model]['mmd'][i] = mmd(Mxx, Mxy, Myy, 1)
                self.score[conv_model]['knn'][i]=knn(Mxx, Mxy, Myy, 1, False).acc
                if self.needwasserstein:
                    self.score[conv_model]['wasserstein'][i] = wasserstein(Mxy, True)
            # if conv_model.find('vgg') < 0:
            self.score[conv_model]['fid'] = fid(self.predA[conv_model]['conv'], self.predB[conv_model]['conv'])
            # else:
            #     self.score[conv_model]['fid'] = fid(self.predA[conv_model]['conv'].view((self.predA[conv_model]['conv'].size(0),-1,7,7)).mean(3).mean(2), self.predB[conv_model]['conv'].view((self.predA[conv_model]['conv'].size(0),-1,7,7)).mean(3).mean(2))

            if self.needinception:
                self.score[conv_model]['inception'] = inception_score(self.predB[conv_model]['smax'])
            if self.needmode:
                self.score[conv_model]['mode']= mode_score(self.predA[conv_model]['smax'],self.predB[conv_model]['smax'])
        return self.score

class ConvNetFeatureSaver(object):
    def __init__(self, model,cuda, workers=4, batchSize=64):
        '''
        model: inception_v3, vgg13, vgg16, vgg19, resnet18, resnet34,
               resnet50, resnet101, or resnet152
        '''
        self.model = model
        self.cuda = cuda
        self.batch_size = batchSize
        self.workers = workers
        if self.model.find('vgg') >= 0:
            self.vgg = getattr(models, model)(pretrained=True).eval()
            if self.cuda:
                self.vgg.cuda()
            self.trans = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ])
        elif self.model.find('resnet') >= 0:
            resnet = getattr(models, model)(pretrained=True).eval()
            resnet_feature = nn.Sequential(resnet.conv1, resnet.bn1,
                                           resnet.relu,
                                           resnet.maxpool, resnet.layer1,
                                           resnet.layer2, resnet.layer3,
                                           resnet.layer4).eval()
            if self.cuda:
                resnet.cuda()
                resnet_feature.cuda()
            self.resnet = resnet
            self.resnet_feature = resnet_feature
            self.trans = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ])
        elif self.model == 'inception' or self.model == 'inception_v3':
            inception = models.inception_v3(
                pretrained = True, transform_input = False).eval()
            inception_feature = nn.Sequential(inception.Conv2d_1a_3x3,
                                              inception.Conv2d_2a_3x3,
                                              inception.Conv2d_2b_3x3,
                                              nn.MaxPool2d(3, 2),
                                              inception.Conv2d_3b_1x1,
                                              inception.Conv2d_4a_3x3,
                                              nn.MaxPool2d(3, 2),
                                              inception.Mixed_5b,
                                              inception.Mixed_5c,
                                              inception.Mixed_5d,
                                              inception.Mixed_6a,
                                              inception.Mixed_6b,
                                              inception.Mixed_6c,
                                              inception.Mixed_6d,
                                              inception.Mixed_6e,
                                              inception.Mixed_7a,
                                              inception.Mixed_7b,
                                              inception.Mixed_7c,
                                              nn.AdaptiveAvgPool2d(output_size=(1, 1))
                                              ).eval()
            if self.cuda:
                inception.cuda()
                inception_feature.cuda()
            self.inception = inception
            self.inception_feature = inception_feature
            self.trans = transforms.Compose([
                transforms.Resize(299),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 减去mean，除以std
            ])
        else:
            raise NotImplementedError

    def make_from_ndarray (self, imgs):
        # imgs :bn*h*w*c  取值0-255，长宽未知，ndarray
        pretran=transforms.ToPILImage()
        transimgs =[]
        for img in imgs:
            img=pretran(img)
            img=self.trans(img)
            transimgs.append(img)
        transimgs=torch.stack(transimgs,0)
        return self.makeit(transimgs)

    def makeit(self,imgs):
        #imgs：归一化后、resize后的bs*c*h*w的tensor
        with torch.no_grad():
            input = imgs if not self.cuda else imgs.cuda()
            # if self.model == 'vgg' or self.model == 'vgg16':\
            if self.model.find('vgg') >= 0:
                fconv = self.vgg.features(input)#.view(input.size(0), -1)
                flogit = self.vgg.classifier(fconv.view(input.size(0), -1))
                fconv=fconv.mean(3).mean(2)
            elif self.model.find('resnet') >= 0:
                fconv = self.resnet_feature(
                    input).mean(3).mean(2)#.squeeze()
                flogit = self.resnet.fc(fconv)
            elif self.model == 'inception' or self.model == 'inception_v3':
                fconv = self.inception_feature(
                    input).squeeze(3).squeeze(2)#mean(3).mean(2)#.squeeze()
                flogit = self.inception.fc(fconv)  # logit: 可简单理解为未归一化的概率
            else:
                raise NotImplementedError
            fsmax = F.softmax(flogit)
        return imgs,fconv.data.cpu(),flogit.data.cpu(),fsmax.data.cpu()



def distance(X, Y, sqrt):  #M（i，j）是Xi与Yj的向量的距离平方（即xi-yj的自己点乘）
    nX = X.size(0)
    nY = Y.size(0)
    X = X.view(nX,-1)  #改变形状
    X2 = (X*X).sum(1).resize_(nX,1)
    Y = Y.view(nY,-1)
    Y2 = (Y*Y).sum(1).resize_(nY,1)

    M = torch.zeros(nX, nY)
    M.copy_(X2.expand(nX, nY) + Y2.expand(nY, nX).transpose(0, 1) -
            2 * torch.mm(X, Y.transpose(0, 1)))  #torch.mm: 矩阵乘法     expand: 仅能扩展尺寸为1的维度，改变视图而不会分配内存

    del X, X2, Y, Y2

    if sqrt:
        M = ((M + M.abs()) / 2).sqrt()

    return M


def wasserstein(M, sqrt):
    if sqrt:
        M = M.abs().sqrt()
    emd = ot.emd2([], [], M.numpy())

    return emd


class Score_knn:
    acc = 0
    acc_real = 0
    acc_fake = 0
    precision = 0
    recall = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0


def knn(Mxx, Mxy, Myy, k, sqrt):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1)))
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat(
        (Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1))
                ).topk(k, 0, False)   #返回指定维度最大/最小k个值及其索引，这里是最小

    count = torch.zeros(n0 + n1)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])   # 每列最小值在x区域则加一，y区域则加零，什么意思？？  1和0是标记，通过加标记的方式获得k近邻中有几个x有几个y
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1)).float()   #ge 大于等于；此处判断k近邻中来自x的多还是y的多

    s = Score_knn()
    s.tp = (pred * label).sum()  #true postive，X中k近邻X多的个数
    s.fp = (pred * (1 - label)).sum()  #Y中近邻X多的个数
    s.fn = ((1 - pred) * label).sum()  #X中近邻Y多的个数
    s.tn = ((1 - pred) * (1 - label)).sum()  #Y中近邻Y多的个数
    s.precision = s.tp / (s.tp + s.fp + 1e-10)  #
    s.recall = s.tp / (s.tp + s.fn + 1e-10)
    s.acc_real = s.tp / (s.tp + s.fn)
    s.acc_fake = s.tn / (s.tn + s.fp)
    s.acc = torch.eq(label, pred).float().mean().item()
    s.k = k

    return s


def mmd(Mxx, Mxy, Myy, sigma):
    scale = Mxx.mean()
    Mxx = torch.exp(-Mxx / (scale * 2 * sigma * sigma))
    Mxy = torch.exp(-Mxy / (scale * 2 * sigma * sigma))
    Myy = torch.exp(-Myy / (scale * 2 * sigma * sigma))
    mmd = math.sqrt(Mxx.mean() + Myy.mean() - 2 * Mxy.mean())

    return mmd

eps = 1e-20
def inception_score(X):
    kl = X * ((X+eps).log()-(X.mean(0)+eps).log().expand_as(X))
    score = np.exp(kl.sum(1).mean())

    return score.item()

def mode_score(X, Y):
    kl1 = X * ((X+eps).log()-(X.mean(0)+eps).log().expand_as(X))
    kl2 = X.mean(0) * ((X.mean(0)+eps).log()-(Y.mean(0)+eps).log())
    score = np.exp(kl1.sum(1).mean() - kl2.sum())

    return score.item()


def fid(X, Y):
    m = X.mean(0)
    m_w = Y.mean(0)
    X_np = X.numpy()
    Y_np = Y.numpy()

    C = np.cov(X_np.transpose())
    C_w = np.cov(Y_np.transpose())
    C_C_w_sqrt = linalg.sqrtm(C.dot(C_w), True).real

    score = m.dot(m) + m_w.dot(m_w) - 2 * m_w.dot(m) + \
        np.trace(C + C_w - 2 * C_C_w_sqrt)
    return score.item()  #np.sqrt(score)


