import torch
import numpy as np
from PIL import Image
import os
import os.path as osp

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif',
]

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def get_edges(t: torch.Tensor):
    edge = torch.zeros(t.shape).bool().to(t.device)
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()


def label_to_one_hot(targets: torch.Tensor, n_class: int, with_255: bool = False):
    """
    get one-hot tensor from targets, ignore the 255 label
    :param targets: long tensor[bs, 1, h, w]
    :param nlabels: int
    :return: float tensor [bs, nlabel, h, w]
    """
    # batch_size, _, h, w = targets.size()
    # res = torch.zeros([batch_size, nlabels, h, w])
    targets = targets.squeeze(dim=1)
    # print(targets.shape)
    zeros = torch.zeros(targets.shape).long().to(targets.device)

    # del 255.
    targets_ignore = targets >= n_class
    # print(targets_ignore)
    targets = torch.where(targets < n_class, targets, zeros)

    one_hot = torch.nn.functional.one_hot(targets, num_classes=n_class)
    if with_255:
        one_hot[targets_ignore] = 0
    else:
        one_hot[targets_ignore] = 255
    # print(one_hot[targets_ignore])
    one_hot = one_hot.transpose(3, 2)
    one_hot = one_hot.transpose(2, 1)
    # print(one_hot.size())
    return one_hot.float()


def create_dir(dir_path):
    if not osp.exists(dir_path):
        os.mkdir(dir_path)


def merge_imgs(origin_path,merge_path,target_path,id_layer,size):
# id_layer比本身低
    img_list = make_dataset(origin_path)
    for img in img_list:
        img_name = os.path.split(img)[1]
        img_name_split = img_name.split("_")
        index_x = int(img_name_split[1])
        index_y = int(img_name_split[2].split(".")[0])

        B1_path = os.path.join(merge_path,
                               str(id_layer + 1) + "_" + str(index_x * 2) + "_" + str(index_y * 2)+".png")
        B2_path = os.path.join(merge_path,
                               str(id_layer + 1) + "_" + str(index_x * 2) + "_" + str(index_y * 2+1)+".png")
        B3_path = os.path.join(merge_path,
                               str(id_layer + 1) + "_" + str(index_x * 2+1) + "_" + str(index_y * 2)+".png")
        B4_path = os.path.join(merge_path,
                               str(id_layer + 1) + "_" + str(index_x * 2+1) + "_" + str(index_y * 2+1)+".png")
        B1 = Image.open(B1_path).convert('RGB')
        B2 = Image.open(B2_path).convert('RGB')
        B3 = Image.open(B3_path).convert('RGB')
        B4 = Image.open(B4_path).convert('RGB')

        B = Image.new('RGB', (2 * size, 2 * size))  # 创建一个新图
        B.paste(B1, (0*size, 0*size))
        B.paste(B2, (0*size, 1*size))
        B.paste(B3, (1*size, 0*size))
        B.paste(B4, (1*size, 1*size))
        B = B.resize((size, size))
        B.save(os.path.join(target_path, img_name))


def crop_imgs(origin_path,target_path,id_layer,size=256):
    img_list = make_dataset(origin_path)
    # print(len(img_list))
    for img in img_list:
        img_name = os.path.split(img)[1]
        img_name_split = img_name.split("_")
        index_x = int(img_name_split[1])
        index_y = int(img_name_split[2].split(".")[0])

        B1_path = os.path.join(target_path,
                               str(id_layer + 1) + "_" + str(index_x * 2) + "_" + str(index_y * 2)+".png")
        B2_path = os.path.join(target_path,
                               str(id_layer + 1) + "_" + str(index_x * 2) + "_" + str(index_y * 2+1)+".png")
        B3_path = os.path.join(target_path,
                               str(id_layer + 1) + "_" + str(index_x * 2+1) + "_" + str(index_y * 2)+".png")
        B4_path = os.path.join(target_path,
                               str(id_layer + 1) + "_" + str(index_x * 2+1) + "_" + str(index_y * 2+1)+".png")

        B = Image.open(img).convert('RGB')
        B = B.resize((2* size,2*size))
        B1 = B.crop((0*size, 0*size,1*size,1*size))
        B2 = B.crop((0*size, 1*size,1*size,2*size))
        B3 = B.crop((1*size, 0*size,2*size,1*size))
        B4 = B.crop((1*size, 1*size,2*size,2*size))
        B1.save(B1_path)
        B2.save(B2_path)
        B3.save(B3_path)
        B4.save(B4_path)


def from_std_tensor_save_image(filename, data, std=[0.5, 0.5, 0.5], mean=[0.5, 0.5, 0.5]): # 三通道为图像预测值(c*h*w)，而一通道为已经转换为0-255的图像
# def from_std_tensor_save_image(filename, data, std=[0.229, 0.224, 0.225], mean=[0.485, 0.456, 0.406]):
    if len(data.shape)==3:
        std = np.array(std).reshape((3, 1, 1))
        mean = np.array(mean).reshape((3, 1, 1))
        img = data.clone().numpy()
        img = ((img * std + mean).transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
        img = Image.fromarray(img)
        img.save(filename)
    elif len(data.shape)==2:
        # img=data.clone().numpy()
        # img = img.clip(0, 255).astype("uint8")
        # img = Image.fromarray(img)
        # img.save(filename)
        img = data.clone().numpy()
        mean = np.array([0.5])
        std = np.array([0.5])
        inp = std*img+mean

        inp = np.clip(inp,0,1)
        inp = inp*255.0
        inp = inp.astype("uint8")
        img = Image.fromarray(inp)
        img.save(filename)



def get_encode_features(E: torch.nn.Module, imgs: torch.Tensor, instances: torch.Tensor, labels: torch.Tensor):
    """
    get instance-wise pooling feature from encoder, this function is also built in encode
    :param E:
    :param imgs:
    :param instances:
    :param labels:
    :return:
    """
    assert imgs.dim() == 4
    encode_features = E(imgs)
    batch_size = imgs.size(0)
    class_feature_dict = {}
    for b in range(batch_size):
        encode_feature = encode_features[b]
        instance = instances[b]
        label = labels[b]
        for i in instance.unique():
            mask = (instance == i).expand_as(encode_feature)
            cls = int(label[mask].unique())
            mean_feature = encode_feature[mask] / mask.float().sum()
            encode_feature[mask] = mean_feature
            if cls not in class_feature_dict:
                class_feature_dict[cls] = []
            class_feature_dict[cls].append(mean_feature.cpu().numpy())
    return encode_features, class_feature_dict

if __name__ == '__main__':
    from pix2pixHD.train_voc import *
    pass