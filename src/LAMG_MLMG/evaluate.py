import argparse
from pathlib import Path
import numpy as np
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from skimage.measure import compare_psnr, compare_ssim, shannon_entropy
import os
from PIL import Image
import cv2


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif',
]


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


def sam(y_true, y_pred):
    """Spectral Angle Mapper"""
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    y_true_prod = np.sum(np.sqrt(y_true ** 2), axis=0)
    y_pred_prod = np.sum(np.sqrt(y_pred ** 2), axis=0)
    true_pred_prod = np.sum(y_true * y_pred, axis=0)
    ratio = true_pred_prod / (y_true_prod * y_pred_prod)
    angle = np.mean(np.arccos(ratio))
    return angle


def ergas(y_true, y_pred, scale_factor=16):
    errors = []
    for i in range(y_true.shape[0]):
        errors.append(rmse(y_true[i], y_pred[i]))
        errors[i] /= np.mean(y_pred[i])
    return 100.0 / scale_factor * sqrt(np.mean(errors))


def evaluate(y_true, y_pred, func):
    # print(y_true.shape)
    assert y_true.shape == y_pred.shape
    if y_true.ndim == 2:
        y_true = y_true[np.newaxis, :]
        y_pred = y_pred[np.newaxis, :]
    metrics = []
    for i in range(y_true.shape[0]):
        metrics.append(func(y_true[i], y_pred[i]))
    return metrics


def mae(y_true, y_pred):
    return evaluate(y_true, y_pred,
                    lambda x, y: mean_absolute_error(x.ravel(), y.ravel()))


def rmse(y_true, y_pred):
    return evaluate(y_true, y_pred,
                    lambda x, y: sqrt(mean_squared_error(x.ravel(), y.ravel())))


def r2(y_true, y_pred):
    return evaluate(y_true, y_pred,
                    lambda x, y: r2_score(x.ravel(), y.ravel()))


def kge(y_true, y_pred):
    def compute(x, y):
        im_true = x.ravel()
        im_pred = y.ravel()
        r = np.corrcoef(im_true, im_pred)[1, 0]
        m_true = np.mean(im_true)
        m_pred = np.mean(im_pred)
        std_true = np.std(im_true)
        std_pred = np.std(im_pred)
        return 1 - np.sqrt((r - 1) ** 2
                           + (std_pred / std_true - 1) ** 2
                           + (m_pred / m_true - 1) ** 2)

    return evaluate(y_true, y_pred, compute)


def psnr(y_true, y_pred, data_range=255):
    return evaluate(y_true, y_pred,
                    lambda x, y: compare_psnr(x, y, data_range=data_range))


def ssim(y_true, y_pred, data_range=255):
    return evaluate(y_true, y_pred,
                    lambda x, y: compare_ssim(x, y, data_range=data_range))


def entropy(image):
    if image.ndim == 2:
        return shannon_entropy(image)
    if image.ndim == 3:
        entropies = []
        for i in range(image.shape[0]):
            entropies.append(shannon_entropy(image[i]))
        return entropies


if __name__ == '__main__':
    old_path = r"/data/multilayer_map_project/align_data2_2/test/A/4"
    fake_path = r"/data/multilayer_map_project/align_data2_2/test/B1/4"
    # fake_path = r"/data/multilayer_map_project/inter1_2/fake_result/1"
    gt_path = r"/data/multilayer_map_project/align_data2_2/test/C/4"
    f_imgs = make_dataset(fake_path)
    o_imgs = make_dataset(old_path)
    gt_imgs = make_dataset(gt_path)
    rmse_value_all = 0
    ssim_value_all = 0
    psnr_value_all = 0
    num_img = 0
    for img_path in o_imgs:
        num_img = num_img + 1
        print(img_path)
        f_img = cv2.imread(img_path)
        f_img = np.transpose(f_img, (2, 0, 1))
        # f_img = np.array(Image.open(img_path))
        img_name = os.path.split(img_path)[1]

        gt_img_path = os.path.join(gt_path, img_name)
        print(gt_img_path)
        gt_img = cv2.imread(gt_img_path)
        gt_img = np.transpose(gt_img, (2, 0, 1))
        # gt_img = np.array(Image.open(gt_img_path))
        rmse_value = rmse(gt_img, f_img)
        rmse_value = np.mean(rmse_value)

        rmse_value_all = rmse_value_all + rmse_value
        ssim_value = ssim(gt_img, f_img)
        ssim_value = np.mean(ssim_value)

        ssim_value_all = ssim_value_all + ssim_value
        psnr_value = psnr(gt_img, f_img)
        psnr_value = np.mean(psnr_value)
        psnr_value_all = psnr_value_all + psnr_value


    print(rmse_value_all / num_img)
    print(ssim_value_all / num_img)
    print(psnr_value_all / num_img)
