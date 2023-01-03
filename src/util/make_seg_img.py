import numpy as np
import shutil
import os
from PIL import Image
from data.image_folder import make_dataset
from src.util.my_util import get_inner_path
import cv2

from src.pix2pixHD.myutils import gray2rgb

def labelpixels_old_not_use(img3D, label_list=[[239,238,236],[255,242,175],[170,218,255],[208,236,208],[255,255,255]]):
    # scale array
    s = 256**np.arange(img3D.shape[-1])

    # Reduce image and labels to 1D
    img1D = img3D.reshape(-1,img3D.shape[-1]).dot(s)
    label1D = np.dot(label_list, s)

    ret1D=np.zeros(img1D.shape)
    for i,num in enumerate(label1D):
        ret1D+=(img1D==num)*(i+1)
    ret2D=ret1D.reshape(img3D.shape[:-1])
    return ret2D

def labelpixels(img3D, label_list=[(15,23),(24,26),(101,105),(58,62),(0,0)]):  # 接受参数为ndarray
    img3D = cv2.cvtColor(img3D, cv2.COLOR_RGB2HSV)
    ret2D=np.zeros(img3D.shape[:-1])
    # img3D = cv2.cvtColor(img3D, cv2.COLOR_HSV2BGR)
    # cv2.imshow('hsv', img3D)
    # cv2.waitKey()
    # lower_reds = []
    # upper_reds = []
    masks=[]
    for tmp in label_list:
        # lower_reds.append(np.array([tmp[0], 0, 0]))
        # upper_reds.append(np.array([tmp[1], 255, 255]))
        masks.append(cv2.inRange(img3D, np.array([tmp[0], 0, 0]), np.array([tmp[1], 255, 255])))
    for i,mask in enumerate(masks):
        mask=(mask==255)
        ret2D+=mask*(i+1)
    mask=(ret2D==0)
    ret2D+=mask*256
    ret2D=ret2D-1
    ret2D=ret2D.astype(np.uint8)
    return ret2D



def make_seg_img_from_maps(maps_path,segs_path):
    maps=make_dataset(maps_path)
    if os.path.isdir(segs_path):
        segs=make_dataset(segs_path)
        if len(maps)==len(segs):
            return
        else:
            shutil.rmtree(segs_path)
    os.makedirs(segs_path)
    for map in maps:
        map_np=np.array(Image.open(map).convert("RGB"))
        seg_np=labelpixels(map_np).astype(np.uint8)
        seg_np_rgb=gray2rgb(seg_np)
        seg_pil=Image.fromarray(seg_np_rgb)
        seg_path=os.path.join(segs_path,get_inner_path(map,maps_path))
        seg_pil.save(seg_path)

    pass

if __name__=="__main__":
    pass
    # label_list = np.array([[[239, 238, 236], [255, 242, 175], [170, 218, 255], [208, 236, 208], [255, 255, 255]]]).astype(np.uint8)
    # label_list =cv2.cvtColor(label_list,cv2.COLOR_RGB2HSV)


    maps_path="D:\\map_translate\\数据集\\20191117第三批数据\\导出数据\\rs\\15_tiny"
    segs_path="D:\\map_translate\\数据集\\20191117第三批数据\\导出数据\\rs\\seg15_tiny"
    make_seg_img_from_maps(maps_path, segs_path)