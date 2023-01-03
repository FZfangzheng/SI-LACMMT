"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
import os
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
class MYDATAPPDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        if is_train:
            parser.set_defaults(load_size=256)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=3)
        parser.set_defaults(contain_dontcare_label=False)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        parser.set_defaults(no_instance=True)
        return parser

    def get_paths(self, opt):
        croot = r'/data3/fangzheng/map_generator/experiment/pp_data/pp_pix2pixHD'
        # croot = r'/BIT/BIT/fangzheng/data/data_for_SPADE_RIO'

        c_image_dir = os.path.join(croot, '%sA' % opt.phase)
        c_image_paths = sorted(make_dataset(c_image_dir, recursive=True))

        s_image_dir = os.path.join(croot, '%sB' % opt.phase)
        s_image_paths = sorted(make_dataset(s_image_dir, recursive=True))

        if opt.phase == 'train':
            for i in range(2):
                s_image_paths = s_image_paths + s_image_paths



        length = min(len(c_image_paths), len(s_image_paths))
        c_image_paths = c_image_paths[:length]
        s_image_paths = s_image_paths[:length]
        instance_paths = []
        return c_image_paths, s_image_paths, instance_paths


    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        label = label.convert('RGB')

        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params)
        label_tensor = transform_label(label)

        # input image (real images)
        image_path = self.image_paths[index]
        assert self.paths_match(label_path, image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)
        image = Image.open(image_path)
        image = image.convert('RGB')


        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_path = self.instance_paths[index]
            instance = Image.open(instance_path)
            if instance.mode == 'L':
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_label(instance)

        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict
