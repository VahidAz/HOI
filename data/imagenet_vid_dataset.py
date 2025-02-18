import numpy as np
from skimage import transform as sktsf
import cv2 as cv

import torch
from torchvision import transforms as tvtsf

from data.imagenet_vid_parser import IMGNETVIDPARSER
from data import util


def pytorch_normalze(_img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(_img))

    return img.numpy()


def caffe_normalize(_img):
    """
    return appr -125-125 BGR
    """
    img = _img[[2, 1, 0], :, :]  # RGB->BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)

    return img


def inverse_normalize(_img, _caffe=True):
    if _caffe:
        img = _img + (
            np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        # return img[::-1, :, :]

        # For OpenCV, img already is in BGR
        # C H W -> H W C, np.float32 -> np.uint8
        img = img.transpose((1, 2, 0))
        img = img.astype(np.uint8, copy=False)

        return img

    # approximate un-normalize for visualize
    return (_img * 0.225 + 0.45).clip(min=0, max=1) * 255


def preprocess(_img, _min_size=600, _max_size=1000, 
    _pretrained=False, _pretrained_caffe=False):
    
    # TODO: Crop the image if w >> h or h >> w

    C, H, W = _img.shape
    img = _img / 255.

    scale = _min_size / float(min(H, W))
    if np.round(scale * max(H, W)) > _max_size:
        scale = _max_size / float(max(H, W))
    
    img = sktsf.resize(img, (C, H * scale, W * scale), anti_aliasing=False)

    if _pretrained:
        if _pretrained_caffe:
            normalize = caffe_normalize
    else:
        normalize = pytorch_normalze

    ###<<< DEBUG
    # print('\nGet example in voc preprocess>>>')
    # print(img.shape)
    # # C H W -> H W C, RGB -> BGR, np.float32 -> np.uint8
    # img_ = img.transpose((1, 2, 0))
    # img_ = img_[...,::-1].copy()
    # img_ = img_ * 255
    # img_ = img_.astype(np.uint8, copy=False)

    # cv.imwrite('ge_voc_preprocess.jpg', img_)
    ###>>>

    return normalize(img), scale


class Transform(object):
    def __init__(self, _min_size=600, _max_size=1000, 
        _pretrained=False, _pretrained_caffe=True):
        self.min_size = _min_size
        self.max_size = _max_size
        self.pretrained = _pretrained
        self.pretrained_caffe = _pretrained_caffe


    def __call__(self, _in_data):
        img, bbox, label = _in_data

        _, H, W = img.shape

        img, scale = preprocess(img, self.min_size, self.max_size, 
            self.pretrained, self.pretrained_caffe)

        _, o_H, o_W = img.shape

        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W), scale)

        # Horizontally flip
        img, params = util.random_flip(img, x_random=False, 
            return_param=True)
        bbox = util.flip_bbox(bbox, (o_H, o_W), 
            x_flip=params['x_flip'])

        return img, bbox, label, o_H, o_W, scale


# TODO : Find a better way for implementing this Dataset
class Dataset:
    def __init__(self, _cfg):
        self.cfg = _cfg
        self.db = IMGNETVIDPARSER(
            _data_dir=self.cfg.imgnet_vid_dataset_train, 
            _split='train', _time_win=self.cfg.time_win, _class_num=3) #30)
        self.tsf = Transform(self.cfg.min_size, self.cfg.max_size, 
            self.cfg.pretrained, self.cfg.pretrained_caffe)


    def __getitem__(self, _idx):
        ori_img, bbox, label, occluded, corrupted = self.db.get_example(_idx)

        final_img = []
        final_bbox = []
        final_label = []
        final_bbox_num = []
        final_im_info = []
        final_corrupted = []

        final_corrupted.append(corrupted)

        if corrupted == 0: # False
            for cur_img, cur_bbox, cur_label in zip(ori_img, bbox, label):
                img, bbox, label, o_H, o_W, scale = self.tsf(
                    (cur_img, cur_bbox, cur_label))

                tmp_im_info = np.zeros(shape=(1,3))
                tmp_im_info[0][0] = o_H
                tmp_im_info[0][1] = o_W
                tmp_im_info[0][2] = scale

                final_img.append(img)
                final_bbox.append(bbox)
                final_label.append(label)
                final_bbox_num.append([bbox.shape[0]])
                final_im_info.append(tmp_im_info)

            return np.asarray(final_img).copy(), \
                np.asarray(final_bbox).copy(), \
                np.asarray(final_label).copy(), \
                np.asarray(final_im_info).copy(), \
                np.asarray(final_bbox_num).copy(), \
                np.asarray(final_corrupted).copy()

        else:
            # To address toch issue for squzee
            # Retrun dummy values
            return np.asarray(final_corrupted).copy(), \
                np.asarray(final_corrupted).copy(), \
                np.asarray(final_corrupted).copy(), \
                np.asarray(final_corrupted).copy(), \
                np.asarray(final_corrupted).copy(), \
                np.asarray(final_corrupted).copy()


    def __len__(self):
        return len(self.db)


    def num_classes(self):
        return self.db.num_classes()
