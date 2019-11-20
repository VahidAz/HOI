from data.imgnvid_dataset import ImgVidDataset
from skimage import transform as sktsf
from torchvision import transforms as tvtsf

import torch as t
import torch

import numpy as np

from data import util


def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(t.from_numpy(img))
    return img.numpy()


def caffe_normalize(img):
    """
    return appr -125-125 BGR
    """
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)
    return img


def preprocess(img, min_size=600, max_size=1000, pretrained_flag=False, pretrained_caffe_flag=False, gt_boxes=None):
    """Preprocess an image for feature extraction.
    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.
    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.
    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.
    Returns:
        ~numpy.ndarray: A preprocessed image.
    """
    # C, H, W = img.shape

    # print('\n at beggining \n')
    # print('H, W: ', H, W)

    # ratio = W/H
    # data_width = W
    # data_height = H

    # print('\n ratiooooo: ', ratio)

    # if ratio > 2.0 or ratio < 0.5:

    #     if ratio < 1:
    #         print('\n****** clip ration < 1 ********')
    #         print('ratio: ', ratio)
    #         print('gt_boxes: ', gt_boxes)
    #         # this means that data_width << data_height, we need to crop the
    #         # data_height
    #         gt_boxes =  torch.from_numpy(gt_boxes)

    #         min_y = int(torch.min(gt_boxes[:,1]))
    #         max_y = int(torch.max(gt_boxes[:,3]))
    #         trim_size = int(np.floor(data_width / ratio))
    #         if trim_size > data_height:
    #             trim_size = data_height                
    #         box_region = max_y - min_y + 1
    #         if min_y == 0:
    #             y_s = 0
    #         else:
    #             if (box_region-trim_size) < 0:
    #                 y_s_min = max(max_y-trim_size, 0)
    #                 y_s_max = min(min_y, data_height-trim_size)
    #                 if y_s_min == y_s_max:
    #                     y_s = y_s_min
    #                 else:
    #                     y_s = np.random.choice(range(y_s_min, y_s_max))
    #             else:
    #                 y_s_add = int((box_region-trim_size)/2)
    #                 if y_s_add == 0:
    #                     y_s = min_y
    #                 else:
    #                     y_s = np.random.choice(range(min_y, min_y+y_s_add))
    #         # crop the image
    #         img = img[:, y_s:(y_s + trim_size), :]

    #         # shift y coordiante of gt_boxes
    #         gt_boxes[:, 1] = gt_boxes[:, 1] - float(y_s)
    #         gt_boxes[:, 3] = gt_boxes[:, 3] - float(y_s)

    #         # update gt bounding box according the trip
    #         gt_boxes[:, 1].clamp_(0, trim_size - 1)
    #         gt_boxes[:, 3].clamp_(0, trim_size - 1)

    #         gt_boxes = gt_boxes.numpy()



    #     else:
    #         # this means that data_width >> data_height, we need to crop the
    #         # data_width
    #         print('\n******** clip ratio > 1 *****')
    #         # print('\n****** clip ration < 1 ********')
    #         print('ratio: ', ratio)
    #         print('gt_boxes: ', gt_boxes)

    #         gt_boxes =  torch.from_numpy(gt_boxes)

    #         min_x = int(torch.min(gt_boxes[:,0]))
    #         max_x = int(torch.max(gt_boxes[:,2]))
    #         trim_size = int(np.ceil(data_height * ratio))
    #         if trim_size > data_width:
    #             trim_size = data_width                
    #         box_region = max_x - min_x + 1
    #         if min_x == 0:
    #             x_s = 0
    #         else:
    #             if (box_region-trim_size) < 0:
    #                 x_s_min = max(max_x-trim_size, 0)
    #                 x_s_max = min(min_x, data_width-trim_size)
    #                 if x_s_min == x_s_max:
    #                     x_s = x_s_min
    #                 else:
    #                     x_s = np.random.choice(range(x_s_min, x_s_max))
    #             else:
    #                 x_s_add = int((box_region-trim_size)/2)
    #                 if x_s_add == 0:
    #                     x_s = min_x
    #                 else:
    #                     x_s = np.random.choice(range(min_x, min_x+x_s_add))
    #         # crop the image
    #         img = img[:, :, x_s:(x_s + trim_size)]

    #         # shift x coordiante of gt_boxes
    #         gt_boxes[:, 0] = gt_boxes[:, 0] - float(x_s)
    #         gt_boxes[:, 2] = gt_boxes[:, 2] - float(x_s)
    #         # update gt bounding box according the trip
    #         gt_boxes[:, 0].clamp_(0, trim_size - 1)
    #         gt_boxes[:, 2].clamp_(0, trim_size - 1)

    #         gt_boxes = gt_boxes.numpy()

    # print('\n After clip \n')
    # print(img.shape)
    # print(gt_boxes)
    # print('\n---------------------\n')

    # print('\n ANNNNNNNNNNN \n')
    C, H, W = img.shape
    #print('C, H, W: ', C, H, W)
    #scale1 = min_size / min(H, W)
    #scale2 = max_size / max(H, W)
    # print('scale1: ', scale1)
    # print('scale2: ', scale2)
    #scale = min(scale1, scale2)
    # print('scale: ', scale)
    scale = max_size / float(max(H, W))
    img = img / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)

    # print('Image after scaling: ', img.shape)

    # print('img.shape after resize: ', img.shape)
    # both the longer and shorter should be less than
    # max_size and min_size
    if pretrained_flag:
        if pretrained_caffe_flag:
            normalize = caffe_normalize
    else:
        normalize = pytorch_normalze

    return normalize(img), scale #, gt_boxes #, int(np.ceil(H * scale)), int(np.ceil((W * scale))), scale


class Transform(object):

    def __init__(self, min_size=600, max_size=1000, pretrained_flag=False, pretrained_caffe_flag=False):
        self.min_size = min_size
        self.max_size = max_size
        self.pretrained_flag = pretrained_flag
        self.pretrained_caffe_flag = pretrained_caffe_flag

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        # img, scH, scW, scale = preprocess(img, self.min_size, self.max_size)
        img, scale = preprocess(img, self.min_size, self.max_size, self.pretrained_flag, self.pretrained_caffe_flag, bbox)
        _, o_H, o_W = img.shape
        #scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        img, params = util.random_flip(
            img, x_random=True, return_param=True)
        bbox = util.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])


        # gt_boxes_padding =np.zeros(shape=(20, 5))
        
        # for ii, boxi in enumerate(bbox):
        #     gt_boxes_padding[ii] = boxi


        # print('\nGGGGGGGGG')
        # print(bbox)
        # print(gt_boxes_padding)
        # exit(0)


        return img, bbox, label, o_H, o_W, scale #, gt_boxes_padding


class Dataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.db = ImgVidDataset(data_dir='./datasets/ILSVRC2015/', split='train', num=30)
        self.tsf = Transform(cfg.min_size, cfg.max_size, cfg.pretrained_flag, cfg.pretrained_caffe_flag)

    def __getitem__(self, idx):
        # print('\n****************************************\n')
        # print('\n ___get_item__ \n')
        # print('idx: ', idx)

        ori_img, bbox, label, occluded = self.db.get_example(idx)

        final_img = []
        final_bbox = []
        final_label = []
        final_bbox_num = []
        final_im_info = []

        for cur_img, cur_bbox, cur_label in zip(ori_img, bbox, label):
            print('\n In THE LOOP \n')
            print('\n BEFORE TRANSFORM \n')
            print(cur_img.shape)
            print(cur_bbox.shape)
            print(cur_label.shape)
            img, bbox, label, o_H, o_W, scale= self.tsf((cur_img, cur_bbox, cur_label))

            print('\n AFTER TRANSFORM \n')
            print(img.shape)
            print(bbox.shape)
            print(label.shape)
            print(o_H)
            print(o_W)
            print(scale)
            
            # TODO: check whose stride is negative to fix this instead copy all
            # some of the strides of a given numpy array are negative.
            tmp_im_info = np.zeros(shape=(1,3))
            tmp_im_info[0][0] = o_H
            tmp_im_info[0][1] = o_W
            tmp_im_info[0][2] = scale

            final_img.append(img)
            final_bbox.append(bbox)
            final_label.append(label)
            final_bbox_num.append(bbox.shape[0])
            final_im_info.append(tmp_im_info)

        # return img.copy(), bbox.copy(), label.copy(), tmp_im_info, bbox.shape[0], idx, id_
        return np.asarray(final_img), np.asarray(final_bbox), np.asarray(final_label), np.asarray(final_im_info), np.asarray(final_bbox_num)

    def __len__(self):
        return len(self.db)
