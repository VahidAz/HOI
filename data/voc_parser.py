import os

import xml.etree.ElementTree as ET
import numpy as np
import cv2 as cv

from data.util import read_image


class VOCPARSER:
    def __init__(self, _data_dir, _split='trainval',
                 _use_difficult=False, _return_difficult=False):
        id_file = os.path.join(
            _data_dir, 'ImageSets/Main/{0}.txt'.format(_split))

        self.ids = [id_.strip() for id_ in open(id_file)]
        self.data_dir = _data_dir
        self.use_difficult = _use_difficult
        self.return_difficult = _return_difficult
        self.label_names = VOC_BBOX_LABEL_NAMES


    def __len__(self):
        return len(self.ids)


    def num_classes(self):
        return len(self.label_names)


    def get_example(self, _idx):
        id_ = self.ids[_idx]

        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))

        bbox = list()
        label = list()
        difficult = list()

        for obj in anno.findall('object'):
            if not self.use_difficult and int(
                obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))

            name = obj.find('name').text.lower().strip()
            label.append(self.label_names.index(name))

            # Subtract 1 to make pixel indexes 0-based
            bndbox_anno = obj.find('bndbox')            
            tmp_bbox = [int(bndbox_anno.find(tag).text) - 1 
                for tag in ('xmin', 'ymin', 'xmax', 'ymax')]
            tmp_bbox.append(int(self.label_names.index(name)))
            bbox.append(tmp_bbox)

        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.uint8)
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool

        # Load the image
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_image(img_file, color=True)

        ###<<< DEBUG
        # print('\nGet example in voc parser>>>')
        # print(id_)
        # print(img.shape)
        # # C H W -> H W C, RGB -> BGR, np.float32 -> np.uint8
        # img_ = img.transpose((1, 2, 0))
        # img_ = img_[...,::-1].copy()
        # img_ = img_.astype(np.uint8, copy=False)
        
        # cv.imwrite('ge_voc_parser.jpg', img_)
        ###>>>

        # if self.return_difficult:
            # return img, bbox, label, difficult

        return img, bbox, label, difficult


VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')
