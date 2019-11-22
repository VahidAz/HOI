import os

import xml.etree.ElementTree as ET
import fnmatch
import numpy as np
from pprint import pprint
import pdb
import cv2 as cv

from data.util import read_image


class IMGNETVIDPARSER:
    def __init__(self, _data_dir, _split='train', 
        _class_num=30, _time_win= 5, _use_occluded=True):
        self.data_dir = _data_dir
        self.split = _split
        self.class_num = _class_num
        self.tw = _time_win
        
        # Please note that some of images only
        # have one bounding box which is occluded
        # it is handled in this version, we ignore
        # the set without bounding box
        self.use_occluded = _use_occluded

        self.ids = []
        self.label_names = IMGNET_VID_BBOX_LABEL_NAMES

        # For debugging
        total_ids_num = 0

        for ii in range(self.class_num):
            ii += 1
            print('ii: ', ii)
            
            id_file = os.path.join(
                self.data_dir, 'ImageSets/VID/', self.split + 
                '_' + str(ii) + '.txt')
            print('id_file: ', id_file)

            cur_ids = [id_.strip()[:-2] for id_ in open(id_file)]
            print('cur_ids: ', cur_ids)

            cur_lbl = ii
            print('lbl: ', cur_lbl)

            for name in cur_ids:
                imgs_path = os.path.join(
                    self.data_dir, 'Data/VID/' + self.split + '/' + name)
                num_imgs = len(fnmatch.filter(
                    os.listdir(imgs_path), '*.JPEG'))
                print('Name: ', name, '\tNum_imgs: ', num_imgs)

                # The number of valid sets, ignoring teh remaining ones
                num_usable = int(num_imgs/self.tw) * self.tw
                print('Num_usuable: ', num_usable)

                total_ids_num += int(num_imgs/self.tw)

                for jj in range(num_usable):
                    print('jj: ', jj)

                    if jj % self.tw == 0:
                        if jj != 0:
                            self.ids.append(cur_id)

                        cur_id = []
                        cur_id.append(imgs_path + '/' + format(jj, '06'))
                    else:
                        cur_id.append(imgs_path + '/' + format(jj, '06'))

                self.ids.append(cur_id) # Last set

                print('Len all ids: ', len(self.ids))
                print('Total ids num: ', total_ids_num)


        if len(self.ids) != total_ids_num:
            print('ID len is not correct!')
            exit(0)


    def __len__(self):
        return len(self.ids)


    def num_classes(self):
        return len(self.label_names)


    def get_example(self, _idx):
        id_ = self.ids[_idx]

        img_list = list()
        bbox_list = list()
        label_list =list()
        occluded_list = list()
        # Flag, True if a set is not proper for training
        corrupted = 0

        # Each id has some images(tw)
        for name in id_:
            file_name = name + '.JPEG'
            print('Img path: ', file_name)
            img = read_image(file_name, color=True)

            anno_path = name.replace('Data', 'Annotations') + '.xml'
            print('Annot path: ', anno_path)
            anno = ET.parse(anno_path)

            # Current name info
            bbox_tmp = list()
            label_tmp = list()
            occluded_tmp = list()
            
            for obj in anno.findall('object'):
                # When in not using occluded split, and the object is
                # occluded, skipt it.
                if not self.use_occluded and int(
                    obj.find('occluded').text) == 1:
                    continue

                occluded_tmp.append(int(obj.find('occluded').text))

                name = obj.find('name').text.lower().strip()
                label_tmp.append(self.label_names.index(name))

                # Imagenet is zero based
                bndbox_anno = obj.find('bndbox')
                tmp_bbox = [int(bndbox_anno.find(tag).text)
                    for tag in ('xmin', 'ymin', 'xmax', 'ymax')]
                tmp_bbox.append(int(self.label_names.index(name)))
                bbox_tmp.append(tmp_bbox)

            # Current id info
            img_list.append(img)
            bbox_list.append(bbox_tmp)
            label_list.append(label_tmp)
            occluded_list.append(occluded_tmp)


        try:
            bbox_np = np.stack(bbox_list).astype(np.float32)
        except:
            min_len = len(min(bbox_list, key=len))
            print('min_len: ', min_len)

            bbox_list_tmp = list()
            label_list_tmp = list()
            occluded_list_tmp = list()

            for cur_bbox, cur_lbl, cur_occl in zip(bbox_list, label_list, occluded_list):
                bbox_list_tmp.append(cur_bbox[:min_len])
                label_list_tmp.append(cur_lbl[:min_len])
                occluded_list_tmp.append(cur_occl[:min_len])

            bbox_list = bbox_list_tmp
            label_list = label_list_tmp
            occluded_list = occluded_list_tmp


        if len(min(bbox_list, key=len)) == 0:
            print('WARNING, this id is corrupted!')
            corrupted = 1 # True

        img_np = np.stack(img_list).astype(np.float32)
        bbox_np = np.stack(bbox_list).astype(np.float32)
        label_np = np.stack(label_list).astype(np.uint8)
        occluded_np = np.array(occluded_list, dtype=np.bool).astype(np.uint8)

        ###<<< DEBUG
        # for ind, img_ in enumerate(img_list):
        #     print('\nGet example in imgnetvid parser>>>')
        #     print(id_)
        #     print(img_.shape)
        #     # C H W -> H W C, RGB -> BGR, np.float32 -> np.uint8
        #     img_ = img_.transpose((1, 2, 0))
        #     img_ = img_[...,::-1].copy()
        #     img_ = img_.astype(np.uint8, copy=False)
            
        #     cv.imwrite('ge_voc_parser_' + str(ind) + '.jpg', img_)
        ###>>>

        return img_np, bbox_np, label_np, occluded_np, corrupted


IMGNET_VID_BBOX_LABEL_NAMES_REAL = (
    'airplane',
    'antelope',
    'bear',
    'bicycle',
    'bird',
    'bus',
    'car',
    'cattle',
    'dog',
    'domestic_cat',
    'elephant',
    'fox',
    'giant_panda',
    'hamster',
    'horse',
    'lion',
    'lizard',
    'monkey',
    'motorcycle',
    'rabbit',
    'red_panda',
    'sheep',
    'snake',
    'squirrel',
    'tiger',
    'train',
    'turtle',
    'watercraft',
    'whale',
    'zebra'
)


IMGNET_VID_BBOX_LABEL_NAMES = (
    'n02691156',
    'n02419796',
    'n02131653',
    'n02834778',
    'n01503061',
    'n02924116',
    'n02958343',
    'n02402425',
    'n02084071',
    'n02121808',
    'n02503517',
    'n02118333',
    'n02510455',
    'n02342885',
    'n02374451',
    'n02129165',
    'n01674464',
    'n02484322',
    'n03790512',
    'n02324045',
    'n02509815',
    'n02411705',
    'n01726692',
    'n02355227',
    'n02129604',
    'n04468005',
    'n01662784',
    'n04530566',
    'n02062744',
    'n02391049'
)


# if __name__ == '__main__':
#    test_obj = IMGNETVIDPARSER('./datasets/ILSVRC2015/')
#    examp = test_obj.get_example(0)
#    pdb.set_trace()
