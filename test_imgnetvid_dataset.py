#!/usr/bin/env python3.7.5
# -*- coding: utf-8 -*-
"""
"""


import pdb
from tqdm import tqdm
import numpy as np
import torch
import cv2 as cv

from configs.config import cfg
from data.imagenet_vid_dataset import Dataset, inverse_normalize
from models.libs.utils import array_tool as at 


def train(**kwargs):
    cfg._parse(kwargs)

    if torch.cuda.is_available() and not cfg.cuda:
        print('WARNING: You have a CUDA device, so you \
                should probably run with --cuda=True')
        print('ERROR: CUDA is the only supported device in this version.')
        exit(0)

    # Train dataset loader
    train_dataset = Dataset(cfg)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=1)
    
    print('Len Train Data: ', len(train_dataloader))
    print('Num classes: ', train_dataset.num_classes())

    for epoch in range(1):
        for step, (img, bbox, lbls, 
            im_info, num_bbox, corrupted) in tqdm(enumerate(train_dataloader)):

            if corrupted[0] > 0:
                print('\n\n CORRUPTED \n\n')
                continue

            print('\n\n IN MAIN LOOP >>>')
            print(img.shape)
            print(bbox.shape)
            print(lbls.shape)
            print(im_info.shape)
            print(num_bbox.shape)
            print(corrupted.shape)

            print(bbox[0])
            print(lbls[0])
            print(im_info[0])
            print(num_bbox[0])
            print(corrupted[0])

            for idx, (img_, bbox_) in enumerate(zip(img[0], bbox[0])):
                img_ = at.tonumpy(img_)
                img_ = inverse_normalize(img_).copy()

                bbox_ = at.tonumpy(bbox_)
                for bb in bbox_:
                    cv.rectangle(img_, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 0, 255), 3)

                cv.imwrite('img_in_train_loop_' + str(idx) + '.jpg', img_)
            # exit(0)

            # pdb.set_trace()
            # pass


if __name__ == '__main__':
    import fire

    fire.Fire()
