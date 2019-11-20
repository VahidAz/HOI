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
from data.voc_dataset import Dataset, inverse_normalize
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

    for epoch in range(1):
        for step, (img, bbox, lbls, 
            im_info, num_bbox) in tqdm(enumerate(train_dataloader)):

            img = at.tonumpy(img[0])
            img = inverse_normalize(img).copy()

            bbox = at.tonumpy(bbox[0])
            for bb in bbox:
                cv.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 0, 255), 3)

            cv.imwrite('img_in_train_loop.jpg', img)

            pdb.set_trace()
            # pass


if __name__ == '__main__':
    import fire

    fire.Fire()
