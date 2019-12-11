#!/usr/bin/env python3.7.5
# -*- coding: utf-8 -*-
"""
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import pdb
from tqdm import tqdm
import numpy as np
import torch
import cv2 as cv

from configs.config import cfg
from data.voc_dataset import TestDataset, inverse_normalize
from models.libs.utils import array_tool as at

from models.rpn.rpn_vgg16 import RPN_VGG16
from models.libs.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient


torch.backends.cudnn.benchmark = True


def test(**kwargs):
    cfg._parse(kwargs)

    if torch.cuda.is_available() and not cfg.cuda:
        print('WARNING: You have a CUDA device, so you \
                should probably run with --cuda=True')
        print('ERROR: CUDA is the only supported device in this version.')
        exit(0)

    np.random.seed(cfg.RNG_SEED)


    # TODO:
    # Adding number of maximum boxes
    # Evaluate having sampler
    # Flipped flag is hardcoded
    # Multiple GPUs


    # Parameters
    start_epoch = 1
    disp_interval = 100
    checkpoint_interval = 10000
    save_dir = './checkpoints/'
    optimizer = 'sgd'
    lr = 0.001
    TRAIN_MOMENTUM = 0.9
    lr_decay_step = 5
    lr_decay_gamma = 0.1
    session = 1
    resume = True # Resume checkpoint or not
    checksession = 1 # Checksession to load model
    checkepoch = 15 # Checkepoch to load model
    checkpoint = 11539 # Checkpoint to load model


    output_dir = (save_dir + "/" + cfg.backend_model + 
        "_RPN_" + cfg.voc_dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # Train dataset loader
    test_dataset = TestDataset(cfg)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=1)
    print('Len Train Data: ', len(test_dataloader))
    iters_per_epoch = len(test_dataloader) / cfg.batch_size


    if cfg.backend_model == 'vgg16':
        rpn_vgg16 = RPN_VGG16(cfg)
        rpn_vgg16 = rpn_vgg16.cuda()
    rpn_vgg16.create_architecture()


    load_name = os.path.join(output_dir,
      'rpn_vgg16_{}_{}_{}.pth'.format(
        checksession, checkepoch, checkpoint))
    print("loading checkpoint %s" % (load_name))

    checkpoint = torch.load(load_name)
    rpn_vgg16.load_state_dict(checkpoint['model'])
    print("loaded checkpoint %s" % (load_name))


    rpn_vgg16.eval()

    for step, (img, bbox, lbls, 
        im_info, num_bbox) in tqdm(enumerate(test_dataloader)):

        # CUDA
        img = img.cuda()
        bbox = bbox.cuda()
        lbls = lbls.cuda()
        im_info = im_info.view(-1, 3).cuda()
        num_bbox = num_bbox.cuda()

        rois, rpn_loss_cls, rpn_loss_box, _ = rpn_vgg16(
            img, im_info, bbox, num_bbox)

        img = at.tonumpy(img[0])
        img = inverse_normalize(img).copy()

        bbox = at.tonumpy(rois[0])
        for ind, bb in enumerate(bbox):
            print(ind, '\t', bb)
            bb = bb[1:]
            img_ = img.copy()
            cv.rectangle(img_, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 0, 255), 3)

            cv.imwrite('img_in_eval_loop.jpg', img_)

            pdb.set_trace()


if __name__ == '__main__':
    import fire

    fire.Fire()
