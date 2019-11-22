#!/usr/bin/env python3.7.5
# -*- coding: utf-8 -*-
"""
"""

import os
import time

import pdb
from tqdm import tqdm
import numpy as np
import torch
import cv2 as cv

from configs.config import cfg
from data.voc_dataset import Dataset, inverse_normalize
from models.libs.utils import array_tool as at

from models.rpn.rpn_vgg16 import RPN_VGG16
from models.libs.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient


def train(**kwargs):
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
    resume = False # Resume checkpoint or not
    checksession = 0 # Checksession to load model
    checkepoch = 1 # Checkepoch to load model
    checkpoint = 1 # Checkpoint to load model


    output_dir = (save_dir + "/" + cfg.backend_model + 
        "/" + cfg.voc_dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # Train dataset loader
    train_dataset = Dataset(cfg)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=1)
    print('Len Train Data: ', len(train_dataloader))
    iters_per_epoch = len(train_dataloader) / cfg.batch_size


    if cfg.backend_model == 'vgg16':
        rpn_vgg16 = RPN_VGG16(cfg)
    rpn_vgg16 = rpn_vgg16.cuda()
    rpn_vgg16.create_architecture()


    params = []
    for key, value in dict(rpn_vgg16.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params':[value],'lr':lr*(cfg.TRAIN_DOUBLE_BIAS + 1), \
                        'weight_decay': cfg.TRAIN_BIAS_DECAY and cfg.TRAIN_WEIGHT_DECAY or 0}]
        else:
            params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN_WEIGHT_DECAY}]


    if optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=TRAIN_MOMENTUM)


    if resume:
        load_name = os.path.join(output_dir,
          'rpn_vgg16_{}_{}_{}.pth'.format(
            checksession, checkepoch, checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        session = checkpoint['session']
        start_epoch = checkpoint['epoch']
        rpn_vgg16.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        print("loaded checkpoint %s" % (load_name))


    if cfg.use_tfboard:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter("logs")


    for epoch in range(cfg.epoch):
        # Setting to train mode
        rpn_vgg16.train()
        loss_temp = 0
        start = time.time()

        if epoch % (lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, lr_decay_gamma)
            lr *= lr_decay_gamma

        for step, (img, bbox, lbls, 
            im_info, num_bbox) in tqdm(enumerate(train_dataloader)):

            # CUDA
            img = img.cuda()
            bbox = bbox.cuda()
            lbls = lbls.cuda()
            im_info = im_info.view(-1, 3)
            im_info = im_info.cuda()
            num_bbox = num_bbox.cuda()


            # pdb.set_trace()

            rpn_vgg16.zero_grad()
            rois, rpn_loss_cls, rpn_loss_box, _ = rpn_vgg16(
                img, im_info, bbox, num_bbox)

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean()
            loss_temp += loss.item()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            if cfg.backend_model == "vgg16":
              clip_gradient(rpn_vgg16, 10.)
            optimizer.step()

            if step % disp_interval == 0:
                end = time.time()
                if step > 0:
                  loss_temp /= (disp_interval + 1)

                loss_rpn_cls = rpn_loss_cls.item()
                loss_rpn_box = rpn_loss_box.item()


            print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                    % (session, epoch, step, iters_per_epoch, loss_temp, lr))
            print("\t\t\trpn_cls: %.4f, rpn_box: %.4f" % (loss_rpn_cls, loss_rpn_box))

            if cfg.use_tfboard:
              info = {
                'loss': loss_temp,
                'loss_rpn_cls': loss_rpn_cls,
                'loss_rpn_box': loss_rpn_box
              }
              logger.add_scalars("logs_s_{}/losses".format(session), info, (epoch - 1) * iters_per_epoch + step)

            loss_temp = 0
            start = time.time()

    save_name = os.path.join(output_dir, 'rpn_vgg16_{}_{}_{}.pth'.format(session, epoch, step))
    save_checkpoint({
      'session': session,
      'epoch': epoch + 1,
      'model': rpn_vgg16.state_dict(),
      'optimizer': optimizer.state_dict(),
    }, save_name)
    print('save model: {}'.format(save_name))

    if cfg.use_tfboard:
        logger.close()


if __name__ == '__main__':
    import fire

    fire.Fire()
