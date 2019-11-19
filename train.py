#!/usr/bin/env python3.7.5
# -*- coding: utf-8 -*-
"""
VA
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
from tqdm import tqdm

import torch
from torch import nn

from torchvision.models import vgg16

from cfg.config import cfg
from data.dataset import Dataset
from model.rpn.rpnn import _RPNN

import torch.optim as optim
from model.utils.net_utils import adjust_learning_rate, clip_gradient,  save_checkpoint

import time
import os


def train(**kwargs):
    cfg._parse(kwargs)

    if torch.cuda.is_available() and not cfg.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda=True")

    # Backend model
    feat_extractor = None

    if cfg.backend_model == 'vgg16':
        vgg16_model = vgg16()

        if cfg.pretrained_flag:
            print("Loading pretrained weights from %s" %(cfg.pretrained_weight_path))

            # Map location should be like parameter
            state_dict = torch.load(cfg.pretrained_weight_path,
                                    map_location='cuda')
            vgg16_model.load_state_dict({k:v for k,v in state_dict.items() if k in vgg16_model.state_dict()})

            # not using the last maxpool layer
            feat_extractor = nn.Sequential(*list(vgg16_model.features._modules.values())[:-1])

            # pdb.set_trace()
            
            # Freeze top4 conv
            for layer in range(10):
                for p in feat_extractor[layer].parameters(): p.requires_grad = False

    if cfg.cuda:
        feat_extractor = feat_extractor.cuda()

    # Train dataset loader
    train_dataset = Dataset(cfg)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    
    # RPN
    rpn_net = _RPNN(feat_extractor=feat_extractor)

    print(cfg.WEIGHT_DECAY)
    params = []
    lr = 0.001
    momentum = 0.9
    for key, value in dict(rpn_net.named_parameters()).items():
        # print('key: ', key)
        # print('value: ', value)
        # pdb.set_trace()
        # continue

        if value.requires_grad:
          if 'bias' in key:
            params += [{'params':[value],'lr':lr*(cfg.DOUBLE_BIAS + 1), \
                    'weight_decay': cfg.BIAS_DECAY and cfg.WEIGHT_DECAY or 0}]
          else:
            params += [{'params':[value],'lr':lr, 'weight_decay': cfg.WEIGHT_DECAY}]


    optimizer = torch.optim.SGD(params, momentum)

    if cfg.use_tfboard:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter("logs")

    # exit(0)
    lr_decay_gamma = 0.1
    lr_decay_step = 5
    
    disp_interval = 10
    batch_size = 1.0
    iters_per_epoch = len(train_dataloader) /batch_size

    print('\nHH : ', len(train_dataloader))
    pdb.set_trace()
    # counter = 0

    session = 1

    for epoch in range(cfg.epoch):
        loss_temp = 0
        start = time.time()

        if epoch % (lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, lr_decay_gamma)
            lr *= lr_decay_gamma

        for step, (img, bbox, lbls, im_info, num_bbox, idx, id_) in enumerate(train_dataloader):
            
            img = img.cuda().float()
            bbox = bbox.cuda()
            lbls = lbls.cuda()
            num_bbox = num_bbox.cuda()
           
            im_info = im_info.view(-1, 3)
            im_info = im_info.cuda()

            # print('\nImage Info')
            # print("IDX: ", idx)
            # print('id_: ', id_)

            # cur_feat = feat_extractor(img)

            # print(ii)
            # pdb.set_trace()


            # print('------------------------------------')
            # print('\n Before FORWARD \n')
            # print('Step: ', step + 1)
            # print("IDX: ", idx)
            # print('id_: ', id_)
            # print('bbox: ', bbox)
            # print('lbl: ', lbls)
            # print('im_info: ', im_info)
            # print('num_bbox: ', num_bbox)
            # print('im shape: ', img.shape)




            rpn_net.zero_grad()

            cur_roi, cur_rpn_loss_cls, cur_rpn_loss_box = rpn_net.forward(img, im_info, bbox, num_bbox)

            # print('\n===============\n')
            # print('ii: ', ii)
            # print(cur_roi.shape)
            # print(cur_rpn_loss_cls)
            # print(cur_rpn_loss_box)

            # print('\nANNNNNNNNN\n')
            # pdb.set_trace()

            loss = cur_rpn_loss_cls.mean() + cur_rpn_loss_box.mean()
            loss_temp += loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            if cfg.backend_model == "vgg16":
              clip_gradient(rpn_net, 10.)
            optimizer.step()

            if (step) % disp_interval == 0:
                end = time.time()
                if (step) > 0:
                  loss_temp /= (disp_interval + 1)

                # if args.mGPUs:
                # if False:
                    # pass
                    # loss_rpn_cls = cur_rpn_loss_cls.mean().item()
                    # loss_rpn_box = cur_rpn_loss_box.mean().item()
                    # loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    # loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    # fg_cnt = torch.sum(rois_label.data.ne(0))
                    #  bg_cnt = rois_label.data.numel() - fg_cnt
                # else:
                cur_loss_rpn_cls = cur_rpn_loss_cls.item()
                cur_loss_rpn_box = cur_rpn_loss_box.item()
                    # loss_rcnn_cls = RCNN_loss_cls.item()
                    # loss_rcnn_box = RCNN_loss_bbox.item()
                    # fg_cnt = torch.sum(rois_label.data.ne(0))
                    # bg_cnt = rois_label.data.numel() - fg_cnt

                # print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                        # % (session, epoch, step, iters_per_epoch, loss_temp, lr))
                # print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
                # print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                #               % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
                print("\t\t\trpn_cls: %.4f, rpn_box: %.4f" \
                              % (cur_loss_rpn_cls, cur_loss_rpn_box))
                if cfg.use_tfboard:
                    info = {
                    'loss': loss_temp,
                    'loss_rpn_cls': cur_loss_rpn_cls,
                    'loss_rpn_box': cur_loss_rpn_box
                    # 'loss_rcnn_cls': loss_rcnn_cls,
                    # 'loss_rcnn_box': loss_rcnn_box
                  }
                    logger.add_scalars("logs_s_{}/losses".format(session), info, (epoch - 1) * iters_per_epoch + step)
                    # for tag, value in info.items():
                        # logger.add_scalars(tag, value, step+1)

                loss_temp = 0
                start = time.time()

            # counter = counter + 1


        save_name = os.path.join('./chk/', 'faster_rcnn_{}_{}_{}.pth'.format(session, epoch, step))
        save_checkpoint({
          'session': session,
          'epoch': epoch + 1,
          'model': rpn_net.state_dict(),
          'optimizer': optimizer.state_dict(),
          'pooling_mode': 'gg',
          'class_agnostic': 'bb',
        }, save_name)
        print('save model: {}'.format(save_name))

        if cfg.use_tfboard:
            logger.close()


if __name__ == '__main__':
    import fire

    fire.Fire()
