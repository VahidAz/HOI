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

from configs.config import cfg
from data.dataset_imgnvid import Dataset
from models.rpn.rpnn import _RPNN

import torch.optim as optim
from models.libs.utils.net_utils import adjust_learning_rate, clip_gradient,  save_checkpoint

import time
import os

from models.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from models.libs.roi_align.modules.roi_align import RoIAlignAvg

from torch.autograd import Variable
from models.libs.utils.cdist import fast_cdist, my_cdist

from models.rpn.bbox_transform import clip_boxes, bbox_overlaps_batch, bbox_transform_batch
import torch.nn.functional as F
from models.libs.utils.net_utils import _smooth_l1_loss


def train(**kwargs):
    cfg._parse(kwargs)

    if torch.cuda.is_available() and not cfg.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda=True")

    # Backend model
    feat_extractor = None
    classifier = None

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

            classifier = nn.Sequential(*list(vgg16_model.classifier._modules.values())[:-1])

            print('\n VGGGGGGGGGGGG \n')
            pdb.set_trace()
            
            # Freeze top4 conv
            for layer in range(10):
                for p in feat_extractor[layer].parameters(): p.requires_grad = False

    if cfg.cuda:
        feat_extractor = feat_extractor.cuda()
        classifier = classifier.cuda()

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

    # print('\nHH : ', len(train_dataloader))
    # pdb.set_trace()
    # counter = 0

    RCNN_cls_score = nn.Linear(4096, 30).cuda()

    session = 1
    POOLING_SIZE = 7


    proposal_target_layer_here = _ProposalTargetLayer(30)
    roi_align_here= RoIAlignAvg(POOLING_SIZE, POOLING_SIZE, 1.0/16.0)


    for epoch in range(cfg.epoch):
        loss_temp = 0
        start = time.time()

        if epoch % (lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, lr_decay_gamma)
            lr *= lr_decay_gamma

        for step, (img, bbox, lbls, im_info, num_bbox) in enumerate(train_dataloader):
            
            img = img.cuda().float()
            bbox = bbox.cuda()
            lbls = lbls.cuda()
            num_bbox = num_bbox.cuda()
           
            im_info = im_info.view(-1, 3)
            im_info = im_info.cuda()

            print('\n\n CHECKKKKKKKKKK \n\n')
            pdb.set_trace()

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

            img = img.view(img.shape[1], img.shape[2], img.shape[3], img.shape[4])
            bbox = bbox.view(bbox.shape[1], bbox.shape[2], bbox.shape[3])
            lbls = lbls.view(lbls.shape[1], lbls.shape[2])
            # im_info = im_info.view()


            cur_roi, cur_rpn_loss_cls, cur_rpn_loss_box, base_feat = rpn_net.forward(img, im_info, bbox, num_bbox)

            print('\n QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ \n')
            pdb.set_trace()

            roi_data = proposal_target_layer_here(cur_roi, bbox, num_bbox)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))

            rois = Variable(rois)

            pooled_feat = roi_align_here(base_feat, rois.view(-1, 5))

            pool5_flat = pooled_feat.view(pooled_feat.size(0), -1)
            pooled_feat = classifier(pool5_flat)

            pooled_feat2 = pooled_feat.view(5, 128, pooled_feat.shape[1])
            # pooled_feat2 = pooled_feat.view(rois.shape[0], 
            #     rois.shape[1], pooled_feat.shape[1], 
            #     pooled_feat.shape[2], pooled_feat.shape[3])


            tmpp = torch.zeros([4, 128], dtype=torch.float32)
            tmpp1 = torch.zeros([4, 128], dtype=torch.int32)

            for ll in range(rois.shape[0] - 1):
                print('\n KHAR TULE INSIDE LOOP \n')
                print('Index: ', ll)

                first_feat = pooled_feat2[ll]
                sec_feat = pooled_feat2[ll+1]

                first_rois = rois[ll]
                sec_rois = rois[ll+1]

                first_rois = first_rois.view(1, first_rois.shape[0], first_rois.shape[1])
                sec_rois = sec_rois[:, :4]

                # pdb.set_trace()

                # print('\n KHAR TU:LE')
                cdist_res = fast_cdist(first_feat, sec_feat)
                overlaps_res = bbox_overlaps_batch(sec_rois, first_rois)
                overlaps_res = overlaps_res.view(overlaps_res.shape[1], overlaps_res.shape[2])


                mul_res = cdist_res * overlaps_res

                val, indices = torch.max(mul_res, 0)

                print('\n LL: ', ll)
                print(val)
                print(indices)

                print(cdist_res)
                print(overlaps_res)

                tmpp[ll][:] = val
                tmpp1[ll][:] = indices



                # pdb.set_trace()
                
                # cdist_res1 = my_cdist(first_feat, sec_feat)

                # ctmp  = cdist_res.argmin(0)

                # tmpp[ll][:] = ctmp


                # pdb.set_trace()

            # print('\n===============\n')
            # print('ii: ', ii)
            # print(cur_roi.shape)
            # print(cur_rpn_loss_cls)
            # print(cur_rpn_loss_box)

            # for ll in range(rois.shape[0] - 1):
            #     l

            final_feat = torch.zeros([pooled_feat2.shape[1], pooled_feat2.shape[2]], dtype=torch.float32)
            final_lbl = torch.zeros([pooled_feat2.shape[1], pooled_feat2.shape[2]], dtype=torch.float32)
            countt = 0

            for nnc in range(rois.shape[1]):
                f0 = tmpp[0][nnc]
                f0_ind = tmpp1[0][nnc]

                f1 = tmpp[1][nnc]
                f1_ind = tmpp1[1][nnc]

                f2 = tmpp[2][nnc]
                f2_ind = tmpp1[2][nnc]

                f3 = tmpp[3][nnc]
                f3_ind = tmpp1[3][nnc]

                # # tmp_feat = 0
                # if f0 > 0.2 and f1 > 0.2 and f2 > 0.2 and f3 > 0.2:
                tmp_feat = ((pooled_feat2[0][nnc] +
                    pooled_feat2[1][nnc] +
                    pooled_feat2[2][nnc] +
                    pooled_feat2[3][nnc] +
                    pooled_feat2[4][nnc]) / 5.0)

                final_feat[nnc] = tmp_feat
                final_lbl[nnc] = 0
                countt += 1


            final_feat = final_feat[:countt][:]
            final_feat = final_feat.cuda()


            # compute object classification probability
            cls_score = RCNN_cls_score(final_feat)
            cls_prob = F.softmax(cls_score, 1)

            print('\nANNNNNNNNN\n')
            pdb.set_trace()

            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label[:128])

            print('\nO)O)O)o0o0o0o0o0\n')
            pdb.set_trace()

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

            print('\nGGGGGGGG\n')
            pdb.set_trace()


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
