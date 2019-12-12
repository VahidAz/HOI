import random
import time

import numpy as np
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

from models.rpn.rpn import _RPN
from models.libs.roi_pooling.modules.roi_pool import _RoIPooling
from models.libs.roi_crop.modules.roi_crop import _RoICrop
from models.libs.roi_align.modules.roi_align import RoIAlignAvg
from models.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from models.libs.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
from models.vid_obj_det.tube_maker import *


class _VIDOBJDET(nn.Module):
    def __init__(self, _cfg,  _n_classes, _class_agnostic):
        super(_VIDOBJDET, self).__init__()

        self.cfg = _cfg
        # self.classes = _classes
        # self.n_classes = len(_classes)
        self.n_classes = _n_classes
        self.class_agnostic = _class_agnostic
        self.dout_base_model = 512

        # Loss
        self._VIDOBJDET_loss_cls = 0
        self._VIDOBJDET_loss_bbox = 0

        # define rpn
        self._VIDOBJDET_rpn = _RPN(din=self.dout_base_model)
        self._VIDOBJDET_proposal_target = _ProposalTargetLayer(self.n_classes)
        self._VIDOBJDET_roi_pool = _RoIPooling(self.cfg.POOLING_SIZE, self.cfg.POOLING_SIZE, 1.0/16.0)
        self._VIDOBJDET_roi_align = RoIAlignAvg(self.cfg.POOLING_SIZE, self.cfg.POOLING_SIZE, 1.0/16.0)

        self.grid_size = self.cfg.POOLING_SIZE * 2 if self.cfg.CROP_RESIZE_WITH_MAX_POOL else self.cfg.POOLING_SIZE
        self._VIDOBJDET_roi_crop = _RoICrop()

        # self.tube_threshold = 1.0


    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # Feed image data to base model to obtain base feature map
        base_feat = self._VIDOBJDET_base(im_data)

        # Feed base feature map to RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self._VIDOBJDET_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # If it is training phase, then use ground truth bboxes for refining
        if self.training:
            roi_data = self._VIDOBJDET_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if self.cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self._VIDOBJDET_roi_crop(base_feat, Variable(grid_yx).detach())
            if self.cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif self.cfg.POOLING_MODE == 'align':
            pooled_feat = self._VIDOBJDET_roi_align(base_feat, rois.view(-1, 5))
        elif self.cfg.POOLING_MODE == 'pool':
            pooled_feat = self._VIDOBJDET_roi_pool(base_feat, rois.view(-1,5))

        # Feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)


        # Making Tube
        if self.training: # Tube in training mode
            start_time = time.time()
            tube_pooled_feat, tube_rois_label = make_tube_ff_ov_feat(pooled_feat, rois_label, rois, self.cfg, im_data)
            torch.cuda.synchronize()
            print('make_tube_bf_ov time: {time.time() - start_time:.2f}s')

            exit(0)
        else: # Tube in eval mode
            # Making tube in eval mode is different and we have only rois and features
            pass


        # Compute bbox offset
        bbox_pred = self._VIDOBJDET_bbox_pred(tube_pooled_feat)

        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self._VIDOBJDET_cls_score(tube_pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        _VIDOBJDET_loss_cls = 0
        _VIDOBJDET_loss_bbox = 0

        if self.training:
            # classification loss
            _VIDOBJDET_loss_cls = F.cross_entropy(cls_score, tube_rois_label.long())

            # bounding box regression L1 loss
            _VIDOBJDET_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        # TODO: doreally need this!

        # cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        # bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)


        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, _VIDOBJDET_loss_cls, _VIDOBJDET_loss_bbox, rois_label


    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self._VIDOBJDET_rpn.RPN_Conv, 0, 0.01, self.cfg.TRAIN_TRUNCATED)
        normal_init(self._VIDOBJDET_rpn.RPN_cls_score, 0, 0.01, self.cfg.TRAIN_TRUNCATED)
        normal_init(self._VIDOBJDET_rpn.RPN_bbox_pred, 0, 0.01, self.cfg.TRAIN_TRUNCATED)
        normal_init(self._VIDOBJDET_cls_score, 0, 0.01, self.cfg.TRAIN_TRUNCATED)
        normal_init(self._VIDOBJDET_bbox_pred, 0, 0.001, self.cfg.TRAIN_TRUNCATED)


    def create_architecture(self):
        self._init_modules()
        self._init_weights()
