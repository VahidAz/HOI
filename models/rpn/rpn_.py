from __future__ import absolute_import

import numpy as np
import math
import pdb
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from configs.config import cfg
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from models.libs.utils.net_utils import _smooth_l1_loss


class RPN_(nn.Module):
    def __init__(self, _cfg, _din=512):
        super(RPN_, self).__init__()
        
        self.cfg = _cfg
        self.din = _din  # Get depth of input feature map, e.g., 512
        self.anchor_scales = self.cfg.anchor_scales
        self.anchor_ratios = self.cfg.anchor_scales
        self.feat_stride = 16

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # Define the convrelu layers processing input feature map
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True).cuda()

        # Define bg/fg classifcation score layer
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2 # 2(bg/fg) * 9 (anchors)
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0).cuda()

        # Define anchor box offset prediction layer
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4 # 4(coords) * 9 (anchors)
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0).cuda()

        # Define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # Define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)


    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x


    # Def forward(self, base_feat, im_info, gt_boxes, num_boxes):
    def forward(self, img, im_info, gt_boxes, num_boxes):

        batch_size = img.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # Feed image data to base model to obtain base feature map
        base_feat = self.RPN__base(img)

        # Return feature map after convrelu layer
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)
        # Get rpn classification score
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        # get rpn offsets to the anchor boxes
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)

        # proposal layer
        cfg_key = 'TRAIN' #if self.training else 'TEST'

        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                 im_info, cfg_key))

        # Generating training labels and build the rpn loss
        # if self.training:
        if True:
            assert gt_boxes is not None

            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))

            # Compute classification loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)

            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0,
                                               rpn_keep.cuda())
            rpn_cls_score = rpn_cls_score.cuda()
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label.cuda())
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            # Compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                            rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])

        return rois, self.rpn_loss_cls, self.rpn_loss_box, base_feat


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

        normal_init(self.RPN_Conv, 0, 0.01, self.cfg.TRAIN_TRUNCATED)
        normal_init(self.RPN_cls_score, 0, 0.01, self.cfg.TRAIN_TRUNCATED)
        normal_init(self.RPN_bbox_pred, 0, 0.01, self.cfg.TRAIN_TRUNCATED)


    def create_architecture(self):
        self._init_modules()
        self._init_weights()
