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
from models.libs.utils.cdist import cdist_v2
from models.rpn.bbox_transform import clip_boxes, bbox_overlaps_batch, bbox_transform_batch


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

        self.tube_threshold = 1.0


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
        pooled_feat_reshaped = pooled_feat.view(rois.shape[0], rois.shape[1], pooled_feat.shape[1])

        rois_label_reshaped = rois_label.view(rois.shape[0], rois.shape[1])


        # Making tube
        # TODO: Replace it with a function
        # TODO: Try LSTM data associatation
        # TODO: Try to implement it better
        final_val_all_frames = torch.zeros([rois.shape[0] - 1, rois.shape[1]], dtype=torch.float32).cuda()
        final_idx_all_frames = torch.zeros([rois.shape[0] - 1, rois.shape[1]], dtype=torch.int32).cuda()

        for ii in range(self.cfg.time_win - 1):
            # First value in rois is batch number, bbox is [1:]
            first_feat = pooled_feat_reshaped[ii]
            first_rois = (rois[ii])[:, 1:]
            first_rois_reshaped = first_rois.view(1, first_rois.shape[0], first_rois.shape[1])

            sec_feat = pooled_feat_reshaped[ii+1]
            sec_rois = (rois[ii+1])[:, 1:]
            sec_rois_reshaped = sec_rois.view(1, sec_rois.shape[0], sec_rois.shape[1])

            # Dist_res, 0 max, 1 min
            # Dist_res_reverse, 0 min, 1 max
            # Overlap, 0 min, 1 max
            dist_res, dist_res_reverse = cdist_v2(first_feat, sec_feat)
            overlaps_res = bbox_overlaps_batch(first_rois, sec_rois_reshaped)
            overlaps_res_reshaped = overlaps_res.squeeze()

            # Final matrix
            final_metric = overlaps_res_reshaped + dist_res_reverse
            final_val, final_idx = torch.max(final_metric, 0)

            final_val_all_frames[ii, :] = final_val
            final_idx_all_frames[ii, :] = final_idx


        # Making pooled_feat for tube
        # TODO: Write a vectorized version or a better one
        # TODO: Current aggregation is sum, we need to try other aggrigation methods
        # We cannot use label for matching since we don't have at test time
        tube_pooled_feat = torch.zeros(pooled_feat.shape[0], pooled_feat.shape[1], dtype=torch.float32).cuda()
        tube_rois_label = torch.zeros(pooled_feat.shape[0], dtype=torch.float32).cuda()

        count_tube = 0
        for ii in range(rois.shape[1]):
            tmp_feat = torch.zeros(pooled_feat.shape[1], dtype=torch.float32).cuda()
            tmp_feat += pooled_feat_reshaped[0][ii]
            tmp_flag = True
            ref_lbl = rois_label_reshaped[0][ii]
            sum_lbl = 0
            sum_lbl += ref_lbl
            for jj in range(rois.shape[0] - 1):
                if (final_val_all_frames[jj][ii] >= self.tube_threshold):
                    tmp_feat += pooled_feat_reshaped[jj+1][final_idx_all_frames[jj][ii]]
                    sum_lbl += rois_label_reshaped[jj+1][ii]
                else:
                    tmp_flag = False
                    break

            if tmp_flag:
                tube_pooled_feat[count_tube][:] = tmp_feat
                if int(sum_lbl/rois.shape[0]) == ref_lbl:
                    tube_rois_label[count_tube] = ref_lbl
                else:
                    tube_rois_label[count_tube] = 0
                count_tube += 1

        # TODO: Check it causes problem to give many zeros!
        # Uncommenting this causes problem because of batch size

        # tube_pooled_feat = tube_pooled_feat[:count_tube]
        # tube_rois_label = tube_rois_label[:count_tube]


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
