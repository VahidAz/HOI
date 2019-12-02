import numpy as np
import pdb
import cv2 as cv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

from data.imagenet_vid_dataset import inverse_normalize
from models.libs.utils import array_tool as at
from models.libs.utils.cdist import cdist_v2
from models.rpn.bbox_transform import clip_boxes, bbox_overlaps_batch, bbox_transform_batch


def make_tube_1(pooled_feat, rois_label, rois, cfg, im_data):
    pooled_feat_reshaped = pooled_feat.view(rois.shape[0], rois.shape[1], pooled_feat.shape[1])

    rois_label_reshaped = rois_label.view(rois.shape[0], rois.shape[1])


    # Making tube
    # TODO: Replace it with a function
    # TODO: Try LSTM data associatation
    # TODO: Try to implement it better

    final_val_all_frames = torch.zeros([rois.shape[0] - 1, rois.shape[1]], dtype=torch.float32).cuda()
    final_idx_all_frames = torch.zeros([rois.shape[0] - 1, rois.shape[1]], dtype=torch.int32).cuda()

    # final_val_all_frames_cdist = torch.zeros([rois.shape[0] - 1, rois.shape[1]], dtype=torch.float32).cuda()
    # final_idx_all_frames_cdist = torch.zeros([rois.shape[0] - 1, rois.shape[1]], dtype=torch.int32).cuda()

    # final_val_all_frames_ov = torch.zeros([rois.shape[0] - 1, rois.shape[1]], dtype=torch.float32).cuda()
    # final_idx_all_frames_ov = torch.zeros([rois.shape[0] - 1, rois.shape[1]], dtype=torch.int32).cuda()


    for ii in range(cfg.time_win - 1):
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

        # Clipping by threshold
        overlaps_res_reshaped[overlaps_res_reshaped < 0.7] = 0
        dist_res_reverse[dist_res_reverse < 0.5] = 0

        # Final matrix
        final_metric = overlaps_res_reshaped + dist_res_reverse
        final_val, final_idx = torch.max(final_metric, 0)

        final_val_all_frames[ii, :] = final_val
        final_idx_all_frames[ii, :] = final_idx

        # final_val_ov, final_idx_ov = torch.max(overlaps_res_reshaped, 0)
        # final_val_all_frames_ov[ii, :] = final_val_ov
        # final_idx_all_frames_ov[ii, :] = final_idx_ov

        # final_val_cdist, final_idx_cdist = torch.max(dist_res_reverse, 0)
        # final_val_all_frames_cdist[ii, :] = final_val_cdist
        # final_idx_all_frames_cdist[ii, :] = final_idx_cdist


    print('\nAfter finding max\n')
    pdb.set_trace()


    # Making pooled_feat for tube
    # TODO: Write a vectorized version or a better one
    # TODO: Current aggregation is sum, we need to try other aggrigation methods
    # We cannot use label for matching since we don't have at test time
    tube_pooled_feat = torch.zeros(pooled_feat.shape[0], pooled_feat.shape[1], dtype=torch.float32).cuda()
    tube_rois_label = torch.zeros(pooled_feat.shape[0], dtype=torch.float32).cuda()

    count_tube = 0
    tube_threshold = 1.2

    for ii in range(rois.shape[1]):
        tmp_feat = torch.zeros(pooled_feat.shape[1], dtype=torch.float32).cuda()
        tmp_feat += pooled_feat_reshaped[0][ii]
        tmp_flag = True
        ref_lbl = rois_label_reshaped[0][ii]
        sum_lbl = 0
        sum_lbl += ref_lbl

        tmp_idx_list = []
        tmp_idx_list.append(ii)
        tmp_val_list = []

        for jj in range(rois.shape[0] - 1):
            if (final_val_all_frames[jj][ii] >= tube_threshold):
                tmp_feat += pooled_feat_reshaped[jj+1][final_idx_all_frames[jj][ii]]
                sum_lbl += rois_label_reshaped[jj+1][ii]
                tmp_idx_list.append(final_idx_all_frames[jj][ii])
                tmp_val_list.append(final_val_all_frames[jj][ii])
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

            # Debugging
            print(tmp_val_list)
            for count, idx in enumerate(tmp_idx_list):
                cur_img = im_data[count]
                bb = rois[count][idx][1:]

                img = at.tonumpy(cur_img)
                img = inverse_normalize(img).copy()

                cv.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 0, 255), 3)
                cv.imwrite('img_in_train_loop_tube_' + str(ii) + '_' + str(count) + '.jpg', img)

            print('\n FLAG IS TREU \n')
            pdb.set_trace()


    # TODO: Check it causes problem to give many zeros!
    # Uncommenting this causes problem because of batch size

    # tube_pooled_feat = tube_pooled_feat[:count_tube]
    # tube_rois_label = tube_rois_label[:count_tube]


    print('\n After making tube \n')
    pdb.set_trace()


    return tube_pooled_feat, tube_rois_label
