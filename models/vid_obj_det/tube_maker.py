import numpy as np
import pdb
import cv2 as cv

import torch

from data.imagenet_vid_dataset import inverse_normalize
from models.libs.utils import array_tool as at
from models.libs.utils.cdist import cdist_v2
from models.rpn.bbox_transform import bbox_overlaps_batch, bbox_overlaps, bbox_iou


# Debug flag
DEBUG = False

if DEBUG:
    torch.set_printoptions(profile="full")


#@torch.jit.script
def make_tube_ff_ov_feat(pooled_feat, rois_label, rois, cfg, im_data, rois_target, rois_inside_ws, rois_outside_ws):
    '''
    this function makes tube by integrating overlap and cdist
    the tracking happens frame by frame
    Now only overlap
    '''

    # TODO: Try LSTM data associatation
    # TODO: Try to implement it better
    # TODO: Write a vectorized version or a better one


    batch_size = rois.shape[0]
    num_rois = rois.shape[1]
    feat_dim = pooled_feat.shape[1]

    pooled_feat_view = pooled_feat.view(batch_size, num_rois, feat_dim)
    rois_label_view = rois_label.view(batch_size, num_rois)


    ov_max_val = torch.zeros([cfg.time_win - 1, num_rois], dtype=torch.float32).cuda()
    ov_max_idx = torch.zeros([cfg.time_win - 1, num_rois], dtype=torch.int32).cuda()

    # dist_max_val = torch.zeros([cfg.time_win - 1, num_rois], dtype=torch.float32).cuda()
    # dist_max_idx = torch.zeros([cfg.time_win - 1, num_rois], dtype=torch.int32).cuda()


    for ii in range(cfg.time_win - 1):
        # First value in rois is batch number, bbox is [1:]
        # first_feat = pooled_feat_view[ii]
        first_rois = (rois[ii])[:, 1:]

        # sec_feat = pooled_feat_view[ii+1]
        sec_rois = (rois[ii+1])[:, 1:]

        # Dist_res, 0 max, 1 min
        # Dist_res_reverse, 0 min, 1 max
        # Overlap, 0 min, 1 max
        # dist_res, dist_res_reverse = cdist_v2(first_feat, sec_feat)
        overlaps_res = bbox_overlaps(first_rois, sec_rois)

        # dist_max_val[ii, :], dist_max_idx[ii, :] = torch.max(dist_res_reverse, 1)
        ov_max_val[ii, :], ov_max_idx[ii, :] = torch.max(overlaps_res, 1)


    # TODO: Current aggregation is sum, we need to try other aggrigation methods


    # Making pooled_feat for tube
    tube_pooled_feat = torch.zeros(batch_size * num_rois, feat_dim, dtype=torch.float32).cuda()
    tube_rois_label = torch.zeros(batch_size * num_rois, dtype=torch.float32).cuda()

    count_tube = 0
    ov_threshold = 0.5
    # dist_threshold = 0.7

    for ii in range(num_rois):

        tmp_feat = pooled_feat_view[0][ii]
        ref_lbl = rois_label_view[0][ii]

        tmp_flag = True

        # Debugging
        tmp_idx_list = []
        tmp_idx_list.append(ii)

        kk = ii

        if DEBUG:
            print('\n============================> ', ii)
            print('ref lbl: ', ref_lbl)

        for jj in range(cfg.time_win - 1):

            kk = ov_max_idx[jj][kk]

            if DEBUG:
                print('jj: ', jj)
                print('kk: ', kk)
                print('ov_idx: ', ov_max_val[jj][kk])
                print('ov_val: ', ov_max_val[jj][kk])
                print('lbl: ', rois_label_view[jj+1][kk])


            if (ov_max_val[jj][kk] >= ov_threshold and 
                rois_label_view[jj+1][kk] == ref_lbl):

                tmp_feat += pooled_feat_view[jj+1][kk]

                tmp_idx_list.append(kk)
            else:
                tmp_flag = False
                break

        if tmp_flag:

            if DEBUG:
                print('<<<<<<<<<< TUBE! >>>>>>>>>>')

            tube_pooled_feat[count_tube][:] = tmp_feat
            tube_rois_label[count_tube] = ref_lbl
            count_tube += 1

            # Debugging
            if DEBUG:
                for count, idx in enumerate(tmp_idx_list):
                    idx = int(idx)

                    cur_img = im_data[count]

                    bb = rois[count][idx][1:]

                    img = at.tonumpy(cur_img)
                    img = inverse_normalize(img).copy()

                    cv.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 0, 255), 3)
                    cv.imwrite('make_tube_ff_ov_feat_' + str(ii) + '_' + str(count) + '.jpg', img)


    print('\n STOP \n')
    pdb.set_trace()

    # TODO: Check it causes problem to give many zeros!
    # Uncommenting this causes problem because of batch size

    # tube_pooled_feat = tube_pooled_feat[:count_tube]
    # tube_rois_label = tube_rois_label[:count_tube]

    return tube_pooled_feat, tube_rois_label
