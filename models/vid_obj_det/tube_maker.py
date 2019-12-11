import numpy as np
import pdb
import cv2 as cv

import torch

from data.imagenet_vid_dataset import inverse_normalize
from models.libs.utils import array_tool as at
from models.libs.utils.cdist import cdist_v2
from models.rpn.bbox_transform import bbox_overlaps_batch, bbox_overlaps


# Debug flag
DEBUG = True


#@torch.jit.script
def make_tube_ff_feat(pooled_feat, rois_label, rois, cfg, im_data, tube_threshold=0.7):
    '''
    Feature dist frame by frame
    '''
    pooled_feat_view = pooled_feat.view(rois.shape[0], rois.shape[1], pooled_feat.shape[1]).cuda()
    rois_label_view = rois_label.view(rois.shape[0], rois.shape[1]).cuda()

    final_max_val = torch.zeros([rois.shape[0] - 1, rois.shape[1]], dtype=torch.float32).cuda()
    final_max_idx = torch.zeros([rois.shape[0] - 1, rois.shape[1]], dtype=torch.int32).cuda()

    for ii in range(cfg.time_win - 1):
        first_feat = pooled_feat_view[ii]
        sec_feat = pooled_feat_view[ii+1]
        
        # Dist_res, 0 max, 1 min
        # Dist_res_reverse, 0 min, 1 max
        dist_res, dist_res_reverse = cdist_v2(first_feat, sec_feat)

        final_val, final_idx = torch.max(dist_res_reverse, 0)

        final_max_val[ii, :] = final_val
        final_max_idx[ii, :] = final_idx

    count_tube = 0
    tube_pooled_feat = torch.zeros(pooled_feat.shape[0], pooled_feat.shape[1], dtype=torch.float32).cuda()
    tube_rois_label = torch.zeros(pooled_feat.shape[0], dtype=torch.float32).cuda()

    for ii in range(rois.shape[1]):
        tmp_feat = torch.zeros(pooled_feat.shape[1], dtype=torch.float32).cuda()
        tmp_feat += pooled_feat_view[0][ii]

        ref_lbl = rois_label_view[0][ii]
        sum_lbl = 0
        sum_lbl += ref_lbl

        tmp_flag = True
        
        tmp_idx_list = []
        tmp_idx_list.append(ii)
        tmp_val_list = []

        for jj in range(rois.shape[0] - 1):
            if (final_max_val[jj][ii] >= tube_threshold):
                tmp_feat += pooled_feat_view[jj+1][final_max_idx[jj][ii]]
                sum_lbl += rois_label_view[jj+1][final_max_idx[jj][ii]]
                tmp_idx_list.append(final_max_idx[jj][ii])
                tmp_val_list.append(final_max_val[jj][ii])
            else:
                tmp_flag = False
                break

            if tmp_flag:
                if int(sum_lbl/rois.shape[0]) == int(ref_lbl):
                    tube_pooled_feat[count_tube][:] = tmp_feat
                    tube_rois_label[count_tube] = ref_lbl
                    count_tube += 1

                    # Debugging
                    for count, idx in enumerate(tmp_idx_list):
                        idx = int(idx)

                        cur_img = im_data[count]

                        bb = rois[count][idx][1:]

                        img = at.tonumpy(cur_img)
                        img = inverse_normalize(img).copy()

                        cv.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 0, 255), 3)
                        cv.imwrite('make_tube_ff_feat_' + str(ii) + '_' + str(count) + '.jpg', img)


    return tube_pooled_feat, tube_rois_label


#@torch.jit.script
def make_tube_bf_feat(pooled_feat, rois_label, rois, cfg, im_data, tube_threshold=0.7):
    '''
    Feature dist only with base rois
    '''
    pooled_feat_view = pooled_feat.view(rois.shape[0], rois.shape[1], pooled_feat.shape[1])
    rois_label_view = rois_label.view(rois.shape[0], rois.shape[1])

    final_max_val = torch.zeros([rois.shape[0] - 1, rois.shape[1]], dtype=torch.float32).cuda()
    final_max_idx = torch.zeros([rois.shape[0] - 1, rois.shape[1]], dtype=torch.int32).cuda()

    base_feat = pooled_feat_view[0]

    for ii in range(1, cfg.time_win):
        sec_feat = pooled_feat_view[ii]
        
        # Dist_res, 0 max, 1 min
        # Dist_res_reverse, 0 min, 1 max
        dist_res, dist_res_reverse = cdist_v2(base_feat, sec_feat)

        final_val, final_idx = torch.max(dist_res_reverse, 0)

        final_max_val[ii-1, :] = final_val
        final_max_idx[ii-1, :] = final_idx

    count_tube = 0
    tube_pooled_feat = torch.zeros(pooled_feat.shape[0], pooled_feat.shape[1], dtype=torch.float32).cuda()
    tube_rois_label = torch.zeros(pooled_feat.shape[0], dtype=torch.float32).cuda()

    for ii in range(rois.shape[1]):
        tmp_feat = torch.zeros(pooled_feat.shape[1], dtype=torch.float32).cuda()
        tmp_feat += pooled_feat_view[0][ii]

        ref_lbl = rois_label_view[0][ii]
        sum_lbl = 0
        sum_lbl += ref_lbl

        tmp_flag = True
        tmp_idx_list = []
        tmp_idx_list.append(ii)
        tmp_val_list = []

        for jj in range(rois.shape[0] - 1):
            if (final_max_val[jj][ii] >= tube_threshold):
                tmp_feat += pooled_feat_view[jj+1][final_max_idx[jj][ii]]
                sum_lbl += rois_label_view[jj+1][final_max_idx[jj][ii]]
                tmp_idx_list.append(final_max_idx[jj][ii])
                tmp_val_list.append(final_max_val[jj][ii])
            else:
                tmp_flag = False
                break

        if tmp_flag:
            if int(sum_lbl/rois.shape[0]) == int(ref_lbl):
                tube_pooled_feat[count_tube][:] = tmp_feat
                tube_rois_label[count_tube] = ref_lbl
                count_tube += 1

                # Debugging
                for count, idx in enumerate(tmp_idx_list):
                    idx = int(idx)

                    cur_img = im_data[count]

                    bb = rois[count][idx][1:]

                    img = at.tonumpy(cur_img)
                    img = inverse_normalize(img).copy()

                    cv.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 0, 255), 3)
                    cv.imwrite('make_tube_bf_feat_' + str(ii) + '_' + str(count) + '.jpg', img)


    return tube_pooled_feat, tube_rois_label


#@torch.jit.script
def make_tube_ff_ov(pooled_feat, rois_label, rois, cfg, im_data, tube_threshold=0.7):
    '''
    Overlap frame by frame
    '''
    pooled_feat_view = pooled_feat.view(rois.shape[0], rois.shape[1], pooled_feat.shape[1])
    rois_label_view = rois_label.view(rois.shape[0], rois.shape[1])

    final_max_val = torch.zeros([rois.shape[0] - 1, rois.shape[1]], dtype=torch.float32).cuda()
    final_max_idx = torch.zeros([rois.shape[0] - 1, rois.shape[1]], dtype=torch.int32).cuda()

    for ii in range(cfg.time_win - 1):
        first_roi = rois[ii][:, 1:]

        sec_roi = rois[ii+1][:, 1:]
        sec_roi_view = sec_roi.view(1, sec_roi.shape[0], sec_roi.shape[1])

        overlap_res = bbox_overlaps_batch(first_roi, sec_roi_view)
        overlap_res = overlap_res.squeeze()

        max_val, max_idx = torch.max(overlap_res, 0)

        final_max_val[ii, :] = max_val
        final_max_idx[ii, :] = max_idx

    count_tube = 0
    tube_pooled_feat = torch.zeros(pooled_feat.shape[0], pooled_feat.shape[1], dtype=torch.float32).cuda()
    tube_rois_label = torch.zeros(pooled_feat.shape[0], dtype=torch.float32).cuda()

    for ii in range(rois.shape[1]):
        tmp_feat = torch.zeros(pooled_feat.shape[1], dtype=torch.float32).cuda()
        tmp_feat += pooled_feat_view[0][ii]

        ref_lbl = rois_label_view[0][ii]
        sum_lbl = 0
        sum_lbl += ref_lbl

        tmp_flag = True
        tmp_idx_list = []
        tmp_idx_list.append(ii)
        tmp_val_list = []

        for jj in range(rois.shape[0] - 1):
            if (final_max_val[jj][ii] >= tube_threshold):
                tmp_feat += pooled_feat_view[jj+1][final_max_idx[jj][ii]]
                sum_lbl += rois_label_view[jj+1][final_max_idx[jj][ii]]
                tmp_idx_list.append(final_max_idx[jj][ii])
                tmp_val_list.append(final_max_val[jj][ii])
            else:
                tmp_flag = False
                break

            if tmp_flag:
                if int(sum_lbl/rois.shape[0]) == int(ref_lbl):
                    tube_pooled_feat[count_tube][:] = tmp_feat
                    tube_rois_label[count_tube] = ref_lbl
                    count_tube += 1

                    # Debugging
                    for count, idx in enumerate(tmp_idx_list):
                        idx = int(idx)

                        cur_img = im_data[count]

                        bb = rois[count][idx][1:]

                        img = at.tonumpy(cur_img)
                        img = inverse_normalize(img).copy()

                        cv.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 0, 255), 3)
                        cv.imwrite('make_tube_ff_ov_' + str(ii) + '_' + str(count) + '.jpg', img)


    return tube_pooled_feat, tube_rois_label


#@torch.jit.script
def make_tube_bf_ov(pooled_feat, rois_label, rois, cfg, im_data, tube_threshold=0.7):
    '''
    Overlap only with base rois
    '''
    pooled_feat_view = pooled_feat.view(rois.shape[0], rois.shape[1], pooled_feat.shape[1])
    rois_label_view = rois_label.view(rois.shape[0], rois.shape[1])

    final_max_val = torch.zeros([rois.shape[0] - 1, rois.shape[1]], dtype=torch.float32).cuda()
    final_max_idx = torch.zeros([rois.shape[0] - 1, rois.shape[1]], dtype=torch.int32).cuda()

    base_rois = rois[0][:, 1:]

    for ii in range(1, cfg.time_win):
        target_rois = rois[ii][:, 1:]
        target_rois_view = target_rois.view(1, target_rois.shape[0], target_rois.shape[1])

        overlaps_res = bbox_overlaps_batch(base_rois, target_rois_view)
        overlaps_res = overlaps_res.squeeze()

        max_val, max_idx = torch.max(overlaps_res, 0)

        final_max_val[ii-1, :] = max_val
        final_max_idx[ii-1, :] = max_idx

    count_tube = 0
    tube_pooled_feat = torch.zeros(pooled_feat.shape[0], pooled_feat.shape[1], dtype=torch.float32).cuda()
    tube_rois_label = torch.zeros(pooled_feat.shape[0], dtype=torch.float32).cuda()

    for ii in range(rois.shape[1]):
        tmp_feat = torch.zeros(pooled_feat.shape[1], dtype=torch.float32).cuda()
        tmp_feat += pooled_feat_view[0][ii]

        ref_lbl = rois_label_view[0][ii]
        sum_lbl = 0
        sum_lbl += ref_lbl

        tmp_flag = True
        tmp_idx_list = []
        tmp_idx_list.append(ii)
        tmp_val_list = []

        for jj in range(rois.shape[0] - 1):
            if (final_max_val[jj][ii] >= tube_threshold):
                tmp_feat += pooled_feat_view[jj+1][final_max_idx[jj][ii]]
                sum_lbl += rois_label_view[jj+1][final_max_idx[jj][ii]]
                tmp_idx_list.append(final_max_idx[jj][ii])
                tmp_val_list.append(final_max_val[jj][ii])
            else:
                tmp_flag = False
                break

        if tmp_flag:
            if int(sum_lbl/rois.shape[0]) == int(ref_lbl):
                tube_pooled_feat[count_tube][:] = tmp_feat
                tube_rois_label[count_tube] = ref_lbl
                count_tube += 1

                # Debugging
                for count, idx in enumerate(tmp_idx_list):
                    idx = int(idx)

                    cur_img = im_data[count]

                    bb = rois[count][idx][1:]

                    img = at.tonumpy(cur_img)
                    img = inverse_normalize(img).copy()

                    cv.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 0, 255), 3)
                    cv.imwrite('make_tube_bf_ov_' + str(ii) + '_' + str(count) + '.jpg', img)


    return tube_pooled_feat, tube_rois_label


#@torch.jit.script
def make_tube_ff_ov_feat(pooled_feat, rois_label, rois, cfg, im_data):
    '''
    this function makes tube by integrating overlap and cdist
    the tracking happens frame by frame
    '''

    batch_size = rois.shape[0]
    num_rois = rois.shape[1]
    feat_dim = pooled_feat.shape[1]

    pooled_feat_view = pooled_feat.view(batch_size, num_rois, feat_dim)
    rois_label_view = rois_label.view(batch_size, num_rois)


    # Making tube
    # TODO: Try LSTM data associatation
    # TODO: Try to implement it better


    ov_max_val = torch.zeros([cfg.time_win - 1, num_rois], dtype=torch.float32).cuda()
    ov_max_idx = torch.zeros([cfg.time_win - 1, num_rois], dtype=torch.int32).cuda()

    dist_max_val = torch.zeros([cfg.time_win - 1, num_rois], dtype=torch.float32).cuda()
    dist_max_idx = torch.zeros([cfg.time_win - 1, num_rois], dtype=torch.int32).cuda()

    print('\nTM Debug1')
    pdb.set_trace()

    for ii in range(cfg.time_win - 1):
        # First value in rois is batch number, bbox is [1:]
        first_feat = pooled_feat_view[ii]
        first_rois = (rois[ii])[:, 1:]

        sec_feat = pooled_feat_view[ii+1]
        sec_rois = (rois[ii+1])[:, 1:]

        # Dist_res, 0 max, 1 min
        # Dist_res_reverse, 0 min, 1 max
        # Overlap, 0 min, 1 max
        dist_res, dist_res_reverse = cdist_v2(first_feat, sec_feat)
        overlaps_res = bbox_overlaps(first_rois, sec_rois)

        dist_max_val[ii, :], dist_max_idx[ii, :] = torch.max(dist_res_reverse, 0)
        ov_max_val[ii, :], ov_max_idx[ii, :] = torch.max(overlaps_res, 0)


    # TODO: Write a vectorized version or a better one
    # TODO: Current aggregation is sum, we need to try other aggrigation methods


    # Making pooled_feat for tube
    tube_pooled_feat = torch.zeros(batch_size, feat_dim, dtype=torch.float32).cuda()
    tube_rois_label = torch.zeros(rois_label.shape[0], dtype=torch.float32).cuda()

    count_tube = 0
    ov_threshold = 0.7
    dist_threshold = 0.5

    for ii in range(num_rois):
        print('\n==========================> ', ii)
        tmp_feat = torch.zeros(feat_dim, dtype=torch.float32).cuda()

        tmp_feat += pooled_feat_view[0][ii]
        ref_lbl = rois_label_view[0][ii]
        print('ref lbl: ', ref_lbl)

        tmp_flag = True

        # Debugging
        tmp_idx_list = []
        tmp_idx_list.append(ii)

        for jj in range(cfg.time_win - 1):
            print('\tjj: ', jj)
            print(rois_label_view[jj+1][ii])
            print(ov_max_idx[jj][ii], dist_max_idx[jj][ii])
            print(ov_max_val[jj][ii])
            print(dist_max_val[jj][ii])
            if (rois_label_view[jj+1][ii] == ref_lbl and
                ov_max_idx[jj][ii] == dist_max_idx[jj][ii] and
                ov_max_val[jj][ii] >= ov_threshold and 
                dist_max_val[jj][ii] >= dist_threshold):

                tmp_feat += pooled_feat_view[jj+1][ov_max_idx[jj][ii]]

                tmp_idx_list.append(ov_max_idx[jj][ii])
            else:
                tmp_flag = False
                break

        if tmp_flag:
            tube_pooled_feat[count_tube][:] = tmp_feat
            tube_rois_label[count_tube] = ref_lbl
            count_tube += 1

            # Debugging
            for count, idx in enumerate(tmp_idx_list):
                idx = int(idx)

                cur_img = im_data[count]

                bb = rois[count][idx][1:]

                img = at.tonumpy(cur_img)
                img = inverse_normalize(img).copy()

                cv.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 0, 255), 3)
                cv.imwrite('make_tube_ff_ov_feat_' + str(ii) + '_' + str(count) + '.jpg', img)


    print('\n RIDIIII \n')
    pdb.set_trace()

    # TODO: Check it causes problem to give many zeros!
    # Uncommenting this causes problem because of batch size

    # tube_pooled_feat = tube_pooled_feat[:count_tube]
    # tube_rois_label = tube_rois_label[:count_tube]

    return tube_pooled_feat, tube_rois_label


#@torch.jit.script
def make_tube_bf_ov_feat(pooled_feat, rois_label, rois, cfg, im_data, tube_threshold=0.7):
    '''
    this function makes tube by integrating overlap and cdist
    the tracking happens base to frames
    '''
    pooled_feat_view = pooled_feat.view(rois.shape[0], rois.shape[1], pooled_feat.shape[1]).cuda()
    rois_label_view = rois_label.view(rois.shape[0], rois.shape[1]).cuda()

    final_max_val = torch.zeros([rois.shape[0] - 1, rois.shape[1]], dtype=torch.float32).cuda()
    final_max_idx = torch.zeros([rois.shape[0] - 1, rois.shape[1]], dtype=torch.int32).cuda()

    first_feat = pooled_feat_view[0]
    first_rois = rois[0][:, 1:]

    for ii in range(1, cfg.time_win):
        sec_feat = pooled_feat_view[ii]
        sec_rois = (rois[ii])[:, 1:]
        sec_rois_view = sec_rois.view(1, sec_rois.shape[0], sec_rois.shape[1])

        # Dist_res, 0 max, 1 min
        # Dist_res_reverse, 0 min, 1 max
        # Overlap, 0 min, 1 max
        dist_res, dist_res_reverse = cdist_v2(first_feat, sec_feat)
        overlaps_res = bbox_overlaps_batch(first_rois, sec_rois_view)
        overlaps_res = overlaps_res.squeeze()

        # Clipping by threshold
        overlaps_res[overlaps_res < 0.7] = 0
        dist_res_reverse[dist_res_reverse < 0.5] = 0

        # Final matrix
        final_metric = overlaps_res + dist_res_reverse
        final_val, final_idx = torch.max(final_metric, 0)

        final_max_val[ii-1, :] = final_val
        final_max_idx[ii-1, :] = final_idx


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
        tmp_feat += pooled_feat_view[0][ii]

        ref_lbl = rois_label_view[0][ii]
        sum_lbl = 0
        sum_lbl += ref_lbl

        tmp_flag = True
        tmp_idx_list = []
        tmp_idx_list.append(ii)
        tmp_val_list = []

        for jj in range(rois.shape[0] - 1):
            if (final_max_val[jj][ii] >= tube_threshold):
                tmp_feat += pooled_feat_view[jj+1][final_max_idx[jj][ii]]
                sum_lbl += rois_label_view[jj+1][final_max_idx[jj][ii]]
                tmp_idx_list.append(final_max_idx[jj][ii])
                tmp_val_list.append(final_max_val[jj][ii])
            else:
                tmp_flag = False
                break

        if tmp_flag:
            if int(sum_lbl/rois.shape[0]) == int(ref_lbl):
                tube_pooled_feat[count_tube][:] = tmp_feat
                tube_rois_label[count_tube] = ref_lbl
                count_tube += 1

                # Debugging
                for count, idx in enumerate(tmp_idx_list):
                    idx = int(idx)

                    cur_img = im_data[count]

                    bb = rois[count][idx][1:]

                    img = at.tonumpy(cur_img)
                    img = inverse_normalize(img).copy()

                    cv.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 0, 255), 3)
                    cv.imwrite('make_tube_bf_ov_feat_' + str(ii) + '_' + str(count) + '.jpg', img)


    # TODO: Check it causes problem to give many zeros!
    # Uncommenting this causes problem because of batch size

    # tube_pooled_feat = tube_pooled_feat[:count_tube]
    # tube_rois_label = tube_rois_label[:count_tube]

    return tube_pooled_feat, tube_rois_label
