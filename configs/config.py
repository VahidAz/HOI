#!/usr/bin/env python3.7.5
# -*- coding: utf-8 -*-
"""
"""


from pprint import pprint


### TODO: soem of the pathes are hardcoded, they should move here


class Config(object):
    # Dataset configs
    voc_dataset_name = 'pascal_voc2012'
    voc_dataset_trainval = None
    voc_dataset_test = None

    imgnet_vid_dataset_train = './datasets/ILSVRC2015/'

    # Dataset properties
    min_size = None
    max_size = None
    anchor_scales = None
    anchor_ratios = None

    time_win = 5

    # Backend model
    backend_model = 'vgg16'
    pretrained = True
    pretrained_caffe = True
    pretrained_path = None

    # Pytorch configs
    cuda = True

    # Training configs
    # For reproducibility
    RNG_SEED = 3
    
    epoch = 2
    batch_size = 1

    # Weight decay, for regularization
    TRAIN_WEIGHT_DECAY = 0.0005

    # Whether to double the learning rate for bias
    TRAIN_DOUBLE_BIAS = True

    # Whether to have weight decay on bias as well
    TRAIN_BIAS_DECAY = False

    # Whether to initialize the weights with truncated normal distribution
    TRAIN_TRUNCATED = False



    TRAIN_HAS_RPN = True
    # IOU >= thresh: positive example
    TRAIN_RPN_POSITIVE_OVERLAP = 0.7
    # IOU < thresh: negative example
    TRAIN_RPN_NEGATIVE_OVERLAP = 0.3
    # If an anchor statisfied by positive and negative conditions set to negative
    TRAIN_RPN_CLOBBER_POSITIVES = False
    # Max number of foreground examples
    TRAIN_RPN_FG_FRACTION = 0.5
    # Total number of examples
    TRAIN_RPN_BATCHSIZE = 256
    # NMS threshold used on RPN proposals
    TRAIN_RPN_NMS_THRESH = 0.7
    # Number of top scoring boxes to keep before apply NMS to RPN proposals
    TRAIN_RPN_PRE_NMS_TOP_N = 12000
    # Number of top scoring boxes to keep after applying NMS to RPN proposals
    TRAIN_RPN_POST_NMS_TOP_N = 2000
    # Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
    TRAIN_RPN_MIN_SIZE = 8
    # Deprecated (outside weights)
    TRAIN_RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
    # Give the positive RPN examples weight of p * 1 / {num positives}
    # and give negatives a weight of (1 - p)
    # Set to -1.0 to use uniform example weighting
    TRAIN_RPN_POSITIVE_WEIGHT = -1.0
    # Whether to use all ground truth bounding boxes for training,
    # For COCO, setting USE_ALL_GT to False will exclude boxes that are flagged as ''iscrowd''
    TRAIN_USE_ALL_GT = True

    # Whether to tune the batch normalization parameters during training
    TRAIN_BN_TRAIN = False

    USE_GPU_NMS = True

    # Make tensorboard logs
    use_tfboard = True


    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        # VOC_2012
        if self.voc_dataset_name == 'pascal_voc2012':
            self.voc_dataset_trainval = (
                './datasets/PASCAL_VOC/VOC2012/VOC2012_trainval/')
            self.voc_dataset_test = './datasets/PASCAL_VOC/VOC2012/VOC2012_test/'
            self.min_size = 600
            self.max_size = 1000
            self.anchor_scales = [8, 16, 32]
            self.anchor_ratios = [0.5, 1, 2]

        # VGG16
        if self.backend_model == 'vgg16' and self.pretrained:
            # Load caffe pretrained vgg16 since it has better performance
            # Otherwise pytorch pretrained will be used(NOT TESTED)
            if self.pretrained_caffe:
                self.pretrained_path = './pretrained_models/vgg16_caffe.pth'

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')


    def _state_dict(self):
        return {k: getattr(
            self, k) for k, _ in Config.__dict__.items() if not k.startswith('_')}


cfg = Config()
