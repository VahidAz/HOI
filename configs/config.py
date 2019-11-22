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
    epoch = 2
    batch_size = 1

    # Weight decay, for regularization
    WEIGHT_DECAY = 0.0005

    # Whether to double the learning rate for bias
    DOUBLE_BIAS = True

    # Whether to have weight decay on bias as well
    BIAS_DECAY = False

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
