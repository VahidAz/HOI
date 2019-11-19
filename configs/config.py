#!/usr/bin/env python3.7.5
# -*- coding: utf-8 -*-
"""
"""


from pprint import pprint


class Config(object):
    # Dataset configs
    dataset_name = 'pascal_voc2012'
    dataset_trainval = None
    dataset_test = None

    # Dataset properties
    min_size = None
    max_size = None
    anchor_scales = None
    anchor_ratios = None

    # Backend model
    backend_model = 'vgg16'
    pretrained_flag = True
    pretrained_caffe_flag = True
    pretrained_weight_path = None

    # Pytorch configs
    cuda = True

    # Training configs
    epoch = 2

    # Weight decay, for regularization
    # WEIGHT_DECAY = 0.0005
    WEIGHT_DECAY = 0.0005

    # Whether to double the learning rate for bias
    DOUBLE_BIAS = True

    # Whether to have weight decay on bias as well
    BIAS_DECAY = False

    use_tfboard = True


    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        # VOC_2012
        if self.dataset_name == 'pascal_voc2012':
            self.dataset_trainval = './datasets/pascal_voc/VOC2012/VOC2012_trainval/'
            self.dataset_test = './datasets/pascal_voc/VOC2012/VOC2012_test/'
            self.min_size = 600
            self.max_size = 1000
            self.anchor_scales = [8, 16, 32]
            self.anchor_ratios = [0.5, 1, 2]

        # VGG16
        if self.backend_model == 'vgg16' and self.pretrained_flag:
            # Load caffe pretrained vgg16 since it has better performance
            if self.pretrained_caffe_flag:
                self.pretrained_weight_path = './pretrained_model/vgg16_caffe.pth'
            else:
                self.pretrained_weight_path = None

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')


    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


cfg = Config()
