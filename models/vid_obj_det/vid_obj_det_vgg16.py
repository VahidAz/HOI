from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

from models.vid_obj_det.vid_obj_det import _VIDOBJDET


class VID_OBJ_DET_VGG16(_VIDOBJDET):
    def __init__(self, _cfg, _n_classes, _class_agnostic=False):
        self.cfg = _cfg
        self.n_classes = _n_classes
        self.class_agnostic = _class_agnostic
        self.dout_base_model = 512

        _VIDOBJDET.__init__(self, _cfg, _n_classes, _class_agnostic)


    def _init_modules(self):
    
        # TODO: Already just caffe pretrained is supported
        # pytorch vgg16 will be added later

        vgg = models.vgg16().cuda()

        if self.cfg.pretrained:
            if self.cfg.pretrained_caffe:
                print("Loading caffe pretrained weights from %s" %(
                    self.cfg.pretrained_path))
                state_dict = torch.load(self.cfg.pretrained_path)
                vgg.load_state_dict(
                    {k:v for k,v in state_dict.items() if k in vgg.state_dict()})

        vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

        # Not using the last maxpool layer
        self._VIDOBJDET_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

        # Fix the layers before conv3:
        for layer in range(10):
            for p in self._VIDOBJDET_base[layer].parameters(): p.requires_grad = False

        self._VIDOBJDET_top = vgg.classifier

        # not using the last maxpool layer
        self._VIDOBJDET_cls_score = nn.Linear(4096, self.n_classes)

        if self.class_agnostic:
            self._VIDOBJDET_bbox_pred = nn.Linear(4096, 4)
        else:
            self._VIDOBJDET_bbox_pred = nn.Linear(4096, 4 * self.n_classes)      


    def _head_to_tail(self, pool5):
        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self._VIDOBJDET_top(pool5_flat)

        return fc7
