from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

from models.rpn.rpn_ import RPN_


class RPN_VGG16(RPN_):
  def __init__(self, _cfg):
    self.cfg = _cfg
    self.dout_base_model = 512


    # TODO, check the possibility of adding class agnostic


    RPN_.__init__(self, _cfg=_cfg, _din=self.dout_base_model)


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

    # Not using the last maxpool layer
    self.RPN__base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

    # Fix the layers before conv3:
    for layer in range(10):
        for p in self.RPN__base[layer].parameters(): p.requires_grad = False
