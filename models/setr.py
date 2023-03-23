import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from mmcv.runner import build_runner
import math
import numpy as np
from torch.autograd import Variable
import mmcv
from .mmseg.models import build_segmentor
from mmseg.models import backbones
from mmseg.models.builder import BACKBONES, SEGMENTORS, HEADS
import os
import logging
logger = logging.getLogger(__name__)
from .iou_loss import IOU
import random
from torch.nn import init
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
import thop

def init_weights(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.BatchNorm2d:
        # print(layer)
        nn.init.normal_(layer.weight, mean=1.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)

class BBCEWithLogitLoss(nn.Module):
    '''
    Balanced BCEWithLogitLoss
    '''
    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)

        return loss

def _iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)

    return iou.mean()

@register('setr')
class SETR(nn.Module):
    def __init__(self, inp_size=None, encoder_mode=None, loss=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if 'vpt' in encoder_mode['name']:
            backbone = dict(type='VisionTransformerVPT', img_size=inp_size, pos_embed_interp=True, drop_rate=0.,
                  model_name='vit_base_patch16_384', embed_dim=768, depth=12, num_heads=12
                  )
        elif 'adaptformer' in encoder_mode['name']:
            backbone = dict(type='VisionTransformerAdaptFormer', img_size=inp_size, pos_embed_interp=True, drop_rate=0.,
                  model_name='vit_base_patch16_384', embed_dim=768, depth=12, num_heads=12
                  )
        elif 'evp' in encoder_mode['name']:
            backbone = dict(type='VisionTransformerEVP', img_size=inp_size, pos_embed_interp=True, drop_rate=0.,
                  model_name='vit_base_patch16_384', embed_dim=768,depth=12, num_heads=12,
                  scale_factor=encoder_mode['scale_factor'], input_type=encoder_mode['input_type'],
                  freq_nums=encoder_mode['freq_nums'], prompt_type=encoder_mode['prompt_type'],
                  tuning_stage=encoder_mode['tuning_stage'], handcrafted_tune=encoder_mode['handcrafted_tune'],
                  embedding_tune=encoder_mode['embedding_tune'], adaptor=encoder_mode['adaptor'], )
        else:
            backbone = dict(type='VisionTransformer', img_size=inp_size, pos_embed_interp=True, drop_rate=0.,
                  model_name='vit_base_patch16_384', embed_dim=768, depth=12, num_heads=12
                  )

        model_config = dict(
            type='EncoderDecoder',
            backbone=backbone,
            decode_head=dict(
                type='VisionTransformerUpHead',
                in_channels=768,
                channels=512,
                in_index=0,
                img_size=inp_size,
                embed_dim=768,
                num_classes=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                num_conv=4,
                upsampling_method='bilinear',
                num_upsampe_layer=4,
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1)),
            auxiliary_head=[dict(
                type='VisionTransformerUpHead',
                in_channels=1024,
                channels=512,
                in_index=0,
                img_size=inp_size,
                embed_dim=1024,
                num_classes=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                num_conv=2,
                upsampling_method='bilinear',
                num_upsampe_layer=2,
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
                dict(
                type='VisionTransformerUpHead',
                in_channels=1024,
                channels=512,
                in_index=0,
                img_size=inp_size,
                embed_dim=1024,
                num_classes=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                num_conv=2,
                upsampling_method='bilinear',
                num_upsampe_layer=2,
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
                dict(
                type='VisionTransformerUpHead',
                in_channels=1024,
                channels=512,
                in_index=0,
                img_size=inp_size,
                embed_dim=1024,
                num_classes=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                num_conv=2,
                upsampling_method='bilinear',
                num_upsampe_layer=2,
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
                dict(
                type='VisionTransformerUpHead',
                in_channels=1024,
                channels=512,
                in_index=0,
                img_size=inp_size,
                embed_dim=1024,
                num_classes=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                num_conv=2,
                upsampling_method='bilinear',
                num_upsampe_layer=2,
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
            ],
                train_cfg=dict(),
                test_cfg=dict(mode='whole')
        )

        model = build_segmentor(
            model_config,
            # train_cfg=dict(),
            # test_cfg=dict(mode='whole')
        )

        self.encoder = model

        if 'vpt' in encoder_mode['name']:
            for k, p in self.encoder.named_parameters():
                if "prompt" not in k and "decode_head" not in k and "auxiliary_head" not in k:
                    p.requires_grad = False
        if 'evp' in encoder_mode['name']:
            for k, p in self.encoder.named_parameters():
                if "prompt" not in k and "decode_head" not in k and "auxiliary_head" not in k:
                    p.requires_grad = False
        if 'adaptformer' in encoder_mode['name']:
            for k, p in self.encoder.named_parameters():
                if "adaptmlp" not in k and "decode_head" not in k and "auxiliary_head" not in k:
                    p.requires_grad = False
        if 'linear' in encoder_mode['name']:
            for k, p in self.encoder.named_parameters():
                if "decode_head" not in k and "auxiliary_head" not in k:
                    p.requires_grad = False

        model_total_params = sum(p.numel() for p in self.encoder.parameters())
        model_grad_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        print('model_grad_params:' + str(model_grad_params),
              '\nmodel_total_params:' + str(model_total_params))

        self.loss_mode = loss
        if self.loss_mode == 'bce':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()

        elif self.loss_mode == 'bbce':
            self.criterionBCE = BBCEWithLogitLoss()

        elif self.loss_mode == 'iou':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
            self.criterionIOU = IOU()

    def set_input(self, input, gt_mask):
        self.input = input.to(self.device)
        self.gt_mask = gt_mask.to(self.device)

    def forward(self):
        self.pred_mask = self.encoder.forward_dummy(self.input)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G = self.criterionBCE(self.pred_mask, self.gt_mask)
        if self.loss_mode == 'iou':
            self.loss_G += _iou_loss(self.pred_mask, self.gt_mask)

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer.step()  # udpate G's weights

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
