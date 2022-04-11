import os.path

import torch
import torch.nn as nn
import logging
import math
import numpy as np
from typing import List
import torch
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from torch import nn

from detectron2.layers import ShapeSpec, batched_nms, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.meta_arch.retinanet import permute_to_N_HWA_K
from detectron2.structures.keypoints import Keypoints
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
import matplotlib.pyplot as plt


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        nn.init.normal_(self.fc1.weight, std=0.01)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        nn.init.normal_(self.fc2.weight, std=0.01)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        nn.init.normal_(self.conv1.weight, std=0.01)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

@META_ARCH_REGISTRY.register()
class ClsNet(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.num_classes = cfg.MODEL.CLSNET.NUM_CLASSES
        self.in_features = cfg.MODEL.CLSNET.IN_FEATURES
        self.bottom_up = build_backbone(cfg)
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4]))

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))
        # self.linear = nn.Linear(int(cfg.MODEL.BCE.INPUTFEATURESIZE), int(cfg.MODEL.BCE.BCECLASS))
        # nn.init.normal_(self.linear.weight, std=0.01)
        self.conv1x1 = nn.Conv2d(in_channels=cfg.MODEL.BCE.INPUTFEATURESIZE,out_channels=cfg.MODEL.BCE.BCECLASS,kernel_size=1,stride=1,padding=0)
        nn.init.normal_(self.conv1x1.weight,std=0.01)


    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images)  # Do not need size_divisibility
        return images

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        gt_labels = [x['label'] for x in batched_inputs]
        gt_labels = torch.as_tensor(gt_labels, dtype=torch.long).to(self.device)
        features = self.bottom_up(images.tensor)
        features = [features[f] for f in self.in_features][0]
        # features = torch.nn.functional.relu(features, inplace=True)
        features = torch.nn.functional.adaptive_avg_pool2d(features,(1,1))
        features = self.conv1x1(features)
        features = features.squeeze()
        # features = self.linear(features)
        if self.training:
            losses = self.losses(gt_labels, features)
            return losses
        else:
            results = self.inference(features)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                processed_results.append({"Image_name":input_per_image['file_name'],"Ground_Truth_classes":input_per_image['label'],"pred_classes": results_per_image})
            return processed_results

    def inference(self, features):
        pred = torch.sigmoid(features)
        return pred

    def losses(self, gt_labels, features):
        return {"loss_cls": self.criterion(features, gt_labels.float())}


@META_ARCH_REGISTRY.register()
class PMTBClsNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_classes = cfg.MODEL.CLSNET.NUM_CLASSES
        self.in_features = cfg.MODEL.CLSNET.IN_FEATURES
        self.bottom_up = build_backbone(cfg)
        self.criterion_1 = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([2,0.8]))
        self.criterion_2 = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.25,0.25,0.5]))
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        # self.conv1x1 = nn.Conv2d(in_channels=cfg.MODEL.BCE.INPUTFEATURESIZE,out_channels=cfg.MODEL.BCE.BCECLASS,kernel_size=1,stride=1,padding=0)
        self.linear1 = nn.Linear(cfg.MODEL.BCE.INPUTFEATURESIZE,cfg.MODEL.BCE.MLPFEATURESIZE)
        self.linear2 = nn.Linear(cfg.MODEL.BCE.MLPFEATURESIZE,cfg.MODEL.BCE.BCECLASS)
        self.dropout = torch.nn.Dropout(p=0.5)
        # nn.init.normal_(self.conv1x1.weight,std=0.01)
        nn.init.normal_(self.linear1.weight,std=0.01)
        nn.init.normal_(self.linear2.weight,std=0.01)
    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [x/255.0 for x in images]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images)  # Do not need size_divisibility
        return images

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        gt_labels = []
        for x in batched_inputs:
            # gt_labels.append(x['label'][:2])
            if x['label'][:2]==[1,0]:
                gt_labels.append(0)
            elif x['label'][:2]==[0,1]:
                gt_labels.append(1)
            else:
                gt_labels.append(2)
        # gt_labels = torch.as_tensor(gt_labels, dtype=torch.float32).to(self.device)
        gt_labels = torch.as_tensor(gt_labels, dtype=torch.long).to(self.device)

        features = self.bottom_up(images.tensor)
        features = [features[f] for f in self.in_features][0]
        features = torch.nn.functional.adaptive_avg_pool2d(features,(1,1))
        features = torch.flatten(features,1)
        features = self.dropout(features)
        features = self.linear1(features)
        features = self.dropout(features)
        features = self.linear2(features)

        if self.training:
            losses = self.losses(gt_labels,features)
            return losses
        else:
            results = self.inference(features)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                processed_results.append({"Image_name":input_per_image['file_name'],
                                          "Ground_Truth_classes":input_per_image['label'],
                                          "pred_classes": results_per_image,
                                          "loss":(self.criterion_2(features,gt_labels)).detach().cpu().numpy()})
            return processed_results

    def inference(self, features):
        # pred = torch.sigmoid(features)
        pred = torch.softmax(features,dim=1)
        return pred

    def losses(self, gt_labels,features):
        # bce_loss = self.criterion_1(features,gt_labels)
        ce_loss = self.criterion_2(features,gt_labels)
        return {"loss_hard_cls":ce_loss}

@META_ARCH_REGISTRY.register()
class PMTB_CBAM_ClsNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_classes = cfg.MODEL.CLSNET.NUM_CLASSES
        self.in_features = cfg.MODEL.CLSNET.IN_FEATURES
        self.bottom_up = build_backbone(cfg)
        self.criterion_1 = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([5,1.5]))
        self.criterion_2 = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.3,0.1,0.6]))
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        #classification head
        self.linear1 = nn.Linear(cfg.MODEL.BCE.INPUTFEATURESIZE,cfg.MODEL.BCE.MLPFEATURESIZE)
        self.linear2 = nn.Linear(cfg.MODEL.BCE.MLPFEATURESIZE,cfg.MODEL.BCE.BCECLASS)
        nn.init.normal_(self.linear1.weight,std=0.01)
        nn.init.normal_(self.linear2.weight,std=0.01)

        #CBAM parameters
        self.inplanes = cfg.MODEL.BCE.INPUTFEATURESIZE
        self.ca = ChannelAttention(self.inplanes)
        self.sa = SpatialAttention()

    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [x/255.0 for x in images]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images)  # Do not need size_divisibility
        return images

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        gt_labels = []
        for x in batched_inputs:
            # gt_labels.append(x['label'][:2])
            if x['label'][:2]==[1,0]:
                gt_labels.append(0)
            elif x['label'][:2]==[0,1]:
                gt_labels.append(1)
            else:
                gt_labels.append(2)
        # gt_labels = torch.as_tensor(gt_labels, dtype=torch.float32).to(self.device)
        gt_labels = torch.as_tensor(gt_labels, dtype=torch.long).to(self.device)

        features = self.bottom_up(images.tensor)
        features = [features[f] for f in self.in_features][0]
        #channel attention and spatial attention
        features = self.ca(features) * features
        features = self.sa(features) * features

        features = torch.nn.functional.adaptive_avg_pool2d(features,(1,1))
        features = torch.flatten(features,1)
        features = self.linear1(features)
        features = self.linear2(features)

        if self.training:
            losses = self.losses(gt_labels,features)
            return losses
        else:
            results = self.inference(features)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                processed_results.append({"Image_name":input_per_image['file_name'],"Ground_Truth_classes":input_per_image['label'],"pred_classes": results_per_image})
            return processed_results

    def inference(self, features):
        pred = torch.softmax(features,dim=1)
        return pred

    def losses(self, gt_labels,features):
        ce_loss = self.criterion_2(features,gt_labels)
        return {"loss_hard_cls":ce_loss}