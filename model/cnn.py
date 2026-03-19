from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
import segmentation_models_pytorch as smp

class CNN(nn.Module):
    def __init__(self, in_chns, seg_class_num, cls_class_num, view_num_classes=4, pretrain_name='resnet50', bottleneck_dim=2048, hidden_dim=1024):
        super(CNN, self).__init__()
        
        self.model = smp.Unet(
            encoder_name=pretrain_name,     # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_chns,            # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=seg_class_num,          # model output channels (number of classes in your dataset)
        )

        # extra cls head
        self.cls_decoder = nn.Sequential(nn.Linear(bottleneck_dim, hidden_dim), nn.ReLU(inplace=True),
                                         nn.Linear(hidden_dim, cls_class_num))

    def forward(self, x, need_fp=False):
        feature = self.model.encoder(x)

        # segmentation
        seg_out = self.model(x)
        # classification
        bottleneck = feature[-1]
        image_embedding = F.adaptive_avg_pool2d(bottleneck, 1).view(bottleneck.size(0), -1)
        cls_out = self.cls_decoder(image_embedding)
        
        return seg_out, cls_out
    
class MIT(nn.Module):
    def __init__(self, in_chns, seg_class_num, cls_class_num, view_num_classes=4, pretrain_name='mit_b2', bottleneck_dim=512, hidden_dim=320):
        super(MIT, self).__init__()
        
        self.model = smp.Segformer(
            encoder_name=pretrain_name,     # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_chns,            # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=seg_class_num,          # model output channels (number of classes in your dataset)
        )

        # extra cls head
        self.cls_decoder = nn.Sequential(nn.Linear(bottleneck_dim, hidden_dim), nn.ReLU(inplace=True),
                                         nn.Linear(hidden_dim, cls_class_num))

    def forward(self, x, need_fp=False):
        feature = self.model.encoder(x)

        # segmentation
        seg_out = self.model(x)
        # classification
        bottleneck = feature[-1]
        image_embedding = F.adaptive_avg_pool2d(bottleneck, 1).view(bottleneck.size(0), -1)
        cls_out = self.cls_decoder(image_embedding)
        
        return seg_out, cls_out
    
