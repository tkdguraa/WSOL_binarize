"""
Original code: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch.utils.model_zoo import load_url

from .method import AcolBase
from .method import ADL
from .method import spg
from .method.util import normalize_tensor
from .util import remove_layer
from .util import replace_layer
from .util import initialize_weights

__all__ = ['vgg16']

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'
}

configs_dict = {
    'cam': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512,
                  512, 'M', 512, 512, 512],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512,
                  512, 512, 512, 512],
    },
    'acol': {
        '14x14': [64, 64, 'M1', 128, 128, 'M1', 256, 256, 256, 'M1', 512, 512,
                  512, 'M1', 512, 512, 512, 'M2'],
        '28x28': [64, 64, 'M1', 128, 128, 'M1', 256, 256, 256, 'M1', 512, 512,
                  512, 'M2', 512, 512, 512, 'M2'],
    },
    'spg': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'],
    },
    'adl': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'A', 512,
                  512, 512, 'A', 'M', 512, 512, 512, 'A'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'A', 512,
                  512, 512, 'A', 512, 512, 512, 'A'],
    }
}


class VggCam(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(VggCam, self).__init__()
        self.features = features
        
        self.large_features = kwargs['large_feature_map']
        self.rate = kwargs['binary_rate']
        self.mode = kwargs['mode']


        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x = self.features(x)
        x = self.conv6(x)
        x = self.relu(x)
        pre_logit = self.avgpool(x)
        pre_logit = pre_logit.view(pre_logit.size(0), -1)
        logits = self.fc(pre_logit)
        
        if return_cam:
            if self.mode != 'vote':
                feature_map = x.detach().clone()
                cam_weights = self.fc.weight[labels].squeeze()
                if self.mode != 'origin':
                    cam_weights[cam_weights<0]=0.
                if self.mode == 'bin':
                    cam_weights[cam_weights>0]=1.
                cams = (cam_weights.view(*feature_map.shape[:2], 1, 1) *
                        feature_map).mean(1, keepdim=False)
                return cams, logits
        
            else:
                cam_list = []
                feature_map = x.detach().clone()
                cam_weights = self.fc.weight[labels].squeeze()
                cams = feature_map
                b, c, h, w = cams.shape
                for i in range(b):
                    cnt = 0
                    lis = []
                    maps = feature_map[i]
                    maps = maps.view(c,-1)
                    for j in range(c):
                        mapj = maps[j]
                        if(cam_weights[i][j]>0):
                            maxi, _ = mapj.topk(1,-1)
                            maxi = maxi * self.rate
                            lis.append(((mapj > maxi) + 0).view(1,h,w))
                    cams = torch.cat(lis, 0)
                    cams = cams.sum(axis=0)
                    cam_list.append(cams.view(1,h,w))
                cams = torch.cat(cam_list,0)
                return cams, logits

        return {'logits': logits}


class VggAcol(AcolBase):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(VggAcol, self).__init__()

        self.features = features
        self.drop_threshold = kwargs['acol_drop_threshold']
        self.rate = kwargs['binary_rate']
        self.mode = kwargs['mode']


        self.classifier_A = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, padding=0),
        )
        self.classifier_B = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, padding=0),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        initialize_weights(self.modules(), init_mode='xavier')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]

        feature = self.features(x)
        feature = F.avg_pool2d(feature, kernel_size=3, stride=1, padding=1)
        logits_dict = self._acol_logits(feature=feature, labels=labels,
                                        drop_threshold=self.drop_threshold)

        if return_cam:

            feature_map_A = self.classifier_A[:-1](feature).clone().detach()
            erased_feat = logits_dict['erased_feat'].clone().detach()
            feature_map_B = self.classifier_B[:-1](erased_feat)
            cam_weights_a = self.classifier_A[-1].weight[labels].squeeze()
            cam_weights_b = self.classifier_B[-1].weight[labels].squeeze()
           
            if self.mode != 'vote':

                if self.mode != 'origin':
                    cam_weights_a[cam_weights_a<0] = 0.
                    cam_weights_b[cam_weights_b<0] = 0.
                if self.mode == 'bin':
                    cam_weights_a[cam_weights_a>0] = 1.
                    cam_weights_b[cam_weights_b>0] = 1.
                
                cams_A = (cam_weights_a.view(*feature_map_A.shape[:2], 1, 1) *
                        feature_map_A).mean(1, keepdim=False)
                cams_B = (cam_weights_b.view(*feature_map_B.shape[:2], 1, 1) *
                        feature_map_B).mean(1, keepdim=False)

            else :
                cam_list_a = []
                cam_list_b = []
                cams = feature_map_A
                b, c, h, w = cams.shape
                for i in range(b):
                    lis_a = []
                    lis_b = []
                    maps_a = feature_map_A[i]
                    maps_a = maps_a.view(c,-1)
                    maps_b = feature_map_B[i]
                    maps_b = maps_b.view(c,-1)
                    for j in range(c):
                        mapj_a = maps_a[j]
                        if(cam_weights_a[i][j]>0):
                            maxi_a, _ = mapj_a.topk(1,-1)
                            maxi_a = maxi_a * self.rate
                            lis_a.append(((mapj_a > maxi_a) + 0.).view(1,h,w))
                        mapj_b = maps_b[j]
                        if(cam_weights_b[i][j]>0):
                            maxi_b, _ = mapj_b.topk(1,-1)
                            maxi_b = maxi_b * self.rate
                            lis_b.append(((mapj_b > maxi_b) + 0.).view(1,h,w))
                    cams_a = torch.cat(lis_a, 0)
                    cams_a = cams_a.sum(axis=0)
                    cam_list_a.append(cams_a.view(1,h,w))
                    cams_b = torch.cat(lis_b, 0)
                    cams_b = cams_b.sum(axis=0)
                    cam_list_b.append(cams_b.view(1,h,w))
                cams_A = torch.cat(cam_list_a,0)
                cams_B = torch.cat(cam_list_b,0)

            cams = cams_A + cams_B

            return cams, logits_dict['logits']

        return logits_dict


class VggSpg(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(VggSpg, self).__init__()

        self.features = features
        self.lfs = kwargs['large_feature_map']
        self.rate = kwargs['binary_rate']
        self.mode = kwargs['mode']

        self.SPG_A_1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.SPG_A_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.SPG_A_3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.SPG_A_4 = nn.Conv2d(512, num_classes, kernel_size=1, padding=0)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.SPG_B_1a = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.SPG_B_2a = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.SPG_B_shared = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=1),
        )

        self.SPG_C = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=1),
        )

        initialize_weights(self.modules(), init_mode='xavier')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]

        x = self.features(x)
        x = self.SPG_A_1(x)
        if not self.lfs:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        logits_b1 = self.SPG_B_1a(x)
        logits_b1 = self.SPG_B_shared(logits_b1)

        x = self.SPG_A_2(x)
        logits_b2 = self.SPG_B_2a(x)
        logits_b2 = self.SPG_B_shared(logits_b2)

        x = self.SPG_A_3(x)
        logits_c = self.SPG_C(x)

        feat_map = self.SPG_A_4(x)
        logits = self.avgpool(feat_map)
        logits = logits.flatten(1)

        labels = logits.argmax(dim=1).long() if labels is None else labels
        attention, fused_attention = spg.compute_attention(
            feat_map=feat_map, labels=labels,
            logits_b1=logits_b1, logits_b2=logits_b2)


        if return_cam:
            if self.mode != 'vote':
                feature_map = x.detach().clone()
                cam_weights = self.SPG_A_4.weight[labels].squeeze()
                if self.mode != 'origin':
                    cam_weights[cam_weights<0]=0.
                if self.mode == 'bin':
                    cam_weights[cam_weights>0]=1.
                cams = (cam_weights.view(*feature_map.shape[:2], 1, 1) *
                        feature_map).mean(1, keepdim=False)
                return cams, logits
        
            else:
                cam_list = []
                feature_map = x.detach().clone()
                cam_weights = self.SPG_A_4.weight[labels].squeeze()
                cams = feature_map
                b, c, h, w = cams.shape
                for i in range(b):
                    cnt = 0
                    lis = []
                    maps = feature_map[i]
                    maps = maps.view(c,-1)
                    for j in range(c):
                        mapj = maps[j]
                        if(cam_weights[i][j]>0):
                            maxi, _ = mapj.topk(1,-1)
                            maxi = maxi * self.rate
                            lis.append(((mapj > maxi) + 0).view(1,h,w))
                    cams = torch.cat(lis, 0)
                    cams = cams.sum(axis=0)
                    cam_list.append(cams.view(1,h,w))
                cams = torch.cat(cam_list,0)
                return cams, logits


        return {'attention': attention, 'fused_attention': fused_attention,
                'logits': logits, 'logits_b1': logits_b1,
                'logits_b2': logits_b2, 'logits_c': logits_c}


def adjust_pretrained_model(pretrained_model, current_model):
    def _get_keys(obj, split):
        keys = []
        iterator = obj.items() if split == 'pretrained' else obj
        for key, _ in iterator:
            if key.startswith('features.'):
                keys.append(int(key.strip().split('.')[1].strip()))
        return sorted(list(set(keys)), reverse=True)

    def _align_keys(obj, key1, key2):
        for suffix in ['.weight', '.bias']:
            old_key = 'features.' + str(key1) + suffix
            new_key = 'features.' + str(key2) + suffix
            obj[new_key] = obj.pop(old_key)
        return obj

    pretrained_keys = _get_keys(pretrained_model, 'pretrained')
    current_keys = _get_keys(current_model.named_parameters(), 'model')

    for p_key, c_key in zip(pretrained_keys, current_keys):
        pretrained_model = _align_keys(pretrained_model, p_key, c_key)

    return pretrained_model


def batch_replace_layer(state_dict):
    state_dict = replace_layer(state_dict, 'features.17', 'SPG_A_1.0')
    state_dict = replace_layer(state_dict, 'features.19', 'SPG_A_1.2')
    state_dict = replace_layer(state_dict, 'features.21', 'SPG_A_1.4')
    state_dict = replace_layer(state_dict, 'features.24', 'SPG_A_2.0')
    state_dict = replace_layer(state_dict, 'features.26', 'SPG_A_2.2')
    state_dict = replace_layer(state_dict, 'features.28', 'SPG_A_2.4')
    return state_dict


def load_pretrained_model(model, architecture_type, path=None):
    if path is not None:
        state_dict = torch.load(os.path.join(path, 'vgg16.pth'))
    else:
        state_dict = load_url(model_urls['vgg16'], progress=True)

    if architecture_type == 'spg':
        state_dict = batch_replace_layer(state_dict)
    state_dict = remove_layer(state_dict, 'classifier.')
    state_dict = adjust_pretrained_model(state_dict, model)

    model.load_state_dict(state_dict, strict=False)
    return model


def make_layers(cfg, **kwargs):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M1':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        elif v == 'M2':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        elif v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'A':
            layers += [
                ADL(kwargs['adl_drop_rate'], kwargs['adl_drop_threshold'])]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg16(architecture_type, pretrained=False, pretrained_path=None,
          **kwargs):
    config_key = '28x28' if kwargs['large_feature_map'] else '14x14'
    layers = make_layers(configs_dict[architecture_type][config_key], **kwargs)
    model = {'cam': VggCam,
             'acol': VggAcol,
             'spg': VggSpg,
             'adl': VggCam}[architecture_type](layers, **kwargs)
    if pretrained:
        model = load_pretrained_model(model, architecture_type,
                                      path=pretrained_path)
    return model
