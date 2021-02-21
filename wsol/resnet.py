"""
Original code: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""

import os
import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url

from .method import AcolBase
from .method import ADL
from .method import spg
from .method.util import normalize_tensor
from .util import remove_layer
from .util import replace_layer
from .util import initialize_weights

__all__ = ['resnet50']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

_ADL_POSITION = [[], [], [], [0], [0, 2]]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.))
        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetCam(nn.Module):
    def __init__(self, block, layers, num_classes=1000,
                 large_feature_map=False, **kwargs):
        super(ResNetCam, self).__init__()


        self.rate = kwargs['binary_rate']
        self.mode = kwargs['mode']

        stride_l3 = 1 if large_feature_map else 2
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride_l3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        initialize_weights(self.modules(), init_mode='xavier')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        pre_logit = self.avgpool(x)
        pre_logit = pre_logit.reshape(pre_logit.size(0), -1)
        logits = self.fc(pre_logit)
        if self.mode != 'vote':
            if return_cam:
                feature_map = x.detach().clone()
                cam_weights = self.fc.weight[labels].squeeze()
                if self.mode != 'origin':
                    cam_weights[cam_weights<0]=0.
                if self.mode == 'bin':
                    cam_weights[cam_weights>0]=1.
                cams = (cam_weights.view(*feature_map.shape[:2], 1, 1) *
                        feature_map).mean(1, keepdim=False)
                return cams, logits
                #cam_list = []
                #total_chl = []
                #feature_map = x.detach().clone()
                #cam_weights = self.fc.weight[labels].squeeze()
                #sorted_weight, weight_idx = cam_weights.topk(2048, -1)
                #cams = feature_map
                #b, c, h, w = cams.shape
                #for i in range(b):
                #    cnt = 0
                #    lis = []
                #    chl = []
                #    maps = feature_map[i]
                #    maps = maps.view(c,-1)
                #    for j in range(c):
                #        idx = weight_idx[i][j]
                #        mapj = maps[idx]
                #        #if(cam_weights[i][idx]>0):
                #        chl.append(mapj.view(1,h,w))
                #        maxi, _ = mapj.topk(1,-1)
                #        maxi = maxi * self.rate
                #        lis.append(((mapj > maxi) + 0).view(1,h,w))
                #    cams = torch.cat(lis, 0)
                #    channel = torch.cat(chl, 0)
                #    cams = cams.sum(axis=0)
                #    total_chl.append(channel.view(1, 2048, h, w))
                #    cam_list.append(cams.view(1,h,w))
                #cams = torch.cat(cam_list,0)
                #channel = torch.cat(total_chl, 0)
                #return cams, logits, channel, sorted_weight
        else:
            if return_cam:
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



        #if return_cam:
        #    pos_weight_list = []
        #    neg_weight_list = []
        #    cam_weights = self.fc.weight[labels]
        #    for i in range(batch_size):
        #        num_pos = ((cam_weights[i]>0)+0.).sum()
        #        num_neg = len(cam_weights[i]) - num_pos
        #        pos_weight_list.append(num_pos)
        #        neg_weight_list.append(num_neg)
        #    return pos_weight_list, neg_weight_list

        #    return cams
        return {'logits': logits}

    def _make_layer(self, block, planes, blocks, stride):
        layers = self._layer(block, planes, blocks, stride)
        return nn.Sequential(*layers)

    def _layer(self, block, planes, blocks, stride):
        downsample = get_downsampling_layer(self.inplanes, block, planes,
                                            stride)

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers


class ResNetAcol(AcolBase):
    def __init__(self, block, layers, num_classes=1000,
                 large_feature_map=False, **kwargs):
        super(ResNetAcol, self).__init__()

        stride_l3 = 1 if large_feature_map else 2
        self.inplanes = 64

        self.label = None
        self.drop_threshold = kwargs['acol_drop_threshold']

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride_l3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        self.classifier_A = nn.Sequential(
            nn.Conv2d(512 * block.expansion, 1024, 3, 1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, 1024, 3, 1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, num_classes, 1, 1, padding=0),
        )
        self.classifier_B = nn.Sequential(
            nn.Conv2d(512 * block.expansion, 1024, 3, 1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, 1024, 3, 1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, num_classes, 1, 1, padding=0),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature = self.layer4(x)

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
                cams = cams_A + cams_B

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

    def _make_layer(self, block, planes, blocks, stride):
        layers = self._layer(block, planes, blocks, stride)
        return nn.Sequential(*layers)

    def _layer(self, block, planes, blocks, stride):
        downsample = get_downsampling_layer(self.inplanes, block, planes,
                                            stride)

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers


class ResNetSpg(nn.Module):
    def __init__(self, block, layers, num_classes=1000,
                 large_feature_map=False, **kwargs):
        super(ResNetSpg, self).__init__()

        self.rate = kwargs['binary_rate']
        self.mode = kwargs['mode']


        stride_l3 = 1 if large_feature_map else 2
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block=block, planes=64,
                                       blocks=layers[0],
                                       stride=1, split=False)
        self.layer2 = self._make_layer(block=block, planes=128,
                                       blocks=layers[1],
                                       stride=2, split=False)
        self.SPG_A1, self.SPG_A2 = self._make_layer(block=block, planes=256,
                                                    blocks=layers[2],
                                                    stride=stride_l3,
                                                    split=True)
        self.layer4 = self._make_layer(block=block, planes=512,
                                       blocks=layers[3],
                                       stride=1, split=False)
        self.SPG_A4 = nn.Conv2d(512 * block.expansion, num_classes,
                                kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.SPG_B_1a = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.SPG_B_2a = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.SPG_B_shared = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1, kernel_size=1),
        )

        self.SPG_C = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=1),
        )

        initialize_weights(self.modules(), init_mode='xavier')

    def _make_layer(self, block, planes, blocks, stride, split=None):
        downsample = get_downsampling_layer(self.inplanes, block, planes,
                                            stride)

        first_layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        other_layers = []
        for _ in range(1, blocks):
            other_layers.append(block(self.inplanes, planes))

        if split:
            return nn.Sequential(*first_layers), nn.Sequential(*other_layers)
        else:
            return nn.Sequential(*(first_layers + other_layers))

    def _layer(self, block, planes, blocks, stride):
        downsample = get_downsampling_layer(self.inplanes, block, planes,
                                            stride)

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.SPG_A1(x)

        logits_b1 = self.SPG_B_1a(x)
        logits_b1 = self.SPG_B_shared(logits_b1)

        x = self.SPG_A2(x)
        logits_b2 = self.SPG_B_2a(x)
        logits_b2 = self.SPG_B_shared(logits_b2)

        x = self.layer4(x)
        feat_map = self.SPG_A4(x)

        logits_c = self.SPG_C(x)

        logits = self.avgpool(feat_map)
        logits = logits.view(logits.shape[0:2])

        labels = logits.argmax(dim=1).long() if labels is None else labels
        attention, fused_attention = spg.compute_attention(
            feat_map=feat_map, labels=labels,
            logits_b1=logits_b1, logits_b2=logits_b2)

        if self.mode != 'vote':
            if return_cam:
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
            if return_cam:
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


        return {'attention': attention, 'fused_attention': fused_attention,
                'logits': logits, 'logits_b1': logits_b1,
                'logits_b2': logits_b2, 'logits_c': logits_c}


class ResNetAdl(nn.Module):
    def __init__(self, block, layers, num_classes=1000,
                 large_feature_map=False, **kwargs):
        super(ResNetAdl, self).__init__()

        self.rate = kwargs['binary_rate']
        self.mode = kwargs['mode']


        self.stride_l3 = 1 if large_feature_map else 2
        self.inplanes = 64

        self.adl_drop_rate = kwargs['adl_drop_rate']
        self.adl_threshold = kwargs['adl_drop_threshold']

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0],
                                       stride=1,
                                       split=_ADL_POSITION[1])
        self.layer2 = self._make_layer(block, 128, layers[1],
                                       stride=2,
                                       split=_ADL_POSITION[2])
        self.layer3 = self._make_layer(block, 256, layers[2],
                                       stride=self.stride_l3,
                                       split=_ADL_POSITION[3])
        self.layer4 = self._make_layer(block, 512, layers[3],
                                       stride=1,
                                       split=_ADL_POSITION[4])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        initialize_weights(self.modules(), init_mode='xavier')

    def forward(self, x, labels=None, return_cam=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        pre_logit = self.avgpool(x)
        pre_logit = pre_logit.reshape(pre_logit.size(0), -1)
        logits = self.fc(pre_logit)

        if self.mode != 'vote':
            if return_cam:
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
            if return_cam:
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

    def _make_layer(self, block, planes, blocks, stride, split=None):
        layers = self._layer(block, planes, blocks, stride)
        for pos in reversed(split):
            layers.insert(pos + 1, ADL(self.adl_drop_rate, self.adl_threshold))
        return nn.Sequential(*layers)

    def _layer(self, block, planes, blocks, stride):
        downsample = get_downsampling_layer(self.inplanes, block, planes,
                                            stride)

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return layers


def get_downsampling_layer(inplanes, block, planes, stride):
    outplanes = planes * block.expansion
    if stride == 1 and inplanes == outplanes:
        return
    else:
        return nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1, stride, bias=False),
            nn.BatchNorm2d(outplanes),
        )


def align_layer(state_dict):
    keys = [key for key in sorted(state_dict.keys())]
    for key in reversed(keys):
        move = 0
        if 'layer' not in key:
            continue
        key_sp = key.split('.')
        layer_idx = int(key_sp[0][-1])
        block_idx = key_sp[1]
        if not _ADL_POSITION[layer_idx]:
            continue

        for pos in reversed(_ADL_POSITION[layer_idx]):
            if pos < int(block_idx):
                move += 1

        key_sp[1] = str(int(block_idx) + move)
        new_key = '.'.join(key_sp)
        state_dict[new_key] = state_dict.pop(key)
    return state_dict


def batch_replace_layer(state_dict):
    state_dict = replace_layer(state_dict, 'layer3.0.', 'SPG_A1.0.')
    state_dict = replace_layer(state_dict, 'layer3.1.', 'SPG_A2.0.')
    state_dict = replace_layer(state_dict, 'layer3.2.', 'SPG_A2.1.')
    state_dict = replace_layer(state_dict, 'layer3.3.', 'SPG_A2.2.')
    state_dict = replace_layer(state_dict, 'layer3.4.', 'SPG_A2.3.')
    state_dict = replace_layer(state_dict, 'layer3.5.', 'SPG_A2.4.')
    return state_dict


def load_pretrained_model(model, wsol_method, path=None, **kwargs):
    strict_rule = True

    if path:
        state_dict = torch.load(os.path.join(path, 'resnet50.pth'))
    else:
        state_dict = load_url(model_urls['resnet50'], progress=True)

    if wsol_method == 'adl':
        state_dict = align_layer(state_dict)
    elif wsol_method == 'spg':
        state_dict = batch_replace_layer(state_dict)

    if kwargs['dataset_name'] != 'ILSVRC' or wsol_method in ('acol', 'spg'):
        state_dict = remove_layer(state_dict, 'fc')
        strict_rule = False

    model.load_state_dict(state_dict, strict=strict_rule)
    return model


def resnet50(architecture_type, pretrained=False, pretrained_path=None,
             **kwargs):
    model = {'cam': ResNetCam,
             'acol': ResNetAcol,
             'spg': ResNetSpg,
             'adl': ResNetAdl}[architecture_type](Bottleneck, [3, 4, 6, 3],
                                                  **kwargs)
    if pretrained:
        model = load_pretrained_model(model, architecture_type,
                                      path=pretrained_path, **kwargs)
    return model
