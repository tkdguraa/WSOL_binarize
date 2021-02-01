"""
Original code: https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url

from .method import AcolBase
from .method import ADL
from .method import normalize_tensor
from .method import spg
from .util import initialize_weights
from .util import remove_layer

__all__ = ['inception_v3']

model_urls = {
    'inception_v3_google':
        'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, 1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, 1)
        self.branch5x5_2 = BasicConv2d(48, 64, 5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, 1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, 3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, 3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, 1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2, padding=0):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size,
                                     stride=stride, padding=padding)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, 1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, 3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, 3,
                                          stride=stride, padding=padding)

        self.stride = stride

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=self.stride,
                                   padding=1)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, 1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, 1)
        self.branch7x7_2 = BasicConv2d(c7, c7, (1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, (7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, 1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, (7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, (1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, (7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, (1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, 1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionCam(nn.Module):
    def __init__(self, num_classes=1000, large_feature_map=False, **kwargs):
        super(InceptionCam, self).__init__()


        self.rate = kwargs['binary_rate']
        self.mode = kwargs['mode']

        self.large_feature_map = large_feature_map

        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, 3, stride=2, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, 3, stride=1, padding=0)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, 3, stride=1, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, 1, stride=1, padding=0)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, 3, stride=1, padding=0)

        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)

        self.Mixed_6a = InceptionB(288, kernel_size=3, stride=1, padding=1)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        self.SPG_A3_1b = nn.Sequential(
            nn.Conv2d(768, 1024, 3, padding=1),
            nn.ReLU(True),
        )
        self.SPG_A3_2b = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU(True),
        )
        self.SPG_A4 = nn.Conv2d(1024, num_classes, 1, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        initialize_weights(self.modules(), init_mode='xavier')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]

        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)

        if not self.large_feature_map:
            x = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)

        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        feat = self.Mixed_6e(x)

        x = F.dropout(feat, 0.5, self.training)
        x = self.SPG_A3_1b(x)
        x = F.dropout(x, 0.5, self.training)
        x = self.SPG_A3_2b(x)
        x = F.dropout(x, 0.5, self.training)
        feat_map = self.SPG_A4(x)
        logits = self.avgpool(feat_map)
        logits = logits.view(logits.shape[0:2])

        if return_cam:
            if self.mode != 'vote':
                feature_map = x.detach().clone()
                cam_weights = self.SPG_A4.weight[labels].squeeze()
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
                cam_weights = self.SPG_A4.weight[labels].squeeze()
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
        #    #feature_map = feat_map.clone().detach()
        #    #cams = feature_map[range(batch_size), labels]
        #    #return cams, logits

        #    cam_pos_list = []
        #    feature_map = x.detach().clone()
        #    cam_weights = self.SPG_A4.weight[labels].squeeze()
        #    cam_weights[cam_weights<0] = 0.
        #    cams = feature_map
        #    b, c, h, w = cams.shape

            #for i in range(b):
            #    lis_pos = []
            #    maps = feature_map[i]
            #    maps = maps.view(c,-1)
            #    for j in range(c):
            #        mapj = maps[j]
            #        if(cam_weights[i][j]>0):
            #            #lis_pos.append((cam_weights[i][j] * mapj).view(1,h,w))
            #            maxi, _ = mapj.topk(1,-1)
            #            maxi = maxi * 0.15
            #            lis_pos.append(((mapj > maxi) + 0).view(1,h,w))
            #            #lis_pos.append(mapj.view(1,h,w))
            #    cams_pos = torch.cat(lis_pos, 0)
            #    pos = cams_pos.sum(axis=0)
            #    cam_pos_list.append(pos.view(1,h,w))
            #cams_pos = torch.cat(cam_pos_list,0)
            #for i in range(b):
            #  cm_pos = cams_pos[i]
            #  w, d = cm_pos.shape
            #  cm_pos = cm_pos.view(-1)
            #  maxi_pos = cm_pos.max()
            #  maxi_pos = maxi_pos * 0.2 #0.25 is best now
            #  idx_pos = ((cm_pos > maxi_pos) + 0)
            #  cams_pos[i] = cams_pos[i] * idx_pos.view(w, d) 
            #cams = cams_pos


            #cams = (cam_weights.view(*feature_map.shape[:2], 1, 1) *
            #        feature_map).mean(1, keepdim=False)

            #return cams

        return {'logits': logits}

    def get_loss(self, logits, target):
        loss_cls = nn.CrossEntropyLoss()(logits, target.long())
        return loss_cls


class InceptionAcol(AcolBase):
    def __init__(self, num_classes=1000, large_feature_map=False, **kwargs):
        super(InceptionAcol, self).__init__()

        self.rate = kwargs['binary_rate']
        self.mode = kwargs['mode']


        self.large_feature_map = large_feature_map

        self.drop_threshold = kwargs['acol_drop_threshold']

        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, 3, stride=2, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, 3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, 3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, 1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, 3)

        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)

        self.Mixed_6a = InceptionB(288, kernel_size=3, stride=1, padding=1)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        self.classifier_A = nn.Sequential(
            nn.Conv2d(768, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(1024, num_classes, kernel_size=1, padding=0)
        )
        self.classifier_B = nn.Sequential(
            nn.Conv2d(768, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(1024, num_classes, kernel_size=1, padding=0)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        initialize_weights(self.modules(), init_mode='xavier')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]

        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)

        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        if not self.large_feature_map:
            x = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)

        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        feature = self.Mixed_6e(x)

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


class InceptionSpg(nn.Module):
    def __init__(self, num_classes=1000, large_feature_map=False, **kwargs):
        super(InceptionSpg, self).__init__()


        self.rate = kwargs['binary_rate']
        self.mode = kwargs['mode']

        self.large_feature_map = large_feature_map

        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, 3, stride=2, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, 3, stride=1, padding=0)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, 3, stride=1, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, 1, stride=1, padding=0)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, 3, stride=1, padding=0)

        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)

        self.Mixed_6a = InceptionB(288, kernel_size=3, stride=1, padding=1)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        self.SPG_A3_1b = nn.Sequential(
            nn.Conv2d(768, 1024, kernel_size=3, padding=1),
            nn.ReLU(True),
        )
        self.SPG_A3_2b = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(True),
        )

        self.SPG_A4 = nn.Conv2d(1024, num_classes, kernel_size=1, padding=0)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.SPG_B_1a = nn.Sequential(
            nn.Conv2d(288, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.SPG_B_2a = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.SPG_B_shared = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=1, padding=0),
        )
        self.SPG_C = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=1),
        )

        initialize_weights(self.modules(), init_mode='xavier')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]

        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)

        if not self.large_feature_map:
            x = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)

        logits_b1 = self.SPG_B_1a(x)
        logits_b1 = self.SPG_B_shared(logits_b1)

        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        feat = self.Mixed_6e(x)

        logits_b2 = self.SPG_B_2a(x)
        logits_b2 = self.SPG_B_shared(logits_b2)

        x = F.dropout(feat, 0.5, self.training)
        x = self.SPG_A3_1b(x)
        x = F.dropout(x, 0.5, self.training)
        x = self.SPG_A3_2b(x)
        x = F.dropout(x, 0.5, self.training)
        feat_map = self.SPG_A4(x)

        logits_c = self.SPG_C(x)

        logits = self.avgpool(feat_map)
        logits = logits.view(logits.shape[0:2])

        labels = logits.argmax(dim=1).long() if labels is None else labels
        attention, fused_attention = spg.compute_attention(
            feat_map=feat_map, labels=labels,
            logits_b1=logits_b1, logits_b2=logits_b2)

        if return_cam:
            if self.mode != 'vote':
                feature_map = x.detach().clone()
                cam_weights = self.SPG_A4.weight[labels].squeeze()
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
                cam_weights = self.SPG_A4.weight[labels].squeeze()
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
                return cams, logits

        return {'attention': attention, 'fused_attention': fused_attention,
                'logits': logits, 'logits_b1': logits_b1,
                'logits_b2': logits_b2, 'logits_c': logits_c}


class InceptionAdl(nn.Module):
    def __init__(self, num_classes=1000, large_feature_map=False, **kwargs):
        super(InceptionAdl, self).__init__()
        self.large_feature_map = large_feature_map

        self.rate = kwargs['binary_rate']
        self.mode = kwargs['mode']

        self.adl_drop_rate = kwargs['adl_drop_rate']
        self.adl_threshold = kwargs['adl_drop_threshold']

        self.ADL_5d = ADL(self.adl_drop_rate, self.adl_threshold)
        self.ADL_6e = ADL(self.adl_drop_rate, self.adl_threshold)
        self.ADL_A3_2b = ADL(self.adl_drop_rate, self.adl_threshold)

        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, 3, stride=2, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, 3, stride=1, padding=0)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, 3, stride=1, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, 1, stride=1, padding=0)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, 3, stride=1, padding=0)

        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)

        self.Mixed_6a = InceptionB(288, kernel_size=3, stride=1, padding=1)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        self.SPG_A3_1b = nn.Sequential(
            nn.Conv2d(768, 1024, 3, padding=1),
            nn.ReLU(True),
        )
        self.SPG_A3_2b = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU(True),
        )

        self.SPG_A4 = nn.Conv2d(1024, num_classes, 1, padding=0)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        initialize_weights(self.modules(), init_mode='xavier')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]

        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)

        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.ADL_5d(x)
        if not self.large_feature_map:
            x = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)

        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.ADL_6e(x)

        x = self.SPG_A3_1b(x)
        feat = self.SPG_A3_2b(x)
        x = self.ADL_A3_2b(feat)
        feat_map = self.SPG_A4(x)

        logits = self.avgpool(feat_map)
        logits = logits.view(logits.shape[0:2])



        if return_cam:
            if self.mode != 'vote':
                feature_map = x.detach().clone()
                cam_weights = self.SPG_A4.weight[labels].squeeze()
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
                cam_weights = self.SPG_A4.weight[labels].squeeze()
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


def load_pretrained_model(model, path=None):
    if path:
        state_dict = torch.load(
            os.path.join(path, 'inception_v3.pth'))
    else:
        state_dict = load_url(model_urls['inception_v3_google'],
                              progress=True)

    remove_layer(state_dict, 'Mixed_7')
    remove_layer(state_dict, 'AuxLogits')
    remove_layer(state_dict, 'fc.')

    model.load_state_dict(state_dict, strict=False)
    return model


def inception_v3(architecture_type, pretrained=False, pretrained_path=None,
                 **kwargs):
    model = {'cam': InceptionCam,
             'acol': InceptionAcol,
             'spg': InceptionSpg,
             'adl': InceptionAdl}[architecture_type](**kwargs)
    if pretrained:
        model = load_pretrained_model(model, pretrained_path)
    return model

