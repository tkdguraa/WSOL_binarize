"""
Original repository: https://github.com/xiaomengyc/ACoL
"""

import torch
import torch.nn as nn

from .util import get_attention

__all__ = ['AcolBase']


class AcolBase(nn.Module):
    def _acol_logits(self, feature, labels, drop_threshold, return_cam):
        feat_map_a, logits = self._branch(feature=feature,
                                          classifier=self.classifier_A, return_cam=return_cam, labels=labels)
        labels = logits.argmax(dim=1).long() if labels is None else labels
        attention = get_attention(feature=feat_map_a, label=labels, return_cam=return_cam)
        erased_feature = _erase_attention(
            feature=feature, attention=attention, drop_threshold=drop_threshold)
        feat_map_b, logit_b = self._branch(feature=erased_feature,
                                           classifier=self.classifier_B, return_cam=return_cam, labels=labels)
        return {'logits': logits, 'logit_b': logit_b,
                'feat_map_a': feat_map_a, 'feat_map_b': feat_map_b}

    def _branch(self, feature, classifier, return_cam, labels):
        feature = classifier[0](feature)
        feature = classifier[1](feature)
        feature = classifier[2](feature)
        feature = classifier[3](feature)
        feat_map = classifier[4](feature)
        logits = self.avgpool(feat_map)
        logits = logits.view(logits.size(0), -1)


        if return_cam:
          cam_pos_list = []
          cam_weights = classifier[4].weight[labels].squeeze()
          cam_weights = cam_weights.view(-1, 1024)
          feature_map = feature.detach().clone()
          b, c, h, w = feature_map.shape
          for i in range(b):
              lis_pos = []
              maps = feature_map[i]
              maps = maps.view(c,-1)
              for j in range(c):
                  mapj = maps[j]
                  if(cam_weights[i][j]>0):
                      #maxi, _ = mapj.topk(1,-1)
                      #maxi = maxi * 0.6
                      #lis_pos.append(((mapj > maxi) + 0).view(1,h,w))
                      lis_pos.append(mapj.view(1,h,w))
              cams_pos = torch.cat(lis_pos, 0)
              pos = cams_pos.sum(axis=0)
              cam_pos_list.append(pos.view(1,h,w))
          cams_pos = torch.cat(cam_pos_list,0)
          #for i in range(b):
          #  cm_pos = cams_pos[i]
          #  w, d = cm_pos.shape
          #  cm_pos = cm_pos.view(-1)
          #  maxi_pos = cm_pos.max()
          #  maxi_pos = maxi_pos * 0.2 #0.25 is best now
          #  idx_pos = ((cm_pos > maxi_pos) + 0)
          #  cams_pos[i] = cams_pos[i] * idx_pos.view(w, d) 
          #cams = cams_pos
          return cams_pos, logits


        return feat_map, logits


def _erase_attention(feature, attention, drop_threshold):
    b, _, h, w = attention.size()
    pos = torch.ge(attention, drop_threshold)
    mask = attention.new_ones((b, 1, h, w))
    mask[pos.data] = 0.
    erased_feature = feature * mask
    return erased_feature


def get_loss(output_dict, gt_labels, **kwargs):
    return nn.CrossEntropyLoss()(output_dict['logits'], gt_labels.long()) + \
           nn.CrossEntropyLoss()(output_dict['logit_b'], gt_labels.long())
