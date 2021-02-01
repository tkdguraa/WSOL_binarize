"""
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import cv2
import numpy as np
import os
from os.path import join as ospj
from os.path import dirname as ospd

from evaluation_top1 import BoxEvaluator
from evaluation_top1 import MaskEvaluator
from evaluation_top1 import configure_metadata
from util import t2n

_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
_RESIZE_LENGTH = 224


def normalize_scoremap(cam):
    """
    Args:
        cam: numpy.ndarray(size=(H, W), dtype=np.float)
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(cam).any():
        return np.zeros_like(cam)
    if cam.min() == cam.max():
        return np.zeros_like(cam)
    cam -= cam.min()
    cam /= cam.max()
    return cam


class CAMComputer(object):
    def __init__(self, model, loader, metadata_root, mask_root,
                 iou_threshold_list, dataset_name, split,
                 multi_contour_eval, cam_curve_interval=.001, log_folder=None):
        self.model = model
        self.model.eval()
        self.loader = loader
        self.split = split
        self.log_folder = log_folder

        metadata = configure_metadata(metadata_root)
        cam_threshold_list = list(np.arange(0, 1, cam_curve_interval))

        self.evaluator = {"OpenImages": MaskEvaluator,
                          "CUB": BoxEvaluator,
                          "ILSVRC": BoxEvaluator
                          }[dataset_name](metadata=metadata,
                                          dataset_name=dataset_name,
                                          split=split,
                                          cam_threshold_list=cam_threshold_list,
                                          iou_threshold_list=iou_threshold_list,
                                          mask_root=mask_root,
                                          multi_contour_eval=multi_contour_eval)

    def compute_and_evaluate_cams(self):
        print("Computing and evaluating cams.")
        cnt = 0
        for images, targets, image_ids in self.loader:
            cnt = cnt + 1
            image_size = images.shape[2:]
            images = images.cuda()
            #modified
            cams, logits = self.model(images, targets, return_cam=True)
            #cams, logits, tps = self.model(images, targets, return_cam=True)
            #np.save("clustering/" + str(cnt), t2n(tps))
            cams = t2n(cams)
            #channel = t2n(channel)
            #weight = t2n(weight)

            for cam, image_id, logit, target in zip(cams, image_ids, logits, targets):
                cam_resized = cv2.resize(cam, image_size,
                                         interpolation=cv2.INTER_CUBIC)
                cam_normalized = normalize_scoremap(cam_resized)
                #for i in range(0, 2048):
                #   ch = cv2.resize(chl[i], image_size, interpolation=cv2.INTER_CUBIC)
                #   np.save("channels2/" + image_id.replace("/","") + str(i), ch)

                #np.save("channels2/" + image_id.replace("/","") + "weight", w)

                if self.split in ('val', 'test'):
                    cam_path = ospj(self.log_folder, 'scoremaps', image_id)
                    if not os.path.exists(ospd(cam_path)):
                        os.makedirs(ospd(cam_path))
                    np.save(ospj(cam_path), cam_normalized)
                _, _top5 = logit.topk(5,dim=0)
                _top1 = _top5[0]
                top5 = (_top5 == target).sum().item()
                top1 = (_top1 == target).sum().item()
                self.evaluator.accumulate(cam_normalized, image_id, top1, top5)
            """
            for feature_map, image_id in zip(feature_maps, image_ids):
                acc = np.zeros(feature_map[0].shape)
                c = feature_map.shape[0]
                for i in range(0, c):
                  cam = t2n(feature_map[i])
                  acc = acc + (cam > cam.max() * 0.15) + 0
                  cam_resized = cv2.resize(acc, image_size,
                                           interpolation=cv2.INTER_CUBIC)
                  #cam_normalized = normalize_scoremap(cam_resized)
                  cam_normalized = acc
                  if self.split in ('val', 'test'):
                      cam_path = ospj(self.log_folder, 'scoremaps', image_id)
                      if not os.path.exists(ospd(cam_path)):
                          os.makedirs(ospd(cam_path))
                      np.save(ospj(cam_path + "_" + str(i)), cam_normalized)
             """

        return self.evaluator.compute()
