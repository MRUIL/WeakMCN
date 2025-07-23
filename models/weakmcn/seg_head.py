# coding=utf-8
# Copyright 2022 The SimREC Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from .layer import aspp_decoder
import torch.nn.functional as F
from utils.utils import compute_project_term, mask_iou_batch

class REShead(nn.Module):
    """
    segmentation layer
    """

    def __init__(self, in_ch, input_img_size, iou_thresh):
        super(REShead, self).__init__()
        self.sconv = nn.Sequential(aspp_decoder(in_ch, in_ch, 1),
                                   nn.UpsamplingBilinear2d(scale_factor=8)
                                   )
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        self.iou_thres = iou_thresh
        self.scale_factor_h, self.scale_factor_w = input_img_size

    def forward(self, yin, x_label=None, y_label=None, pred_boxes=None, epoch=None):
        seg_weight = None
        batchsize = yin.shape[0]
        mask = self.sconv(yin)
        if x_label is None:  # not training
            mask = (mask.sigmoid() > 0.5).float().squeeze(1)
            box = torch.zeros(batchsize, 5, device=yin.device)
            return box, mask
        if pred_boxes is not None:
            mask_score = torch.sigmoid(mask)
            per_im_bitmasks = []
            pred_boxes = pred_boxes.squeeze(1)
            for per_box in pred_boxes:
                bitmask = torch.zeros((self.scale_factor_h, self.scale_factor_w), device=per_box.device).float()
                bitmask[int(per_box[1]):int(per_box[3] + 1), int(per_box[0]):int(per_box[2] + 1)] = 1.0
                per_im_bitmasks.append(bitmask)
            box_bitmasks = torch.stack(per_im_bitmasks, dim=0).unsqueeze(1)
            if epoch >= 15:
                iou_mask = mask_iou_batch(box_bitmasks, mask.sigmoid())
                seg_weight = iou_mask > self.iou_thres
            loss_prj_term = compute_project_term(mask_score, box_bitmasks)
        if seg_weight is not None:
            loss_seg = F.binary_cross_entropy_with_logits(mask, y_label, reduction='none')
            # Apply weights to each sample in the batch
            loss_seg = loss_seg * seg_weight.view(-1, 1, 1, 1)  # Reshape seg_weight to match loss dimensions
            # Average the loss over the batch
            loss_seg = loss_seg.mean() * batchsize
        else:
            loss_seg = F.binary_cross_entropy_with_logits(mask, y_label, reduction='mean') * batchsize
        loss_seg = loss_seg + loss_prj_term
        return loss_seg
