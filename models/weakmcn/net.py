# coding=utf-8
import torch
import torch.nn as nn
from models.language_encoder import language_encoder
from models.visual_encoder import visual_encoder
from models.weakmcn.head import WeakREChead
from models.network_blocks import MultiScaleFusion, SimpleFusion, GaranAttention
from models.weakmcn.seg_head import REShead
from EfficientSAM.efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
from utils.utils import  clip_boxes_to_image
import math
import torch.nn.functional as F
from transformers import Dinov2Model


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=256, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, positions):
        y_embed = positions[:, :, 1:] * self.scale
        x_embed = positions[:, :, :1] * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=positions.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)

        # dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :] / dim_t
        pos_y = y_embed[:, :, :] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2)
        return pos


class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size):
        super(Net, self).__init__()
        self.select_num = __C.SELECT_NUM
        self.visualize = False
        self.visual_encoder = visual_encoder(__C).eval()
        self.lang_encoder = language_encoder(__C, pretrained_emb, token_size)
        self.scale_factor_h, self.scale_factor_w = __C.INPUT_SHAPE
        self.num_points = __C.NUM_POINTS

        self.linear_vs = nn.Linear(__C.WREC_DIM, __C.HIDDEN_SIZE)
        self.linear_ts = nn.Linear(512, __C.HIDDEN_SIZE)
        self.head = WeakREChead(__C)
        self.seg_head = REShead(__C.WRES_DIM,  __C.INPUT_SHAPE, __C.IOU_THRESH)
        self.multi_scale_manner = MultiScaleFusion(v_planes=(256, 512, 1024), hiden_planes=__C.WREC_DIM, scaled=True)
        self.multi_scale_manner_sup = MultiScaleFusion(v_planes=(__C.WRES_DIM, __C.WRES_DIM, __C.WRES_DIM), hiden_planes=__C.WRES_DIM,
                                                       scaled=True)
        self.fusion_manner = nn.ModuleList(
            [
                SimpleFusion(v_planes=__C.WREC_DIM, out_planes=__C.WRES_DIM, q_planes=512),
                SimpleFusion(v_planes=__C.WREC_DIM, out_planes=__C.WRES_DIM, q_planes=512),
                SimpleFusion(v_planes=__C.WREC_DIM, out_planes=__C.WRES_DIM, q_planes=512)
            ]
        )
        self.attention_manner = GaranAttention(512, __C.WRES_DIM)

        # load ESAM model
        if __C.USE_VITS:
            self.efficientsam = build_efficient_sam_vits()
        else:
            self.efficientsam = build_efficient_sam_vitt()
        self.linear_sam_rec = nn.Linear(256, __C.WREC_DIM)
        self.linear_sam_res = nn.Linear(256, __C.WREC_DIM)

        # load DINO model
        self.dino_model = Dinov2Model.from_pretrained('facebook/dinov2-base')
        self.linear_dino_rec = nn.Linear(768, __C.WREC_DIM)
        self.linear_dino_res = nn.Linear(768, __C.WREC_DIM)

        self.linear_router_rec = nn.Linear(__C.WREC_DIM, 2)
        self.linear_router_res = nn.Linear(__C.WREC_DIM, 2)

        self.class_num = __C.CLASS_NUM
        self.pixel_mean = torch.tensor(__C.MEAN).view(-1, 1, 1)
        self.pixel_std = torch.tensor(__C.STD).view(-1, 1, 1)
        self.pos_encoder = PositionEmbeddingSine()

        if __C.VIS_FREEZE:
            self.frozen(self.visual_encoder)
            self.frozen(self.efficientsam)
            self.frozen(self.dino_model)

    def frozen(self, module):
        if getattr(module, 'module', False):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False

    def reverse_normalization(self, x):
        # 原始标准化参数
        original_mean = self.pixel_mean.view(1, 3, 1, 1).to(x.device)
        original_std = self.pixel_std.view(1, 3, 1, 1).to(x.device)
        # 反标准化
        x_unnormalized = x * original_std + original_mean
        return x_unnormalized

    def generate_masks(self, batched_images, pts_sampled, pts_labels, model):
        """
        Arguments:
          batched_images: A tensor of shape [B, 3, H, W]
          batched_points: A tensor of shape [B, num_queries, max_num_pts, 2]
          batched_point_labels: A tensor of shape [B, num_queries, max_num_pts]
        """
        batch_size, num_channels, img_H, img_W = batched_images.size()
        pts_sampled = torch.reshape(pts_sampled, [batch_size, 1, -1, 2])  # [B, num_queries, max_num_pts, 2]
        pts_labels = torch.reshape(pts_labels, [batch_size, 1, -1])  # [B, num_queries, max_num_pts]
        predicted_logits, predicted_iou = model(
            batched_images,
            pts_sampled,
            pts_labels,
        )

        sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
        predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
        predicted_logits = torch.take_along_dim(
            predicted_logits, sorted_ids[..., None, None], dim=2
        )

        predict_mask = torch.ge(predicted_logits[:, :, 0, :, :], 0).int()
        return predict_mask.detach()

    def generate_prompts(self, boxes, using_gt=False, point_prompt=False):
        """
        Sample random points within ground truth bounding boxes.

        Args:
            box_gt (torch.Tensor): Tensor of shape (B, 4) representing bounding boxes in YOLO format.

        Returns:
            tuple: A tuple containing:
                - point_prompt (torch.Tensor): Sampled points of shape (B, num_points, 2).
                - pts_labels (torch.Tensor): Labels for the sampled points of shape (B, num_points).
        """
        # Scale the ground truth boxes
        if using_gt:
            boxes[..., 0] = boxes[..., 0] * self.scale_factor_h
            boxes[..., 1] = boxes[..., 1] * self.scale_factor_w
            boxes[..., 2] = boxes[..., 2] * self.scale_factor_h
            boxes[..., 3] = boxes[..., 3] * self.scale_factor_w

        if point_prompt:
            # Sample points within the bounding boxes
            prompt = self.sample_points_in_boxes(boxes, self.num_points)

            # Create labels for the sampled points
            pts_labels = torch.ones(boxes.size(0), self.num_points).to(point_prompt.device)

        else:
            boxes_top_left = torch.cat([boxes[..., None, 0], boxes[..., None, 1]], dim=-1)
            boxes_bottom_right = torch.cat([boxes[..., None, 2], boxes[..., None, 3]], dim=-1)
            prompt = torch.stack([boxes_top_left, boxes_bottom_right], dim=-2)
            pts_labels = torch.tensor([[2, 3]]).to(boxes.device)
            pts_labels = pts_labels[None, :].repeat(boxes.size(0), 1, 1)
        return prompt, pts_labels

    def get_position_embedding(self, yolov3_output):
        bbox = yolov3_output[..., :2].mean(2)  # [64, 17, 2] bbox midpoints [batch, num_anchors, x_center, y_center]
        scaled_bbox = bbox / self.scale_factor_h  # Normalize coordinates to [0, 1]
        position_embeddings = self.pos_encoder(scaled_bbox)
        return position_embeddings

    def forward(self, x, y, box_gt=None, mask_gt=None, info_iter=None, gpu_tracker=None, epoch=None):
        # Vision and Language Encodingå
        with torch.no_grad():
            boxes_all, x_, boxes_sml = self.visual_encoder(x)

            resized_image_feature_dino = F.interpolate(x, size=(364, 364), mode='bilinear', align_corners=False)
            dino_feature = self.dino_model(resized_image_feature_dino).last_hidden_state.to(x.device)

            resized_image_feature_sam = self.reverse_normalization(x)
            sam_feature = self.efficientsam.get_image_embeddings(resized_image_feature_sam).to(x.device)

        y_ = self.lang_encoder(y)

        # Vision Multi Scale Fusion
        s, m, l = x_
        x_input = [l, m, s]
        l_new, m_new, s_new = self.multi_scale_manner(x_input)

        # Dynamic routing
        rec_feature = F.adaptive_avg_pool2d(s_new, (1, 1)).permute(0, 2, 3, 1).squeeze(1)  # (64, 1,1024)
        res_feature = F.adaptive_avg_pool2d(l_new, (1, 1)).permute(0, 2, 3, 1).squeeze(1)  # (64, 1,1024)

        # load dino model
        dino_feature = dino_feature[:, 1:, :]
        dino_feature = dino_feature.transpose(1, 2).contiguous().view(dino_feature.size(0), dino_feature.size(2), 26, 26)
        dino_feature_rec = F.avg_pool2d(dino_feature, kernel_size=2, stride=2)
        dino_feature_res = F.interpolate(dino_feature, size=(52, 52), mode='bilinear', align_corners=False)
        dino_feature_rec = self.linear_dino_rec(dino_feature_rec.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        dino_feature_res = self.linear_dino_res(dino_feature_res.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # load sam model
        sam_feature_rec = self.linear_sam_rec(sam_feature.permute(0, 2, 3, 1))
        sam_feature_rec = sam_feature_rec.permute(0, 3, 1, 2)
        sam_feature_rec = F.avg_pool2d(sam_feature_rec, kernel_size=2, stride=2)
        sam_feature_res = self.linear_sam_res(sam_feature.permute(0, 2, 3, 1))
        sam_feature_res = sam_feature_res.permute(0, 3, 1, 2)
        sam_feature_res = F.interpolate(sam_feature_res, size=(52, 52), mode='bilinear', align_corners=False)

        # Calculate the probability distribution of router_logits
        router_logits = self.linear_router_rec(rec_feature.detach()).squeeze(1)
        router_logits = torch.softmax(router_logits, dim=-1)
        s_new = s_new + dino_feature_rec * router_logits[:, 0][:, None, None, None] + sam_feature_rec * router_logits[:, 1][:, None, None, None]
        # s_new = s_new * router_logits[:, 0][:, None, None, None] + dino_feature_rec * router_logits[:, 1][:, None, None, None] + sam_feature_rec * router_logits[:, 2][:, None, None, None]

        # Calculate the probability distribution of router_logits
        router_logits = self.linear_router_res(res_feature.detach()).squeeze(1)
        router_logits = torch.softmax(router_logits, dim=-1)
        l_new = l_new + dino_feature_res * router_logits[:, 0][:, None, None, None] + sam_feature_res * router_logits[:, 1][:, None, None, None]
        # l_new = l_new * router_logits[:, 0][:, None, None, None] + dino_feature_res * router_logits[:, 1][:, None, None, None] + sam_feature_res * router_logits[:, 2][:, None, None, None]

        x_ = [s_new, m_new, l_new]
        # Anchor Selection
        boxes_sml_new = []
        mean_i = torch.mean(boxes_sml[0], dim=2, keepdim=True)
        mean_i = mean_i.squeeze(2)[:, :, 4]
        vals, indices = mean_i.topk(k=int(self.select_num), dim=1, largest=True, sorted=True)
        bs, gridnum, anncornum, ch = boxes_sml[0].shape
        bs_, selnum = indices.shape
        box_sml_new = boxes_sml[0].masked_select(
            torch.zeros(bs, gridnum).to(boxes_sml[0].device).scatter(1, indices, 1).bool().unsqueeze(2).unsqueeze(
                3).expand(bs, gridnum, anncornum, ch)).contiguous().view(bs, selnum, anncornum, ch)
        boxes_sml_new.append(box_sml_new)

        batchsize, dim, h, w = x_[0].size()
        i_new = x_[0].view(batchsize, dim, h * w).permute(0, 2, 1)
        bs, gridnum, ch = i_new.shape
        i_new = i_new.masked_select(
            torch.zeros(bs, gridnum).to(i_new.device).scatter(1, indices, 1).
            bool().unsqueeze(2).expand(bs, gridnum, ch)).contiguous().view(bs, selnum, ch)

        # Anchor-based Contrastive Learning
        x_new = self.linear_vs(i_new)
        position_embedding = self.get_position_embedding(boxes_sml_new[0])
        x_new = x_new + position_embedding
        y_new = self.linear_ts(y_['flat_lang_feat'].unsqueeze(1))

        x_sup = [l_new, m_new, s_new]
        for i in range(len(self.fusion_manner)):
            x_sup[i] = self.fusion_manner[i](x_sup[i], y_['flat_lang_feat'])
        x_sup = self.multi_scale_manner_sup(x_sup)
        seg_emb, _, _ = self.attention_manner(y_['flat_lang_feat'], x_sup[0])

        if self.training:
            loss_det = self.head(x_new, y_new)
            predictions_s = self.head.getPrediction(x_new, y_new)
            predictions_list = [predictions_s]
            pred_boxes = self.get_boxes(boxes_sml_new, predictions_list, self.class_num)
            pred_boxes = clip_boxes_to_image(pred_boxes, info_iter)
            prompt, pts_labels = self.generate_prompts(pred_boxes, using_gt=False)
            x = self.reverse_normalization(x)
            predict_masks = self.generate_masks(x, prompt, pts_labels, self.efficientsam)
            predict_masks = self.ensure_float32(predict_masks)
            loss_seg = self.seg_head(seg_emb, box_gt, predict_masks, pred_boxes, epoch)
            return loss_det, loss_seg
        else:
            predictions_s = self.head(x_new, y_new)
            predictions_list = [predictions_s]
            box_pred = self.get_boxes(boxes_sml_new, predictions_list, self.class_num)
            _, mask_pred = self.seg_head(seg_emb)
            return box_pred, mask_pred


    def ensure_float32(self, tensor):
        """
        Check if the input tensor is of type float32.
        If not, convert it to float32.

        Args:
            tensor (torch.Tensor): Input tensor to check and convert.

        Returns:
            torch.Tensor: Tensor converted to float32 if it was not already.
        """
        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)
        return tensor

    def sample_points_in_boxes(self, boxes, num_points=5):
        """
        Randomly sample points within bounding boxes.

        Args:
            boxes (torch.Tensor): Tensor of shape (B, 4) representing bounding boxes (x1, y1, x2, y2).
            num_points (int): Number of points to sample within each box.

        Returns:
            torch.Tensor: Tensor of shape (B, num_points, 2) with sampled points (x, y).
        """
        # Extract box coordinates
        x1 = boxes[..., 0]  # x1 coordinates
        y1 = boxes[..., 1]  # y1 coordinates
        x2 = boxes[..., 2]  # x2 coordinates
        y2 = boxes[..., 3]  # y2 coordinates

        # Generate random points within the boxes
        random_x = x1 + (x2 - x1) * torch.rand(boxes.size(0), num_points).to(boxes.device)
        random_y = y1 + (y2 - y1) * torch.rand(boxes.size(0), num_points).to(boxes.device)

        # Stack points into a tensor of shape (B, num_points, 2)
        sampled_points = torch.stack((random_x, random_y), dim=-1)

        return sampled_points

    def get_boxes(self, boxes_sml, predictionslist, class_num):
        batchsize = predictionslist[0].size()[0]
        pred = []
        for i in range(len(predictionslist)):
            mask = predictionslist[i].squeeze(1)
            masked_pred = boxes_sml[i][mask]
            refined_pred = masked_pred.view(batchsize, -1, class_num + 5)
            refined_pred[:, :, 0] = refined_pred[:, :, 0] - refined_pred[:, :, 2] / 2
            refined_pred[:, :, 1] = refined_pred[:, :, 1] - refined_pred[:, :, 3] / 2
            refined_pred[:, :, 2] = refined_pred[:, :, 0] + refined_pred[:, :, 2]
            refined_pred[:, :, 3] = refined_pred[:, :, 1] + refined_pred[:, :, 3]
            pred.append(refined_pred.data)
        boxes = torch.cat(pred, 1)
        score = boxes[:, :, 4]
        max_score, ind = torch.max(score, -1)
        ind_new = ind.unsqueeze(1).unsqueeze(1).repeat(1, 1, 5)
        box_new = torch.gather(boxes, 1, ind_new)
        if self.training:
            box_new = box_new[..., :4]
        return box_new
