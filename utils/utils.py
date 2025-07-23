import cv2
import torch
import os
import random
import torch.optim.lr_scheduler as lr_scheduler
import math
import numpy as np


class EMA(object):
    '''
        apply expontential moving average to a model. This should have same function as the `tf.train.ExponentialMovingAverage` of tensorflow.
        usage:
            model = resnet()
            model.train()
            ema = EMA(model, 0.9999)
            ....
            for img, lb in dataloader:
                loss = ...
                loss.backward()
                optim.step()
                ema.update_params() # apply ema
            evaluate(model)  # evaluate with original model as usual
            ema.apply_shadow() # copy ema status to the model
            evaluate(model) # evaluate the model with ema paramters
            ema.restore() # resume the model parameters
        args:
            - model: the model that ema is applied
            - alpha: each parameter p should be computed as p_hat = alpha * p + (1. - alpha) * p_hat
            - buffer_ema: whether the model buffers should be computed with ema method or just get kept
        methods:
            - update_params(): apply ema to the model, usually call after the optimizer.step() is called
            - apply_shadow(): copy the ema processed parameters to the model
            - restore(): restore the original model parameters, this would cancel the operation of apply_shadow()
    '''

    def __init__(self, model, alpha, buffer_ema=True):
        self.step = 0
        self.model = model
        self.alpha = alpha
        self.buffer_ema = buffer_ema
        self.shadow = self.get_model_state()
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]

    def update_params(self):
        decay = min(self.alpha, (self.step + 1) / (self.step + 10))
        state = self.model.state_dict()
        for name in self.param_keys:
            self.shadow[name].copy_(
                decay * self.shadow[name]
                + (1 - decay) * state[name]
            )
        for name in self.buffer_keys:
            if "efficientsam" in name:
                continue
            if "clip" in name:
                continue
            if self.buffer_ema:
                self.shadow[name].copy_(
                    decay * self.shadow[name]
                    + (1 - decay) * state[name]
                )
            else:
                self.shadow[name].copy_(state[name])
        self.step += 1

    def apply_shadow(self):
        self.backup = self.get_model_state()
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.backup)

    def get_model_state(self):
        return {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }


def setup_unique_version(__C):
    # if __C.RESUME:
    #     __C.VERSION = __C.RESUME_VERSION
    #     return
    while True:
        version = random.randint(0, 99999)
        # version = 77263
        if not (os.path.exists(os.path.join(__C.LOG_PATH, str(version)))):
            __C.VERSION = str(version)
            break


def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def show_anns_ours(mask, ax):
    ax.set_autoscale_on(False)
    img = np.ones((mask.shape[0], mask.shape[1], 4))
    img[:, :, 3] = 0
    color_mask = [0, 1, 0, 0.7]
    img[np.logical_not(mask)] = color_mask
    ax.imshow(img)


def yolobox2xyxy(labels):
    """
    Convert YOLO box format (xc, yc, w, h) to xyxy format (x1, y1, x2, y2).

    Args:
        labels (numpy.ndarray): label data whose shape is :math:`(N, 1, 4)`.
            Each label consists of [class, xc, yc, w, h] where
                class (float): class index.
                xc, yc (float): center of bbox (values range from 0 to 1).
                w, h (float): size of bbox (values range from 0 to 1).

    Returns:
        numpy.ndarray: Converted label data with size :math:`(N, 1, 4)`.
            Each label consists of [x1, y1, x2, y2].
    """
    # 提取中心坐标和宽高
    xc = labels[..., 0]
    yc = labels[..., 1]
    w = labels[..., 2]
    h = labels[..., 3]

    # 计算左上角 (x1, y1) 和右下角 (x2, y2) 坐标
    x1 = xc - (w / 2)
    y1 = yc - (h / 2)
    x2 = xc + (w / 2)
    y2 = yc + (h / 2)

    # Stack into xyxy format
    xyxy_labels = torch.stack((x1, y1, x2, y2), dim=-1)

    return xyxy_labels


def draw_box(image, bbox, color):
    # color: bgr
    # bbox: [x1, y1, x2, y2]
    thickness = 2
    image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=thickness)
    return image


def draw_mask(image, mask, color):
    # color: bgr
    # mask: [h, w]
    alpha = 0.3
    draw_contours = True
    coutour_thickness = 2

    # mask = mask * 255
    # mask = mask.astype(np.uint8)
    mask = mask > 0
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    overlay = image.copy()
    overlay[mask] = color
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0.0)
    if draw_contours:
        image = cv2.drawContours(image, contours, -1, (255, 255, 255), coutour_thickness)

    return image


def convert_box_format(box):
    """
    Convert a (B, N, 4) tensor of bounding boxes in the format (x1, y1, x2, y2)
    to a (B, N, 2, 2) tensor representing the top-left and bottom-right corners.

    Args:
        box (torch.Tensor): Tensor of shape (B, N, 4) with bounding boxes.

    Returns:
        torch.Tensor: Converted tensor of shape (B, N, 2, 2).
    """
    # Reshape to (B, N, 2, 2)
    box_reshaped = box.view(box.size(0), box.size(1), 2, 2)
    return box_reshaped


def batch_box_iou(box1, box2, threshold=0.5, iou_out=False):
    """
    :param box1:  B,N,4,85
    :param box2:  B,1,1,4
    :return: N
    """
    in_h = torch.min(box1[..., 2], box2[..., 2]) - torch.max(box1[..., 0], box2[..., 0])
    in_w = torch.min(box1[..., 3], box2[..., 3]) - torch.max(box1[..., 1], box2[..., 1])
    in_h = in_h.clamp(min=0.)
    in_w = in_w.clamp(min=0.)
    inter = in_h * in_w
    union = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1]) + \
            (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1]) - inter
    iou = inter / union
    if iou_out:
        return iou > threshold, iou
    else:
        return iou > threshold


def label2yolobox(labels, info_img, maxsize, lrflip=False):
    """
    Transform coco labels to yolo box labels
    Args:
        labels (numpy.ndarray): label data whose shape is :math:`(N, 5)`.
            Each label consists of [class, x, y, w, h] where \
                class (float): class index.
                x, y, w, h (float) : coordinates of \
                    left-top points, width, and height of a bounding box.
                    Values range from 0 to width or height of the image.
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
        maxsize (int): target image size after pre-processing
        lrflip (bool): horizontal flip flag

    Returns:
        labels:label data whose size is :math:`(N, 5)`.
            Each label consists of [class, xc, yc, w, h] where
                class (float): class index.
                xc, yc (float) : center of bbox whose values range from 0 to 1.
                w, h (float) : size of bbox whose values range from 0 to 1.
    """
    h, w, nh, nw, dx, dy, _ = info_img
    x1 = labels[:, 0] / w
    y1 = labels[:, 1] / h
    x2 = (labels[:, 0] + labels[:, 2]) / w
    y2 = (labels[:, 1] + labels[:, 3]) / h
    labels[:, 0] = (((x1 + x2) / 2) * nw + dx) / maxsize
    labels[:, 1] = (((y1 + y2) / 2) * nh + dy) / maxsize

    labels[:, 2] = (labels[:, 2] * (nw / w / maxsize))
    labels[:, 3] = (labels[:, 3] * (nh / h / maxsize))

    # labels[:, 2] *= nw / w / maxsize
    # labels[:, 3] *= nh / h / maxsize
    labels[:, :4] = np.clip(labels[:, :4], 0., 0.99)
    if lrflip:
        labels[:, 0] = 1 - labels[:, 0]
    return labels


def yolobox2label(box, info_img):
    """
    Transform yolo box labels to yxyx box labels.
    Args:
        box (list): box data with the format of [yc, xc, w, h]
            in the coordinate system after pre-processing.
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
    Returns:
        label (list): box data with the format of [y1, x1, y2, x2]
            in the coordinate system of the input image.
    """
    # h, w, nh, nw, dx, dy,_ = info_img
    # y1, x1, y2, x2 = box
    # box_h = ((y2 - y1) / nh) * h
    # box_w = ((x2 - x1) / nw) * w
    # y1 = ((y1 - dy) / nh) * h
    # x1 = ((x1 - dx) / nw) * w
    # label = [y1, x1, y1 + box_h, x1 + box_w]
    h, w, nh, nw, dx, dy, _ = info_img
    x1, y1, x2, y2 = box[:4]
    box_h = ((y2 - y1) / nh) * h
    box_w = ((x2 - x1) / nw) * w
    y1 = ((y1 - dy) / nh) * h
    x1 = ((x1 - dx) / nw) * w
    label = [x1, y1, x1 + box_w, y1 + box_h]
    return np.concatenate([np.array(label), box[4:]])


def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)


def get_lr_scheduler(__C, optimizer):
    if __C.SCHEDULER == 'step':
        t, T = __C.WARMUP, __C.EPOCHS

        def lr_func(epoch):
            coef = 1.
            if epoch <= t:
                coef = float(epoch) / float(t + 1)
            else:
                for i, deps in enumerate(__C.DECAY_EPOCHS):
                    if epoch >= deps:
                        coef = __C.LR_DECAY_R ** (i + 1)
            return coef

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_func(epoch))
    elif __C.SCHEDULER == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=__C.EPOCHS)
    else:
        t, T = __C.WARMUP, __C.EPOCHS
        n_t = 0.5
        lr_func = lambda epoch: (0.9 * epoch / t + __C.LR) if epoch < t else __C.LR if n_t * (
                1 + math.cos(math.pi * (epoch - t) / (T - t))) < __C.LR else n_t * (
                1 + math.cos(math.pi * (epoch - t) / (T - t)))
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    return scheduler


def get_lr_scheduler(__C, optimizer, n_iter_per_epoch):
    num_steps = int(__C.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(__C.WARMUP * n_iter_per_epoch)
    if __C.SCHEDULER == 'step':
        # default step lr
        t, T = __C.WARMUP * n_iter_per_epoch, __C.EPOCHS * n_iter_per_epoch

        def lr_func(step):
            coef = 1.
            if step <= t:
                coef = float(step) / float(t + 1)
            else:
                for i, deps in enumerate(__C.DECAY_EPOCHS):
                    if step >= deps * n_iter_per_epoch:
                        coef = __C.LR_DECAY_R ** (i + 1)
            return coef

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif __C.SCHEDULER == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=__C.EPOCHS * n_iter_per_epoch)
    else:
        t, T = __C.WARMUP * n_iter_per_epoch, __C.EPOCHS * n_iter_per_epoch
        n_t = 0.5
        warm_step_lr = (__C.LR - __C.WARMUP_LR) / t
        lr_func = lambda step: (step * warm_step_lr + __C.WARMUP_LR) / __C.LR if step < t \
            else (__C.MIN_LR + n_t * (__C.LR - __C.MIN_LR) * (1 + math.cos(math.pi * (step - t) / (T - t)))) / __C.LR

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    return scheduler


def getLayers(model):
    """
    get each layer's name and its module
    :param model:
    :return: each layer's name and its module
    """
    layers = []

    def unfoldLayer(model):
        """
        unfold each layer
        :param model: the given model or a single layer
        :param root: root name
        :return:
        """

        # get all layers of the model
        layer_list = list(model.named_children())
        for item in layer_list:
            module = item[1]
            sublayer = list(module.named_children())
            sublayer_num = len(sublayer)

            # if current layer contains sublayers, add current layer name on its sublayers
            if sublayer_num == 0 and len(list(module.parameters())) > 0:
                layers.append(module)
            # if current layer contains sublayers, unfold them
            elif isinstance(module, nn.Module):
                unfoldLayer(module)

    unfoldLayer(model)
    return layers


def get_module_ops(obj):
    ret = []
    for m in obj:
        if hasattr(obj, 'children'):
            ret += get_module_ops(m)
        else:
            ret.append(m)
    return ret


def split_weights(net):
    """split network weights into to categlories,
    one are weights in conv layer and linear layer,
    others are other learnable paramters(conv bias,
    bn weights, bn bias, linear bias)
    Args:
        net: network architecture

    Returns:
        a dictionary of params splite into to categlories
    """

    decay = []
    no_decay = []
    total = getLayers(net)
    # for (i,j) in net.named_parameters():
    #     print(i)
    for m in total:
        # print(m)
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            decay.append(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                no_decay.append(m.bias)
        else:
            no_decay += list(m.parameters())
            # print(m)
            # if hasattr(m, 'weight'):
            #     no_decay.append(m.weight)
            # elif hasattr(m, 'bias'):
            #     no_decay.append(m.bias)
            # else:
            #     no_decay.append(m)

    # print(len(list(net.parameters())),len(total),len(decay) , len(no_decay))
    assert len(list(net.parameters())) == len(decay) + len(no_decay)

    return [dict(params=filter(lambda p: p.requires_grad, decay)),
            dict(params=filter(lambda p: p.requires_grad, no_decay), weight_decay=0)]


def lr_step_decay(lr_start=0.001, steps=[30, 40]):
    def get_lr(epoch):
        decay_rate = len(steps)
        for i, e in enumerate(steps):
            if epoch < e:
                decay_rate = i
                break
        lr = lr_start / (10 ** (decay_rate))
        return lr

    return get_lr


def lr_power_decay(lr_start=2.5e-4, lr_power=0.9, warm_up_lr=0., step_all=45 * 1414, warm_up_step=1000):
    # step_per_epoch=3286
    def warm_up(base_lr, lr, cur_step, end_step):
        return base_lr + (lr - base_lr) * cur_step / end_step

    def get_learningrate(epoch):

        if epoch < warm_up_step:
            lr = warm_up(warm_up_lr, lr_start, epoch, warm_up_step)
        else:
            lr = lr_start * ((1 - float(epoch - warm_up_step) / (step_all - warm_up_step)) ** lr_power)
        return lr
        # print("learning rate is", lr)

    return get_learningrate


def normed2original(image, mean=None, std=None, transpose=True):
    """
    :param image: 3,h,w
    :param mean: 3
    :param std: 3
    :return:
    """
    if std is not None:
        std = torch.from_numpy(np.array(std)).to(image.device).float()
        image = image * std.unsqueeze(-1).unsqueeze(-1)
    if mean is not None:
        mean = torch.from_numpy(np.array(mean)).to(image.device).float()
        image = image + mean.unsqueeze(-1).unsqueeze(-1)
    if transpose:
        image = image.permute(1, 2, 0)
    return image.cpu().numpy()


def draw_visualization(image, sent, pred_box, gt_box, draw_text=True, savepath=None):
    # image=(image*255).astype(np.uint8)
    image = np.ascontiguousarray(image)
    left, top, right, bottom, _ = (pred_box).astype('int32')
    gt_left, gt_top, gt_right, gt_bottom = (gt_box).astype('int32')
    colors = [(255, 0, 0), (0, 255, 0), (0, 191, 255)]

    cv2.rectangle(image, (left, top), (right, bottom), colors[0], 2)
    cv2.rectangle(image, (gt_left, gt_top), (gt_right, gt_bottom), colors[1], 2)
    # cv2.imwrite(savepath+str(k)+'.jpg',img)

    if draw_text:
        cv2.putText(image,
                    '{:%.2f}' % pred_box[-1],
                    (left, max(top - 3, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, colors[0], 2)
        cv2.putText(image,
                    'ground_truth',
                    (gt_left, max(gt_top - 3, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, colors[1], 2)
        cv2.putText(image,
                    str(sent),
                    (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, colors[2], 2)
    return image


def nms(bbox, thresh, score=None, limit=None):
    """Suppress bounding boxes according to their IoUs and confidence scores.
    Args:
        bbox (array): Bounding boxes to be transformed. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
        thresh (float): Threshold of IoUs.
        score (array): An array of confidences whose shape is :math:`(R,)`.
        limit (int): The upper bound of the number of the output bounding
            boxes. If it is not specified, this method selects as many
            bounding boxes as possible.
    Returns:
        array:
        An array with indices of bounding boxes that are selected. \
        They are sorted by the scores of bounding boxes in descending \
        order. \
        The shape of this array is :math:`(K,)` and its dtype is\
        :obj:`numpy.int32`. Note that :math:`K \\leq R`.

    from: https://github.com/chainer/chainercv
    """

    if len(bbox) == 0:
        return np.zeros((0,), dtype=np.int32)

    if score is not None:
        order = score.argsort()[::-1]
        bbox = bbox[order]
    bbox_area = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)

    selec = np.zeros(bbox.shape[0], dtype=bool)
    for i, b in enumerate(bbox):
        tl = np.maximum(b[:2], bbox[selec, :2])
        br = np.minimum(b[2:], bbox[selec, 2:])
        area = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)

        iou = area / (bbox_area[i] + bbox_area[selec] - area)
        if (iou >= thresh).any():
            continue

        selec[i] = True
        if limit is not None and np.count_nonzero(selec) >= limit:
            break

    selec = np.where(selec)[0]
    if score is not None:
        selec = order[selec]
    return selec.astype(np.int32)


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    """
    Postprocess for the output of YOLO model
    perform box transformation, specify the class for each detection,
    and perform class-wise non-maximum suppression.
    Args:
        prediction (torch tensor): The shape is :math:`(N, B, 4)`.
            :math:`N` is the number of predictions,
            :math:`B` the number of boxes. The last axis consists of
            :math:`xc, yc, w, h` where `xc` and `yc` represent a center
            of a bounding box.
        num_classes (int):
            number of dataset classes.
        conf_thre (float):
            confidence threshold ranging from 0 to 1,
            which is defined in the config file.
        nms_thre (float):
            IoU threshold of non-max suppression ranging from 0 to 1.

    Returns:
        output (list of torch tensor):

    """
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        # class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1)
        # class_pred = class_pred[0]
        conf_mask = (image_pred[:, 4] >= conf_thre).squeeze()
        image_pred = image_pred[conf_mask]

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get detections with higher confidence scores than the threshold
        ind = (image_pred[:, 5:] * image_pred[:, 4][:, None] >= conf_thre).nonzero()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((
            image_pred[ind[:, 0], :5],
            image_pred[ind[:, 0], 5 + ind[:, 1]].unsqueeze(1),
            ind[:, 1].float().unsqueeze(1)
        ), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            nms_in = detections_class.cpu().numpy()
            nms_out_index = nms(
                nms_in[:, :4], nms_thre, score=nms_in[:, 4] * nms_in[:, 5])
            detections_class = detections_class[nms_out_index]
            if output[i] is None:
                output[i] = detections_class
            else:
                output[i] = torch.cat((output[i], detections_class))

    return output


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    # top left
    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def mask_iou(mask1, mask2):
    """
    :param mask1:  l
    :param mask2:  l
    :return: iou
    """
    mask1 = mask1.reshape([-1])
    mask2 = mask2.reshape([-1])
    t = np.array(mask1 > 0.5)
    p = mask2 > 0.
    intersection = np.logical_and(t, p)
    union = np.logical_or(t, p)
    iou = (np.sum(intersection > 0) + 1e-10) / (np.sum(union > 0) + 1e-10)

    ap = dict()
    thresholds = np.arange(0.5, 1, 0.05)
    s = []
    for thresh in thresholds:
        ap[thresh] = float(iou > thresh)
    return iou, ap


def mask_iou_batch(mask1, mask2):
    # 将掩码转换为布尔值
    mask1 = mask1 > 0.5
    mask2 = mask2 > 0.5
    mask1 = mask1.bool()
    mask2 = mask2.bool()

    # 计算交集
    intersection = (mask1 & mask2).float().sum((1, 2, 3))

    # 计算并集
    union = (mask1 | mask2).float().sum((1, 2, 3))

    # 避免除以零
    iou = intersection / union
    iou[union == 0] = 0

    return iou


def mask_processing(mask, info_img):
    # print(info_img)
    h, w, nh, nw, dx, dy, _ = info_img
    # print(info_img)
    # print(mask)
    mask = mask[dy:dy + nh, dx:dx + nw, None]
    mask = cv2.resize(mask, (int(w), int(h)))
    return mask


def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss


def compute_project_term(mask_scores, gt_bitmasks, seg_weight=None):
    mask_losses_y = dice_coefficient(
        mask_scores.max(dim=2, keepdim=True)[0],
        gt_bitmasks.max(dim=2, keepdim=True)[0]
    )
    mask_losses_x = dice_coefficient(
        mask_scores.max(dim=3, keepdim=True)[0],
        gt_bitmasks.max(dim=3, keepdim=True)[0]
    )
    loss = mask_losses_x + mask_losses_y
    if seg_weight is not None:
        # loss = (loss * seg_weight).sum() / (seg_weight.sum() + 1e-6)
        loss = loss * seg_weight
        return loss.mean()
    return loss.mean()


def clip_boxes_to_image(pred_boxes, info_iter):
    """
    Clips the bounding boxes to fit within the image dimensions.

    Args:
        pred_boxes (torch.Tensor): Tensor of shape (N, 1, 4) containing the boxes (x1, y1, x2, y2).
        info_iter (torch.Tensor): Tensor of shape (N, 4) with image info (nh, nw, dx, dy).

    Returns:
        torch.Tensor: Tensor of clipped boxes.
    """
    clipped_boxes = pred_boxes[..., :4].clone()

    for id, box in enumerate(pred_boxes):
        # x1, y1, x2, y2 = box[..., :4]
        h, w, nh, nw, dx, dy, iid = info_iter[id]

        # Clip box coordinates
        x1 = torch.clamp(box[..., 0], min=dx, max=dx + nw).unsqueeze(-1)
        y1 = torch.clamp(box[..., 1], min=dy, max=dy + nh).unsqueeze(-1)
        x2 = torch.clamp(box[..., 2], min=dx, max=dx + nw).unsqueeze(-1)
        y2 = torch.clamp(box[..., 3], min=dy, max=dy + nh).unsqueeze(-1)

        # Update the box

        clipped_boxes[id] = torch.cat([x1, y1, x2, y2], dim=-1)

    return clipped_boxes
