import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import zipfile
import os
import cv2
# os.chdir("EfficientSAM")

from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits


def run_ours_box_or_points(img_path, pts_sampled, pts_labels, model):
    image_np = np.array(Image.open(img_path))
    img_tensor = ToTensor()(image_np)
    pts_sampled = torch.reshape(torch.tensor(pts_sampled), [1, 1, -1, 2])
    pts_labels = torch.reshape(torch.tensor(pts_labels), [1, 1, -1])
    predicted_logits, predicted_iou = model(
        img_tensor[None, ...],
        pts_sampled,
        pts_labels,
    )

    sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
    predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
    predicted_logits = torch.take_along_dim(
        predicted_logits, sorted_ids[..., None, None], dim=2
    )

    return torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()


def run_ours_box_or_points_debug(img, pts_sampled, pts_labels, model):
    img_tensor = ToTensor()(img)
    pts_sampled = torch.reshape(torch.tensor(pts_sampled), [1, 1, -1, 2])
    pts_labels = torch.reshape(torch.tensor(pts_labels), [1, 1, -1])
    predicted_logits, predicted_iou = model(
        img_tensor[None, ...],
        pts_sampled,
        pts_labels,
    )

    sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
    predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
    predicted_logits = torch.take_along_dim(
        predicted_logits, sorted_ids[..., None, None], dim=2
    )

    return torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.8])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="yellow", facecolor=(0, 0, 0, 0), lw=5)
    )


def show_anns_ours(mask, ax):
    ax.set_autoscale_on(False)
    img = np.ones((mask.shape[0], mask.shape[1], 4))
    img[:, :, 3] = 0
    color_mask = [0, 1, 0, 0.7]
    img[np.logical_not(mask)] = color_mask
    ax.imshow(img)


def sample_points_in_boxes(boxes, num_points=5):
    """
    Randomly sample points within bounding boxes.

    Args:
        boxes (torch.Tensor): Tensor of shape (B, 4) representing bounding boxes (x1, y1, x2, y2).
        num_points (int): Number of points to sample within each box.

    Returns:
        torch.Tensor: Tensor of shape (B, num_points, 2) with sampled points (x, y).
    """
    # Extract box coordinates
    x1 = boxes[:, 0]  # x1 coordinates
    y1 = boxes[:, 1]  # y1 coordinates
    x2 = boxes[:, 2]  # x2 coordinates
    y2 = boxes[:, 3]  # y2 coordinates

    # Generate random points within the boxes
    random_x = x1.unsqueeze(1) + (x2 - x1).unsqueeze(1) * torch.rand(boxes.size(0), num_points)
    random_y = y1.unsqueeze(1) + (y2 - y1).unsqueeze(1) * torch.rand(boxes.size(0), num_points)

    # Stack points into a tensor of shape (B, num_points, 2)
    sampled_points = torch.stack((random_x, random_y), dim=-1)

    return sampled_points.numpy()


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
    h, w, nh, nw, dx, dy = info_img
    x1 = labels[:, 0] / w
    y1 = labels[:, 1] / h
    x2 = (labels[:, 0] + labels[:, 2]) / w
    y2 = (labels[:, 1] + labels[:, 3]) / h
    labels[:, 0] = x1 * nw + dx
    labels[:, 1] = y1 * nh + dy

    labels[:, 2] = x2 * nw + dx
    labels[:, 3] = y2 * nh + dy

    # labels[:, 2] *= nw / w / maxsize
    # labels[:, 3] *= nh / h / maxsize
    # labels[:, :4] = np.clip(labels[:, :4], 0., 0.99)
    # if lrflip:
    #     labels[:, 0] = 1 - labels[:, 0]
    return labels


def preprocess_info(img, box, lr_flip=False):
    h, w, _ = img.shape
    imgsize = 416
    new_ar = w / h
    if new_ar < 1:
        nh = imgsize
        nw = nh * new_ar
    else:
        nw = imgsize
        nh = nw / new_ar
    nw, nh = int(nw), int(nh)
    dx = (imgsize - nw) // 2
    dy = (imgsize - nh) // 2
    img = cv2.resize(img, (nw, nh))
    sized = np.ones((imgsize, imgsize, 3), dtype=np.uint8) * 127
    sized[dy:dy + nh, dx:dx + nw, :] = img
    info_img = (h, w, nh, nw, dx, dy)
    sized_box = label2yolobox(box, info_img, 416, lrflip=lr_flip)
    return sized, sized_box


efficient_sam_vitt_model = build_efficient_sam_vitt()
efficient_sam_vitt_model.eval()

# Since EfficientSAM-S checkpoint file is >100MB, we store the zip file.
with zipfile.ZipFile("weights/efficient_sam_vits.pt.zip", 'r') as zip_ref:
    zip_ref.extractall("weights")
efficient_sam_vits_model = build_efficient_sam_vits()
efficient_sam_vits_model.eval()


fig, ax = plt.subplots(1, 3, figsize=(30, 30))
image_path = '/data/luogen/csl/RefCLIP/data/images/train2014/COCO_train2014_000000197251.jpg'
image_iter = cv2.imread(image_path)
image_iter = cv2.cvtColor(image_iter, cv2.COLOR_BGR2RGB)
bbox = [219.2400, 107.9500, 245.6200, 265.9500]
box_prompt = torch.tensor([bbox])

image_iter, box_prompt = preprocess_info(image_iter, box_prompt, lr_flip=False)
box_prompt_list = box_prompt.tolist()[0]
print(box_prompt_list)
num_points = 5
input_point = sample_points_in_boxes(box_prompt, num_points)
input_point = input_point.reshape((-1,2))


print(input_point.shape)
image = np.array(image_iter)
input_label = np.ones(num_points)

show_points(input_point, input_label, ax[0])
show_box(box_prompt_list, ax[0])
ax[0].imshow(image)


ax[1].imshow(image)
mask_efficient_sam_vitt = run_ours_box_or_points_debug(image, input_point, input_label, efficient_sam_vitt_model)
show_anns_ours(mask_efficient_sam_vitt, ax[1])
ax[1].title.set_text("EfficientSAM (VIT-tiny)")
ax[1].axis('off')

ax[2].imshow(image)
mask_efficient_sam_vits = run_ours_box_or_points_debug(image, input_point, input_label, efficient_sam_vits_model)
show_anns_ours(mask_efficient_sam_vits, ax[2])
ax[2].title.set_text("EfficientSAM (VIT-small)")
ax[2].axis('off')

# plt.show()
plt.savefig("./results.jpg", dpi=300)

