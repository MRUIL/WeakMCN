from utils.distributed import *
import torch.multiprocessing as mp
from utils.ckpt import *
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.logging import *
import argparse
import time
from utils import config
from datasets.dataloader import loader, RefCOCODataSet
from tensorboardX import SummaryWriter
from utils.utils import *
import torch.optim as Optim
from importlib import import_module
import cv2
import numpy as np


class ModelLoader:
    def __init__(self, __C):
        self.model_use = __C.MODEL
        model_moudle_path = 'models.' + self.model_use + '.net'
        self.model_moudle = import_module(model_moudle_path)

    def Net(self, __arg1, __arg2, __arg3):
        return self.model_moudle.Net(__arg1, __arg2, __arg3)


def draw_box(image, bbox, color):
    # color: bgr
    # bbox: [x1, y1, x2, y2]
    thickness = 2
    image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=thickness)
    return image


# ssh -L 16006:127.0.0.1:6006 username@ip

def truncate_bounding_boxes(boxes, image_width, image_height):
    """
    Truncates bounding box coordinates to ensure they do not exceed image width and height.
    Values less than 0 are truncated to 0.

    Parameters:
        boxes (numpy.ndarray): A numpy array of shape (N, 4) where each row represents a bounding box
                               with coordinates (x_min, y_min, x_max, y_max).
        image_width (int): The width of the image.
        image_height (int): The height of the image.

    Returns:
        numpy.ndarray: The truncated bounding boxes.
    """
    # Ensure coordinates are not less than 0
    boxes[:, 0] = np.maximum(boxes[:, 0], 0)  # x_min
    boxes[:, 1] = np.maximum(boxes[:, 1], 0)  # y_min
    boxes[:, 2] = np.maximum(boxes[:, 2], 0)  # x_max
    boxes[:, 3] = np.maximum(boxes[:, 3], 0)  # y_max

    # Ensure coordinates do not exceed image dimensions
    boxes[:, 0] = np.minimum(boxes[:, 0], image_width)  # x_min
    boxes[:, 1] = np.minimum(boxes[:, 1], image_height)  # y_min
    boxes[:, 2] = np.minimum(boxes[:, 2], image_width)  # x_max
    boxes[:, 3] = np.minimum(boxes[:, 3], image_height)  # y_max

    return boxes


def validate_box_and_mask(__C,
                          net,
                          loader,
                          writer,
                          epoch,
                          rank,
                          ix_to_token,
                          save_ids=None,
                          prefix='Val',
                          ema=None):
    if ema is not None:
        ema.apply_shadow()
    net.eval()

    batches = len(loader)
    batch_time = AverageMeter('Time', ':6.5f')
    data_time = AverageMeter('Data', ':6.5f')
    losses = AverageMeter('Loss', ':.4f')
    box_ap = AverageMeter('BoxIoU@0.5', ':6.2f')
    mask_ap = AverageMeter('MaskIoU', ':6.2f')
    inconsistency_error = AverageMeter('IE', ':6.2f')
    mask_aps = {}
    for item in np.arange(0.5, 1, 0.05):
        mask_aps[item] = []
    # meters = [batch_time, data_time, losses, box_ap]
    meters = [batch_time, data_time, losses, box_ap, mask_ap, inconsistency_error]
    meters_dict = {meter.name: meter for meter in meters}
    progress = ProgressMeter(__C.VERSION, __C.EPOCHS, len(loader), meters, prefix=prefix + ': ')
    with th.no_grad():
        end = time.time()
        for ith_batch, data in enumerate(loader):
            # ref_iter, image_iter, box_iter, gt_box_iter, info_iter, ref_txt, img_path = data
            ref_iter, image_iter, mask_iter, box_iter, gt_box_iter, mask_id, info_iter, ref_txt, img_path = data
            # print(ref_txt)
            mask_iter = mask_iter.cuda(non_blocking=True)
            ref_iter = ref_iter.cuda(non_blocking=True)
            image_iter = image_iter.cuda(non_blocking=True)
            box_iter = box_iter.cuda(non_blocking=True)
            gt_box_iter = gt_box_iter.cuda(non_blocking=True)
            box, mask = net(image_iter, ref_iter, box_gt=box_iter, mask_gt=mask_iter, info_iter=info_iter)

            # timelist = []
            # for i in range(100):
            #     t1 = time.time()
            #     box, mask = net(image_iter, ref_iter, box_gt=box_iter, mask_gt=mask_iter, info_iter=info_iter)
            #     t2 = time.time()
            #     timelist.append(t2-t1)
            # print("Mean FPS:", 1. / np.mean(timelist))
            gt_box_iter = gt_box_iter.squeeze(1)
            gt_box_iter[:, 2] = (gt_box_iter[:, 0] + gt_box_iter[:, 2])
            gt_box_iter[:, 3] = (gt_box_iter[:, 1] + gt_box_iter[:, 3])
            gt_box_iter = gt_box_iter.cpu().numpy()
            info_iter = info_iter.cpu().numpy()
            box = box.squeeze(1).cpu().numpy()
            pred_box_vis = box.copy()

            # predictions to gt
            for i in range(len(gt_box_iter)):
                # Adjust the horizontal and vertical coordinates based on the resized dimensions.
                box[i] = yolobox2label(box[i], info_iter[i])
            box_iou = batch_box_iou(torch.from_numpy(gt_box_iter), torch.from_numpy(box)).cpu().numpy()
            box_ap.update((box_iou > 0.5).astype(np.float32).mean() * 100., box_iou.shape[0])

            seg_iou = []
            mask = mask.cpu().numpy().astype(np.uint8)
            # mask = mask_iter.squeeze(1).cpu().numpy()
            for i, mask_pred in enumerate(mask):
                if writer is not None and save_ids is not None and ith_batch * __C.BATCH_SIZE + i in save_ids:
                    ixs = ref_iter[i].cpu().numpy()
                    words = []
                    for ix in ixs:
                        if ix > 0:
                            words.append(ix_to_token[ix])
                    sent = ' '.join(words)
                    box_iter = box_iter.view(box_iter.shape[0], -1) * __C.INPUT_SHAPE[0]
                    box_iter[:, 0] = box_iter[:, 0] - 0.5 * box_iter[:, 2]
                    box_iter[:, 1] = box_iter[:, 1] - 0.5 * box_iter[:, 3]
                    box_iter[:, 2] = box_iter[:, 0] + box_iter[:, 2]
                    box_iter[:, 3] = box_iter[:, 1] + box_iter[:, 3]
                    det_image = draw_visualization(normed2original(image_iter[i], __C.MEAN, __C.STD), sent,
                                                   pred_box_vis[i], box_iter[i].cpu().numpy())
                    writer.add_image('image/' + str(ith_batch * __C.BATCH_SIZE + i) + '_det', det_image, epoch,
                                     dataformats='HWC')
                    writer.add_image('image/' + str(ith_batch * __C.BATCH_SIZE + i) + '_seg',
                                     (mask[i, None] * 255).astype(np.uint8))

                # from pydensecrf import densecrf

                mask_gt = np.load(os.path.join(__C.MASK_PATH[__C.DATASET], '%d.npy' % mask_id[i]))
                mask_pred = mask_processing(mask_pred, info_iter[i])

                single_seg_iou, single_seg_ap = mask_iou(mask_gt, mask_pred)
                for item in np.arange(0.5, 1, 0.05):
                    mask_aps[item].append(single_seg_ap[item] * 100.)
                seg_iou.append(single_seg_iou)
            seg_iou = np.array(seg_iou).astype(np.float32)
            ie = (box_iou >= 0.5).astype(np.float32) * (seg_iou < 0.5).astype(np.float32) + (box_iou < 0.5).astype(
                np.float32) * (seg_iou >= 0.5).astype(np.float32)
            inconsistency_error.update(ie.mean() * 100., ie.shape[0])
            box_ap.update((box_iou > 0.5).astype(np.float32).mean() * 100., box_iou.shape[0])
            mask_ap.update(seg_iou.mean() * 100., seg_iou.shape[0])

            reduce_meters(meters_dict, rank, __C)
            if (ith_batch % __C.PRINT_FREQ == 0 or ith_batch == (len(loader) - 1)) and main_process(__C, rank):
                progress.display(epoch, ith_batch)
            batch_time.update(time.time() - end)
            end = time.time()

        if main_process(__C, rank) and writer is not None:
            writer.add_scalar("Acc/BoxIoU@0.5", box_ap.avg_reduce, global_step=epoch)
            writer.add_scalar("Acc/MaskIoU", mask_ap.avg_reduce, global_step=epoch)
            writer.add_scalar("Acc/IE", inconsistency_error.avg_reduce, global_step=epoch)
            for item in mask_aps:
                writer.add_scalar("Acc/MaskIoU@%.2f" % item, np.array(mask_aps[item]).mean(), global_step=epoch)
    if ema is not None:
        ema.restore()
    return box_ap.avg_reduce, mask_ap.avg_reduce


def validate(__C,
             net,
             loader,
             writer,
             epoch,
             rank,
             ix_to_token,
             save_ids=None,
             prefix='Val',
             ema=None):
    if ema is not None:
        ema.apply_shadow()
    net.eval()

    batches = len(loader)
    batch_time = AverageMeter('Time', ':6.5f')
    data_time = AverageMeter('Data', ':6.5f')
    losses = AverageMeter('Loss', ':.4f')
    box_ap = AverageMeter('BoxIoU@0.5', ':6.2f')
    meters = [batch_time, data_time, losses, box_ap]
    meters_dict = {meter.name: meter for meter in meters}
    progress = ProgressMeter(__C.VERSION, __C.EPOCHS, len(loader), meters, prefix=prefix + ': ')
    with th.no_grad():
        end = time.time()
        for ith_batch, data in enumerate(loader):
            # ref_iter, image_iter, box_iter, gt_box_iter, info_iter, ref_txt, img_path = data
            ref_iter, image_iter, mask_iter, box_iter, gt_box_iter, mask_id, info_iter, ref_txt, img_path = data
            # print(ref_txt)
            # mask_iter = mask_iter.cuda(non_blocking=True)
            ref_iter = ref_iter.cuda(non_blocking=True)
            image_iter = image_iter.cuda(non_blocking=True)
            box_iter = box_iter.cuda(non_blocking=True)
            gt_box_iter = gt_box_iter.cuda(non_blocking=True)

            box, _ = net(image_iter, ref_iter, __C.USING_CHECKPOINT, gt_box_iter, info_iter)
            ######## Visualization ##########
            # box = box.squeeze(1)
            # # box = net(image_iter, ref_iter)  # [left_bottom_x, left_bottom_y, right_top_x, right_top_y]
            # for i in range(box.size(0)):
            #     # 将张量转换为PIL图像
            #     to_pil = transforms.ToPILImage()
            #     pixel_mean = torch.tensor(__C.MEAN).view(-1, 1, 1).to(image_iter[i].device)
            #     pixel_std = torch.tensor(__C.STD).view(-1, 1, 1).to(image_iter[i].device)
            #     # pil_image = to_pil(x[i])
            #     pil_image = to_pil(image_ori[i] * pixel_std + pixel_mean)
            #     box_tmp = box[i].tolist()
            #     box_tmp = np.array(box_tmp)
            #     box_tmp = np.expand_dims(box_tmp, axis=0)
            #     box_tmp = truncate_bounding_boxes(box_tmp, image_width=416, image_height=416)
            #     for j in range(box_tmp.shape[0]):
            #         res = np.array(pil_image)
            #         box_tmp_ = yolobox2label(box_tmp[j], info_iter[i]).astype(int)
            #         res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            #         res = draw_box(res, box_tmp_, (0, 255, 0))
            #         cv2.imwrite("PRED_{}.jpg".format(__C.SEED), res)

            gt_box_iter = gt_box_iter.squeeze(1)
            gt_box_iter[:, 2] = (gt_box_iter[:, 0] + gt_box_iter[:, 2])
            gt_box_iter[:, 3] = (gt_box_iter[:, 1] + gt_box_iter[:, 3])
            gt_box_iter = gt_box_iter.cpu().numpy()
            info_iter = info_iter.cpu().numpy()
            box = box.squeeze(1).cpu().numpy()
            ######## Visualization ##########
            # for j in range(gt_box_iter.shape[0]):
            #     res = np.array(pil_image)
            #     box_tmp_ = gt_box_iter[j].astype(int)
            #     # box_tmp_ = yolobox2label(box_tmp[j], info_iter[i]).astype(int)
            #     res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            #     res = draw_box(res, box_tmp_, (0, 255, 0))
            #     cv2.imwrite("GT_{}.jpg".format(__C.SEED), res)

            # predictions to gt
            for i in range(len(gt_box_iter)):
                # Adjust the horizontal and vertical coordinates based on the resized dimensions.
                box[i] = yolobox2label(box[i], info_iter[i])
            box_iou = batch_box_iou(torch.from_numpy(gt_box_iter), torch.from_numpy(box)).cpu().numpy()
            box_ap.update((box_iou > 0.5).astype(np.float32).mean() * 100., box_iou.shape[0])

            reduce_meters(meters_dict, rank, __C)
            if (ith_batch % __C.PRINT_FREQ == 0 or ith_batch == (len(loader) - 1)) and main_process(__C, rank):
                progress.display(epoch, ith_batch)
            batch_time.update(time.time() - end)
            end = time.time()

        if main_process(__C, rank) and writer is not None:
            writer.add_scalar("Acc/BoxIoU@0.5", box_ap.avg_reduce, global_step=epoch)
    if ema is not None:
        ema.restore()
    return box_ap.avg_reduce


def main_worker(gpu, __C):
    global best_det_acc
    best_det_acc = 0.
    if __C.MULTIPROCESSING_DISTRIBUTED:
        if __C.DIST_URL == "env://" and __C.RANK == -1:
            __C.RANK = int(os.environ["RANK"])
        if __C.MULTIPROCESSING_DISTRIBUTED:
            __C.RANK = __C.RANK * len(__C.GPU) + gpu
        dist.init_process_group(backend=dist.Backend('NCCL'), init_method=__C.DIST_URL, world_size=__C.WORLD_SIZE,
                                rank=__C.RANK)

    train_set = RefCOCODataSet(__C, split='train')
    train_loader = loader(__C, train_set, gpu, shuffle=(not __C.MULTIPROCESSING_DISTRIBUTED))

    loaders = []
    prefixs = ['val']
    val_set = RefCOCODataSet(__C, split='val')
    val_loader = loader(__C, val_set, gpu, shuffle=True)
    loaders.append(val_loader)
    if __C.DATASET == 'refcoco' or __C.DATASET == 'refcoco+':
        testA = RefCOCODataSet(__C, split='testA')
        testA_loader = loader(__C, testA, gpu, shuffle=False)
        testB = RefCOCODataSet(__C, split='testB')
        testB_loader = loader(__C, testB, gpu, shuffle=False)
        prefixs.extend(['testA', 'testB'])
        loaders.extend([testA_loader, testB_loader])
    elif __C.DATASET == 'referit':
        test = RefCOCODataSet(__C, split='test')
        test_loader = loader(__C, test, gpu, shuffle=False)
        prefixs.append('test')
        loaders.append(test_loader)

    net = ModelLoader(__C).Net(
        __C,
        train_set.pretrained_emb,
        train_set.token_size
    )

    # optimizer
    std_optim = getattr(Optim, __C.OPT)
    params = filter(lambda p: p.requires_grad, net.parameters())  # split_weights(net)
    eval_str = 'params, lr=%f' % __C.LR
    for key in __C.OPT_PARAMS:
        eval_str += ' ,' + key + '=' + str(__C.OPT_PARAMS[key])
    optimizer = eval('std_optim' + '(' + eval_str + ')')

    if __C.MULTIPROCESSING_DISTRIBUTED:
        torch.cuda.set_device(gpu)
        net = DDP(net.cuda(), device_ids=[gpu], find_unused_parameters=True)
    elif len(gpu) == 1:
        net.cuda()
    else:
        net = DP(net.cuda())
    if main_process(__C, gpu):
        print(__C)
        total = sum([param.nelement() for param in net.parameters()])
        print('  + Number of all params: %.2fM' % (total / 1e6))  # 每一百万为一个单位
        total = sum([param.nelement() for param in net.parameters() if param.requires_grad])
        print('  + Number of trainable params: %.2fM' % (total / 1e6))  # 每一百万为一个单位

    if os.path.isfile(__C.RESUME_PATH):
        checkpoint = torch.load(__C.RESUME_PATH, map_location=lambda storage, loc: storage.cuda())
        new_dict = {}
        for k in checkpoint['state_dict']:
            if 'module.' in k:
                new_k = k.replace('module.', '')
                new_dict[new_k] = checkpoint['state_dict'][k]
        if len(new_dict.keys()) == 0:
            new_dict = checkpoint['state_dict']
        net.load_state_dict(new_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])

        if main_process(__C, gpu):
            print("==> loaded checkpoint from {}\n".format(__C.RESUME_PATH) +
                  "==> epoch: {} lr: {} ".format(checkpoint['epoch'], checkpoint['lr']))

    if __C.AMP:
        assert th.__version__ >= '1.6.0', \
            "Automatic Mixed Precision training only supported in PyTorch-1.6.0 or higher"
        scalar = th.cuda.amp.GradScaler()
    else:
        scalar = None

    SAVE_PATH = os.path.join(__C.LOG_PATH, __C.MODEL, 'seed_{}'.format(__C.SEED), str(__C.VERSION))
    if main_process(__C, gpu):
        writer = SummaryWriter(log_dir=SAVE_PATH)
        # writer = SummaryWriter(log_dir=os.path.join(__C.LOG_PATH, str(__C.VERSION)))
    else:
        writer = None

    save_ids = np.random.randint(1, len(val_loader) * __C.BATCH_SIZE, 100) if __C.LOG_IMAGE else None
    for loader_, prefix_ in zip(loaders, prefixs):
        box_ap, mask_ap = validate_box_and_mask(__C, net, loader_, writer, 0, gpu, val_set.ix_to_token,
                                                save_ids=save_ids, prefix=prefix_)
        print("Box_AP:", box_ap)
        print("Mask_AP:", mask_ap)


def main():
    parser = argparse.ArgumentParser(description="RefCLIP")
    parser.add_argument('--config', type=str, default='config/refcoco.yaml')
    parser.add_argument('--eval-weights', type=str, default='')
    args = parser.parse_args()
    assert args.config is not None
    __C = config.load_cfg_from_cfg_file(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in __C.GPU)
    setup_unique_version(__C)
    seed_everything(__C.SEED)
    N_GPU = len(__C.GPU)
    __C.RESUME_PATH = args.eval_weights

    SAVE_PATH = os.path.join(__C.LOG_PATH, __C.MODEL, 'seed_{}'.format(__C.SEED), str(__C.VERSION))
    if not os.path.exists(SAVE_PATH):
        os.makedirs(os.path.join(SAVE_PATH, 'ckpt'), exist_ok=True)
    # if not os.path.exists(os.path.join(__C.LOG_PATH, str(__C.VERSION))):
    #     os.makedirs(os.path.join(__C.LOG_PATH, str(__C.VERSION), 'ckpt'), exist_ok=True)

    if N_GPU == 1:
        __C.MULTIPROCESSING_DISTRIBUTED = False
    else:
        # turn on single or multi node multi gpus training
        __C.MULTIPROCESSING_DISTRIBUTED = True
        __C.WORLD_SIZE *= N_GPU
        __C.DIST_URL = f"tcp://127.0.0.1:{find_free_port()}"
    if __C.MULTIPROCESSING_DISTRIBUTED:
        mp.spawn(main_worker, args=(__C,), nprocs=N_GPU, join=True)
    else:
        main_worker(__C.GPU, __C)


if __name__ == '__main__':
    main()
