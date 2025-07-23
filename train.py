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
from importlib import import_module
import torch.optim as Optim
from test import validate_box_and_mask, validate
from utils.utils import EMA
import torch.nn as nn


class ModelLoader:
    def __init__(self, __C):
        self.model_use = __C.MODEL
        model_moudle_path = 'models.' + self.model_use + '.net'
        self.model_moudle = import_module(model_moudle_path)

    def Net(self, __arg1, __arg2, __arg3):
        return self.model_moudle.Net(__arg1, __arg2, __arg3)


def train_one_epoch(__C,
                    net,
                    optimizer,
                    scheduler,
                    loader,
                    scalar,
                    writer,
                    epoch,
                    rank,
                    ema=None):
    net.train()
    if __C.MULTIPROCESSING_DISTRIBUTED:
        loader.sampler.set_epoch(epoch)
    batches = len(loader)
    batch_time = AverageMeter('Time', ':6.5f')
    data_time = AverageMeter('Data', ':6.5f')
    losses = AverageMeter('Loss', ':.4f')
    losses_det = AverageMeter('LossDet', ':.4f')
    losses_mask = AverageMeter('LossSeg', ':.4f')
    lr = AverageMeter('lr', ':.5f')
    meters = [batch_time, data_time, losses, losses_det, losses_mask, lr]
    meters_dict = {meter.name: meter for meter in meters}
    progress = ProgressMeter(__C.VERSION, __C.EPOCHS, len(loader), meters, prefix='Train: ')
    end = time.time()

    for ith_batch, data in enumerate(loader):
        data_time.update(time.time() - end)
        # ref_iter, image_iter, box_iter, gt_box_iter, info_iter, ref_txt = data
        ref_iter, image_iter, mask_iter, box_iter, gt_box_iter, mask_id, info_iter, ref_txt, img_path = data
        # print(ref_txt)
        mask_iter = mask_iter.cuda(non_blocking=True)
        ref_iter = ref_iter.cuda(non_blocking=True)
        image_iter = image_iter.cuda(non_blocking=True)
        box_iter = box_iter.cuda(non_blocking=True)

        # mask_ori = mask_iter.cpu().numpy()
        # for i, mask_pred in enumerate(mask_ori):
        #     if writer is not None:
        #         ixs = ref_iter[i].cpu().numpy()
        #         words = []
        #         for ix in ixs:
        #             if ix > 0:
        #                 words.append(loader.dataset.ix_to_token[ix])
        #         sent = ' '.join(words)
        #         box_iter = box_iter.view(box_iter.shape[0], -1) * __C.INPUT_SHAPE[0]
        #         box_iter[:, 0] = box_iter[:, 0] - 0.5 * box_iter[:, 2]
        #         box_iter[:, 1] = box_iter[:, 1] - 0.5 * box_iter[:, 3]
        #         box_iter[:, 2] = box_iter[:, 0] + box_iter[:, 2]
        #         box_iter[:, 3] = box_iter[:, 1] + box_iter[:, 3]
        #         # det_image = draw_visualization(normed2original(image_iter[i], __C.MEAN, __C.STD), sent,
        #         #                                pred_box_vis[i], box_iter[i].cpu().numpy())
        #         # writer.add_image('image/' + str(ith_batch * __C.BATCH_SIZE + i) + '_det', det_image, epoch,
        #         #                  dataformats='HWC')
        #         writer.add_image('image/' + str(ith_batch * __C.BATCH_SIZE + i) + '_det', image_iter[i], epoch,
        #                          dataformats='HWC')
        #         writer.add_image('image/' + str(ith_batch * __C.BATCH_SIZE + i) + '_seg',
        #                          (mask_ori[i] * 255).astype(np.uint8))

        if scalar is not None:
            with th.cuda.amp.autocast():
                # loss = net(image_iter, ref_iter)
                # loss = net(image_iter, ref_iter, mask_gt=mask_iter)
                loss_det, loss_mask = net(image_iter, ref_iter, __C.USING_CHECKPOINT, box_gt=box_iter,
                                          mask_gt=mask_iter)
                if epoch >= __C.WARM_EPOCH:
                    loss = loss_det + loss_mask
                else:
                    loss = loss_det
        else:
            # loss = net(image_iter, ref_iter, gt_box_iter, info_iter)
            loss_det, loss_mask = net(image_iter, ref_iter, box_gt=box_iter, mask_gt=mask_iter, info_iter=info_iter, epoch=epoch)
            if loss_mask is not None:
                if epoch >= __C.WARM_EPOCH:
                    loss = loss_det + loss_mask
                else:
                    loss = loss_det
            else:
                loss = loss_det

        import gc
        gc.collect()
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        if scalar is not None:
            scalar.scale(loss).backward()
            scalar.step(optimizer)
            if __C.GRAD_NORM_CLIP > 0:
                nn.utils.clip_grad_norm_(
                    net.parameters(),
                    __C.GRAD_NORM_CLIP
                )
            scalar.update()
        else:
            loss.backward()
            if __C.GRAD_NORM_CLIP > 0:
                nn.utils.clip_grad_norm_(
                    net.parameters(),
                    __C.GRAD_NORM_CLIP
                )
            optimizer.step()

            # with open('parameters_log_w_queue.txt', 'w') as f:
            #     # 记录参数信息
            #
            #     # 梯度裁剪
            #     if __C.GRAD_NORM_CLIP > 0:
            #         nn.utils.clip_grad_norm_(
            #             net.parameters(),
            #             __C.GRAD_NORM_CLIP
            #         )
            #
            #     optimizer.step()
            #
            #     f.write("============= 更新之后 ===========\n")
            #
            #     # 记录更新后的参数信息
            #     for name, parms in net.named_parameters():
            #         if parms.requires_grad is False and parms.grad is None:
            #             f.write(f'--> name: {name}\n')
            #             f.write("===\n")
        scheduler.step()
        if ema is not None:
            ema.update_params()
        losses.update(loss.item(), image_iter.size(0))
        losses_det.update(loss_det.item(), image_iter.size(0))
        if loss_mask is not None:
            losses_mask.update(loss_mask.item(), image_iter.size(0))

        lr.update(optimizer.param_groups[0]["lr"], -1)
        reduce_meters(meters_dict, rank, __C)
        if main_process(__C, rank):
            global_step = epoch * batches + ith_batch
            writer.add_scalar("loss/train", losses.avg_reduce, global_step=global_step)
            writer.add_scalar("loss_det/train", losses_det.avg_reduce, global_step=global_step)
            if loss_mask is not None:
                writer.add_scalar("loss_mask/train", losses_mask.avg_reduce, global_step=global_step)
            if ith_batch % __C.PRINT_FREQ == 0 or ith_batch == len(loader):
                progress.display(epoch, ith_batch)
        # break
        batch_time.update(time.time() - end)
        end = time.time()


def main_worker(gpu, __C):
    global best_det_acc
    best_det_acc = 0.
    best_seg_acc = 0.
    if __C.MULTIPROCESSING_DISTRIBUTED:
        if __C.DIST_URL == "env://" and __C.RANK == -1:
            __C.RANK = int(os.environ["RANK"])
        if __C.MULTIPROCESSING_DISTRIBUTED:
            __C.RANK = __C.RANK * len(__C.GPU) + gpu
        dist.init_process_group(backend=dist.Backend('NCCL'), init_method=__C.DIST_URL, world_size=__C.WORLD_SIZE,
                                rank=__C.RANK)

    train_set = RefCOCODataSet(__C, split='train')
    train_loader = loader(__C, train_set, gpu, shuffle=(not __C.MULTIPROCESSING_DISTRIBUTED), drop_last=True)

    val_set = RefCOCODataSet(__C, split='val')
    val_loader = loader(__C, val_set, gpu, shuffle=False)

    net = ModelLoader(__C).Net(
        __C,
        train_set.pretrained_emb,
        train_set.token_size
    )
    # optimizer
    params = filter(lambda p: p.requires_grad, net.parameters())  # split_weights(net)
    std_optim = getattr(Optim, __C.OPT)

    eval_str = 'params, lr=%f' % __C.LR
    for key in __C.OPT_PARAMS:
        eval_str += ' ,' + key + '=' + str(__C.OPT_PARAMS[key])
    optimizer = eval('std_optim' + '(' + eval_str + ')')

    ema = None

    if os.path.isfile(__C.PRETRAIN_WEIGHT) and not os.path.isfile(__C.RESUME_PATH):

        checkpoint = torch.load(__C.PRETRAIN_WEIGHT, map_location=torch.device('cpu'))
        new_dict = {}

        for k in checkpoint:
            new_k = ''.join(['visual_encoder.', k])
            new_dict[new_k] = checkpoint[k]
        net.load_state_dict(new_dict, strict=False)
        if main_process(__C, gpu):
            print("==> loaded checkpoint from {}\n".format(__C.PRETRAIN_WEIGHT))

    if __C.MULTIPROCESSING_DISTRIBUTED:
        torch.cuda.set_device(gpu)
        net = DDP(net.cuda(), device_ids=[gpu], find_unused_parameters=True)
    elif len(gpu) == 1:
        net.cuda()
    else:
        net = DP(net.cuda())

    if main_process(__C, gpu):
        print(__C)
        # print(net)
        total = sum([param.nelement() for param in net.parameters()])
        print('  + Number of all params: %.2fM' % (total / 1e6))  # 每一百万为一个单位
        total = sum([param.nelement() for param in net.parameters() if param.requires_grad])
        print('  + Number of trainable params: %.2fM' % (total / 1e6))  # 每一百万为一个单位

    scheduler = get_lr_scheduler(__C, optimizer, len(train_loader))

    start_epoch = 0

    if os.path.isfile(__C.RESUME_PATH):
        checkpoint = torch.load(__C.RESUME_PATH, map_location=lambda storage, loc: storage.cuda())
        new_dict = {}
        for k in checkpoint['state_dict']:
            if 'module.' in k:
                new_k = k.replace('module.', '')
                new_dict[new_k] = checkpoint['state_dict'][k]
        if len(new_dict.keys()) == 0:
            new_dict = checkpoint['state_dict']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
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
    else:
        writer = None

    save_ids = np.random.randint(1, len(val_loader) * __C.BATCH_SIZE, 100) if __C.LOG_IMAGE else None

    best_det_acc_list = []
    for ith_epoch in range(start_epoch, __C.EPOCHS):
        if __C.USE_EMA and ema is None:
            ema = EMA(net, 0.9997)
        train_one_epoch(__C, net, optimizer, scheduler, train_loader, scalar, writer, ith_epoch, gpu, ema)
        # box_ap = validate(__C, net, val_loader, writer, ith_epoch, gpu, val_set.ix_to_token, save_ids=save_ids,
        #                            ema=ema)
        box_ap, mask_ap = validate_box_and_mask(__C, net, val_loader, writer, ith_epoch, gpu, val_set.ix_to_token,
                                                save_ids=save_ids,
                                                ema=ema)
        best_det_acc_list.append(box_ap)
        if main_process(__C, gpu):
            if ema is not None:
                ema.apply_shadow()
            torch.save({'epoch': ith_epoch + 1, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'lr': optimizer.param_groups[0]["lr"], },
                       os.path.join(SAVE_PATH, 'ckpt', 'last.pth'))
            # torch.save({'epoch': ith_epoch + 1, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(),
            #             'scheduler': scheduler.state_dict(), 'lr': optimizer.param_groups[0]["lr"], },
            #            os.path.join(__C.LOG_PATH, str(__C.VERSION), 'ckpt', 'last.pth'))
            if box_ap > best_det_acc:
                best_det_acc = box_ap
                torch.save({'epoch': ith_epoch + 1, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(), 'lr': optimizer.param_groups[0]["lr"], },
                           os.path.join(SAVE_PATH, 'ckpt', 'det_best.pth'))
                # torch.save({'epoch': ith_epoch + 1, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(),
                #             'scheduler': scheduler.state_dict(), 'lr': optimizer.param_groups[0]["lr"], },
                #            os.path.join(__C.LOG_PATH, str(__C.VERSION), 'ckpt', 'det_best.pth'))
            if mask_ap > best_seg_acc:
                best_seg_acc = mask_ap
                torch.save({'epoch': ith_epoch + 1, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(), 'lr': optimizer.param_groups[0]["lr"], },
                           os.path.join(SAVE_PATH, 'ckpt', 'seg_best.pth'))
                # torch.save({'epoch': ith_epoch + 1, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(),
                #             'scheduler': scheduler.state_dict(), 'lr': optimizer.param_groups[0]["lr"], },
                #            os.path.join(__C.LOG_PATH, str(__C.VERSION), 'ckpt', 'seg_best.pth'))
            if ema is not None:
                ema.restore()
        # if main_process(__C, gpu):
        #     if ema is not None:
        #         ema.apply_shadow()
        #     torch.save({'epoch': ith_epoch + 1, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(),
        #                 'scheduler': scheduler.state_dict(), 'lr': optimizer.param_groups[0]["lr"], },
        #                os.path.join(__C.LOG_PATH, str(__C.VERSION), 'ckpt', 'last.pth'))
        #     if box_ap > best_det_acc:
        #         best_det_acc = box_ap
        #         torch.save({'epoch': ith_epoch + 1, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(),
        #                     'scheduler': scheduler.state_dict(), 'lr': optimizer.param_groups[0]["lr"], },
        #                    os.path.join(__C.LOG_PATH, str(__C.VERSION), 'ckpt', 'det_best.pth'))
        #     if ema is not None:
        #         ema.restore()
    torch.save(best_det_acc_list, "{}/best_det_acc_list.pth".format(SAVE_PATH))
    if __C.MULTIPROCESSING_DISTRIBUTED:
        cleanup_distributed()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, default='./config/refcoco.yaml')
    args = parser.parse_args()
    assert args.config is not None
    __C = config.load_cfg_from_cfg_file(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in __C.GPU)
    setup_unique_version(__C)
    seed_everything(__C.SEED)
    N_GPU = len(__C.GPU)

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
