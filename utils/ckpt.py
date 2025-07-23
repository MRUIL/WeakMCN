# coding=utf-8

import os
import warnings

import torch
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP


def save_ckpt(net, optimizer,scheduler, misc, __C):
    path = __C.CKPTs_PATH
    if not os.path.exists(path):
        os.mkdir(path)
    path += '/' + __C.VERSION
    if not os.path.exists(path):
        os.mkdir(path)
    assert isinstance(misc, dict)
    if isinstance(net, DP) or isinstance(net, DDP):
        path += '/' + 'dist_'
    path += str(misc['epoch']) + '.pth.tar'
    ckpt = {
        'net_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler':scheduler.state_dict(),
        'epoch':misc['epoch'],
        'lr':optimizer.param_groups[0]["lr"],
    }
    torch.save(ckpt, path)


def load_ckpt(net, optimizer,scheduler, path, rank=None):
    loc = f'cuda:{rank}' if rank is not None else None
    ckpt = torch.load(path, map_location=loc)

    flag = isinstance(net, DP) or isinstance(net, DDP)
    if '_dist' in path:
        if not flag:
            for name in ckpt['net_state_dict']:
                assert name.startswith('module.')
                ckpt['net_state_dict'][name.lstrip('module.')] = ckpt['net_state_dict'].pop(name)
    else:
        if flag:
            for name in ckpt['net_state_dict']:
                ckpt['net_state_dict']['module.' + name] = ckpt['net_state_dict'].pop(name)

    optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    scheduler.load_state_dict(ckpt['scheduler'])

    missing, unexpected = net.load_state_dict(ckpt['net_state_dict'], strict=False)
    if unexpected.__len__ != 0:
        warnings.warn(f'Current model misses {unexpected.__len__} parameters from checkpointing model')
        for name in missing:
            print('\n' + name + '\n')
    if missing.__len__ != 0:
        warnings.warn(f'Current model contains {missing.__len__} parameters that checkpointing model doesn\'t contain')
        for name in unexpected:
            print('\n' + name + '\n')

    return ckpt