#!/usr/bin/env bash


srun -p 'INTERN2' --gres=gpu:1 --nodes=1 --ntasks-per-node=1 --cpus-per-task=12 --kill-on-bad-exit=1  --quotatype="reserved" --job-name='eval_luogen' python train.py --config config/RefCLIP_WRECS_SimREC_one_lang_no_detach_visual_share_pred_box_as_prompt_proj_dynamic_iou_consis.yaml

srun -p 'INTERN2' --gres=gpu:1 --nodes=1 --ntasks-per-node=1 --cpus-per-task=12 --kill-on-bad-exit=1  --quotatype="reserved" --job-name='eval_luogen' python train.py --config config/refcoco+_WRECS_SimREC_one_lang_no_detach_visual_share_pred_box_as_prompt_proj_dynamic_iou_consis.yaml

srun -p 'INTERN2' --gres=gpu:1 --nodes=1 --ntasks-per-node=1 --cpus-per-task=12 --kill-on-bad-exit=1  --quotatype="reserved" --job-name='eval_luogen' python train.py --config config/RefCLIP_WRECS_SimREC_clip_proj_dynamic_iou_consis.yaml

srun -p 'INTERN2' --gres=gpu:1 --nodes=1 --ntasks-per-node=1 --cpus-per-task=12 --kill-on-bad-exit=1  --quotatype="reserved" --job-name='eval_luogen' python train_semi.py --config config/refcoco+_WRECS_proj_dynamic_active_semi.yaml

srun -p 'INTERN2' --gres=gpu:1 --nodes=1 --ntasks-per-node=1 --cpus-per-task=12 --kill-on-bad-exit=1  --quotatype="reserved" --job-name='eval_luogen' python train_semi.py --config config/RefCLIP_WRECS_proj_dynamic_active_semi.yaml

srun -p 'INTERN2' --gres=gpu:1 --nodes=1 --ntasks-per-node=1 --cpus-per-task=12 --kill-on-bad-exit=1  --quotatype="reserved" --job-name='eval_luogen' python train_semi.py --config config/refcocog_WRECS_proj_dynamic_active_semi.yaml