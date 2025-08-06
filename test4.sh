#!/usr/bin/env bash

#set -x
#GPUS=1  # less than

# 定义要遍历的目录
BASE_DIR="./logs/refcoco+_weakmcn_vits/weakmcn/seed_123456"

#BASE_DIR="./logs/refcoco+_WRECS_SimREC_one_lang_no_detach_visual_share_pred_box_as_prompt/RefCLIP_WRECS_SimREC_one_lang_no_detach_visual_share_pred_box_as_prompt/seed_234567"

# 遍历所有子目录
for dir in "$BASE_DIR"/*; do
    if [ -d "$dir" ]; then  # Check if it is a directory
        echo "Traversing directory: $dir"

        # Specify the directory to look for checkpoints
        checkpoint_dir="$dir/ckpt"
        checkpoint="$checkpoint_dir/seg_best.pth"

        # Check if the ckpt directory exists and look for seg_best.pth file
        if [ -d "$checkpoint_dir" ] && [ -f "$checkpoint" ]; then
            echo "Found checkpoint file: $checkpoint"
            # Run srun command
            python test.py --config config/refcoco+_tuning_v2.yaml --eval-weights "$checkpoint"
        fi
    fi
done

