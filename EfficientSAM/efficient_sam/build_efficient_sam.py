# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .efficient_sam import build_efficient_sam
import os

current_directory = os.path.abspath(os.getcwd())
print(current_directory)


def build_efficient_sam_vitt():
    return build_efficient_sam(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        checkpoint=os.path.join(current_directory, "EfficientSAM/weights/efficient_sam_vitt.pt"),
    ).eval()


def build_efficient_sam_vits():
    return build_efficient_sam(
        encoder_patch_embed_dim=384,
        encoder_num_heads=6,
        checkpoint=os.path.join(current_directory, "EfficientSAM/weights/efficient_sam_vits.pt"),
    ).eval()
