# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


def add_memory_config(cfg):
    """
    Add config for Memory bank.
    """
    cfg.MODEL.ROI_HEADS.TEMP_S = 48
    cfg.MODEL.ROI_HEADS.MIN_CACHE = 20
    cfg.MODEL.ROI_HEADS.MAX_CACHE = 60
    cfg.MODEL.ROI_HEADS.RANDOM_SELECT = False
    cfg.MODEL.ROI_HEADS.CACHE_CAT_FILE = "lvis0.5_rare_cats.txt"
    cfg.MODEL.ROI_HEADS.CLS_LAYER = "cosine"
    cfg.MODEL.ROI_HEADS.RUN = 1

