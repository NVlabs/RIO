# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import inspect, pdb
import pickle
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import copy
import torch
from torch import nn

from detectron2.layers import ShapeSpec, cat
from detectron2.structures import Boxes, Instances, ImageList
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.utils.events import get_event_storage

from detectron2.config import configurable
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.fast_rcnn import (
    FastRCNNOutputs,
    FastRCNNOutputLayers,
)

from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads

from .fast_rcnn import MemoryFastRCNNOutputLayers
from .lvis_v0_5_categories import get_image_count_frequency

logger = logging.getLogger(__name__)


@ROI_HEADS_REGISTRY.register()
class MemoryROIHeads(StandardROIHeads):
    """
    RIO's memory bank is implemented in classification branch after ROIs are extracted. All memory bank operations (Sample, Dequeue, Push) are located here. 
    """

    @configurable
    def __init__(
        self,
        *,
        temp_S=None,
        min_cache=None,
        max_cache=None,
        cls_layer=None,
        random_select=None,
        cache_category_file=None,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:   
            temp_S (int): s factor for cosine layer.
            min_cache (int): the number of samples to sample from the memory bank each time.
            max_cache (int): the maximum number of samples to keep in memory bank per class. 
            cls_layer (str): the type of classification layer used. Either "linear" or "cosine".
            random_select (bool): by default the memory bank samples from the top. If True, memory bank will randomly select samples instead.
            cache_category_file (str): filename containing the target classes that are resampled via object resampling. This is used to create the memory bank.

            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask head.
                None if not using mask head.
            mask_pooler (ROIPooler): pooler to extra region features for mask head
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(**kwargs)
        self.cls_layer = cls_layer
        if "lvis" in cache_category_file:
            self.cache_categories = np.array(
                [int(x.rstrip()) - 1 for x in open(cache_category_file)]
            )
        else:
            logging.critical("not using lvis bank category file")

        if "lvis0" in cache_category_file:
            assert 1230 not in self.cache_categories
            assert self.num_classes == 1230
        elif "lvis1" in cache_category_file:
            assert 1203 not in self.cache_categories
            assert self.num_classes == 1203

        self.memory_cache = {
            c: {
                "box_features": np.empty((max_cache, 1024), dtype=np.float32),
                "proposals": np.empty((max_cache,), dtype=Instances),
            }
            for c in self.cache_categories
        }
        self.memory_cache_max_idx = np.zeros(self.num_classes, dtype=int)
        self.min_cache = min_cache
        self.max_cache = max_cache
        self.random_select = random_select

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["cls_layer"] = cfg.MODEL.ROI_HEADS.CLS_LAYER
        ret["temp_S"] = cfg.MODEL.ROI_HEADS.TEMP_S
        ret["min_cache"] = cfg.MODEL.ROI_HEADS.MIN_CACHE
        ret["max_cache"] = cfg.MODEL.ROI_HEADS.MAX_CACHE
        ret["random_select"] = cfg.MODEL.ROI_HEADS.RANDOM_SELECT
        ret["cache_category_file"] = cfg.MODEL.ROI_HEADS.CACHE_CAT_FILE
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels, height=pooler_resolution, width=pooler_resolution
            ),
        )
        box_predictor = MemoryFastRCNNOutputLayers(cfg, box_head.output_shape)

        freq_info = torch.FloatTensor(get_image_count_frequency())
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    def update_memory_bank(self, box_features, proposals):
        for p_idx in range(len(proposals)):
            p = proposals[p_idx]
            cur_gt_classes = p.gt_classes.cpu().numpy()
            rare_idxs = np.where(np.isin(cur_gt_classes, self.cache_categories))[0]
            if len(rare_idxs) > 0:
                for idx in rare_idxs:
                    c = cur_gt_classes[idx]
                    # append current feat to cache
                    feat = box_features[p_idx * len(p) + idx].detach().cpu().clone()
                    prop = copy.deepcopy(p[int(idx)])

                    # shift if exceeding max cache space
                    if self.memory_cache_max_idx[c] == self.max_cache:
                        self.memory_cache[c]["box_features"][
                            :-1, :
                        ] = self.memory_cache[c]["box_features"][1:, :]
                        self.memory_cache[c]["proposals"][:-1] = self.memory_cache[c][
                            "proposals"
                        ][1:]
                        # reset to last index (allow to grow by 1)
                        self.memory_cache_max_idx[c] = self.max_cache - 1
                    # append to cache
                    self.memory_cache[c]["box_features"][
                        self.memory_cache_max_idx[c]
                    ] = feat
                    self.memory_cache[c]["proposals"][
                        self.memory_cache_max_idx[c]
                    ] = prop

                    self.memory_cache_max_idx[c] += 1

    def use_memory_cache(self, box_features, proposals):
        # check if we're using memory bank
        if not self.min_cache:
            gt_classes = cat([p.gt_classes for p in proposals], dim=0)
            return proposals, box_features, gt_classes

        augmented_proposals = proposals

        # target classes that exist in the current batch
        target_classes = []
        for p_idx in range(len(proposals)):
            p = proposals[p_idx]
            cur_gt_classes = p.gt_classes.cpu().numpy()
            rare_idxs = np.where(np.isin(cur_gt_classes, self.cache_categories))[0]
            target_classes.extend(cur_gt_classes[rare_idxs])

        # count number of instances per category for any targeted category
        target_instances = dict()
        for i in target_classes:
            target_instances[i] = target_instances.get(i, 0) + 1

        new_proposals = []
        new_features = torch.tensor([]).cuda()
        for c in set(target_classes):
            num_samp_cache = self.memory_cache_max_idx[c]
            if num_samp_cache > 0:
                # get from cache x amount of samples. x = num_new_samps
                # use either the designated amount of samples (default 20) or the minimum amount in the current cache
                num_new_samps = min(self.min_cache, num_samp_cache)

                if self.random_select and num_samp_cache > num_new_samps:
                    cache_idxs = np.random.choice(
                        num_samp_cache, num_new_samps, replace=False
                    )
                else:
                    cache_idxs = np.arange(
                        num_samp_cache - num_new_samps, self.memory_cache_max_idx[c]
                    )

                new_feats = torch.from_numpy(
                    self.memory_cache[c]["box_features"][cache_idxs]
                )
                new_features = torch.cat((new_features, new_feats.cuda()), dim=0)
                if None in self.memory_cache[c]["proposals"][cache_idxs]:
                    pdb.set_trace()
                new_proposals.extend(self.memory_cache[c]["proposals"][cache_idxs])

        # update memory bank with current model
        self.update_memory_bank(box_features, proposals)

        if len(new_proposals) > 0:
            box_features = torch.cat((box_features, new_features), dim=0)
            augmented_proposals = copy.deepcopy(proposals)
            augmented_proposals.extend(new_proposals)

        all_gt_classes = cat([p.gt_classes for p in augmented_proposals], dim=0)
        assert len(all_gt_classes) == len(box_features)

        return augmented_proposals, box_features, all_gt_classes

    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.
        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.in_features]  # FPN features p1,etc...
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)

        augmented_proposals = proposals
        if self.training:
            augmented_proposals, box_features, all_gt_classes, = self.use_memory_cache(
                box_features, proposals
            )
            predictions = self.box_predictor(box_features, all_gt_classes, train=True)
        else:
            predictions = self.box_predictor(box_features, gt_classes=None, train=False)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, augmented_proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                        proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            (pred_instances, _,) = self.box_predictor.inference(predictions, proposals)
            return pred_instances
