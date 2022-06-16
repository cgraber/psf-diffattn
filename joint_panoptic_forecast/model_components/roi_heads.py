import logging
import itertools
from typing import Dict, List, Optional, Tuple
import math
import sys
from torchvision.utils import save_image

from numpy.core.fromnumeric import resize

import torch
from torch import nn 

from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.structures import ImageList, Instances, Boxes, PolygonMasks
from .mask_utils import crop_and_resize_2d, polygons_to_bitmask

@ROI_HEADS_REGISTRY.register()
class JointPanopticForecastROIHeads(StandardROIHeads):

    @configurable
    def __init__(
        self,
        *,
        return_pooled_feats=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.return_pooled_feats = return_pooled_feats
        

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        return ret

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        inference_mode = False,
        gt_box_instances = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class: `ROIHeads.forward`.
        """
        del images
        if self.training and not inference_mode:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
                      
            del targets
            return proposals, losses
        else:
            if inference_mode and self.box_head.training:
                set_eval = True
                self.box_head.eval()
                self.box_predictor.eval()
                if self.mask_on:
                    self.mask_head.eval()
            else:
                set_eval = False
            has_some_boxes = gt_box_instances is not None and any([b is not None for b in gt_box_instances])
            if gt_box_instances is None:
                with torch.no_grad():
                    pred_instances = self._forward_box(features, proposals)
                if has_some_boxes:
                    pred_instances = [proposal if gt_box is None else gt_box 
                                    for proposal, gt_box in zip(pred_instances, gt_box_instances)]
            else:
                pred_instances = gt_box_instances
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            if set_eval:
                self.box_head.train()
                self.box_predictor.train()
                if self.mask_on:
                    self.mask_head.train()
            return pred_instances, {}


    def _forward_mask(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        """
        Forward logic of the mask prediction branch.
        For now, only change from base class is that we cache the pooled features (to be used downstream)
        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.
        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(instances, self.num_classes)

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
            if self.return_pooled_feats:
                lens = [len(inst) for inst in instances]
                split_feats = features.split(lens)
                for inst, feat in zip(instances, split_feats):
                    inst.features = feat
        else:
            features = {f: features[f] for f in self.mask_in_features}
        return self.mask_head(features, instances)


    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.
        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.
        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.
        Returns:
            list[Instances]:
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances