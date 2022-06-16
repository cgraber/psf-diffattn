import os
import itertools
import contextlib
import copy
import numpy as np
import warnings
from typing import Callable, Dict, List, Optional, Union, Tuple
import torch
from torch import nn

from detectron2.structures import Boxes, pairwise_iou, ImageList, Instances
from detectron2.modeling import (
    META_ARCH_REGISTRY,
    build_backbone,
    build_proposal_generator,
    build_roi_heads,
    detector_postprocess,
)

from joint_panoptic_forecast.model_components.fg_bg_refine import build_fg_bg_refine


from ..model_components.batched_instance_sequence import BatchedInstanceSequence
from ..model_components.encoders import build_encoder
from ..model_components.decoders import build_decoder
from ..model_components.mask_models import build_mask_encoder, build_mask_decoder


@META_ARCH_REGISTRY.register()
class JointPanopticForecast_RefineOnly(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.use_saved_feat_mode = cfg.MODEL.USE_SAVED_FEAT_MODE

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        shape = self.backbone.output_shape()

        self.encoder = build_encoder(cfg, shape)
        self.decoder = build_decoder(cfg, shape)
        self.use_separate_mask_model = cfg.MODEL.USE_SEPARATE_MASK_MODEL
        if self.use_separate_mask_model:
            self.mask_encoder = build_mask_encoder(cfg)
            self.mask_decoder = build_mask_decoder(cfg)
        self.input_format = cfg.INPUT.FORMAT
        self.seq_len = cfg.INPUT.SEQ_LEN
        self.forecast_len = cfg.MODEL.FORECAST_LEN
        self.input_len = self.seq_len - self.forecast_len
        self.sub_batch_size = cfg.MODEL.SUB_BATCH_SIZE
        self.use_target_frame_inputs_only = cfg.MODEL.USE_TARGET_FRAME_INPUTS_ONLY
        self.use_target_frame_inputs = cfg.MODEL.USE_TARGET_FRAME_INPUTS
        self.target_frame_input_prob = cfg.MODEL.TARGET_FRAME_INPUT_PROB
        self.saved_feat_pred_masks = cfg.MODEL.SAVED_FEAT_PRED_MASKS or self.use_target_frame_inputs_only or self.use_target_frame_inputs
        self.pred_index = cfg.MODEL.PRED_INDEX

        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), False)
        if cfg.MODEL.ODOM_MEAN is not None:
            odom_size = cfg.MODEL.ODOM_SIZE
            self.register_buffer('odom_mean', torch.tensor(cfg.MODEL.ODOM_MEAN).view(1, odom_size), False)
            self.register_buffer('odom_std', torch.tensor(cfg.MODEL.ODOM_STD).view(1, odom_size), False)
        else:
            self.odom_mean = None
            self.odom_std = None


        for param in self.parameters():
            param.requires_grad = False

        self.refine_head = build_fg_bg_refine(cfg)

    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        image_seqs = [[img.to(self.device) for img in seq['images']] for seq in batched_inputs]
        image_seqs = [[(img - self.pixel_mean) / self.pixel_std for img in seq] for seq in image_seqs]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            image_seqs = [ImageList.from_tensors(images, self.backbone.size_divisibility) for images in image_seqs]
        return image_seqs

    def preprocess_odom(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        odom = [o['odometries'].to(self.device) for o in batched_inputs]
        if self.odom_mean is not None:
            odom = [(o - self.odom_mean) / self.odom_std for o in odom]
        return odom

    def forward(self, all_batched_inputs: Tuple[Dict[str, torch.Tensor]]): 
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * images: Tensor, image in (seq_len, C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if self.sub_batch_size is None:
            sub_batch_size = len(all_batched_inputs)
        else:
            sub_batch_size = self.sub_batch_size
        all_results = []
        final_last_frame_proposals = []
        for sub_batch_idx in range(0, len(all_batched_inputs), sub_batch_size):
            batched_inputs = all_batched_inputs[sub_batch_idx:sub_batch_idx+sub_batch_size]
            if self.use_saved_feat_mode:
                if 'images' in batched_inputs[0]:
                    image_seqs = self.preprocess_image(batched_inputs)
                else:
                    image_seqs = [None for _ in range(len(batched_inputs))]
                all_odom = self.preprocess_odom(batched_inputs)
                proposals = [b['proposals'] for b in batched_inputs]
                proposals = [[p.to(self.device) for p in b] for b in proposals] 
                odom = torch.stack(all_odom)
                final_proposals = []
                for b_idx in range(len(proposals)):
                    current_p = proposals[b_idx]
                    feats = torch.cat([p.features for p in current_p])

                    if self.saved_feat_pred_masks:
                        with torch.no_grad():
                            self.roi_heads.mask_head.eval()
                            current_p = self.roi_heads.mask_head(feats, current_p)
                            if self.training:
                                self.roi_heads.mask_head.train()
                    final_proposals.append(current_p)
                proposals = final_proposals
            else:
                image_seqs = self.preprocess_image(batched_inputs)
                if 'odometries' in batched_inputs[0]:
                    all_odom = self.preprocess_odom(batched_inputs)
                    odom = torch.stack(all_odom)
                else:
                    odom = None
                all_proposals = []
                for b_idx, images in enumerate(image_seqs):
                    all_features = []
                    if self.freeze_backbone:
                        with torch.no_grad():
                            features = self.backbone(images.tensor)
                    else:
                        features = self.backbone(images.tensor)
                    if self.use_gt_boxes:
                        proposals = None
                        gt_box_proposals = batched_inputs[b_idx]['seq_instances']
                        gt_box_proposals = [p.to(self.device) for p in gt_box_proposals]
                        for p in gt_box_proposals:
                            p.pred_boxes = p.gt_boxes
                            p.pred_classes = p.gt_classes
                    else:
                        with torch.no_grad():
                            self.proposal_generator.eval() # required since we don't have gt
                            proposals, _ = self.proposal_generator(images, features, None) 
                            if self.training:
                                self.proposal_generator.train()
                        gt_box_proposals = None
                    with torch.no_grad():
                        self.roi_heads.eval() # required since we don't have gt
                        proposals, _ = self.roi_heads(images, features, features, proposals, None,
                                                        inference_mode=True, gt_box_instances=gt_box_proposals) 
                        if self.training:
                            self.roi_heads.train()
                        all_proposals.append(proposals)
                    del features
                proposals = all_proposals
            if self.training and self.use_target_frame_inputs_only:
                target_proposals = [p[-1] for p in proposals]
                for p in target_proposals:
                    p.decoder_pred_masks = p.pred_masks.clone()
                all_results += target_proposals
            else:
                start_idx = self.input_len
                orig_proposals = proposals
                proposals = BatchedInstanceSequence(proposals)
                input_proposals = proposals[:start_idx]
                start_proposals = input_proposals[-1]
                orig_target_proposals = target_proposals = proposals[-1]
                if odom is not None:
                    input_odom = odom[:, :start_idx]
                    target_odom = odom[:, start_idx:]
                else:
                    input_odom = target_odom = None
                with torch.no_grad():
                    encoder_feats = self.encoder(input_proposals, input_odom)
                    pred_instances, self_attns, cross_attns = self.decoder(encoder_feats, input_proposals, start_proposals,
                                                                            target_odom, self.input_len, self.forecast_len)
                    if self.use_separate_mask_model:
                        mask_encoder_feats = self.mask_encoder(input_proposals, input_odom)
                        pred_instances = self.mask_decoder(mask_encoder_feats, input_proposals, start_proposals, pred_instances,
                                                        target_odom, self.input_len, self.forecast_len)
                                                    
                    final_pred_instances = [i[0] for i in pred_instances[self.pred_index].get_instances()]
                    if final_pred_instances[0].has('mask_feats'):
                        feats = torch.cat([p.mask_feats for p in final_pred_instances])
                        final_pred_instances = self.roi_heads.mask_head(feats, final_pred_instances, inference_mode=True, name_prefix='decoder')
                    
                    new_target_proposals = orig_target_proposals.get_instances()


                    all_results += final_pred_instances
                    final_last_frame_proposals += [new_target_proposals[b_idx][self.pred_index] for b_idx in range(len(new_target_proposals))]

        bg_logits = [b['bg_logits'].to(self.device) for b in all_batched_inputs]
        bg_depths = [b['bg_depths'].to(self.device) for b in all_batched_inputs]
        bg_depth_masks = [b['bg_depth_masks'].to(self.device) for b in all_batched_inputs]
        gt_pan_segs = [b['gt_pan_seg'].to(self.device) for b in all_batched_inputs]
        gt_pan_seg_instances = [b['gt_pan_instances'].to(self.device) for b in all_batched_inputs]
        if 'bg_feats' in all_batched_inputs[0]:
            bg_feats = [b['bg_feats'].to(self.device) for b in all_batched_inputs]
        else:
            bg_feats = [None for b in all_batched_inputs]
        img_ids = [b['image_id'] for b in all_batched_inputs]
        if self.refine_head.supervise_with_instances:
            if not final_last_frame_proposals[0].has('pred_masks'):
                    feats = torch.cat([p.features for p in final_last_frame_proposals])
                    with torch.no_grad():
                        self.roi_heads.mask_head.eval()
                        final_last_frame_proposals = self.roi_heads.mask_head(feats, final_last_frame_proposals)
                        if self.training:
                            self.roi_heads.mask_head.train()
            result = self.refine_head(all_results, bg_logits, bg_depths, bg_depth_masks,
                                             final_last_frame_proposals, gt_pan_segs,
                                             gt_pan_seg_instances, bg_feats, img_ids)
        else:
            result = self.refine_head(all_results, bg_logits, bg_depths, bg_depth_masks,
                                    gt_pan_segs, gt_pan_seg_instances)
        if self.training:
            return result
        else:
            segs, instances = result
            return [{'panoptic_segmentation':s,
                     'instances':i} for s,i in zip(segs, instances)]