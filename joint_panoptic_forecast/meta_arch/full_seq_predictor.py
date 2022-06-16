from typing import Callable, Dict, List, Optional, Union, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.structures import Instances, Boxes, pairwise_iou, ImageList
from detectron2.modeling import (
    META_ARCH_REGISTRY,
    build_backbone,
    build_proposal_generator,
    build_roi_heads,
)
from detectron2.modeling.poolers import ROIPooler



@META_ARCH_REGISTRY.register()
class FullSeqPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.input_format = cfg.INPUT.FORMAT
        self.use_gt_boxes = cfg.MODEL.USE_GT_BOXES
        self.use_saved_feat_mode = cfg.MODEL.USE_SAVED_FEAT_MODE
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), False)

        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        input_shape = self.backbone.output_shape()
        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        self.mask_pooler = ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )


    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess_seq_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        image_seqs = [[img.to(self.device) for img in seq['images']] for seq in batched_inputs]
        image_seqs = [[(img - self.pixel_mean) / self.pixel_std for img in seq] for seq in image_seqs]
        #image_seqs = [ImageList.from_tensors(images, self.backbone.size_divisibility) for images in image_seqs]
        return image_seqs

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [b['image'].to(self.device) for b in batched_inputs]
        images = [(img - self.pixel_mean) / self.pixel_std for img in images]
        #image_seqs = [ImageList.from_tensors(images, self.backbone.size_divisibility) for images in image_seqs]
        return images


    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
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
        if self.use_saved_feat_mode:
            proposals = [b['proposals'] for b in batched_inputs]
            proposals = [[p.to(self.device) for p in b] for b in proposals]
            final_results = []
            for b_idx in range(len(proposals)):
                current_p = proposals[b_idx]
                feats = torch.cat([p.features for p in current_p])
                current_p = self.roi_heads.mask_head(feats, current_p)
                embs = self.roi_heads._forward_contrastive(None, current_p, inference_mode=True,
                                                           pooled_features=feats)
                for inst, emb in zip(current_p, embs):
                    inst.input_viz_embs = emb
                final_results.append(
                    {'proposals':current_p}
                )
            return final_results
        else:
            image_seqs = self.preprocess_seq_image(batched_inputs)
            if 'proposals' in batched_inputs[0]:
                instances = [b['proposals'] for b in batched_inputs]
                instances = [[p.to(self.device) for p in b] for b in instances]
            all_results = []
            for b_idx, images in enumerate(image_seqs):
                all_viz_feats = []
                all_proposals = []
                sub_batch_size = 4
                for im_idx in range(0, len(images), 4):
                    current_imgs = ImageList.from_tensors(images[im_idx:im_idx+sub_batch_size], self.backbone.size_divisibility)
                    with torch.no_grad():
                        features = self.backbone(current_imgs.tensor)
                        if self.use_gt_boxes:
                            current_insts = instances[b_idx][im_idx:im_idx+sub_batch_size]
                            proposals, _ = self.roi_heads(current_imgs, features, None, None, inference_mode=True, gt_box_instances=current_insts)
                        else:
                            self.proposal_generator.eval() # required since we don't have gt
                            proposals, _ = self.proposal_generator(current_imgs, features, None) #TODO: trace here, make sure we're getting what we want
                            self.proposal_generator.train()
                            self.roi_heads.eval() # required since we don't have gt
                            proposals, _ = self.roi_heads(current_imgs, features, proposals, None) #TODO: trace here, make sure we're getting what we want
                            self.roi_heads.train()
                            boxes = [i.pred_boxes for i in proposals]
                        #tmp_features = [features[f] for f in self.in_features]
                        #viz_feats = self.mask_pooler(tmp_features, boxes)
                        #all_viz_feats.append(viz_feats)
                        all_proposals.append(proposals)
                #all_viz_feats = torch.cat(all_viz_feats)
                all_proposals = sum(all_proposals, [])
                all_results.append({
                    #'feats':all_viz_feats,
                    'proposals':all_proposals,
                })
        

        if self.training:
            raise NotImplementedError()
        else:
            return all_results