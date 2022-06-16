import os
import itertools
import contextlib
import copy
import numpy as np
import warnings
from typing import Callable, Dict, List, Optional, Union, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

from detectron2.structures import Boxes, pairwise_iou, ImageList, Instances
from detectron2.modeling import (
    META_ARCH_REGISTRY,
    build_backbone,
    build_proposal_generator,
    build_roi_heads,
    detector_postprocess,
)


from .. import utils
from ..model_components.batched_instance_sequence import BatchedInstanceSequence
from ..model_components.encoders import build_encoder
from ..model_components.decoders import build_decoder
from ..model_components.mask_models import build_mask_encoder, build_mask_decoder
from ..model_components.fg_bg_refine import build_fg_bg_refine





@META_ARCH_REGISTRY.register()
class JointPanopticForecast(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.use_saved_feat_mode = cfg.MODEL.USE_SAVED_FEAT_MODE
        if cfg.MODEL.DEPTH_MEAN is not None:
            self.register_buffer("depth_mean", torch.tensor(cfg.MODEL.DEPTH_MEAN), False)
            self.register_buffer("depth_std", torch.tensor(cfg.MODEL.DEPTH_STD), False)
        else:
            self.depth_mean = self.depth_std = None
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        shape = self.backbone.output_shape()
        self.encoder = build_encoder(cfg, self.backbone.output_shape())

        self.decoder = build_decoder(cfg, shape)
        self.use_separate_mask_model = cfg.MODEL.USE_SEPARATE_MASK_MODEL
        if self.use_separate_mask_model:
            self.mask_encoder = build_mask_encoder(cfg)
            self.mask_decoder = build_mask_decoder(cfg)
        self.is_val=False
        self.start_idxs = cfg.MODEL.START_IDXS
        self.input_format = cfg.INPUT.FORMAT
        self.seq_len = cfg.INPUT.SEQ_LEN
        self.forecast_len = cfg.MODEL.FORECAST_LEN
        self.input_len = self.seq_len - self.forecast_len
        self.iou_coef = cfg.MODEL.IOU_COEF
        self.dist_coef = cfg.MODEL.DIST_COEF
        self.emb_match_coef = cfg.MODEL.EMB_MATCH_COEF
        self.emb_loss_coef = cfg.MODEL.EMB_LOSS_COEF
        self.depth_loss_coef = cfg.MODEL.DEPTH_LOSS_COEF
        self.has_id = cfg.INPUT.USE_ID
        self.loss_dist_threshold = cfg.MODEL.LOSS_DIST_THRESHOLD
        self.edge_dist_threshold = cfg.MODEL.EDGE_DIST_THRESHOLD
        self.sub_batch_size = cfg.MODEL.SUB_BATCH_SIZE
        self.predict_encoder_vels = cfg.MODEL.TRAJECTORY_ENCODER.PREDICT_VEL
        self.predict_intermediate_encoder_vels = cfg.MODEL.TRAJECTORY_ENCODER.PREDICT_INTERMEDIATE_VEL
        self.predict_encoder_locs = cfg.MODEL.TRAJECTORY_ENCODER.PREDICT_LOC
        self.weight_dist_by_iou = cfg.MODEL.WEIGHT_DIST_BY_IOU
        self.iou_threshold = cfg.MODEL.IOU_THRESHOLD
        self.emb_threshold = cfg.MODEL.EMB_THRESHOLD
        self.filter_by_depth = cfg.MODEL.FILTER_BY_DEPTH
        self.depth_thresh = cfg.MODEL.DEPTH_THRESH
        self.presence_loss_coef = cfg.MODEL.PRESENCE_LOSS_COEF
        self.use_class_masking = cfg.MODEL.USE_CLASS_MASKING
        self.use_gt_boxes = cfg.MODEL.USE_GT_BOXES
        self.eval_with_start_boxes = cfg.MODEL.EVAL_WITH_START_BOXES
        self.mask_feat_loss_coef = cfg.MODEL.MASK_FEAT_LOSS_COEF
        self.encoder_next_mask_feat_loss_coef = cfg.MODEL.ENCODER_NEXT_MASK_FEAT_LOSS_COEF
        self.decoder_mask_logit_loss_coef = cfg.MODEL.DECODER_MASK_LOGIT_LOSS_COEF
        self.encoder_next_mask_logit_loss_coef = cfg.MODEL.ENCODER_NEXT_MASK_LOGIT_LOSS_COEF
        self.decoder_mask_loss_coef = cfg.MODEL.DECODER_MASK_LOSS_COEF
        self.encoder_next_mask_loss_coef = cfg.MODEL.ENCODER_NEXT_MASK_LOSS_COEF
        self.encoder_mask_loss_coef = cfg.MODEL.ENCODER_MASK_LOSS_COEF
        self.encoder_mask_logit_loss_coef = cfg.MODEL.ENCODER_MASK_LOGIT_LOSS_COEF
        self.eval_inst_seg = cfg.EVAL_INSTANCE_SEGMENTATION or cfg.EVAL_PANOPTIC_SEGMENTATION
        self.eval_pan_seg = cfg.EVAL_PANOPTIC_SEGMENTATION
        self.binarize_loss_masks = cfg.MODEL.BINARIZE_LOSS_MASKS
        self.use_mask_focal_loss = cfg.MODEL.USE_MASK_FOCAL_LOSS
        self.mask_focal_loss_gamma = cfg.MODEL.MASK_FOCAL_LOSS_GAMMA
        self.mask_model_only = cfg.MODEL.MASK_MODEL_ONLY
        self.saved_feat_pred_masks = cfg.MODEL.SAVED_FEAT_PRED_MASKS
        self.add_final_refine = cfg.MODEL.ADD_FINAL_REFINE
        self.use_pred_future_odometry = cfg.MODEL.USE_PRED_FUTURE_ODOMETRY
        self.pred_index = cfg.MODEL.PRED_INDEX
        self.img_width = cfg.INPUT.IMG_WIDTH
        self.img_height = cfg.INPUT.IMG_HEIGHT
        self.use_depths = cfg.MODEL.USE_DEPTHS
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), False)
        
        if cfg.MODEL.ODOM_MEAN is not None:
            odom_size = cfg.MODEL.ODOM_SIZE
            self.register_buffer('odom_mean', torch.tensor(cfg.MODEL.ODOM_MEAN).view(1, odom_size), False)
            self.register_buffer('odom_std', torch.tensor(cfg.MODEL.ODOM_STD).view(1, odom_size), False)
        else:
            self.odom_mean = None
            self.odom_std = None
        #self.meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        self._lossfn = nn.SmoothL1Loss()
        self.freeze_backbone = cfg.MODEL.BACKBONE.FREEZE
        self.freeze_roi = cfg.MODEL.ROI_HEADS.FREEZE
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        if self.freeze_roi:
            for param in self.proposal_generator.parameters():
                param.requires_grad = False
            for param in self.roi_heads.parameters():
                param.requires_grad = False
        
        if self.mask_model_only:
            for param in self.parameters():
                param.requires_grad = False
            for param in itertools.chain(self.mask_encoder.parameters(), self.mask_decoder.parameters()):
                param.requires_grad = True
        if self.add_final_refine:
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

    def preprocess_single_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def preprocess_features(self, features):
        feats = (features - self.pixel_mean) / self.pixel_std
        feats = {"synth": torch.cat(feats)}
        return feats


    def preprocess_odom(self, batched_inputs: Tuple[Dict[str, torch.Tensor]], is_pred=False):
        if is_pred:
            odom = [o['pred_future_odometries'].to(self.device) for o in batched_inputs]
        else:
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
        final_predicted_proposals = []
        final_target_proposals = []
        final_input_proposals = []
        final_start_viz_feats = []
        final_whole_seq_proposals = []
        final_whole_seq_target_proposals = []
        final_target_predictions = []
        final_last_frame_proposals = []
        
        for sub_batch_idx in range(0, len(all_batched_inputs), sub_batch_size):
            batched_inputs = all_batched_inputs[sub_batch_idx:sub_batch_idx+sub_batch_size]
            all_results = []
            if self.use_saved_feat_mode:
                if 'images' in batched_inputs[0]:
                    image_seqs = self.preprocess_image(batched_inputs)
                else:
                    image_seqs = [None for _ in range(len(batched_inputs))]
                all_odom = self.preprocess_odom(batched_inputs)
                proposals = [b['proposals'] for b in batched_inputs]
                proposals = [[p.to(self.device) for p in b] for b in proposals] 
                odom = torch.stack(all_odom)
                if self.use_pred_future_odometry:
                    future_pred_odom = self.preprocess_odom(batched_inputs, is_pred=True)
                    future_pred_odom = torch.stack(future_pred_odom)
                final_proposals = []
                for b_idx in range(len(proposals)):
                    current_p = proposals[b_idx]
                    feats = torch.cat([p.features for p in current_p])

                    if self.decoder_mask_logit_loss_coef is not None or self.decoder_mask_loss_coef is not None or self.saved_feat_pred_masks:
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
                if self.use_pred_future_odometry:
                    future_pred_odom = self.preprocess_odom(batched_inputs, is_pred=True)
                    future_pred_odom = torch.stack(future_pred_odom)
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
                            proposals, _ = self.proposal_generator(images, features, None) #TODO: trace here, make sure we're getting what we want
                            if self.training:
                                self.proposal_generator.train()
                        if not self.training and self.eval_with_start_boxes:
                            gt_box_proposals = [None for _ in range(self.seq_len)]
                            start_boxes = batched_inputs[b_idx]['start_frame_instances']
                            start_boxes.pred_classes = start_boxes.gt_classes
                            start_boxes.pred_boxes = start_boxes.gt_boxes
                            target_boxes = batched_inputs[b_idx]['target_frame_instances']
                            target_boxes.pred_classes = target_boxes.gt_classes
                            target_boxes.pred_boxes = target_boxes.gt_boxes
                            gt_box_proposals[-1] = target_boxes
                            gt_box_proposals[self.seq_len-self.forecast_len-1] = start_boxes
                        else:
                            gt_box_proposals = None
                    if self.freeze_roi:
                        with torch.no_grad():
                            self.roi_heads.eval() # required since we don't have gt
                            proposals, _ = self.roi_heads(images, features, proposals, None,
                                                          inference_mode=True, gt_box_instances=gt_box_proposals) 
                            if self.training:
                                self.roi_heads.train()
                            all_proposals.append(proposals)
                    else:
                        proposals, _ = self.roi_heads(images, features, proposals, None,
                                                      inference_mode=True, gt_box_instances=gt_box_proposals)
                        all_proposals.append(proposals)
                    del features


                proposals = all_proposals

            if not self.training or self.is_val or self.start_idxs is None:
                start_idxs = [self.input_len]
            else:
                start_idxs = self.start_idxs
            proposals = BatchedInstanceSequence(proposals)
            
            for start_idx in start_idxs:
                
                input_proposals = proposals[:start_idx]
                target_proposals = proposals[start_idx:]
                start_proposals = input_proposals[-1]
                if odom is not None:
                    input_odom = odom[:, :start_idx]
                    target_odom = odom[:, start_idx:]
                else:
                    input_odom = target_odom = None
                if self.use_pred_future_odometry:
                    target_odom = future_pred_odom
                if self.mask_model_only:
                    ctx_mgr = torch.no_grad()
                else:
                    ctx_mgr = contextlib.nullcontext()
                with ctx_mgr:
                    encoder_feats = self.encoder(input_proposals, input_odom)
                    pred_instances, self_attns, cross_attns = self.decoder(encoder_feats, input_proposals, start_proposals,
                                                                        target_odom, self.input_len, self.forecast_len,
                                                                        )
                if self.use_separate_mask_model:
                    mask_encoder_feats = self.mask_encoder(input_proposals, input_odom)
                    pred_instances = self.mask_decoder(mask_encoder_feats, input_proposals, start_proposals, pred_instances,
                                                       target_odom, self.input_len, self.forecast_len)
                final_input_proposals += input_proposals.get_instances()
                new_target_proposals = target_proposals.get_instances()
                new_predicted_proposals = pred_instances.get_instances()
                has_mask_pred = False
                if (self.decoder_mask_logit_loss_coef is not None or self.decoder_mask_loss_coef is not None) and pred_instances[0][0].has('mask_feats'):
                    tmp_preds = []
                    for b_proposals in new_predicted_proposals:
                        feats = torch.cat([p.mask_feats for p in b_proposals])
                        new_preds = self.roi_heads.mask_head(feats, b_proposals, inference_mode=True, name_prefix='decoder')
                        tmp_preds.append(new_preds)
                    new_predicted_proposals = tmp_preds
                    has_mask_pred = True
                
                final_predicted_proposals += new_predicted_proposals
                final_target_proposals += new_target_proposals
                new_final_target_predictions = [new_predicted_proposals[b_idx][-1] for b_idx in range(len(new_predicted_proposals))]
                if not has_mask_pred and new_final_target_predictions[0].has('mask_feats'):
                    feats = torch.cat([p.mask_feats for p in new_final_target_predictions])
                    new_final_target_predictions = self.roi_heads.mask_head(feats, new_final_target_predictions, inference_mode=True, name_prefix='decoder')
                final_target_predictions += new_final_target_predictions
                final_last_frame_proposals += [new_target_proposals[b_idx][-1] for b_idx in range(len(new_target_proposals))]
        
        if self.add_final_refine:
            bg_logits = [b['bg_logits'].to(self.device) for b in all_batched_inputs]
            bg_depths = [b['bg_depths'].to(self.device) for b in all_batched_inputs]
            bg_depth_masks = [b['bg_depth_masks'].to(self.device) for b in all_batched_inputs]
            if self.refine_head.supervise_with_instances:
                if not final_last_frame_proposals[0].has('pred_masks'):
                    feats = torch.cat([p.features for p in final_last_frame_proposals])
                    with torch.no_grad():
                        self.roi_heads.mask_head.eval()
                        final_last_frame_proposals = self.roi_heads.mask_head(feats, final_last_frame_proposals)
                        if self.training:
                            self.roi_heads.mask_head.train()
                refine_result = self.refine_head(final_target_predictions, bg_logits, bg_depths, bg_depth_masks,
                                                 final_last_frame_proposals)
            else:
                gt_pan_segs = [b['gt_pan_seg'].to(self.device) for b in all_batched_inputs]
                gt_instances = [b['gt_pan_instances'].to(self.device) for b in all_batched_inputs]
                refine_result = self.refine_head(final_target_predictions, bg_logits, bg_depths, bg_depth_masks,
                                                gt_pan_segs, gt_instances)


        if self.training:
            target = final_target_proposals
            loss_dict = {}
            if not self.mask_model_only:
                if self.predict_encoder_vels:
                    tmp_input_proposals = [inp[1:] + [t[0]] for inp, t in zip(final_input_proposals, final_target_proposals)]
                    loss_dict.update(self._compute_encoder_vel_losses(final_input_proposals, tmp_input_proposals))
                
            loss_dict.update(self._compute_trajectory_losses(final_predicted_proposals, target, final_start_viz_feats))
            if self.add_final_refine:
                loss_dict.update(refine_result)
            first_key = list(loss_dict.keys())[0]
            tmp_loss, coef = loss_dict[first_key]
            for param in self.parameters():
                tmp_loss = tmp_loss + 0*param.sum()
            loss_dict[first_key] = (tmp_loss, coef)
            return loss_dict
        elif self.eval_pan_seg and self.add_final_refine:
            segs, instances = refine_result
            return [{'panoptic_segmentation': s, 'instances':i} for s,i in zip(segs, instances)]
        elif self.eval_inst_seg:
            final_results = []
            for idx, final_proposals in enumerate(final_predicted_proposals):
                target_pred = final_proposals[self.pred_index]
                if target_pred.has('presence'):
                    presence_mask = target_pred.presence.sigmoid() > 0.5
                    target_pred = target_pred[presence_mask]
                target_pred = self._get_final_instance_masks(target_pred)
                final_results.append({'instances':target_pred})
            return final_results
        else:
                
            for b_ind in range(len(final_predicted_proposals)):
                for p in final_input_proposals[b_ind]:
                    try:
                        p.remove('features')
                    except:
                        pass
                for p in final_target_proposals[b_ind]:
                    try:
                        p.remove('features')
                    except:
                        pass
                
                all_results.append({
                    'input_instances': final_input_proposals[b_ind],
                    'input_target_instances': final_target_proposals[b_ind],
                    'target_instances': final_predicted_proposals[b_ind],
                })
                
            return all_results

    def _get_final_instance_masks(self, results):
        if results.has('decoder_mask_logits'):
            results.pred_masks = results.decoder_mask_logits.sigmoid()
        elif results.has('mask_feats'):
            feats = results.mask_feats
            results = self.roi_heads.mask_head(feats, [results])[0]
        return detector_postprocess(results, self.img_height, self.img_width)

    def _compute_loss(self, p_boxes, p_classes, p_depths, p_id,
                            t_boxes, t_classes, t_depths, t_depth_masks,
                            t_id, p_mask_feats=None, t_mask_feats=None,
                            p_mask_logits=None,t_mask_logits=None, t_mask_probs=None, p_presence=None):
        t_len = len(t_boxes)
        p_len = len(p_boxes)
        all_losses = {}
        return_info = None
        t_boxes.tensor = t_boxes.tensor.detach()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            ious = pairwise_iou(p_boxes, t_boxes)
            dists = F.smooth_l1_loss(p_boxes.tensor.unsqueeze(1), t_boxes.tensor.unsqueeze(0), reduction='none').sum(-1)
        p_classes = p_classes.unsqueeze(-1)
        class_mat = (p_classes == t_classes).bool()
        
        matches = torch.nonzero((p_id[:, None] == t_id)*(p_id[:, None] != -1))
        pred_indices = matches[:, 0]
        target_indices = matches[:, 1]
        
        return_info = (
            torch.zeros(p_len, p_len+t_len),
            torch.zeros(p_len, p_len+t_len),
            np.stack([pred_indices.cpu().numpy(), target_indices.cpu().numpy()], axis=1)
        )
        
        if not self.mask_model_only:
            
            dist_loss = dists[pred_indices, target_indices]
            iou_loss = (1-ious)[pred_indices, target_indices]
            all_losses['loss_dist'] = dist_loss
            all_losses['loss_iou'] = iou_loss
            if self.use_depths and p_depths is not None:
                depth_masks = t_depth_masks
                
                depth_masks = depth_masks[target_indices].cpu().numpy()
                d_pred_indices = pred_indices[depth_masks]
                d_target_indices = target_indices[depth_masks]
                p_depths = p_depths[d_pred_indices]
                t_depths = t_depths[d_target_indices]
                depth_loss = F.smooth_l1_loss(p_depths, t_depths, reduction='none')
                all_losses['loss_depth'] = (depth_loss)
            if p_presence is not None:
                target = torch.tensor([i in pred_indices for i in range(len(p_presence))], dtype=torch.float32, device=self.device)
                presence_loss = F.binary_cross_entropy_with_logits(p_presence, target, reduction='none').reshape(-1)
                all_losses['loss_presence'] = presence_loss
        if p_mask_feats is not None:
            p_mask_feats = p_mask_feats[pred_indices]
            t_mask_feats = t_mask_feats[target_indices]
            mask_loss = F.mse_loss(p_mask_feats, t_mask_feats, reduction='none').reshape(p_mask_feats.size(0), 256*14*14).mean(-1)
            all_losses['loss_mask_feat'] = mask_loss
        if t_mask_logits is not None:
            p_mask_logits = p_mask_logits[pred_indices]
            t_mask_logits = t_mask_logits[target_indices]
            mask_loss = F.mse_loss(p_mask_logits, t_mask_logits, reduction='none').reshape(p_mask_logits.size(0), 28*28).mean(-1)
            all_losses['loss_mask_logit'] = mask_loss
        if t_mask_probs is not None:
            p_mask_logits = p_mask_logits[pred_indices]
            t_mask_probs = t_mask_probs[target_indices]
            if self.binarize_loss_masks:
                t_mask_probs = (t_mask_probs >= 0.5).float()
            if self.use_mask_focal_loss:
                mask_loss = utils.focal_loss(p_mask_logits, t_mask_probs, self.mask_focal_loss_gamma).reshape(p_mask_logits.size(0), 28*28).mean(-1)
            else:
                mask_loss = F.binary_cross_entropy_with_logits(p_mask_logits, t_mask_probs, reduction='none').reshape(p_mask_logits.size(0), 28*28).mean(-1)
            all_losses['loss_mask'] = mask_loss

        return all_losses, return_info

        

    def _compute_trajectory_losses(self, all_predicted_proposals, all_target_proposals, all_inp_viz_feats=None,
                                   return_matching_info=False):
        all_losses = defaultdict(list)
        loss_coefs = {
            'decoder/loss_dist': self.dist_coef,
            'decoder/loss_iou': self.iou_coef,
            'decoder/loss_depth': self.depth_loss_coef,
            'decoder/loss_presence': self.presence_loss_coef,
        }
        if self.mask_feat_loss_coef is not None:
            loss_coefs['decoder/loss_mask_feat'] = self.mask_feat_loss_coef
        if self.decoder_mask_logit_loss_coef is not None:
            loss_coefs['decoder/loss_mask_logit'] = self.decoder_mask_logit_loss_coef
        if self.decoder_mask_loss_coef is not None:
            loss_coefs['decoder/loss_mask'] = self.decoder_mask_loss_coef
        if return_matching_info:
            all_viz_scores = []
            all_match_scores = []
            all_matches = []
        for b_idx, (predicted_proposals, target_proposals) in enumerate(zip(all_predicted_proposals, all_target_proposals)):
            if return_matching_info:
                b_viz_scores = []
                b_match_scores = []
                b_matches = []
                all_viz_scores.append(b_viz_scores)
                all_match_scores.append(b_match_scores)
                all_matches.append(b_matches)
            for l_idx, (preds, targets) in enumerate(zip(predicted_proposals, target_proposals)):
                p_boxes = preds.pred_boxes
                t_boxes = targets.pred_boxes
                p_classes = preds.pred_classes
                t_classes = targets.pred_classes
                if preds.has('depths'):
                    p_depths = preds.depths
                    t_depths = targets.depths
                    t_depth_masks = targets.depth_masks
                else:
                    p_depths = t_depths = t_depth_masks = None
                p_id = preds.ids
                t_id = targets.ids
                
                if self.mask_feat_loss_coef is not None:
                    p_mask_feats = preds.mask_feats
                    t_mask_feats = targets.features
                else:
                    p_mask_feats = t_mask_feats = None
                if self.decoder_mask_logit_loss_coef is not None:
                    p_mask_logits = preds.decoder_mask_logits
                    t_mask_logits = targets.pred_mask_logits
                    t_mask_probs = None
                elif self.decoder_mask_loss_coef is not None:
                    if preds.has('decoder_pred_mask_logits'):
                        p_mask_logits = preds.decoder_pred_mask_logits
                    else:
                        p_mask_logits = preds.decoder_mask_logits
                    t_mask_probs = targets.pred_masks
                    t_mask_logits = None
                else:
                    p_mask_logits = t_mask_logits = t_mask_probs = None
                if self.presence_loss_coef is not None:
                    p_presence = preds.presence
                else:
                    p_presence = None
                new_losses, new_return_vals = self._compute_loss(
                    p_boxes, p_classes, p_depths, p_id,
                    t_boxes, t_classes, t_depths, t_depth_masks,
                    t_id, p_mask_feats=p_mask_feats,
                    t_mask_feats=t_mask_feats, p_mask_logits=p_mask_logits,
                    t_mask_logits=t_mask_logits, t_mask_probs=t_mask_probs,
                    p_presence = p_presence,
                )
                for name, loss in new_losses.items():
                    all_losses['decoder/%s'%name].append(loss)
                if return_matching_info:
                    viz_scores, match_scores, matches = new_return_vals
                    b_viz_scores.append(viz_scores.cpu().numpy())
                    b_match_scores.append(match_scores.cpu().numpy())
                    b_matches.append(matches)

        loss_dict = {}
        for loss_name, loss_vals in all_losses.items():
            if len(loss_vals) == 0:
                final_loss = torch.tensor(0., device=self.device)
            else:
                loss_vals = torch.cat(loss_vals)
                if len(loss_vals) == 0:
                    final_loss = torch.tensor(0., device=self.device)
                else:
                    final_loss = loss_vals.mean()
            loss_dict[loss_name] = (final_loss, loss_coefs[loss_name])
        if return_matching_info:
            return loss_dict, all_viz_scores, all_match_scores, all_matches
        else:
            return loss_dict


    def _compute_encoder_vel_losses(self, all_predicted_proposals, all_target_proposals):
        all_losses = defaultdict(list)
        l_name = 'encoder_vel'
        loss_coefs = {
            '%s/loss_dist'%l_name: self.dist_coef,
            '%s/loss_iou'%l_name: self.iou_coef,
            '%s/emb_loss'%l_name: self.emb_loss_coef,
            '%s/loss_depth'%l_name: self.depth_loss_coef,
        }
        if self.encoder_next_mask_feat_loss_coef is not None:
            loss_coefs['%s/loss_mask_feat'%l_name] = self.encoder_next_mask_feat_loss_coef
        if self.encoder_next_mask_logit_loss_coef is not None:
            loss_coefs['%s/loss_mask_logit'%l_name] = self.encoder_next_mask_logit_loss_coef
        if self.encoder_next_mask_loss_coef is not None:
            loss_coefs['%s/loss_mask'%l_name] = self.encoder_next_mask_loss_coef
        for b_idx, (predicted_proposals, target_proposals) in enumerate(zip(all_predicted_proposals, all_target_proposals)):
            for l_idx, (preds, targets) in enumerate(zip(predicted_proposals, target_proposals)):
                p_boxes = preds.encoder_pred_next_boxes
                t_boxes = targets.pred_boxes
                p_classes = preds.pred_classes
                t_classes = targets.pred_classes
                

                if targets.has('depths'):
                    t_depths = targets.depths
                    t_depth_masks = targets.depth_masks
                else: 
                    t_depths = t_depth_masks = None
                if preds.has('encoder_pred_next_depths'):
                    p_depths = preds.encoder_pred_next_depths
                else:
                    p_depths = None

                p_id = preds.ids
                t_id = targets.ids
                
                if self.encoder_next_mask_feat_loss_coef is not None:
                    p_mask_feats = preds.encoder_mask_feats
                    t_mask_feats = targets.features
                else:
                    p_mask_feats = t_mask_feats = None
                if self.encoder_next_mask_logit_loss_coef is not None:
                    p_mask_logits = preds.encoder_mask_logits
                    t_mask_logits = targets.pred_mask_logits
                    t_mask_probs = None
                elif self.encoder_next_mask_loss_coef is not None:
                    p_mask_logits = preds.encoder_mask_logits
                    t_mask_probs = targets.pred_masks
                    t_mask_logits = None
                else:
                    p_mask_logits = t_mask_logits = t_mask_probs = None
                new_losses, _ = self._compute_loss(
                    p_boxes, p_classes, p_depths, p_id,
                    t_boxes, t_classes, t_depths, t_depth_masks,
                    t_id, p_mask_feats=p_mask_feats,
                    t_mask_feats=t_mask_feats, p_mask_logits=p_mask_logits,
                    t_mask_logits=t_mask_logits, t_mask_probs=t_mask_probs
                )
                for name, loss in new_losses.items():
                    all_losses['%s/%s'%(l_name,name)].append(loss)

        loss_dict = {}
        for loss_name, loss_vals in all_losses.items():
            if len(loss_vals) == 0:
                final_loss = torch.tensor(0., device=self.device)
            else:
                loss_vals = torch.cat(loss_vals)
                if len(loss_vals) == 0:
                    final_loss = torch.tensor(0., device=self.device)
                else:
                    final_loss = loss_vals.mean()
            loss_dict[loss_name] = (final_loss, loss_coefs[loss_name])
        return loss_dict

