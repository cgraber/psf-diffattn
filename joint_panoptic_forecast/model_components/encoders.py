
from typing import Callable, Dict, List, Optional, Union, Tuple
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec
from .feat_models import BoxVizTimeEncoder
from .transformers import TransformerEncoder, TransformerEncoderLayer
from .. import utils




def build_encoder(cfg, input_shape):
    return TrajectoryTransformerEncoder(cfg, input_shape)



class TrajectoryTransformerEncoder(nn.Module):
    def __init__(self, cfg, input_shape: Optional[ShapeSpec]):
        super().__init__()
        emb_size = cfg.MODEL.TRAJECTORY_ENCODER.EMB_SIZE
        num_layers = cfg.MODEL.TRAJECTORY_ENCODER.NUM_LAYERS
        self.nhead = nhead = cfg.MODEL.TRAJECTORY_ENCODER.NUM_ATTENTION_HEADS
        dim_feedforward = cfg.MODEL.TRAJECTORY_ENCODER.DIM_FEEDFORWARD
        dropout = cfg.MODEL.TRAJECTORY_ENCODER.DROPOUT
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        use_odom = cfg.MODEL.TRAJECTORY_ENCODER.USE_ODOM
        attention_type = cfg.MODEL.TRAJECTORY_ENCODER.ATTENTION_TYPE
        aggregation_type = cfg.MODEL.TRAJECTORY_ENCODER.AGGREGATION_TYPE
        use_bbox_cwh = cfg.MODEL.USE_BOX_CWH
        box_mean = cfg.MODEL.BOX_MEAN
        box_std = cfg.MODEL.BOX_STD
        no_viz = cfg.MODEL.TRAJECTORY_ENCODER.NO_VIZ
        temperature = cfg.MODEL.TRAJECTORY_ENCODER.TEMPERATURE
        attention_module = cfg.MODEL.TRAJECTORY_ENCODER.ATTENTION_MODULE
        viz_dropout = cfg.MODEL.TRAJECTORY_ENCODER.VIZ_DROPOUT
        box_embed_size = cfg.MODEL.TRAJECTORY_ENCODER.BOX_EMBED_SIZE
        self.use_causal_mask = cfg.MODEL.TRAJECTORY_ENCODER.USE_CAUSAL_MASK
        self.attn_dist_thresh = cfg.MODEL.TRAJECTORY_ENCODER.ATTENTION_DIST_THRESH
        add_zero_attn = cfg.MODEL.TRAJECTORY_ENCODER.ADD_ZERO_ATTN
        use_no_match_emb = cfg.MODEL.TRAJECTORY_ENCODER.USE_NO_MATCH_EMB
        self.is_agent_aware = cfg.MODEL.TRAJECTORY_ENCODER.IS_AGENT_AWARE
        self.detach_agent_aware_mask = cfg.MODEL.DETACH_AGENT_AWARE_MASK
        self.predict_box_vels = cfg.MODEL.TRAJECTORY_ENCODER.PREDICT_VEL
        self.predict_box_locs = cfg.MODEL.TRAJECTORY_ENCODER.PREDICT_LOC
        self.predict_presence = cfg.MODEL.PREDICT_PRESENCE
        self.odom_size = cfg.MODEL.ODOM_SIZE
        self.use_small_output_init = cfg.MODEL.USE_SMALL_OUTPUT_INIT
        self.use_fixed_box_vels = cfg.MODEL.TRAJECTORY_ENCODER.USE_FIXED_BOX_VELS
        feat_use_relus = cfg.MODEL.TRAJECTORY_ENCODER.FEAT_USE_RELUS
        self.normalize_embs = cfg.MODEL.NORMALIZE_EMBS
        self.agent_aware_temperature = cfg.MODEL.AGENT_AWARE_TEMPERATURE
        key_mode = cfg.MODEL.KEY_MODE
        value_mode = cfg.MODEL.VALUE_MODE
        self.use_class_inp = cfg.MODEL.USE_CLASS_INP
        self.img_width = cfg.INPUT.IMG_WIDTH
        self.img_height = cfg.INPUT.IMG_HEIGHT
        use_cl_mask_emb = cfg.MODEL.USE_CL_MASK_EMB
        self.vel_out_factor = cfg.MODEL.VEL_OUT_FACTOR
        self.use_agent_aware_mask_matching = cfg.MODEL.USE_AGENT_AWARE_MASK_MATCHING
        self.register_buffer("depth_mean", torch.tensor(cfg.MODEL.DEPTH_MEAN), False)
        self.register_buffer("depth_std", torch.tensor(cfg.MODEL.DEPTH_STD), False)
        self.use_depths = cfg.MODEL.USE_DEPTHS
        self.predict_mask_feats = cfg.MODEL.TRAJECTORY_ENCODER.PREDICT_MASK_FEATS
        self.predict_seg_mask = cfg.MODEL.TRAJECTORY_ENCODER.PREDICT_MASK
        self.predict_separate_mask_per_class = cfg.MODEL.TRAJECTORY_ENCODER.PREDICT_SEPARATE_MASK_PER_CLASS
        self.split_output_emb = cfg.MODEL.SPLIT_OUTPUT_EMB
        
        emb_model = BoxVizTimeEncoder
        self.use_seg_mask = cfg.MODEL.TRAJECTORY_ENCODER.USE_SEG_MASK_INPUT
        self.use_seg_mask_probs = cfg.MODEL.TRAJECTORY_ENCODER.USE_SEG_MASK_PROBS
        cl_inp_type = cfg.MODEL.CL_INP_TYPE
        self.box_viz_encoder = emb_model(use_bbox_cwh, in_features,
                                                emb_size, input_shape, use_odom, box_mean, box_std,
                                                no_viz, viz_dropout,
                                                box_embed_size, self.use_depths,
                                                use_relus=feat_use_relus, use_class_inp=self.use_class_inp,
                                                odom_size=self.odom_size,
                                                use_cl_mask_emb=use_cl_mask_emb,
                                                cl_inp_type=cl_inp_type)
        use_pre_norm = cfg.MODEL.USE_PRE_NORM
        if use_pre_norm:
            final_norm = nn.LayerNorm(emb_size)
        else:
            final_norm = None
        encoder_layer = TransformerEncoderLayer(
            emb_size, nhead, dim_feedforward, dropout, use_pre_norm=use_pre_norm,
            temperature=temperature, attention_type=attention_type,
            aggregation_type=aggregation_type, attention_module=attention_module,
            add_zero_attn=add_zero_attn, use_no_match_emb=use_no_match_emb,
            is_agent_aware=self.is_agent_aware, key_mode=key_mode, value_mode=value_mode,
        )
        self.transformer = TransformerEncoder(
            encoder_layer, num_layers, norm=final_norm,
            fix_init=cfg.MODEL.FIX_TRANSFORMER_INIT,
        )
        output_size = 4
        if self.use_depths:
            output_size += 1
        
        
        if self.predict_box_vels:
            self.box_vel_model = nn.Sequential(
                nn.Linear(emb_size, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, output_size)
            )
            if self.use_small_output_init:
                nn.init.normal_(self.box_vel_model[-1].weight, std=0.001)
                nn.init.constant_(self.box_vel_model[-1].bias, 0)

        
        
        if self.predict_presence:
            self.presence_model = nn.Sequential(
                nn.Linear(emb_size, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 1)
            )
        

    def forward(self, proposals, odom):
        features = proposals.features
        boxes = proposals.pred_boxes
        if self.use_depths:
            depths = proposals.depths.unsqueeze(-1)
            depth_masks = proposals.depth_masks.unsqueeze(-1)
            depths = utils.normalize_depths_by_stats(depths, self.depth_mean, self.depth_std)
        else:
            depths = depth_masks = None
        if proposals.has('ids'):
            ids = proposals.ids
        else:
            ids = None
        mask = proposals.get_mask()
        time = proposals.get_time()
        if odom is not None:
            odom = proposals.prepare_odom(odom, self.odom_size)
        classes = proposals.pred_classes
        inp_feats = self.box_viz_encoder(features, boxes, depths, depth_masks, ids, mask,
                                         time, odom, self.img_width, self.img_height, classes=classes)
        inp_feats = inp_feats.transpose(0,1)
        if self.is_agent_aware:
            agent_aware_mask = ids.unsqueeze(-1) == ids.unsqueeze(-2)
        else:
            agent_aware_mask = None
        orig_mask = mask
        mask = mask.unsqueeze(1).expand(-1, mask.size(-1), -1)
        
        ####################
        # PREPARING MASK
        ##################
        dims = mask.shape
        mask = mask + ~orig_mask.unsqueeze(-1)
        mask = (~mask).unsqueeze(1).expand(-1, self.nhead, -1, -1).reshape(self.nhead*dims[0], *dims[1:])
        #################
        # DONE
        ##################
        final_result = result = self.transformer(inp_feats, mask=mask, src_agent_aware_mask=agent_aware_mask)
        

        if self.predict_box_vels:
            box_vels = self.box_vel_model(result).transpose(0,1)
            if self.use_depths:
                box_vels, depth_vels = box_vels[:, :, :4], box_vels[:, :, 4]
                next_depths = depths.squeeze(-1) + depth_vels
                next_depths = utils.unnormalize_depths_by_stats(next_depths, self.depth_mean, self.depth_std)
                proposals.add_batched_entry('encoder_pred_next_depths', next_depths)
            if self.box_viz_encoder.use_cwh:
                tmp_boxes = utils.bbox_ulbr2cwh(boxes)
                tmp_boxes = utils.normalize_box_tensor_by_stats(tmp_boxes, self.box_viz_encoder.box_mean, self.box_viz_encoder.box_std)
                next_boxes = tmp_boxes + box_vels
                next_boxes = utils.unnormalize_box_tensor_by_stats(next_boxes, self.box_viz_encoder.box_mean, self.box_viz_encoder.box_std)
                next_boxes = utils.bbox_cwh2ulbr(next_boxes)
            else:
                tmp_boxes = utils.normalize_box_tensor(boxes, self.img_width, self.img_height)
                next_boxes = tmp_boxes + box_vels*self.vel_out_factor
                next_boxes = utils.unnormalize_box_tensor(next_boxes, self.img_width, self.img_height)
            proposals.add_batched_entry('encoder_pred_next_boxes', next_boxes)
            vels = next_boxes - boxes
            proposals.add_batched_entry('encoder_pred_vels', vels)
        
        if self.predict_presence:
            presence = self.presence_model(result).transpose(0,1)
            proposals.add_batched_entry('encoder_pred_next_presence', presence)
        return final_result




