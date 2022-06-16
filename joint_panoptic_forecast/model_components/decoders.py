from typing import Callable, Dict, List, Optional, Union, Tuple
import torch
from torch import nn
from torch.nn import functional as F


from detectron2.layers import ShapeSpec
from .feat_models import BoxTimeEncoder
from .transformers import TransformerDecoder, TransformerDecoderLayer
from .batched_instance_sequence import BatchedInstanceSequence

from .. import utils


def build_decoder(cfg, input_shape):
    return TrajectoryTransformerDecoder(cfg, input_shape)


class TrajectoryTransformerDecoder(nn.Module):
    def __init__(self, cfg, input_shape: Optional[ShapeSpec]):
        super().__init__()
        emb_size = cfg.MODEL.TRAJECTORY_DECODER.EMB_SIZE
        num_layers = cfg.MODEL.TRAJECTORY_DECODER.NUM_LAYERS
        self.nhead = nhead = cfg.MODEL.TRAJECTORY_DECODER.NUM_ATTENTION_HEADS
        dim_feedforward = cfg.MODEL.TRAJECTORY_DECODER.DIM_FEEDFORWARD
        dropout = cfg.MODEL.TRAJECTORY_DECODER.DROPOUT
        self.use_odom = cfg.MODEL.TRAJECTORY_DECODER.USE_ODOM
        temperature = cfg.MODEL.TRAJECTORY_DECODER.TEMPERATURE
        attention_type = cfg.MODEL.TRAJECTORY_DECODER.ATTENTION_TYPE
        aggregation_type = cfg.MODEL.TRAJECTORY_DECODER.AGGREGATION_TYPE
        use_pre_norm = cfg.MODEL.USE_PRE_NORM
        self.use_cwh = cfg.MODEL.USE_BOX_CWH
        use_positional_encoding = cfg.MODEL.USE_BOX_POSITIONAL_ENCODING
        self.is_agent_aware = cfg.MODEL.TRAJECTORY_DECODER.IS_AGENT_AWARE
        self.predict_offset_from_start = cfg.MODEL.TRAJECTORY_DECODER.PREDICT_OFFSET_FROM_START
        attention_module = cfg.MODEL.TRAJECTORY_DECODER.ATTENTION_MODULE
        self.attn_dist_thresh = cfg.MODEL.TRAJECTORY_DECODER.ATTENTION_DIST_THRESH
        feat_use_relus = cfg.MODEL.TRAJECTORY_DECODER.FEAT_USE_RELUS
        self.predict_presence = cfg.MODEL.TRAJECTORY_DECODER.PREDICT_PRESENCE
        cl_inp_type = cfg.MODEL.CL_INP_TYPE
        odom_size = cfg.MODEL.ODOM_SIZE
        key_mode = cfg.MODEL.KEY_MODE
        value_mode = cfg.MODEL.VALUE_MODE
        self.use_class_inp = cfg.MODEL.USE_CLASS_INP
        assert not (self.use_cwh and use_positional_encoding)
        self.register_buffer("box_mean", torch.Tensor(cfg.MODEL.BOX_MEAN), False)
        self.register_buffer("box_std", torch.Tensor(cfg.MODEL.BOX_STD), False)
        self.register_buffer("depth_mean", torch.tensor(cfg.MODEL.DEPTH_MEAN), False)
        self.register_buffer("depth_std", torch.tensor(cfg.MODEL.DEPTH_STD), False)
        self.use_depths = cfg.MODEL.USE_DEPTHS
        self.use_viz_feats = cfg.MODEL.TRAJECTORY_DECODER.USE_VIZ_FEATS
        add_zero_attn = cfg.MODEL.TRAJECTORY_DECODER.ADD_ZERO_ATTN
        use_no_match_emb = cfg.MODEL.TRAJECTORY_DECODER.USE_NO_MATCH_EMB
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.img_width = cfg.INPUT.IMG_WIDTH
        self.img_height = cfg.INPUT.IMG_HEIGHT
        self.use_agent_aware_mask_matching = cfg.MODEL.USE_AGENT_AWARE_MASK_MATCHING
        self.split_output_emb = cfg.MODEL.SPLIT_OUTPUT_EMB
        self.vel_out_factor = cfg.MODEL.VEL_OUT_FACTOR
        
        mlp_hidden = cfg.MODEL.TRAJECTORY_ENCODER.MLP_HIDDEN
        
        if use_pre_norm:
            final_norm = nn.LayerNorm(emb_size)
        else:
            final_norm = None

        decoder_layer = TransformerDecoderLayer(
            emb_size, nhead, dim_feedforward, dropout, use_pre_norm=use_pre_norm,
            temperature=temperature, attention_type=attention_type,
            aggregation_type=aggregation_type, is_agent_aware=self.is_agent_aware,
            attention_module=attention_module, add_zero_attn=add_zero_attn,
            use_no_match_emb=use_no_match_emb, key_mode=key_mode, value_mode=value_mode,
        )
        self.transformer = TransformerDecoder(
            decoder_layer, num_layers, norm=final_norm,
            fix_init=cfg.MODEL.FIX_TRANSFORMER_INIT,
        )
        emb_model = BoxTimeEncoder
        if self.use_viz_feats:
            box_emb_size = cfg.MODEL.TRAJECTORY_ENCODER.BOX_EMBED_SIZE

        else:
            box_emb_size = cfg.MODEL.TRAJECTORY_DECODER.BOX_EMBED_SIZE
        self.box_emb = emb_model(emb_size, self.use_odom,
                                    self.use_depths, use_relus=feat_use_relus, use_class_inp=self.use_class_inp,
                                    use_viz_feats=self.use_viz_feats, box_embed_size=box_emb_size,
                                    input_shape=input_shape, in_features=in_features,
                                    use_depth_mask=False,
                                    odom_size=odom_size,
                                    cl_inp_type=cl_inp_type)
        
        out_size = 4
        if self.use_depths:
            out_size += 1
        
        mlp_hidden.insert(0, emb_size)
        mlp_layers = []
        for s1, s2 in zip(mlp_hidden[:-1], mlp_hidden[1:]):
            mlp_layers.append(nn.Linear(s1, s2))
            mlp_layers.append(nn.ReLU(inplace=True))
        mlp_layers.append(nn.Linear(mlp_hidden[-1], out_size))
        self.mlp_out = nn.Sequential(*mlp_layers)
        
        
        if self.predict_presence:
            self.mlp_presence_out = nn.Sequential(
                nn.Linear(emb_size, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 1)
            )
        



    def _build_tgt_mask_for_step(self, num_instances, step, inst_mask, ids, other_mask=None):
        """
        Builds causal mask for decoder. Output is shape [T, T], where the 
        row index represents the input and the column index represents what
        it can attend to.
        """
        device = inst_mask.device
        step = step + 1
        mask = (torch.triu(torch.ones(step, step, device=device)) == 1).transpose(0,1)
        mask = mask.reshape(-1, 1).expand(-1, num_instances).reshape(step, step*num_instances)
        mask = mask.unsqueeze(1).repeat(1,num_instances, 1).reshape(step*num_instances, step*num_instances)
        mask = mask.unsqueeze(0)*(inst_mask.unsqueeze(1).expand(-1, inst_mask.size(-1), -1).repeat(1, step, step))
        if other_mask is not None:
            mask = (mask*other_mask)
        return mask

    def _build_mem_mask_for_step(self, num_instances, step, mem_mask, current_mask):
        step = step + 1
        result = (mem_mask).unsqueeze(1).expand(-1, current_mask.size(-1)*step, -1)
        return result

    def _build_context_id_mask(self, context_ids, input_ids):
        mask = input_ids.unsqueeze(2) == context_ids.unsqueeze(1)
        return mask


    def _build_self_agent_aware_mask_for_step(self, b, num_instances, step, device):
        step = step + 1
        mask = torch.eye(num_instances, device=device, dtype=torch.bool).repeat(step, step)
        mask = mask.unsqueeze(0).expand(b, -1, -1)
        return mask

    def _build_context_agent_aware_mask(self, ids, context_ids):
        mask = ids.unsqueeze(-1) == context_ids.unsqueeze(-2)
        return mask

    def forward(self, context: torch.Tensor, context_instances, start_instances, odom, time_offset, num_steps):
        context_boxes = context_instances.pred_boxes
        context_centers = context_boxes[:, :, :2] + 0.5*(context_boxes[:, :, 2:] - context_boxes[:, :, :2])
        memory_mask = context_instances.get_mask()
        current_mask = start_instances.get_mask()
        start_boxes = boxes = start_instances.pred_boxes
        classes = start_instances.pred_classes
        new_tgt_centers = tgt_centers = boxes[:, :, :2] + 0.5*(boxes[:, :, 2:] - boxes[:, :, :2])
        if self.use_cwh:
            boxes = utils.bbox_ulbr2cwh(boxes)
            boxes = utils.normalize_box_tensor_by_stats(boxes, self.box_mean, self.box_std)
        else:
            boxes = utils.normalize_box_tensor(boxes, self.img_width, self.img_height)
        if self.use_depths:
            depths = start_instances.depths.unsqueeze(-1)
            depths = utils.normalize_depths_by_stats(depths, self.depth_mean, self.depth_std)
            boxes = torch.cat([boxes, depths], dim=-1)
        if self.use_odom:
            odom = odom.unsqueeze(2).expand(-1, -1, boxes.size(1), -1)
        if self.use_viz_feats:
            viz_feats = start_instances.features
        else:
            viz_feats = None
        all_box_embs = None
        num_instances = boxes.size(1)
        final_boxes = []
        current_boxes = boxes
        all_self_attns = []
        all_cross_attns = []
        final_embs = []
        final_presence = []
        final_mask_feats = []
        final_seg_masks = []
        ids = context_ids = None
        context_id_mask = None
        
        if self.is_agent_aware:
            if ids is None:
                ids = start_instances.ids
                context_ids = context_instances.ids
            base_context_agent_aware_mask = self._build_context_agent_aware_mask(ids, context_ids)
            context_agent_aware_mask = base_context_agent_aware_mask
        else:
            context_agent_aware_mask = self_agent_aware_mask = None
        dist_context_mask = None
        for step in range(num_steps):
            # embed current boxes
            if self.use_odom:
                inp_odom = odom[:, step]
            else:
                inp_odom = None
            box_embs = self.box_emb(viz_feats, current_boxes, inp_odom,
                                    self.img_width, self.img_height, time_offset=time_offset+step,
                                    classes=classes)
            
            box_embs = box_embs.transpose(0,1)
            # append to current input list
            if all_box_embs is None:
                all_box_embs = box_embs
            else:
                all_box_embs = torch.cat([all_box_embs, box_embs], dim=0)

            if self.is_agent_aware:
                self_agent_aware_mask = self._build_self_agent_aware_mask_for_step(boxes.size(0), num_instances, step, box_embs.device)
            # pass through decoder - don't forget to mask!
            current_mem_mask = self._build_mem_mask_for_step(num_instances, step, memory_mask, current_mask)
            tgt_mask = self._build_tgt_mask_for_step(num_instances, step, current_mask, ids) 
            if context_id_mask is not None:
                current_mem_mask = current_mem_mask*context_id_mask
            
            #####
            # PREPARING MASKS FOR TRANSFORMER
            #####
            dims = current_mem_mask.shape
            invalid_mask = (~current_mask).unsqueeze(-1).repeat(1, step+1, 1)
            final_current_mem_mask = current_mem_mask + invalid_mask
            final_current_mem_mask = (~final_current_mem_mask).unsqueeze(1).expand(-1, self.nhead, -1, -1).reshape(self.nhead*dims[0], *dims[1:])
            final_tgt_mask = tgt_mask + invalid_mask
            dims = tgt_mask.shape
            final_tgt_mask = (~final_tgt_mask).unsqueeze(1).expand(-1, self.nhead, -1, -1).reshape(self.nhead*dims[0], *dims[1:])
            ##DONE
            decoded_boxes, self_attns, cross_attns = self.transformer(all_box_embs, context, 
                                                                      memory_mask=final_current_mem_mask, tgt_mask=final_tgt_mask,
                                                                      tgt_agent_aware_mask=self_agent_aware_mask,
                                                                      memory_agent_aware_mask=context_agent_aware_mask)
            assert (not torch.any(torch.isnan(decoded_boxes)))
            all_self_attns.append([a[:, :, -num_instances:] for a in self_attns])
            all_cross_attns.append([a[:, :, -num_instances:] for a in cross_attns])
            decoded_boxes = decoded_boxes[-num_instances:].transpose(0,1)
            

            

            # use mlp to get output
            current_vel = self.mlp_out(decoded_boxes)
            if self.vel_out_factor != 1.0:
                current_vel[:, :, :4] = current_vel[:, :, :4]*self.vel_out_factor
            current_boxes = current_boxes + current_vel
            final_boxes.append(current_boxes)
            
            if self.predict_presence:
                current_presence = self.mlp_presence_out(decoded_boxes)
                final_presence.append(current_presence)
            if self.is_agent_aware:
                context_agent_aware_mask = torch.cat([context_agent_aware_mask, base_context_agent_aware_mask], dim=1)
        final_boxes = torch.stack(final_boxes, 1)

        if self.use_depths:
            final_boxes, final_depths = final_boxes[:, :, :, :4], final_boxes[:, :, :, 4]
            final_depths = utils.unnormalize_depths_by_stats(final_depths, self.depth_mean, self.depth_std)
        else:
            final_depths = None
        if self.use_cwh:
            final_boxes = utils.unnormalize_box_tensor_by_stats(final_boxes, self.box_mean, self.box_std)
            final_boxes = utils.bbox_cwh2ulbr(final_boxes)
        else:
            final_boxes = utils.unnormalize_box_tensor(final_boxes, self.img_width, self.img_height)
        final_vels = final_boxes - torch.cat([start_boxes.unsqueeze(1), final_boxes[:, :-1]], dim=1)
        result_instances = [start_instances.clone(['pred_classes', 'ids']) for _ in range(num_steps)]
        for t_idx in range(num_steps):
            result_instances[t_idx].add_batched_entry('pred_boxes', final_boxes[:, t_idx])
            result_instances[t_idx].add_batched_entry('pred_vels', final_vels[:, t_idx])
            if final_depths is not None:
                result_instances[t_idx].add_batched_entry('depths', final_depths[:, t_idx])
            if self.predict_presence:
                result_instances[t_idx].add_batched_entry('presence', final_presence[t_idx].squeeze(-1))
        result_instances = BatchedInstanceSequence.time_cat(result_instances)
        return result_instances, all_self_attns, all_cross_attns
