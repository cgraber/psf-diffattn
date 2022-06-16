from typing import Callable, Dict, List, Optional, Union, Tuple
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from detectron2.layers import ShapeSpec

from .conv_transformers import ConvTransformerEncoderLayer, ConvTransformerDecoderLayer
from .transformers import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from .. import utils
from .mask_feat_models import build_mask_feat_encoder, build_mask_feat_decoder
from .batched_instance_sequence import BatchedInstanceSequence


def build_mask_encoder(cfg):
    return MaskTransformerEncoder(cfg)

def build_mask_decoder(cfg):
    return MaskTransformerDecoder(cfg)

class MaskTransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.prediction_type = cfg.MODEL.MASK_ENCODER.PREDICTION_TYPE
        emb_size = cfg.MODEL.MASK_ENCODER.EMB_SIZE
        num_layers = cfg.MODEL.MASK_ENCODER.NUM_LAYERS
        self.nhead = nhead = cfg.MODEL.MASK_ENCODER.NUM_ATTENTION_HEADS
        dim_feedforward = cfg.MODEL.MASK_ENCODER.DIM_FEEDFORWARD
        dropout = cfg.MODEL.MASK_ENCODER.DROPOUT
        attention_type = cfg.MODEL.MASK_ENCODER.ATTENTION_TYPE
        aggregation_type = cfg.MODEL.MASK_ENCODER.AGGREGATION_TYPE
        attention_module = cfg.MODEL.MASK_ENCODER.ATTENTION_MODULE
        self.is_agent_aware = cfg.MODEL.MASK_ENCODER.IS_AGENT_AWARE
        self.odom_size = cfg.MODEL.ODOM_SIZE
        self.agent_aware_temperature = cfg.MODEL.AGENT_AWARE_TEMPERATURE
        self.predict_mask_feats = cfg.MODEL.MASK_ENCODER.PREDICT_MASK_FEATS
        self.predict_seg_mask = cfg.MODEL.MASK_ENCODER.PREDICT_MASK
        self.transformer_type = cfg.MODEL.MASK_ENCODER.TRANSFORMER_TYPE
        self.use_agent_aware_mask_matching = cfg.MODEL.USE_AGENT_AWARE_MASK_MATCHING
        kernel_size = cfg.MODEL.MASK_ENCODER.KERNEL_SIZE
        attn_kernel_size = cfg.MODEL.MASK_ENCODER.ATTN_KERNEL_SIZE
        ff_kernel_size = cfg.MODEL.MASK_ENCODER.FEEDFORWARD_KERNEL_SIZE
        use_pre_norm = cfg.MODEL.USE_PRE_NORM
        key_dim = cfg.MODEL.MASK_ENCODER.KEY_DIM
        
        if self.transformer_type == 'standard':
            encoder_layer = TransformerEncoderLayer(
                emb_size, nhead, dim_feedforward, dropout, use_pre_norm=use_pre_norm,
                attention_type=attention_type,
                aggregation_type=aggregation_type, attention_module=attention_module,
                is_agent_aware=self.is_agent_aware, key_mode='standard', value_mode='standard',
                key_dim=key_dim,
            )
            if use_pre_norm:
                final_norm = nn.LayerNorm(emb_size)
            else:
                final_norm = None
        elif 'conv' in self.transformer_type:
            encoder_layer = ConvTransformerEncoderLayer(
                emb_size, nhead, attn_kernel_size, ff_kernel_size, dim_feedforward, dropout,
                is_agent_aware=self.is_agent_aware, use_pre_norm=use_pre_norm,
                attention_type=cfg.MODEL.MASK_ENCODER.TRANSFORMER_TYPE,
            )
            if use_pre_norm:
                final_norm = MyGroupNorm(nhead, emb_size)
            else:
                final_norm = None
        self.transformer = TransformerEncoder(
            encoder_layer, num_layers, norm=final_norm,
            fix_init=cfg.MODEL.FIX_TRANSFORMER_INIT,
        )
        self.input_feat_module = build_mask_feat_encoder(cfg)


    def forward(self, proposals, odom):
        features = proposals.features
        if proposals.has('pred_masks'):
            masks = proposals.pred_masks
        else:
            masks = None
        
        if proposals.has('ids'):
            ids = proposals.ids
        else:
            ids = None
        mask = proposals.get_mask()
        time = proposals.get_time()
        if odom is not None:
            odom = proposals.prepare_odom(odom, self.odom_size)
        classes = proposals.pred_classes
        inp_feats = self.input_feat_module(features=features, time=time, masks=masks)
        
        if 'conv' not in self.transformer_type:
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
        
        return final_result




class MaskTransformerDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        emb_size = cfg.MODEL.MASK_DECODER.EMB_SIZE
        num_layers = cfg.MODEL.MASK_DECODER.NUM_LAYERS
        self.nhead = nhead = cfg.MODEL.MASK_DECODER.NUM_ATTENTION_HEADS
        dim_feedforward = cfg.MODEL.MASK_DECODER.DIM_FEEDFORWARD
        dropout = cfg.MODEL.MASK_DECODER.DROPOUT
        attention_type = cfg.MODEL.MASK_DECODER.ATTENTION_TYPE
        use_pre_norm = cfg.MODEL.USE_PRE_NORM
        self.is_agent_aware = cfg.MODEL.MASK_DECODER.IS_AGENT_AWARE
        attention_module = cfg.MODEL.MASK_DECODER.ATTENTION_MODULE
        self.agent_aware_temperature = cfg.MODEL.AGENT_AWARE_TEMPERATURE
        self.use_odom = cfg.MODEL.MASK_DECODER.USE_ODOM
        key_dim = cfg.MODEL.MASK_DECODER.KEY_DIM
        self.transformer_type = cfg.MODEL.MASK_DECODER.TRANSFORMER_TYPE
        self.use_agent_aware_mask_matching = cfg.MODEL.USE_AGENT_AWARE_MASK_MATCHING
        attn_kernel_size = cfg.MODEL.MASK_DECODER.ATTN_KERNEL_SIZE
        ff_kernel_size = cfg.MODEL.MASK_DECODER.FEEDFORWARD_KERNEL_SIZE
        if self.transformer_type == 'standard':
            decoder_layer = TransformerDecoderLayer(
                emb_size, nhead, dim_feedforward, dropout, use_pre_norm=use_pre_norm,
                attention_type=attention_type,
                is_agent_aware=self.is_agent_aware,
                attention_module=attention_module,
                key_mode='standard', value_mode='standard',
                key_dim=key_dim,
            )
            if use_pre_norm:
                final_norm = nn.LayerNorm(emb_size)
            else:
                final_norm = None
        elif 'conv' in self.transformer_type:
            decoder_layer = ConvTransformerDecoderLayer(
                emb_size, nhead, attn_kernel_size, ff_kernel_size, dim_feedforward, dropout,
                use_pre_norm=use_pre_norm, is_agent_aware=self.is_agent_aware,
                attention_type=self.transformer_type,
            )
            if use_pre_norm:
                final_norm = MyGroupNorm(nhead, emb_size)
            else:
                final_norm = None
        else:
            raise ValueError('Transformer type not recognized: ',self.transformer_type)
        self.transformer = TransformerDecoder(
            decoder_layer, num_layers, norm=final_norm,
            fix_init=cfg.MODEL.FIX_TRANSFORMER_INIT,
        )
        self.input_feat_model = build_mask_feat_encoder(cfg)
        self.output_feat_model = build_mask_feat_decoder(cfg)
        self.prediction_type = cfg.MODEL.MASK_DECODER.PREDICTION_TYPE
        

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

    def forward(self, context: torch.Tensor, context_instances, start_instances, result_instances, odom, time_offset, num_steps):
        memory_mask = context_instances.get_mask() #TODO: CHECK THESE ARE CORRECT
        current_mask = start_instances.get_mask()
        if self.prediction_type == 'mask_feats':
            current_feats = start_instances.features
        elif self.prediction_type == 'mask_logits':
            current_feats = start_instances.pred_masks
        b = current_feats.size(0)
        n = current_feats.size(1)
        
        ids = context_ids = None
        
        if self.is_agent_aware:
            if ids is None:
                ids = start_instances.ids
                context_ids = context_instances.ids
            base_context_agent_aware_mask = self._build_context_agent_aware_mask(ids, context_ids)
            context_agent_aware_mask = base_context_agent_aware_mask
        else:
            context_agent_aware_mask = self_agent_aware_mask = None
        output_feats = []
        all_mask_embs = None
        for step in range(num_steps):
            # embed current boxes
            current_times = torch.empty(b, n, device=current_feats.device).fill_(time_offset+step)
            if self.prediction_type == 'mask_feats':
                mask_embs = self.input_feat_model(features=current_feats, time=current_times)
            elif self.prediction_type == 'mask_logits':
                mask_embs = self.input_feat_model(masks=current_feats, time=current_times)
            if 'conv' not in self.transformer_type:
                mask_embs = mask_embs.transpose(0,1)
            # append to current input list
            if all_mask_embs is None:
                all_mask_embs = mask_embs
            else:
                if 'conv' in self.transformer_type:
                    all_mask_embs = torch.cat([all_mask_embs, mask_embs], dim=1)
                else:
                    all_mask_embs = torch.cat([all_mask_embs, mask_embs], dim=0) #TODO: shape here?

            if self.is_agent_aware:
                self_agent_aware_mask = self._build_self_agent_aware_mask_for_step(mask_embs.size(0), n, step, mask_embs.device)
            # pass through decoder - don't forget to mask!
            current_mem_mask = self._build_mem_mask_for_step(n, step, memory_mask, current_mask)
            tgt_mask = self._build_tgt_mask_for_step(n, step, current_mask, ids) #TODO: faster to just append to previous mask?
            
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
            decoded_masks, self_attns, cross_attns = self.transformer(all_mask_embs, context, 
                                                                      memory_mask=final_current_mem_mask, tgt_mask=final_tgt_mask,
                                                                      tgt_agent_aware_mask=self_agent_aware_mask,
                                                                      memory_agent_aware_mask=context_agent_aware_mask)
            if 'conv' in self.transformer_type:
                decoded_masks = decoded_masks[:, -n:]
            else:
                decoded_masks = decoded_masks[-n:].transpose(0, 1)
            

            

            # use mlp to get output
            current_feats = self.output_feat_model(decoded_masks)
            output_feats.append(current_feats)
            if self.prediction_type == 'mask_logits':
                current_feats = current_feats.sigmoid()
            
            if self.is_agent_aware:
                context_agent_aware_mask = torch.cat([context_agent_aware_mask, base_context_agent_aware_mask], dim=1)

        final_instances = []
        for t_idx in range(num_steps):
            current_instances = result_instances[t_idx]
            if self.prediction_type == 'mask_feats':
                current_instances.add_batched_entry('mask_feats', output_feats[t_idx])
            elif self.prediction_type == 'mask_logits':
                current_instances.add_batched_entry('decoder_mask_logits', output_feats[t_idx])
            else:
                raise ValueError('Prediction type not recognized: ', self.prediction_type)
            final_instances.append(current_instances)
        result_instances = BatchedInstanceSequence.time_cat(final_instances)
        return result_instances


class MyGroupNorm(nn.GroupNorm):
    def forward(self, x):
        shape = x.shape
        x = x.view(np.prod(shape[:-3]), shape[-3], shape[-2], shape[-1])
        x = super().forward(x)
        return x.view(shape)