import warnings
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from detectron2.structures import Boxes, pairwise_iou, ROIMasks

from .pix_transformer import PixMHA
from .. import utils

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024 ** 3  # 1 GB memory limit

def build_fg_bg_refine(cfg):
    return FGBGRefiner(cfg)



# here, we let the values be a trainable fn of the logits
class FGBGRefiner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.supervise_with_instances = True
        self.num_things_classes = cfg.MODEL.NUM_THINGS_CLASSES
        self.num_stuff_classes = cfg.MODEL.NUM_STUFF_CLASSES
        self.prediction_stride = cfg.MODEL.FGBGREFINE.PREDICTION_STRIDE
        self.iou_threshold = cfg.MODEL.FGBGREFINE.IOU_MATCHING_THRESHOLD
        self.use_single_bg_depth = cfg.MODEL.FGBGREFINE.USE_SINGLE_BG_DEPTH
        self.train_paste_into_gt_box = cfg.MODEL.FGBGREFINE.TRAIN_PASTE_INTO_GT_BOX
        self.train_gt_depth = cfg.MODEL.FGBGREFINE.TRAIN_GT_DEPTH
        self.use_instance_features = cfg.MODEL.FGBGREFINE.USE_INSTANCE_FEATURES
        self.max_inst_num = cfg.MODEL.FGBGREFINE.MAX_INST_NUM
        self.sub_batch_size = cfg.MODEL.FGBGREFINE.SUB_BATCH_SIZE
        self.bg_front_weight = cfg.MODEL.FGBGREFINE.BG_FRONT_WEIGHT
        self.use_gt_pan_seg = cfg.MODEL.FGBGREFINE.USE_GT_PAN_SEG
        self.use_focal_loss = cfg.MODEL.FGBGREFINE.USE_FOCAL_LOSS
        self.focal_loss_gamma = cfg.MODEL.FGBGREFINE.FOCAL_LOSS_GAMMA
        self.trust_fg = cfg.MODEL.FGBGREFINE.TRUSTFG
        self.viz_depths = cfg.MODEL.FGBGREFINE.VIZ_DEPTHS
        self.instance_depth_factor = cfg.MODEL.FGBGREFINE.INST_DEPTH_FACTOR

        self.use_depth_refine_model = cfg.MODEL.FGBGREFINE.USE_DEPTH_REFINE_MODEL

        depth_size = 1
        self.register_buffer("depth_mean", torch.tensor(cfg.MODEL.DEPTH_MEAN), False)
        self.register_buffer("depth_std", torch.tensor(cfg.MODEL.DEPTH_STD), False)
        self.depth_size = depth_size

        self.img_width = cfg.INPUT.IMG_WIDTH
        self.img_height = cfg.INPUT.IMG_HEIGHT

        self.pred_width = self.img_width // self.prediction_stride
        self.pred_height = self.img_height // self.prediction_stride

        emb_size = cfg.MODEL.FGBGREFINE.EMB_SIZE
        num_heads = cfg.MODEL.FGBGREFINE.NUM_HEADS
        softmax_full_matrix = cfg.MODEL.FGBGREFINE.TRANSFORMER.SOFTMAX_FULL_MATRIX
        self.paste_full_rez_instances = cfg.MODEL.FGBGREFINE.PASTE_FULL_REZ_INSTANCES
        self.use_bg_feats = cfg.MODEL.FGBGREFINE.USE_BG_FEATS

        if self.paste_full_rez_instances:
            init_val = 1.0
        else:
            init_val = 10.0
        self.dyn_model = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.dyn_model.weight.data.fill_(0.)
        self.dyn_model.weight.data[0,0,1,1] = init_val
        self.dyn_model.bias.data.fill_(0.)

        self.static_model = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(self.num_stuff_classes, 1, kernel_size=3, padding=1),
        )
        self.static_model[1].weight.data.fill_(0.)
        self.static_model[1].weight.data[0,0,1,1] = init_val
        self.static_model[1].bias.data.fill_(0.)

        
        self.mha = PixMHA(emb_size, 3, num_heads, softmax_full_matrix=softmax_full_matrix) 

        if self.use_depth_refine_model:
            self.use_depth_offset_bias = cfg.MODEL.FGBGREFINE.USE_DEPTH_OFFSET_BIAS
            self.depth_offset_coef = cfg.MODEL.FGBGREFINE.DEPTH_OFFSET_COEF
            d_size = cfg.MODEL.FGBGREFINE.DYN_HEAD.STATIC_EMB_SIZE
            d_inp = 1 + self.num_stuff_classes
            if self.use_bg_feats:
                d_inp += 256
            self.bg_d1 = nn.Conv2d(d_inp, d_size, 3, padding=1)
            self.bg_sub = nn.Sequential(
                nn.Conv2d(d_size, d_size, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(d_size, d_size, 3, padding=1),
            )
            self.bg_d2 = nn.Sequential(
                nn.Conv2d(d_size, d_size, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(d_size, 1, 1),
            )
            if self.use_depth_offset_bias:
                self.bg_d2_bias = nn.Sequential(
                    nn.Conv2d(d_size, d_size, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(d_size, 1, 1),
                )

    def _filter_instances(self, instances):
        if instances.has('presence'):
            presence = instances.presence.sigmoid() > 0.5
            instances = instances[presence]
        if instances.has('pred_boxes'):
            boxes = instances.pred_boxes
        else:
            boxes = instances.gt_boxes
        boxes = Boxes(boxes.tensor.clone())
        boxes.scale(1.0/self.prediction_stride, 1.0/self.prediction_stride)
        boxes.clip((self.pred_height, self.pred_width))
        mask = boxes.nonempty()
        instances = instances[mask]
        boxes = boxes[mask]
        instances.scaled_boxes = boxes
        return instances

    def _paste_gt_masks(self, instances):
        boxes = instances.scaled_boxes
        masks = instances.pred_masks
        roi_masks = ROIMasks(masks[:, 0, :, :])
        output_masks = roi_masks.to_bitmasks(boxes, self.pred_height,
                                self.pred_width, 0.5).tensor
        return output_masks

    def _paste_masks(self, instances, gt_instances):
        if self.training:
            boxes = gt_instances.pred_boxes
        else:
            boxes = instances.pred_boxes
        masks = instances.decoder_pred_masks
        pasted_masks = self._paste_feats(masks, boxes, (self.img_height, self.img_width))
        return pasted_masks, instances

    def _paste_feats(self, feats, boxes, image_shape):
        img_h, img_w = image_shape
        c = feats.size(1)
        N = len(feats)
        if N == 0:
            return feats.new_empty((0,c,) + image_shape, dtype=torch.uint8)
        if not isinstance(boxes, torch.Tensor):
            boxes = boxes.tensor
        device = boxes.device
        assert len(boxes) == N, boxes.shape 
        if device.type == "cpu" or torch.jit.is_scripting():
            # CPU is most efficient when they are pasted one by one with skip_empty=True
            # so that it performs minimal number of operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks, but may have memory issue
            # int(img_h) because shape may be tensors in tracing
            num_chunks = int(np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
            assert (
                num_chunks <= N
            ), "Default GPU_MEM_LIMIT in mask_ops.py is too small; try increasing it"
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        img_feats = torch.zeros(
            N, c, img_h, img_w, device=device, dtype=torch.float32
        )
        for inds in chunks:
            feats_chunk, spatial_inds = _do_paste_mask(
                feats[inds, :, :, :], boxes[inds], img_h, img_w, skip_empty=device.type == "cpu"
            )
            

            if torch.jit.is_scripting() or len(spatial_inds) == 0:  # Scripting does not use the optimized codepath
                img_feats[inds] = feats_chunk
            else:
                img_feats[inds, :, spatial_inds[0], spatial_inds[1]] = feats_chunk
        return img_feats

    def _discretize_depth(self, depths, new_dim):
        depths = depths.clamp(max=self.depth_buckets[-1])
        new_shape = [1 for _ in range(len(depths.shape))]
        new_shape.insert(new_dim, len(self.depth_buckets)-1)
        db1 = self.depth_buckets[:-1].view(new_shape)
        db2 = self.depth_buckets[1:].view(new_shape)
        depths = depths.unsqueeze(new_dim)
        disc_depths = ((depths >= db1)*(depths < db2)).float()
        return disc_depths

    def _build_gt(self, instances, gt_instances, gt_pan_seg_instances, gt_pan_seg):
        if self.use_gt_pan_seg:
            inst_boxes = gt_instances.pred_boxes
            gt_boxes = gt_pan_seg_instances.gt_boxes
            p_cl = gt_instances.pred_classes
            gt_cl = gt_pan_seg_instances.gt_classes
            class_mat = p_cl.unsqueeze(1) != gt_cl.unsqueeze(0)
            ious = pairwise_iou(inst_boxes, gt_boxes)
            match_scores = (1-ious)
            m = class_mat * (ious > 0.8)
            match_scores.masked_fill_(class_mat, float('inf'))
            match_scores = torch.cat([match_scores, torch.ones(len(class_mat), len(class_mat), device=match_scores.device)], dim=1)
            inst_indices, gt_indices = linear_sum_assignment(match_scores.cpu().detach())
            valid = gt_indices < len(gt_cl)
            inst_indices = inst_indices[valid]
            gt_indices = gt_indices[valid]
            gt_instances, gt_pan_seg_instances = gt_instances[inst_indices], gt_pan_seg_instances[gt_indices]
            

            p_id = instances.ids
            inst_id = gt_instances.ids

            pred_matches = torch.nonzero((p_id[:, None] == inst_id)*(p_id[:, None] != -1))
            pred_indices = pred_matches[:, 0]
            pred_inst_indices = pred_matches[:, 1]
            instances, gt_instances = instances[pred_indices], gt_instances[pred_inst_indices]
            gt_pan_seg_instances = gt_pan_seg_instances[pred_inst_indices]
            final_gt = torch.empty(self.pred_height, self.pred_width, device = pred_indices.device, dtype=torch.long).fill_(255)
            if self.prediction_stride != 1:
                gt_pan_seg = F.interpolate(gt_pan_seg[None, None].float(), scale_factor=(1.0/self.prediction_stride, 1.0/self.prediction_stride),
                                    mode='nearest').squeeze().long()
            gt_ps_ids = gt_pan_seg_instances.ids
            translated_ids = {id.item():idx+1 for idx,id in enumerate(gt_ps_ids)}
            for gt_val in torch.unique(gt_pan_seg):
                gt_val = gt_val.item()
                if gt_val == 255:
                    continue
                elif gt_val < self.num_stuff_classes and gt_val >= 0:
                    final_gt[gt_pan_seg == gt_val] = 0
                elif gt_val not in translated_ids:
                    final_gt[gt_pan_seg == gt_val] = 255
                else:
                    final_gt[gt_pan_seg == gt_val] = translated_ids[gt_val]

        else:
            p_id = instances.ids
            gt_id = gt_instances.ids
            matches = torch.nonzero((p_id[:, None] == gt_id)*(p_id[:, None] != -1))
            pred_indices = matches[:, 0]
            gt_indices = matches[:, 1]
            if self.max_inst_num is not None and len(pred_indices) > self.max_inst_num:
                randperm = torch.randperm(len(pred_indices), device=pred_indices.device)[:self.max_inst_num]
                pred_indices = pred_indices[randperm]
                gt_indices = gt_indices[randperm]
            instances, gt_instances = instances[pred_indices], gt_instances[gt_indices]

            final_gt = torch.zeros(self.pred_height, self.pred_width, device = pred_indices.device, dtype=torch.long)

            depths = gt_instances.depths
            _, inst_order = depths.sort(descending=True)
            gt_masks = self._paste_gt_masks(gt_instances)
            for inst_idx in inst_order:
                final_gt[gt_masks[inst_idx]] = 1+inst_idx
        return instances, gt_instances, final_gt

    def forward(self, instances, bg_logits, bg_depths, bg_depth_masks, gt_instances, gt_pan_segs, gt_pan_seg_instances,
                bg_feats, img_ids):
        results = []
        all_losses = []
        final_instances = []
        all_offset_losses = []
        for all_inst, bg_logit, bg_depth, bg_depth_mask, all_gt_inst, gt_pan_seg, all_gt_pan_seg_inst, bg_feat, img_id in \
                zip(instances, bg_logits, bg_depths, bg_depth_masks, gt_instances, gt_pan_segs, gt_pan_seg_instances,
                    bg_feats, img_ids):
            orig_bg_logit = bg_logit = bg_logit.unsqueeze(0)
            max_bg_depth = bg_depth.max()
            m = bg_depth == -1
            bg_depth.masked_fill_(m, bg_depth.max()+1)
            if self.prediction_stride != 1:
                if self.paste_full_rez_instances:
                    bg_depth_mode = 'bilinear'
                    bg_depth = F.interpolate(bg_depth.unsqueeze(0), scale_factor=1./self.prediction_stride,
                                                mode=bg_depth_mode, align_corners=True, recompute_scale_factor=False).squeeze(0)
                else:
                    bg_depth_mode = 'nearest'
                    bg_depth = F.interpolate(bg_depth.unsqueeze(0), scale_factor=1./self.prediction_stride,
                                                mode=bg_depth_mode, recompute_scale_factor=False).squeeze(0)
            if self.prediction_stride != 4:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    orig_bg_logit = bg_logit = F.interpolate(bg_logit, scale_factor=4/self.prediction_stride,
                                            mode='bilinear', align_corners=True)
                    
            #TODO: add possibility of using "GT" boxes/logits during training
            all_inst = self._filter_instances(all_inst)
            all_gt_inst = self._filter_instances(all_gt_inst)
            all_gt_pan_seg_inst = self._filter_instances(all_gt_pan_seg_inst)
            if self.training:
                all_inst, all_gt_inst, final_gt = self._build_gt(all_inst, all_gt_inst, all_gt_pan_seg_inst, gt_pan_seg)
            if self.training and len(all_inst) == 0:
                continue
            if self.use_depth_refine_model:
                bg_depth = bg_depth.min(0)[0][None, None]
                m = bg_depth > max_bg_depth
                bg_depth = (bg_depth - self.depth_mean)/self.depth_std
                bg_depth.masked_fill_(m, 0)
                orig_bg_depth = bg_depth.clone()

                # do depth processing
                bg_depth = torch.cat([bg_depth, bg_logit], dim=1)
                _, _, h,w = bg_depth.shape
                if self.use_bg_feats:
                    bg_feat = bg_feat.unsqueeze(0)
                    if bg_feat.size(-1) != w:
                        bg_feat = F.interpolate(bg_feat, size=(h, w), mode='bilinear', align_corners=False)
                    bg_depth = torch.cat([bg_depth, bg_feat], dim=1)
                bg_depth = self.bg_d1(bg_depth)
                d_sub = F.interpolate(bg_depth, size=(h//2, w//2), mode='bilinear', align_corners=False)
                d_sub = self.bg_sub(d_sub)
                d_sub = F.interpolate(d_sub, size=(h, w), mode='bilinear', align_corners=False)
                bg_depth = bg_depth + d_sub
                if self.use_depth_offset_bias:
                    d_offset = self.bg_d2(bg_depth)
                    bias = self.bg_d2_bias(bg_depth)
                    orig_bg_depth[m] = bias[m]
                    bg_depth = orig_bg_depth + d_offset
                else:
                    bg_depth = self.bg_d2(bg_depth)
                bg_depth = F.relu(bg_depth*self.depth_std + self.depth_mean) #TODO: CHECK SHAPE
            else:
                with torch.no_grad():
                    bg_depth = bg_depth.min(0)[0][None]
                    bg_depth = bg_depth.reshape(1, -1, self.pred_height, self.pred_width)
            
            bg_inp = self.static_model(bg_logit)
            if self.sub_batch_size is None:
                sb_size = len(all_inst)
            else:
                sb_size = self.sub_batch_size
            all_dyn_depth = []
            all_dyn_inp = []
            if sb_size > 0:
                for sb_idx in range(0, len(all_inst), sb_size):
                    inst = all_inst[sb_idx:sb_idx+sb_size]
                    gt_inst = all_gt_inst[sb_idx:sb_idx+sb_size]

                    fg_logit, inst = self._paste_masks(inst, gt_inst)
                    # prepare depths
                    if self.training and self.train_gt_depth:
                        fg_depth = gt_inst.depths
                    else:
                        fg_depth = inst.depths
                    with torch.no_grad():
                        if self.instance_depth_factor != 1.0:
                            fg_depth = fg_depth*self.instance_depth_factor
                        fg_depth = fg_depth.reshape(fg_depth.size(0), self.depth_size, 1, 1).expand(-1, -1, fg_logit.size(-2), fg_logit.size(-1)).clone()
                        fg_depth.masked_fill_(fg_logit < 0.5, 1000)

                    if self.paste_full_rez_instances:
                        fg_logit = F.interpolate(fg_logit, scale_factor=(1./self.prediction_stride, 1./self.prediction_stride),
                                                 mode='bilinear', align_corners=True, recompute_scale_factor=True)
                        fg_depth = F.interpolate(fg_depth, scale_factor=(1./self.prediction_stride, 1./self.prediction_stride),
                                                 mode='bilinear', align_corners=True, recompute_scale_factor=True)



                    fg_inp = self.dyn_model(fg_logit)
                    all_dyn_inp.append(fg_inp)
                    all_dyn_depth.append(fg_depth)
            final_inp_v = torch.cat([bg_inp, *all_dyn_inp])
            final_inp_kq = torch.cat([bg_depth, *all_dyn_depth])
            final_out = self.mha(final_inp_kq, final_inp_v).squeeze(1)
            if self.training:
                if self.use_focal_loss:
                    loss = utils.mc_focal_loss(final_out.unsqueeze(0), final_gt.unsqueeze(0), gamma=self.focal_loss_gamma, ignore_index=255)
                else:
                    loss = F.cross_entropy(final_out.unsqueeze(0), final_gt.unsqueeze(0), reduction="mean", ignore_index=255)
                all_losses.append(loss)
                if self.use_depth_offset_bias:
                    offset_loss = torch.square(d_offset).mean()
                    all_offset_losses.append(offset_loss)
            else:
                if self.prediction_stride != 1:
                    final_out = F.interpolate(final_out[None], scale_factor=(self.prediction_stride, self.prediction_stride),
                                            mode='bilinear', align_corners=True).squeeze(0)
                    final_bg_logit = F.interpolate(orig_bg_logit, scale_factor=self.prediction_stride,
                                            mode='bilinear', align_corners=True).squeeze(0)
                    if len(all_inst) > 0:
                        final_fg_logit = F.interpolate(fg_logit, scale_factor=(self.prediction_stride, self.prediction_stride),
                                                mode='bilinear', align_corners=True).squeeze(1)
                else:
                    final_bg_logit = orig_bg_logit.squeeze(0)
                    if len(all_inst) > 0:
                        final_fg_logit = fg_logit.squeeze(1)

                pred_inds = final_out.argmax(0)
                final_pred = final_bg_logit.argmax(0)
                if self.paste_full_rez_instances and self.trust_fg:
                    for inst_idx in range(len(all_inst)):
                        m = (pred_inds == inst_idx+1)*(final_fg_logit[inst_idx]>0.5)
                        final_pred[m] = self.num_stuff_classes + inst_idx
                else:
                    m = pred_inds > 0
                    final_pred[m] = pred_inds[m] + self.num_stuff_classes - 1
                results.append(final_pred)
                final_instances.append(all_inst)
                    
        if self.training:
            if len(all_losses) > 0:
                loss = torch.stack(all_losses).mean()
            else:
                loss = torch.tensor(0., device=bg_logit.device, dtype=torch.float32)
                for param in self.parameters():
                    loss = loss + 0*param.sum()
            loss_dict = {'e2e_refine/crossent': (loss, 1.0)}
            if self.use_depth_offset_bias:
                if len(all_offset_losses) > 0:
                    offset_loss = torch.stack(all_offset_losses).mean()
                else:
                    offset_loss = torch.tensor(0., device=bg_logit.device, dtype=torch.float32)
                loss_dict['e2e_refine/offset_loss'] = (offset_loss, self.depth_offset_coef)
            return loss_dict
        else:
            return results, final_instances



def _do_paste_mask(masks, boxes, img_h: int, img_w: int, skip_empty: bool = True):
    """
    Args:
        masks: N, 1, H, W
        boxes: N, 4
        img_h, img_w (int):
        skip_empty (bool): only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.
    Returns:
        if skip_empty == False, a mask of shape (N, img_h, img_w)
        if skip_empty == True, a mask of shape (N, h', w'), and the slice
            object for the corresponding region.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device

    if skip_empty and not torch.jit.is_scripting():
        x0_int, y0_int = torch.clamp(boxes.min(dim=0).values.floor()[:2] - 1, min=0).to(
            dtype=torch.int32
        )
        x1_int = torch.clamp(boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    if not torch.jit.is_scripting():
        if not masks.dtype.is_floating_point:
            masks = masks.float()
    img_masks = F.grid_sample(masks, grid.to(masks.dtype), align_corners=False)

    if skip_empty and not torch.jit.is_scripting():
        return img_masks, (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks, ()