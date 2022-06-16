import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional, Union, Tuple

from detectron2.layers import Conv2d, get_norm
from detectron2.modeling.poolers import ROIPooler
import fvcore.nn.weight_init as weight_init
from detectron2.structures import Instances, Boxes, pairwise_iou, ImageList


from .. import utils

class BoxVizTimeEncoder(nn.Module):
    def __init__(self, use_cwh, in_features,
                 emb_dim, input_shape, use_odom, box_mean, box_std, no_viz,
                 viz_dropout,
                 box_embed_size, use_depths,
                 use_relus=False, use_class_inp=False,
                 class_emb_size=32,
                 odom_size=5, use_cl_mask_emb=False, cl_inp_type='one_hot'):
        super().__init__()
        self.use_cwh = use_cwh
        if use_cwh:
            self.register_buffer("box_mean", torch.Tensor(box_mean), False)
            self.register_buffer("box_std", torch.Tensor(box_std), False)
        self.in_channels = in_channels = [input_shape[f].channels for f in in_features][0]
        self.use_odom = use_odom
        self.synth_mode = True
        self.no_viz = no_viz
        self.use_depths = use_depths
        self.use_class_inp = use_class_inp
        self.use_cl_mask_emb = use_cl_mask_emb
        self.cl_inp_type = cl_inp_type
        assert cl_inp_type in ['one_hot', 'emb']
        box_size = 4
        if self.use_depths:
            box_size += 2
        if self.use_class_inp:
            if self.cl_inp_type == 'one_hot':
                box_size += 8
            elif self.cl_inp_type == 'emb':
                self.class_emb = nn.Embedding(8, class_emb_size)
                box_size += class_emb_size
        if box_embed_size is not None:
            if use_relus:
                self.box_embed = nn.Sequential(
                    nn.Linear(box_size, box_embed_size),
                    nn.ReLU(inplace=True),
                )
            else:
                self.box_embed = nn.Linear(box_size, box_embed_size)
            box_size = box_embed_size
        else:
            self.box_embed = None
        if not no_viz:
            
            cur_channels = in_channels
            conv_dims = [256, 256, 1]
            conv_norm = "" #TODO: what does mask head use?
            feat_convs = []
            for k, conv_dim in enumerate(conv_dims[:-1]):
                conv = Conv2d(
                    cur_channels,
                    conv_dim, kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not conv_norm,
                    norm=get_norm(conv_norm, conv_dim),
                    activation=nn.ReLU(),
                )
                feat_convs.append(conv)
                cur_channels = conv_dim
            all_convs = feat_convs
            conv_size = conv_dims[-2]
            if use_relus:
                self.pool = nn.Sequential(
                    nn.AvgPool2d(14),
                    nn.ReLU(inplace=True),
                )
            else:
                self.pool = nn.AvgPool2d(14)
            if viz_dropout is not None:
                self.viz_dropout = nn.Dropout(viz_dropout)
            else:
                self.viz_dropout = None

            for layer in all_convs:
                weight_init.c2_msra_fill(layer)
            self.feat_convs = nn.Sequential(
                *feat_convs,
            )
            inp_size = box_size + conv_size
        else:
            inp_size = box_size
        if self.use_odom:
            inp_size += odom_size
        
        self.emb_l1 = nn.Linear(inp_size, emb_dim)
        self.emb_l2 = nn.Linear(2*emb_dim, emb_dim)
        self.emb_dim = emb_dim

    def forward(self, features, boxes: torch.Tensor, depths: torch.Tensor, depth_masks, ids: List[torch.Tensor], mask: torch.Tensor, time:torch.Tensor,
                odom: torch.Tensor,
                img_w, img_h, classes=None):
        # boxes: [b, max_len, 4]
        
        if self.use_cwh:
            boxes = utils.bbox_ulbr2cwh(boxes)
            boxes = utils.normalize_box_tensor_by_stats(boxes, self.box_mean, self.box_std)
        boxes = utils.normalize_box_tensor(boxes, img_w, img_h)
        if self.use_depths:
            boxes = torch.cat([boxes, depths, depth_masks.float()], dim=-1)
        if self.use_class_inp:
            if self.cl_inp_type == 'one_hot':
                oh_classes = F.one_hot(classes, 8)
                boxes = torch.cat([boxes, oh_classes], dim=-1)
            elif self.cl_inp_type == 'emb':
                cl_emb = self.class_emb(classes)
                boxes = torch.cat([boxes, cl_emb], dim=-1)
        if self.box_embed is not None:
            boxes = self.box_embed(boxes)
        if self.use_odom:
            boxes = torch.cat([boxes, odom], dim=-1)
        if not self.no_viz:
            all_viz_feats = features
            b, t, _, h, w = all_viz_feats.shape
            all_viz_feats = all_viz_feats.reshape(b*t, self.in_channels, h, w)
            all_viz_feats = self.feat_convs(all_viz_feats)
            all_viz_feats = self.pool(all_viz_feats).reshape(b, t, all_viz_feats.size(1))
            if self.viz_dropout is not None:
                all_viz_feats = self.viz_dropout(all_viz_feats)
            all_feats = torch.cat([boxes, all_viz_feats], dim=-1)
        else:
            all_feats = boxes
        
        all_feats = self.emb_l1(all_feats)
        time_encoding = utils.get_batch_time_encoding(time, self.emb_dim)
        all_feats = torch.cat([all_feats, time_encoding], dim=-1)
        return self.emb_l2(all_feats)



    
def id2one_hot(ids, max_len):
    inp_shape = ids.shape
    ids = ids.reshape(-1)
    result = torch.zeros(len(ids), max_len, device=ids.device)
    valid = ids != -1
    result[torch.arange(len(ids))[valid], ids[valid]] = 1
    result = result.reshape(*inp_shape, max_len)
    return result


class BoxTimeEncoder(nn.Module):
    def __init__(self, emb_dim, use_odom,
                 use_depths,
                 use_relus, use_class_inp=False, class_emb_size=32, 
                 use_viz_feats=False, box_embed_size=None, input_shape=None, in_features=None,
                 use_depth_mask=False, odom_size=5,
                 cl_inp_type='one_hot'):
        super().__init__()
        self.use_odom = use_odom
        self.use_depths = use_depths
        self.use_class_inp = use_class_inp
        self.emb_dim = emb_dim
        self.use_viz_feats = use_viz_feats
        self.use_depth_mask = use_depth_mask
        self.cl_inp_type = cl_inp_type
        assert cl_inp_type in ['one_hot', 'emb']
        inp_size = 4
        if self.use_depths:
            inp_size += 1
            if self.use_depth_mask:
                inp_size += 1
        if self.use_class_inp:
            if self.cl_inp_type == 'one_hot':
                inp_size += 8
            elif self.cl_inp_type == 'emb':
                self.class_emb = nn.Embedding(8, class_emb_size)
                inp_size += class_emb_size
        if box_embed_size is not None:
            if use_relus:
                self.box_embed = nn.Sequential(
                    nn.Linear(inp_size, box_embed_size),
                    nn.ReLU(inplace=True),
                )
            else:
                self.box_embed = nn.Linear(inp_size, box_embed_size)
            inp_size = box_embed_size
        else:
            self.box_embed = None
        if self.use_odom:
            inp_size += odom_size
        if self.use_viz_feats:
            self.in_channels = in_channels = [input_shape[f].channels for f in in_features][0]
            cur_channels = in_channels
            conv_dims = [256, 256, 1]
            conv_norm = "" #TODO: what does mask head use?
            feat_convs = []
            for k, conv_dim in enumerate(conv_dims[:-1]):
                conv = Conv2d(
                    cur_channels,
                    conv_dim, kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not conv_norm,
                    norm=get_norm(conv_norm, conv_dim),
                    activation=nn.ReLU(),
                )
                feat_convs.append(conv)
                cur_channels = conv_dim
            all_convs = feat_convs
            conv_size = conv_dims[-2]
            if use_relus:
                self.pool = nn.Sequential(
                    nn.AvgPool2d(14),
                    nn.ReLU(inplace=True),
                )
            else:
                self.pool = nn.AvgPool2d(14)

            for layer in all_convs:
                weight_init.c2_msra_fill(layer)
            self.feat_convs = nn.Sequential(
                *feat_convs,
            )
            inp_size += conv_size
        
        self.l1 = nn.Linear(inp_size, emb_dim)
        self.l2 = nn.Linear(2*emb_dim, emb_dim)
        


    def forward(self, features: List[Dict[str, torch.Tensor]], boxes: torch.Tensor,
                odom: torch.Tensor, img_w, img_h,
                time_offset: int = 0, classes=None):
        b, n, _ = boxes.shape
        if self.use_depth_mask:
            boxes = torch.cat([boxes, torch.ones(*boxes.shape[:-1], 1, device=boxes.device)], dim=-1)
        if self.use_class_inp:
            if self.cl_inp_type == 'one_hot':
                oh_classes = F.one_hot(classes, 8)
                boxes = torch.cat([boxes, oh_classes], dim=-1)
            elif self.cl_inp_type == 'emb':
                cl_emb = self.class_emb(classes)
                boxes = torch.cat([boxes, cl_emb], dim=-1)
        if self.box_embed is not None:
            boxes = self.box_embed(boxes)
        if self.use_odom:
            boxes = torch.cat([boxes, odom], dim=-1)
        if self.use_viz_feats:
            all_viz_feats = features
            b, t, _, h, w = all_viz_feats.shape
            all_viz_feats = all_viz_feats.reshape(b*t, self.in_channels, h, w)
            all_viz_feats = self.feat_convs(all_viz_feats)
            all_viz_feats = self.pool(all_viz_feats).reshape(b, t, all_viz_feats.size(1))
            boxes = torch.cat([boxes, all_viz_feats], dim=-1)
        box_feats = self.l1(boxes)
        times = torch.empty(b, n, device=boxes.device).fill_(time_offset)
        time_encoding = utils.get_batch_time_encoding(times, self.emb_dim)
        box_feats = torch.cat([box_feats, time_encoding], dim=-1)
        return self.l2(box_feats)


