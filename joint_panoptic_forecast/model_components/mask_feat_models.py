import torch
from torch import nn

from typing import Callable, Dict, List, Optional, Union, Tuple

from detectron2.layers import Conv2d, get_norm
from detectron2.modeling.poolers import ROIPooler
import fvcore.nn.weight_init as weight_init
from .. import utils

def build_mask_feat_encoder(cfg):
    type = cfg.MODEL.MASK_ENCODER.ENC_TYPE
    if type == 'feat':
        return MaskFeatEncoder()
    elif type == 'feat_simple':
        feat_size = cfg.MODEL.MASK_ENCODER.EMB_SIZE
        time_size = cfg.MODEL.MASK_ENCODER.TIME_SIZE
        return MaskFeatSimpleEncoder(feat_size, time_size)
    elif type == 'feat_conv':
        feat_size = cfg.MODEL.MASK_ENCODER.EMB_SIZE
        time_size = cfg.MODEL.MASK_ENCODER.TIME_SIZE
        kernel_size = cfg.MODEL.MASK_ENCODER.FEAT_KERNEL_SIZE
        return MaskFeatConvEncoder(feat_size, time_size, kernel_size)
    elif type == 'mask':
        feat_size = cfg.MODEL.MASK_ENCODER.EMB_SIZE
        time_size = cfg.MODEL.MASK_ENCODER.TIME_SIZE
        if time_size is None:
            time_size = feat_size
        return MaskEncoder(feat_size, time_size)
    else:
        raise ValueError('ENC TYPE NOT RECOGNIZED: ',type)



def build_mask_feat_decoder(cfg):
    type = cfg.MODEL.MASK_DECODER.ENC_TYPE
    if type == 'feat':
        return MaskFeatDecoder()
    elif type == 'feat_simple':
        feat_size = cfg.MODEL.MASK_DECODER.EMB_SIZE
        return MaskFeatSimpleDecoder(feat_size)
    elif type == 'feat_conv':
        feat_size = cfg.MODEL.MASK_DECODER.EMB_SIZE
        kernel_size = cfg.MODEL.MASK_DECODER.FEAT_KERNEL_SIZE
        num_layers = cfg.MODEL.MASK_DECODER.FEAT_NUM_LAYERS
        return MaskFeatConvDecoder(feat_size, kernel_size, num_layers)
    elif type == 'mask':
        feat_size = cfg.MODEL.MASK_DECODER.EMB_SIZE
        return MaskDecoder(feat_size)
    else:
        raise ValueError('DEC TYPE NOT RECOGNIZED: ',type)


class MaskEncoder(nn.Module):
    def __init__(self, feat_size, time_size):
        super().__init__()
        self.time_size = time_size
        self.mask_model = nn.Linear(28*28, feat_size)
        self.time_embed = nn.Linear(feat_size+time_size, feat_size)

    def forward(self, features=None, time=None, masks=None):
        del features
        b, t, _, _, _ = masks.shape
        mask = masks.view(b, t, 28*28)
        mask_emb = self.mask_model(mask)
        time_encoding = utils.get_batch_time_encoding(time, self.time_size)
        all_feats = torch.cat([mask_emb, time_encoding], dim=-1)
        return self.time_embed(all_feats)

class MaskFeatSimpleEncoder(nn.Module):
    def __init__(self, feat_size, time_size):
        super().__init__()
        self.time_size = time_size
        self.feat_model = nn.Linear(256*14*14, feat_size)
        self.time_embed = nn.Linear(feat_size+time_size, feat_size)

    def forward(self, features=None, time=None, masks=None):
        del masks
        b, t, _, _, _ = features.shape
        features = features.view(b, t, 256*14*14)
        emb = self.feat_model(features)
        time_encoding = utils.get_batch_time_encoding(time, self.time_size)
        all_feats = torch.cat([emb, time_encoding], dim=-1)
        return self.time_embed(all_feats)

class MaskFeatConvEncoder(nn.Module):
    def __init__(self, feat_size, time_size, kernel_size):
        super().__init__()
        self.feat_size = feat_size
        self.time_size = time_size
        self.feat_model = nn.Conv2d(256, feat_size, kernel_size, stride=1, padding=kernel_size//2)
        self.time_embed = nn.Conv2d(feat_size+time_size, feat_size, 1, stride=1,padding=0)

    def forward(self, features=None, time=None, masks=None):
        del masks
        b, t, c, h, w = features.shape
        features = features.view(b*t, c, h, w)
        emb = self.feat_model(features)
        time_encoding = utils.get_batch_time_encoding(time, self.time_size)
        time_encoding = time_encoding.view(b*t, self.time_size, 1).expand(-1, -1, h*w).view(b*t, self.time_size, h, w)
        all_feats = torch.cat([emb, time_encoding], dim=1)
        result = self.time_embed(all_feats)
        return result.view(b, t, self.feat_size, h, w)

class MaskFeatEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        conv_norm = None
        self.feat_convs = nn.Sequential(
            Conv2d(
                256, 512, kernel_size=3,
                stride=2, padding=1, bias=not conv_norm,
                norm=get_norm(conv_norm, 512),
                activation=nn.ReLU(),
            ),
            Conv2d(
                512, 1024, kernel_size=3,
                stride=2, padding=1, bias=not conv_norm,
                norm=get_norm(conv_norm, 1024),
                activation=nn.ReLU(),
            ),
            nn.AvgPool2d(4),
        )
        self.time_emb = nn.Linear(2048, 1024)

    def forward(self, features=None, time=None, masks=None):
        del masks
        b, t, c, h, w = features.shape
        features = features.view(b*t, c, h, w)
        feats = self.feat_convs(features)
        feats = feats.reshape(b, t, 1024)
        time_encoding = utils.get_batch_time_encoding(time, 1024)
        all_feats = torch.cat([feats, time_encoding], dim=-1)
        return self.time_emb(all_feats)



class MaskFeatDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.upconvs = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, stride=2, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, stride=2, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, stride=2, kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, feats_in):
        b, t, d = feats_in.shape
        feats_in = feats_in.view(b*t, d, 1, 1)
        result = self.upconvs(feats_in)
        return result.view(b, t, *result.shape[1:])

class MaskFeatSimpleDecoder(nn.Module):
    def __init__(self, feat_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(feat_size, feat_size),
            nn.ReLU(inplace=True),
            nn.Linear(feat_size, 256*14*14)
        )

    def forward(self, feats_in):
        b,t,d = feats_in.shape
        result = self.model(feats_in)
        return result.view(b, t, 256, 14, 14)

class MaskFeatConvDecoder(nn.Module):
    def __init__(self, feat_size, kernel_size, num_layers):
        super().__init__()
        if num_layers == 1:
            self.model = nn.Conv2d(feat_size, 256, kernel_size, padding=kernel_size//2)
        else:
            layers = []
            for l in range(num_layers - 1):
                layers.append(nn.Conv2d(feat_size, feat_size, kernel_size, padding=kernel_size//2))
                layers.append(nn.ReLU())
            layers.append(nn.Conv2d(feat_size, 256, kernel_size, padding=kernel_size//2))
            self.model = nn.Sequential(*layers)

    def forward(self, feats_in):
        b, t, c, h, w = feats_in.shape
        feats_in = feats_in.view(b*t, c, h, w)
        return self.model(feats_in).view(b, t, 256, h, w)

class MaskDecoder(nn.Module):
    def __init__(self, feat_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(feat_size, feat_size),
            nn.ReLU(inplace=True),
            nn.Linear(feat_size, 28*28)
        )

    def forward(self, feats_in):
        b,t,d = feats_in.shape
        result = self.model(feats_in)
        return result.view(b, t, 1, 28, 28)