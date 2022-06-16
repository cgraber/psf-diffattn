import torch
from torch import nn
from typing import List

from detectron2.structures import Instances
from detectron2.modeling.roi_heads.mask_head import ROI_MASK_HEAD_REGISTRY, mask_rcnn_loss, MaskRCNNConvUpsampleHead as OldMaskHead



def mask_rcnn_inference(pred_mask_logits: torch.Tensor, pred_instances: List[Instances], name_prefix=""):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.

    This version is the same as the detectron2 version (https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/mask_head.py),
    except it also stores mask logits

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".
    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1

    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
    else:
        # Select masks corresponding to the predicted classes
        num_masks = pred_mask_logits.shape[0]
        class_pred = torch.cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_logits = pred_mask_logits[indices, class_pred][:, None]
        mask_probs_pred = mask_probs_logits.sigmoid()
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)
    mask_probs_logits = mask_probs_logits.split(num_boxes_per_image, dim=0)

    if len(name_prefix) > 0:
        m_name = name_prefix + '_pred_masks'
        l_name = name_prefix + '_pred_mask_logits'
    else:
        m_name = 'pred_masks'
        l_name = 'pred_mask_logits'
    for prob, instances, logits in zip(mask_probs_pred, pred_instances, mask_probs_logits):
        instances.set(m_name, prob)  # (1, Hmask, Wmask)
        instances.set(l_name, logits)


del ROI_MASK_HEAD_REGISTRY._obj_map['MaskRCNNConvUpsampleHead']
# We are extending this class because we want access to mask logits during inference
@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNConvUpsampleHead(OldMaskHead):

    def forward(self, x, instances: List[Instances], inference_mode=False, name_prefix=''):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.
        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        x = self.layers(x)
        if self.training and not inference_mode:
            return {"loss_mask": mask_rcnn_loss(x, instances, self.vis_period) * self.loss_weight}
        else:
            mask_rcnn_inference(x, instances, name_prefix=name_prefix)
            return instances